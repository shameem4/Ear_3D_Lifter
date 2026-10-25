# Wrapping an ear template onto AudioEar3D point clouds (custom NICP)

Audience: an agent fitting a fixed-topology template mesh to partial,
single-sided point clouds of a body part, where commercial wrappers (which
need meshes) do not apply. Written 2026-10-18 from `ear3d/wrap.py`,
`archive-final:archive/scripts/wrap_cloud.py` and `archive-final:archive/scripts/wrap_cloud_adaptive.py`. Numbers are
measured on this project; hypotheses are marked.

> **Status in this repo (2026-10-20).** This NICP method made `ae3d_v2`, the
> AudioEar fits of the previous corpus generation (esc). The current ones,
> `ae3d_v3`, come from a different route: Poisson-mesh the cloud, run the
> landmark pipeline on that mesh, pose the template from those landmarks and
> let Faceform Wrap fit it (`scripts/corpus/03_rewrap_audioear.py`), keeping the
> old fit when the landmarks collapse or the new fit fails its gates (cloud
> coverage among them). The write-up stands as
> a guide to fitting clouds without a commercial wrapper.

## 1. The problem

AudioEar3D gives ear-only point clouds with per-point normals: front shells,
no head, holes behind the helix and lobe, and a hard scan border. The goal is
one shared template topology (1188-vertex left ear) on every ear, so ears can
enter a PCA / landmark corpus alongside mesh-derived fits.

Key idea: **non-rigid registration does not need faces.** Correspondence is a
nearest-neighbour query and the only other thing a mesh gives is normals,
which the cloud already has. So the project's own NICP runs directly on the
cloud. Where the cloud has points they drive the fit; where it has none,
every correspondence is rejected and the template's own shape, carried by
stiffness, fills the gap smoothly. That is warping, not registration, and it
fills a hole with something ear-shaped instead of a dent.

## 2. Pipeline (code description)

### 2a. Load and mirror
- Read points and normals from the PLY's raw vertex block (normals are not in
  the standard vertex array). Refuse if counts differ.
- Right ears → left form by mirroring **points and normals about Z** (project
  convention; X is wrong here). Outputs stay in that mirrored frame.

### 2b. Rigid stage: pre-align, then rigid ICP without scale
1. **Pre-align by extents:** scale the template so its largest extent matches
   the cloud's, and move its centroid onto the cloud's. Mandatory — ICP from
   the raw placement matched helix to concha on the first pass and never
   recovered (came back at scale 0.75, template shrunk inside the ear).
2. **Rigid ICP, rotation + translation only** (`rigid_icp`, `with_scale=False`):
   - each iteration: nearest cloud point per template vertex, bounded by
     `max_dist = 0.3 × template size`;
   - reject matches whose normals disagree (cos ≤ 0.5);
   - keep the closest 80 % (trim);
   - Kabsch on the survivors; stop when the step is tiny.
   - **Why no scale:** the template is a cut ear whose border runs onto scalp
     and cheek; the cloud is pinna only. Border vertices with no data under
     them pull a scaled fit inward — measured 0.879, a 54.8 mm template inside
     a 60.6 mm ear. The extent pre-scale is the honest size estimate; NICP
     absorbs the rest.
   - The distance bound must be a fraction of the **template** size, set after
     pre-scaling. Called on an un-prescaled template the bound was ~50× the
     ear and the guard did nothing (fits settled 38–52 % of ear size away).

### 2c. Find the cloud's edges (`boundary_points`)
- For each point: take 30 neighbours, project them into the point's tangent
  plane (from its normal), sort by angle, find the largest angular gap. Gap
  > 120° ⇒ the point is on an edge (hole rim or scan border).
- These points are **never accepted as correspondences.** This is the single
  most important trick for clouds: template vertices over a hole still find
  the rim points at sensible distances with agreeing normals, pass every
  other test, and drag the template down into the hole.

### 2d. Non-rigid stage: NICP (`nicp`, after Amberg et al. 2007, optimal-step NICP)
- **Unknowns:** one 3×4 affine per template vertex.
- **Energy per iteration:** stiffness `α · ‖(M ⊗ G) X‖²` (M = edge incidence
  matrix, G = diag(1,1,1,γ)) + data `‖W (D X − U)‖²`. D holds the **original**
  template vertices `[v, 1]`, never the deformed ones (rebuilding D from the
  current shape composes the deformation with itself and runs away). U holds
  the matched cloud points; W the 0/1 weights.
- **Weights = 0 when:** distance > `max_dist` (0.06 × template size by
  default), normal cos ≤ `min_cos`, the matched point is an edge point, or the
  vertex is on the template's own cut border (border is carried by stiffness
  alone).
- **Solve:** stack into one tall sparse system, normal equations, sparse
  direct solve (A^T A is only ~5k square); small ridge 1e-9 to keep it
  positive definite. Up to 12 iterations per stage, stop early on a small step.
- **Stiffness schedule:** α = 50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05. The
  loose tail is what lets the template enter the concha and follow the helix
  fold — and also what lets it buckle (§2e).
- **Work in a normalised frame:** centre on the template and divide by its
  extent before solving, undo after. The `[v, 1]` design means γ (translation
  vs linear part) means something different at every scale; the same mesh in
  metres and in millimetres fitted differently, and a corpus run in
  head-normalised units came out mangled.
- Optional `anchors` (template vertex ↔ target point pairs, weight β decaying
  with √(α/α₀)) steer the coarse stages and hand over to the surface as the
  template loosens. Not needed for AudioEar3D.
- If fewer than 4 correspondences survive, raise: the template is not on the
  cloud.

### 2e. Fold control: the adaptive ladder (`wrap_cloud_adaptive.py`)
Run NICP with escalating settings; stop at the first fold-free result,
otherwise keep the fewest-folds fit (cloud distance breaks ties):

| rung | min_cos | schedule |
|---|---|---|
| default | 0.50 | full |
| cos80 | 0.80 | full |
| cos90 | 0.90 | full |
| cos80_t0.2 | 0.80 | truncated at 0.2 |
| cos90_t0.5 | 0.90 | truncated at 0.5 |
| cos90_t1.0 | 0.90 | truncated at 1.0 |

- Raise `min_cos` **first**: it removes bad matches (the far face of a thin
  rim) without removing freedom. Truncating the schedule removes freedom
  directly and costs concha/helix detail — last resort.
- The rigid stage is independent of the ladder; run it once per ear.
- Fold = adjacent faces > 90° apart. Score folds **relative to the template's
  rest angles** when the template has sharp features of its own.
- Parallelise across ears (process pool); each fit is CPU-bound sparse
  algebra.

### 2f. Report support, not gap
- Per vertex, classify: **measured** (a valid, non-edge, normal-agreeing cloud
  point within range), or carried by the template because of: cut border / no
  point in range / nearest point on a rim / normal disagrees.
- Report the measured fraction and how far the filled parts bridge empty
  space. **The gap where measured is ~0 by construction** (the loose final
  stage interpolates the points), so it says nothing about whether the fit
  found the right anatomy.
- Save the support mask beside the fit (`*.support.npy`); downstream, weight
  or mask landmarks by it.

## 3. Results on AudioEar3D

- Template matters most: with a template that had 39 folds of its own
  (ear_mean_v2) the ladder got 0/10 fold-free, mean 42.5 folds — it was
  inheriting the template's folds. With a fold-free template (v3): **113/113
  fold-free**, cloud→surface 0.31 → 0.22, support 0.40 → 0.44.
- Support is the real limit: on average **40.5 of 55 landmarks per ear are
  measured**; the lobe (32–45 %), inferior crus (33 %) and ascending helix
  (46 %) are mostly template-filled — where the ear curls back and the
  scanner saw nothing.
- An apparent "AudioEar ears are flat in depth" result was a frame artefact
  (span measured on a raw axis 23° off the depth axis). Aligned, depth spans
  match the head datasets (0.352 vs 0.355 / 0.359).

## 4. Do's and don'ts

Do
- Pre-scale and centre before any ICP; set distance bounds as fractions of
  the template size after pre-scaling.
- Reject cloud edge points as correspondences.
- Free the template's cut border (weight 0) so it follows stiffness.
- Normalise coordinates before NICP.
- Use a fold-free template; check the template before blaming the data.
- Track support per vertex and carry it into anything downstream.
- Render every fit **face-on from outside** (camera from the fit's mean
  normal), cloud beside fit, with template-filled vertices coloured. Numbers
  support the visual; several confident wrong conclusions here were caught
  only by a render.
- Run batches under a memory/time guard; snapshot the as-run script first.

Don't
- Don't fit scale in the rigid ICP when the template covers more than the
  cloud does.
- Don't let `max_dist` scale with the scan — it lets the rim grab whatever is
  behind it.
- Don't rebuild the NICP design matrix from deformed vertices.
- Don't report median gap as fit quality on a cloud.
- Don't truncate the stiffness schedule before trying a stricter normal test.
- Don't mirror right ears about X here; mirror normals together with points.
- Don't compare datasets on a raw axis; align frames first.

## 5. Per-ear checklist

1. Points and normals loaded, counts equal; right ears mirrored (Z).
2. Pre-scaled + centred; rigid ICP converged, scale untouched.
3. Edge points found (a few % of the cloud is typical).
4. NICP ran all stages without "every correspondence rejected".
5. Folds 0 (or least on the ladder); rung recorded.
6. Support fraction and per-landmark support saved.
7. Face-on render reviewed.
