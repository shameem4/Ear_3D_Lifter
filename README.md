# Ear 3D Lifter

A head scan in, landmarks in 3D out: 55 per ear, and MediaPipe's 478 on the
face. It runs in Python, or entirely in a browser tab:
**[try it](https://shameem4.github.io/Ear_3D_Lifter/)** on HUTUBS heads,
AudioEar3D ears, or your own scan.

## The story

### It started with ears

A 2D model, [Ear_Landmarker](https://github.com/shameem4/Ear_Landmarker),
finds a 55-point ear scheme in photographs. The question was whether those
points could be put on a 3D head scan without building a 3D model. The plan
was simple: render the ear from many angles, read each picture with the 2D
model, and intersect the rays from each view. Each landmark is wherever the
rays cross.

### First, something to measure against

Before anything could be improved, it had to be measured, and the obvious
measurements were wrong. Projecting 2D landmarks back onto the scan puts
them on the scalp, and "distance to the scan" scores a point on the scalp as
a success. So the ground truth was built another way: one template ear,
marked once, fitted to every ear in the corpus with Faceform Wrap (to
0.002 mm). The template's marks, carried onto each fit, became the 3D
reference. The corpus is 476 head ears from HUTUBS and SONICOM and 112
AudioEar3D ears, all in one frame. Fitting them was its own project: every
fit folds a little and no single setting fixes it, so a per-ear escalation
ladder took the corpus to 98.3% fold-free.

### Lifting alone was not enough

Intersecting rays gave points, but too many were wrong. Almost every fix
that worked used knowledge of what a head and an ear look like, not better
geometry:

- **Find the head the right way up.** FaceMesh never reports an upside-down
  face. Shown one, it puts the forehead on the chin, 17 times out of 17. A
  nose-profile check catches that, and three anatomical checks follow: skull
  above neck, mirror symmetry, ears at a plausible distance from the nose.
  Gross orientation errors across the corpus went from 8 to 0.
- **Make the renders look like the training images.** A bare grey mesh loses
  64.5% of the detections, so every mesh gets synthetic skin. The ear is
  framed to fill the picture as it did in training. One lighting bug shaded
  every right ear from the inside, and a lighting rule had been fitted to
  cancel it; both were removed.
- **Not every view can be trusted for every landmark.** A table learned on
  the corpus picks each landmark's best 8 of 25 views. Error went from 2.70 to
  2.24 mm, and gross misses past 10 mm from 1.04% to 0.065%. Eight random views
  are worse than all 25, so the gain is the choice of views. The table is
  stable across disjoint subjects (Spearman 0.94): it describes the ear's
  geometry, not the training set.
- **Points belong on the skin.** Each one is projected back onto the surface,
  always: any tolerance made it worse.

Then the decisive experiment. Fed the ground truth's own 2D positions, the
lift is accurate to **0.36 mm** and never misses by 20 mm. Fed the 2D
model's predictions, it is 3.73 mm, with 6% of points past 20 mm. **The
remaining error was the 2D model, not the lift.** So the 2D model was
retrained on this repository's own renders:
- render-only training took the held-out error from 1.61 to about 1.05 mm;
- synthetic earrings fixed a real hoop earring (worst point 7.6 to 3.3 mm);
- renders of the ear without its head fixed ear-only scans, where the helix
  root had been missed by 5 mm (now about 2 mm).

The ears now land at a median of **1.03 mm** from hand-corrected landmarks on
32 held-out ears, each scan randomly rotated, with 0.62% of points beyond
5 mm.

### Then the face

A sister project lifted MediaPipe's 478 face points the same way: 33 views
around the face, FaceMesh in each, rays intersected per landmark. It was
merged in, so one run now produces both ears and the face. Its ground truth
is MediaPipe's canonical face, wrapped onto each head with hand-placed
control points. The face taught the same lesson more sharply. Points lifted
one at a time were roughly right but wrong as a face:

- **Each view has a consistent 2D bias.** Per-view offsets learned on the
  training heads are subtracted first: 4.21 to 2.34 mm.
- **Neighbours bunched or spread apart,** because each point settled on its
  own set of views. Fitting all 468 points at once, held to the shape of the
  canonical face mesh, keeps the spacing.
- **FaceMesh puts hidden nostrils on the upper lip** in every view where they
  can't be seen (5.4 mm on one head), so no vote between views can fix it.
  Fitting the canonical face to the scan's own nose shape puts the nose base
  back: 1.58 to 1.07 mm.
- **Mouth and eye corners still collapsed,** closed lips most of all.
  Springs that keep each edge near its typical length took collapsed
  mouth-corner edges from 11% to 0.2%, at no cost in accuracy.

The face lands at about **2.05 mm**, leaving one head out of 50. That
reference is itself a canonical-face fit, so it flatters these methods. The
only independent check is one real scan with anatomical stickers: the four
points FaceMesh defines (nose tip, nasal bridge, glabella, chin) are 0.4 to
2.9 mm off.

The general rule: **triangulation finds roughly where a landmark is;
knowledge of what an ear or a face looks like decides the rest.**

### How the numbers lied

Several confident results were wrong, and each left a rule:

| What happened | Rule |
|---|---|
| A landmark thrown off the head has no distance (NaN), and `d > gate` is False for NaN, so it counted as a success | A NaN distance is a miss |
| Right ears were rendered inside out, and a lighting rule was fitted to cancel the symptom | Never fit a correction to a symptom |
| A left/right average cancelled a real side bias to zero | Split by side before averaging |
| Snapping points to the surface made a surface metric meaningless | Score a snap by distance to the reference |
| A transposed box produced two false conclusions | Draw the geometry on its image first |
| A new reference moved the score more than a model change did | Compare models on one fixed reference |
| The same 2D checkpoint gave different results a day apart | The inference code moves too; re-run the baseline |

Each of these would have been caught by rendering the result and looking at
it. That is now the first check, before any number is reported.

### What did not work

- **Learning depth from a single photograph.** Five learned approaches all
  lost to assuming zero depth on real photographs. What did work: solving the
  pose geometrically from a mean ear, and a general depth model (MoGe-2)
  read at the 2D model's points.
- **Forbidding rays from passing through the ear.** It made held-out heads
  worse (0.99 to 1.07 mm): folds legitimately hide landmarks from their best
  views.
- **The 2D model's confidence as a view filter.** It has no relation to
  error (Spearman -0.018).
- **Wrapping the canonical face onto lifted key points.** It looked better
  against the canonical-fit reference, which shares its shape, and worse on
  the real scan's stickers.

### Where this is going

**This is a stopgap.** Rendering a scan and reading it in 2D is a detour.
The goal is direct 3D landmarking: a model that takes the 3D scan itself and
places the landmarks on it. That model needs training data, many scans with
their landmarks, and no such dataset exists for this scheme. This repository
is the intermediate step that builds it: lifting 2D landmarks onto scans to
produce the 3D labels a direct model can learn from.

## Use it

    python scripts/run_pipeline.py head.obj [out.npz]          # ears
    python scripts/face/lift.py head.ply [out.npz]             # face

or in Python, `Pipeline().run(mesh)` from `ear3d.pipeline` and
`FaceLifter().run(mesh)` from `ear3d.face`.

- **Input:** a head scan in millimetres, in any orientation, or an ear-only
  scan. A mesh under 100 mm on its longest side is treated as one ear: its
  pose and side come from fitting the mean ear, and it gets no face stage.
- **Output:** `L` and `R` (55 x 3 each) and the face's 478 x 3, in the input
  mesh's own coordinates, NaN where nothing was found (`L` is the subject's
  left).
- **Speed:** about 15 s per head for the ears on one GPU. In the browser, both
  ears and the face take about 17-20 s.
- **Environment:** the `mocap` conda env, with `Ear_Landmarker` beside this
  repository (or `$EAR_LANDMARKER`), `BlazeEar` beside it (or `$BLAZEEAR`),
  and MediaPipe's `face_landmarker.task` (`$FACE_LANDMARKER_TASK`).

**Limits.** Validated only on clean HUTUBS and SONICOM scans: hair, holes,
partial heads and other scanners are untested. The iris and eye region are
**unverified**: none of the scans we can use show the eyes in enough detail.
The 2D ear model was trained only on renders: never point it at a photograph
or a real texture.

## How it works

    head scan
      -> orient: FaceMesh face frame, checked (skull above neck, symmetry,
         nose-ear distance), retried turned if a check fails
      -> ears, each: BlazeEar box, framed as in training, 25 views,
         Ear_Landmarker in each, every landmark triangulated from its best
         8 views, RANSAC, re-seated on the surface
      -> face: 33 views, FaceMesh in each, per-view offsets removed,
         triangulated, then the joint fit to the canonical face, the
         nose-base fit to the scan, and the spacing springs

`ear3d/pipeline.py` and `ear3d/face.py` give each step's setting with the
measurement behind it. The same pipeline is also written in Rust (`core/`),
which is what the browser runs.

**Building the data behind it:**
- `scripts/corpus` (steps 01-12) builds a corpus generation: fits the
  template to every ear, composes each fit's transforms, and derives the
  reference landmarks from the fits.
- `scripts/caches` (steps 1-4) renders every corpus ear from the 25 views,
  reads them with the 2D model, and learns the view table.
- `scripts/face` lifts the training heads, builds the face's per-view
  offsets, and scores the face lift leaving one head out.
- `scripts/eval` runs the held-out heads and scores them.

Each folder's README has the exact commands. `EAR3D_CORPUS` selects a corpus
generation (`lmw` by default): fits, reference landmarks and view table
change together and must never be mixed. The 2D model's inference code
changes in its own repository, so re-run a baseline before comparing results
from different days.

## Layout

    ear3d/              the library
      pipeline.py         head mesh -> ear landmarks
      face.py             head mesh -> face landmarks
      twod.py             the only coupling to Ear_Landmarker
      frames, render, ear_view, triangulate, camera, skin, ...
      corpus.py, data.py, gt.py   corpus generations and reference landmarks
      faceform.py         drives Faceform Wrap for the corpus fits
    core/               the pipeline in Rust, for the browser
    demo/               the browser demo
    scripts/            corpus/, caches/, face/, eval/, run_pipeline.py
    tests/

`FINDINGS.md` is the full research record behind this story, with every
measurement and its date. `RESULTS_inherited.md` is the earlier effort's
chronological log.

## Licences and credits

No scan data is committed. Scans, fits and photographs stay local, and their
licences travel with them. The demo's data is published separately, as a
release asset.

- **HUTUBS** (Brinkmann et al. 2019, JAES 67(9)), CC BY 4.0: the demo's
  heads, and the source of the view table and the face's view offsets.
- **AudioEar3D** (Huang et al. 2023, "AudioEar: Single-View Ear
  Reconstruction for Personalized Spatial Audio", AAAI), CC BY 4.0: the
  demo's ears, modified (our template wrapped onto each scan). Their
  photographs are not used, and AudioEar2D is not used at all (it derives
  from FFHQ, which is non-commercial).
- **SaifulULL's U-Net** (CC BY 4.0): the 2D ear landmarker builds on it.
- **BlazeEar** (Apache-2.0): the ear box. An Ultralytics YOLO box is also
  supported, but it is AGPL-3.0: local use only.
- **MediaPipe** FaceLandmarker and canonical face model (Apache-2.0).
- Anything released should use the HUTUBS-only (`ccby`) shape model.
