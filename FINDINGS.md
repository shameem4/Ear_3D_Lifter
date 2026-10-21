# Findings

The research record: what has been measured, what was decided because of it,
the measurement mistakes that produced wrong answers, and what is closed or
open. Items marked *(inherited)* come from the effort that ran before this
repository split from Ear_Landmarker on 2026-10-13; the rest are this
project's. Read inherited items as prior work, not decisions: if a closed
direction looks worth reopening, the evidence is here to argue with.

The long form of the inherited work is `RESULTS_inherited.md` (chronological,
with its retractions; grep it). Numbers carry the date and the reference they
were measured against, because both have changed.

## Where it stands (2026-10-20)

- **Held-out accuracy: 1.46 mm** median per ear vs lm3d_v12 (view_table_v12;
  1.56 with v11's table), 16 HUTUBS heads
  used in no table or reference, each randomly rotated and unskinned; p95
  2.09 mm; 4.38% of points beyond 5 mm, concentrated on the upper helix (0-8)
  and the lobe (15-19). 2D checkpoint v4_fc05_s42 epoch 434 (2026-10-21).
  Against the hand-marked lm3d_v7: 3.070 mm (3.080 with v11's table; the
  table's gain on v7 is across-contour, 21/32 ears, upper helix and lobe). The previous model,
  fixed_conv_s42, scores 1.574 vs lm3d_v11, 1.679 vs v12 and 3.099 vs v7.
  `scripts/eval/validate_pipeline.py OUT 2 16 w N`, then `score_heldout.py`.
- **The reference is coupled to the 2D model, not just the fits** (2026-10-21).
  lm3d_v11's curve shape and spacing came from fixed_conv_s42; the new model
  follows the lobe and helix rim more closely, and v11 charged it for that
  (+0.15 mm). lm3d_v12 re-derives both from v4_fc05_s42 (`scripts/corpus` 01,
  11a, 11, 12). Each model wins on the reference built from its own
  predictions and they tie on v7, so a model-derived reference cannot rank
  models: rank on v7 and by eye. On the lobes v12 moves about halfway from v11
  toward the new predictions; it is a corpus median and does not follow single
  ears. v12 vs v11 on the template: median 1.1 mm, max 2.7 mm.
- **v7 sits inside the rim; the new model is on it** (2026-10-21, rendered
  on 32 held-out ears). Against v7, v4_fc05_s42 is outward across the curve on
  31/32 ears at 0-8 (+0.74 mm) and 29/32 at 15-19 (+0.92). The renders show
  the model on the helix rim edge and lower lobe edge, v7 on the rim's face.
  The user's rule: the placement that follows the anatomy wins. So v7
  penalises an anatomically correct model, and no fully neutral reference
  exists (v12 is model-derived). Cross-view agreement
  (`scripts/eval/cross_view_agreement.py`, best 8): 0.770 -> 0.626 mm, >5 mm
  0.88% -> 0.17%, lobe 5.6% -> 0.7%. The chosen views agree; the remaining
  error is shared placement, not disagreement.
- **Run-to-run noise.** The same ear re-predicted under heavy GPU load moved
  0.10 mm median per ear, with single landmarks jumping a few mm (RANSAC
  inlier flips); serial and concurrent reruns are otherwise bit-identical.
  Model differences of a few hundredths of a mm are inside this.
- **Filament panics on long runs** ("uniform baseColor not found") after
  ~130-150 scans per process, writing all-NaN predictions first. Earlier
  output in the same process re-predicts bit-identically. Re-run; 01 resumes.
- **The remaining error is the 2D model, not the lift.** Fed the ground
  truth's own pixels, the lift is 0.36 mm median and never past 20 mm;
  detected 2D gives 3.73 mm and 6.15% past 20 mm (40 ears x 25 views).
  Further accuracy has to come from Ear_Landmarker; the hand-off is
  `../Ear_Landmarker/handoff/from_3d_lifter_2026-10-19.md`.
- One real head scan (private, local only) ran cleanly at 4 orientations,
  0.1-0.2 mm spread, plausible ear sizes; judged improved. Partial validation:
  it is a clean scan, so hair, holes and partial heads are untested.

## The pipeline, and why each stage is the way it is

    head scan -> face frame -> per ear: box, 25 views -> 2D landmarks
              -> best 8 views per landmark -> RANSAC ray intersection
              -> re-seat on the head surface

**Face frame.** FaceMesh on renders, with checks on the result:
- FaceMesh never reports an upside-down face; shown one, it puts the forehead
  on the chin (17/17 at 180 deg). A geometric nose-profile check on the final
  frame catches it and refits turned over (2026-10-17).
- Then three checks, each retried on a turned mesh, keeping the least-bad
  frame if none passes: girth 60 mm above vs below the nose >= 1.15 (skull
  over neck), mirror symmetry about the nose p90 < 7 mm, nose-ear distance
  >= 110 mm with balance >= 0.7. Full corpus: gross errors 8 -> 0.
- On cached heads the checks pass 35/40; the 5 that trip are right by eye
  (scans cut at the jaw, an off-centre FaceMesh nose). They are guards with a
  fallback, not hard gates. Frames carry a small systematic yaw (median
  +1.5 deg, sd 2.5); left alone.

**Ear box and framing.** BlazeEar (Apache-2.0) at confidence 0.05, well
below its photo operating point. It puts a correct box on nearly every
render but scores many under 0.70 (renders frame the ear far larger than its
training photos). The views a low threshold adds are individually worse
(12.25% vs 2.62% past 10 mm), but the multi-view vote needs the rays: at 0.70
the vote's gross-error rate is 4x higher. Raise it only where a clean single
view matters more than a plentiful one. No fit is needed at inference: chop
the head to the box and PCA it (view axis 2.4-3.1 deg median, up 3.7 deg);
landmark planes, surface planes and silhouette flood fills were each worse.

**Rendering.** Skin every mesh (`ear3d/skin.py`; a raw HUTUBS head loses 64.5%
of detections) and frame on the pinna plane. One world-fixed sun, never
mirrored per side. Mesh colour is ignored by default.

**Multi-view triangulation.** A learned table picks each landmark's best 8 of
25 views (`ear3d/view_table_v11.npz`, rebuilt per corpus generation). At the
time it was built (older reference): single view 2.78 mm, RANSAC over 25 views
2.70, best-8 2.24, re-seat 1.98; gross errors past 10 mm 1.04% -> 0.065%.
- Split-half over disjoint subjects: Spearman +0.937. It is geometry.
- Random-8 is worse than all-25 (2.86): the gain is the selection.
- Smoothing the table, or weighting views softly, makes it worse; best-8
  already matches the oracle on gross errors.
- Depth-cliff snap radius 0.08 of the crop (0.16 trades median for tail).

**Re-seat.** Always project the triangulated point onto the head (gate 0):
paired error rises monotonically with any gate.

**2D test-time flip.** Coordinate fusion of the plain and mirrored passes
beats replacing one with the other (2.72% vs 2.95% past 10 mm, 69 of 112
ears). Now Ear_Landmarker's default; inherited, not re-implemented here.

## Ground truth and the corpus

**The wrap is ground truth.** Faceform Wrap fits the ear template to a scan to
0.002 mm, so the template's marks on each fit are the 3D reference. *(inherited)*
Back-projecting 2D landmarks onto a scan is not a substitute: they land on the
scalp. *(inherited)*

**Reference landmarks** are template marks, (triangle, barycentric)
addresses, evaluated on every fit. **Marks are coupled to the fits**: on a
new fit generation the same marks moved the reference 2.3 mm. Re-derive them
(`scripts/corpus/11-12`) whenever the fits change; expect parity with the
previous reference, not a gain.

**Current generation, lmw (2026-10-18).** 476 head ears re-fitted with the
template posed by the pipeline's own landmarks (Faceform RigidAlignment, then
the automatic wrap), plus 112 AudioEar3D ears. 98.3% fold-free on the
relative gate; the residual folds sit at the canal edge and were accepted by
eye. Surface coverage 84.6 -> 86.9% on the 463 that register. Three
lobe-clipped fits refitted with lobe anchors. Built by `scripts/corpus` 01-12;
caches and view table by `scripts/caches`. `EAR3D_CORPUS` selects a
generation (`lmw` default, `esc` the previous), and caches record which one
made them.

**Previous generation, esc** *(inherited, built 2026-10-12)*: `escalated_v1`
(476 head ears, 99.6% fold-free, coverage 80.0%) plus AudioEar3D cloud fits,
589 rows in `canonical_esc` (588 ears; one duplicated file). Its `canon2scan`
was composed from the older `corpus/canonical`, 0.06-0.09 mm off.

**Fitting the template** (found on esc, still true):
- Every fit folds a little; no single parameter fixes it. A per-ear
  escalation ladder cleared 77 of 92 folded ears (96.8% fold-free), at about
  1pp coverage on the median ear. *(inherited)*
- No control points at all: 1.89 folds per ear over 476, and the gen-1
  bootstrap is unnecessary. Extra control points never helped: the lobe and
  helix gaps are missing template geometry. *(inherited)*
- Cutting the template's 54-face canal tube after the fit: 1.89 -> 0.62 folds,
  free (scans have no canal). *(inherited)*
- Keep the back of the ear's geometry, drop its control points: deleting it is
  3.5x worse. *(inherited)*
- Score the ladder at base resolution (1188 vertices); the subdivided output
  can anti-correlate. *(inherited)*
- Pose from landmarks beats a pre-scaled trimmed ICP, which kept shrinking and
  left helix tops ~15 mm short (2026-10-18). If ICP is used, centre AND
  pre-scale the template first or its `max_dist` guard does nothing. *(inherited)*
- AudioEar3D clouds have no head; fitting them needs a cloud method
  (`docs/WRAP_AUDIOEAR3D_CLOUD.md`, which made esc's) or, as for lmw, a Poisson
  mesh of the cloud landmarked and wrapped like a head ear.
- Hand-written `.wrap` projects need `formatVersion` 69 and the full layout
  envelope; WrapCmd does not validate layout, so a bad one fails oddly.
  *(inherited)*

**Template lineage** (older files are still on disk):
- `ear_mean_v5` / `ear_lmv_v4` are lmw's (0 folds); `v4` was esc's.
- v2 was folded, and that was misread as "clouds are hard". *(inherited)*
- Every v5 and v6 PCA file was built with a Procrustes loop that shrank the
  mean a few percent; v7 is the first size-correct one. *(inherited)*
- Reference landmarks: v7 hand marks, v9 (multiview curve shape, real-ear
  spacing), v11 (v9's recipe re-derived on lmw). v10 (v9 marks on lmw fits)
  is wrong. Points 50-54 are the superior crus, not the lobe. *(inherited)*
- Results dated before October 2026 carry a ~2.5 mm offset between the 2D
  scheme's points and the wrap's (about one index step along each helix
  contour), removed by the v6_spacing re-marking. *(inherited)*

**Conventions that corrupt data if broken:** right ears are stored z-mirrored
into left form (mirror about Z, not X); `lm_basis_*` files are unit-height,
not millimetres, despite older `_mm` names (`../wrap_work/templates/UNITS.md`);
the 55-point scheme is extended by appending indices, never renumbering.
*(inherited)*

## How measurements went wrong (and the rule each one left)

| What happened | Rule |
|---|---|
| Distance to the scan scored a landmark on the scalp as a success | Score against the wrap fit, as an escape **rate** at mm gates |
| `d > gate` is False for NaN, so landmarks thrown off the head counted as successes | A NaN distance is an escape; count it |
| Right-ear heads were rendered inside out (det < 0, winding not reversed), and a sun rule was fitted to cancel it | Reverse winding on any reflection; never fit a correction to a symptom |
| A pooled left/right delta cancelled a real side bias to zero | Stratify by side; score the component the effect lives in |
| Snapping to the surface made the surface metric tautological | Score a re-seat on paired distance only |
| A transposed box (`[ymin, xmin, ...]`) gave two false conclusions | Draw geometry on its image before reasoning from it |
| A new reference moved the score more than the model change did | Compare models on one fixed reference |
| The same checkpoint gave different results a day apart | The 2D code moves too; re-run the baseline |
| An absolute 90 deg fold gate flagged the template's own creases (19 folds over 15 ears; 3 over 2 relative) *(inherited)* | Gate folds on fit angle minus template rest angle |
| A "unit error" at ratio 1.114 was a bounding box compared with a caliper *(inherited)* | Measure units landmark to landmark (every corpus ~1.01) |
| Raw Wrap re-fits mixed HUTUBS-normalised and SONICOM-mm units, and work-scale fits were scored unscaled *(inherited)* | Canonicalise first; divide work-scale fits by `w` (`accuracy()` now raises) |
| `tf_mm_*` omitted the scan centroid (5.56 mm) *(inherited)* | Compose canon -> scan once, verified against the scan (`06_canonical_to_scan`) |
| Nearest-surface distance was used to rank views (+0.012 rank correlation) *(inherited)* | Score against the reference point for point |
| A shape head's mesh was noise, and the landmark metric could not tell *(inherited)* | Supervise the coefficients; check the mesh, not just the landmarks |
| AudioEar3D was called depth-flat; it was a frame artefact *(inherited)* | Real limit: partial support, ~40 of 55 landmarks per ear |
| An 11.9 deg "pose error" was a labelling convention *(inherited)* | Check conventions before blaming the model |

Each of these produced a confident wrong number that a render or a second
comparison would have caught. Render and look before reporting.

**Retracted inherited claims**, kept in `RESULTS_inherited.md`: a corpus
bad-annotation rate (5.40%, then 1.9%, then withdrawn: the instrument measured
visible-ear extent); "shape carries no information" (underpowered,
split-half reliability 0.417); "a screen needs ~10 epochs" (backwards:
reliability 0.417 at 1 epoch, 0.937 at 25, 0.978 at 78). Four traps in the
2D audit are listed at `RESULTS_inherited.md:2555`.

## Closed

- **Learned depth from one photograph** *(inherited)*: five heads (heatmap
  integral, no-image lifter, frozen-2D, shape-coefficient route, one
  multi-task network) all lost to z = 0 on real photographs; the best is 5.1x
  worse on photos than renders, and framing is not the cause.
- **What did transfer, for photos** *(inherited)*: a geometric solve (scaled
  orthography, not PnP; sign from the canonical axis), where a mean ear plus
  a solved pose beat the depth head by ~0.4 skill at 0.053 ms; and MoGe-2 read
  at the 2D model's pixels (fed the ear crop), which beat every ear-trained
  head with nothing fitted. Pose: solve it, never regress it (route B reversed
  sign on photos; solving was worth +0.6 to +0.9).
- **A real-photo depth label from AudioEar3D** *(inherited)*: a stranger's ear
  fits a photo as well as its owner's. Costed and closed.
- **Shape** *(inherited)*: 40 modes cost almost nothing against full 3D, and
  the curve is concave, so a weak shape head still pays. One network for
  landmarks, depth and mesh lost to the specialists. Shape coefficients
  identify the source 93.9% of the time, but depth did not use it (six runs).
- **Implicit surfaces** *(inherited)*: template topology serves every consumer;
  nothing needs watertight.
- **Heatmap confidence as a view filter**: within-landmark Spearman -0.018.
  The softmax measures concentration; amplitude, which Deep-MVLM filters on,
  is unidentifiable under a coordinate loss. A usable per-point confidence
  needs an auxiliary amplitude term in the 2D model.
- **2D annotation QA** *(inherited, closed 2026-10-13)*: zero-mean label noise
  is absorbed by L2 training, so cleaning bought no accuracy.

## Open

- **2D accuracy on the upper helix and lobe** (Ear_Landmarker's side).
- **Scans from the wild**: hair, holes, partial heads, other scanners.
- **Scan texture is unused**: a default, not a decision.
- **Reference points 55-94** keep the older definition; only 0-54 were
  re-derived for lmw.
- **No confidence signal survives view selection**: with best-8 the RANSAC
  inlier count is 8/8 everywhere; reprojection spread is unchecked.
- **RANSAC returns NaN when it fails**, where Deep-MVLM falls back to least
  squares on all rays and logs it.
- **Where the 55 points run out** *(inherited)*: they leave the fossae, the
  lobe and the concha floor undetermined.
- **The corpus as a constraint on 2D training** *(inherited)*: never tried.
- **Training-dynamics signals** *(inherited)*: whether they measure label
  error or image difficulty was never settled.
- `skin.py` exists in both repositories (`ear3d/skin.py` here).

## Tools that exist

*(inherited; built for 2D annotation QA, still working)*
- **SAM 2.1 silhouette check** (`archive/scripts/check_helix_silhouette.py`):
  outer points sit 1.2-1.5% of ear height from the mask, interior points 6-7x
  deeper. Apache-2.0, code and checkpoints.
- **SAM 3 mask deficit**: a free occlusion score; deep deficit marks hair and
  hands (57% of hard rows, 2% of clean). Segmenters stop at the hairline,
  where the annotation extrapolates through hair, and no prompt changes that.
- **Sato dark-ridge filtering** traces the antihelix, crura, antitragus and
  lobe; Canny is brittle; the concha is invisible to both.
- **Depth Anything 3**: Base/Small/Metric are Apache-2.0; Large/Giant/Nested
  are CC BY-NC.
- Every independent instrument has a **~5-6% of ear height noise floor**: it
  can flag a 15% disagreement, not correct a row.

## Where things are

| What | Where |
|---|---|
| Raw third-party scans | `$EAR3D_DIR`, default `../clean_3d_data` |
| Templates, bases, corpora | `../wrap_work/templates`, `../wrap_work/corpus` |
| lmw's as-run rewrap outputs | `../wrap_work/corpus/lmwrap_v1/asrun` |
| The 2D model | `../Ear_Landmarker`, reached only via `ear3d/twod.py` |
| Frozen experiments | `archive/` (index in its README) |

Nothing under the data paths is committed, in either repository. Licences by
source matter for anything released; the HUTUBS-only `ccby` bases exist for
that.
