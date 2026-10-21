# Findings

What this project has established since it split from Ear_Landmarker
(2026-10-13): the decisions behind the shipped pipeline, how the ground truth
is built, the measurement mistakes that produced wrong answers, and what is
closed or still open. The inherited 3D effort is in `RESEARCH_BRIEF.md`; this
is its continuation. Numbers carry the date and the reference they were
measured against, because both have changed.

## Where it stands (2026-10-20)

- **Held-out accuracy: 1.58 mm** median per ear, 16 HUTUBS heads used in no
  table or reference, each randomly rotated and unskinned; p95 2.46 mm; 4.03%
  of points beyond 5 mm, concentrated on the upper helix (points 0-8) and the
  lobe (15-19). Reference lm3d_v11; 2D checkpoint fixed_conv_s42 epoch 433.
  `scripts/eval/validate_pipeline.py OUT 2 16 w N`, then `score_heldout.py`.
- **That baseline predates Ear_Landmarker's 2026-10-20 inference changes**,
  which moved one held-out ear by up to 9.2 mm with the same checkpoint.
  Re-run it on their next shipped default before comparing anything to it.
- **The remaining error is the 2D model, not the lift.** With the ground
  truth's own pixels as input, the lift is 0.36 mm median and never past
  20 mm; detected 2D gives 3.73 mm and 6.15% past 20 mm (40 ears x 25 views).
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

- **The wrap fit is the ground truth surface**: one ear template fitted to
  each scan (0.002 mm to the ear). Back-projection is not ground truth;
  it lands on the scalp.
- **Reference landmarks** are template marks, (triangle, barycentric)
  addresses, evaluated on every fit. **Marks are coupled to the fits**: on a
  new fit generation the same marks moved the reference 2.3 mm. Re-derive
  them (`scripts/corpus/11-12`) whenever the fits change; expect parity with
  the previous reference, not a gain.
- **Current generation, lmw (2026-10-18)**: 476 head ears re-fitted with the
  template posed by the pipeline's own landmarks (Faceform RigidAlignment,
  then the automatic wrap), plus 112 AudioEar3D ears. 98.3% fold-free on the
  relative gate; the residual folds sit at the canal edge and were accepted
  by eye. Surface coverage 84.6 -> 86.9% on the 463 that register. Three
  lobe-clipped fits refitted with lobe anchors.
- Built by `scripts/corpus` 01-12; caches and view table by `scripts/caches`.
  `EAR3D_CORPUS` selects a generation (lmw default, esc the previous), and
  caches record which one made them.
- Conventions that corrupt data if broken: right ears are stored z-mirrored
  into left form; basis files are unit-height, not millimetres.

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

Each of these produced a confident wrong number that a render or a second
comparison would have caught. Render and look before reporting.

## Closed

- **Learned depth from a single image** (five heads): never beat z=0 on
  photographs. A geometric solve and a general depth model (MoGe-2, read at
  the 2D model's pixels) did. Code in `archive/`.
- **Heatmap confidence as a view filter**: within-landmark Spearman -0.018.
  The model's softmax measures concentration; amplitude, which is what
  Deep-MVLM filters on, is unidentifiable under a coordinate loss.
- **Extra wrap control points**: never helped; the lobe and helix gaps are
  missing template, not a weak fit.
- **Implicit surfaces**: not needed; template topology is enough.
- **A single network for landmarks and mesh**: loses to both separately.

## Open

- **2D accuracy on the upper helix and lobe** (Ear_Landmarker's side).
- **Scans from the wild**: hair, holes, partial heads, other scanners.
- **Scan texture is unused**: a choice by default, not a decision.
- **Reference points 55-94** keep the older definition; only 0-54 were
  re-derived for lmw.
- **No confidence signal survives view selection**: with best-8 the RANSAC
  inlier count is 8/8 everywhere; reprojection spread is unchecked.
- **RANSAC returns NaN when it fails**, where Deep-MVLM falls back to least
  squares on all rays and logs it.
