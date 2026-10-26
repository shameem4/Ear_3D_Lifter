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

## Where it stands (2026-10-25)

- **r3p_s42 replaces r2j_s42 as the 3D path's model** (user's choice).
  r3p is r2j's recipe plus pinna-only renders (synthetic_v3p, 25% of the mix).
  Paired r2j -> r3p, same day, same heads and rotations:
  - **Heads vs the user's 32 hand-corrected held-out ears:** clean 1.089 -> 1.031 mm (20/32 better, p=0.15); with jewellery 1.061 -> 1.068 mm. Points 26-28 are +0.10-0.14 mm worse (p~0.045 in both runs).
  - **HUTUBS held-out ears cut to the pinna, vs hand:** landmarks 0-2 4.97 -> 1.93 mm; per ear 1.72 -> 1.04.
  - **Head-cut held-out ears:** 1.50 -> 1.10 mm.
  - **AudioEar3D, vs v12:** wraps 0-2 5.39 -> 2.16 mm; raw clouds per ear 2.74 -> 2.04 mm, >5 mm 40% -> 14%. The cloud lobe stays weak (3.7 mm).
  One seed.
- **Why ear-only meshes failed: no surrounding skin.** r2j missed the helix root (landmarks 0-2) by ~5 mm on any pinna-only mesh.
  Evidence: HUTUBS ears cut to the pinna 5.0 mm; the same ears with a disc of cheek left 2.2 mm; full heads ~1.5 mm.
  Ruled out: the lift (given the reference's own pixels it is exact, 0.00 mm) and mesh density (16x subdivision changes nothing). The model had never seen a pinna without its head.
  This supersedes the 2026-10-24 advice not to fine-tune on ear-only renders: that test kept skin around the ear.
- **A visibility constraint on the lift does not pay** ("rays cannot pass through the ear": replace a point that sits >1.5 mm behind its rays' first surface hit).
  Held-out heads vs v12: 0.991 -> 1.071 mm; >5 mm 0.23% -> 0.80%. Reverted. Folds legitimately hide landmarks 1-3 mm from their best views.

## Where it stood (2026-10-24)

- **Render-only 2D model r2j_s42 is the 3D path's model** (user's choice).
  On the user's 32 hand-corrected held-out ears (16 heads), end to end, view_table_v13:
  - medians: v4 1.613, v5syn25 1.412, r1 (render-only) 1.045, r2j 1.089 mm;
  - share of points beyond 5 mm: v4 4.66%, r2j 0.85%;
  - with synthetic jewellery on every ear: r1 1.112, r2j 1.061 mm;
  - r2j vs r1 on the lobe: -0.21 clean (p=0.034), -0.42 with jewellery (p=0.001);
  - pp21_left's real hoop: worst point 7.6 -> 3.3 mm.
  The control r2 (same recipe without the jewellery set) does not fix the hoop (2D: +7% on hoop renders), so the gain is the jewellery data.
  r2j is +420% on photographs, so it must never read a real texture.
- **Re-skin every scan, textured or not (2026-10-24).**
  Three Artec textured heads (BJ030, CD022, CD048), 4 orientations each: the synthetic-skin + r2j route spread p95 0.13-0.33 mm, their own texture + v4 0.35-1.55 mm (2-9x).
  The two routes sit 1.1-1.9 mm apart (median); no ground truth, so this is stability, not accuracy. User chose synthetic for all after the overlays; `use_mesh_texture=False`.
  The reference is one annotator's placement, which r2j trained toward.
- **Pooled jewellery deltas hide the effect; stratify by kind.**
  r1 under synthetic jewellery looked flat pooled (p=0.72), but on the 15 lobe-hoop ears lobe error rose on 12, up to 14.2 mm, all on points 16-19 (slid onto the hoop).
  Studs and helix piercings: no effect.
- **synthetic_v3 shipped 8,850 broken images (2026-10-23).**
  SONICOM STLs share no vertices, so decimation deleted triangles and vertex noise cracked the surface.
  r1 trained on them. Rule: weld before any mesh edit, and look at every damage mode on every dataset before a bulk render.
  The user then judged noise and holes unrealistic; damage is now decimation only (synthetic_v3r).
- **Orientation fallback (2026-10-24).** When no face search passes, the least-bad frame is kept.
  A missing ear scored inf, so on P0022_right (SONICOM graded: the far ear is never detected) an asymmetric, sideways frame won.
  The ear was rendered on its side and the prediction missed by a median of 34 mm. A single missing ear now scores 1.0, the mildest rejection.
  P0022_right goes 33.8 -> 1.15 mm; the other fallback scans in the corpus are bit-identical.
  No held-out head used the missing-ear path.

## Where it stood (2026-10-20)

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
- **Framing to the model's training occupancy** (2026-10-21). The 3D
  pipeline feeds whole renders to a bare LandmarkPredictor, so occupancy is
  set by our framing: the BlazeEar box + PAD gave 0.50-0.62 landmark extent
  against TRAIN_OCCUPANCY 0.777. Re-sizing each orbit to 0.75 is parity on the
  held-out 16 WITH a table learned at it (vs v12 -0.004 mm p=0.92, vs v7
  -0.010 p=0.89, >5 mm 4.38% -> 3.75% n.s.), and WORSE without one (v7 +0.136
  p=0.004, >5 mm +2.8pp). It fixes most of pp16's attached lobe (point 19
  9.7 -> 5.7 mm). Default since then, with view_table_v13. Stronger AO in
  the ear skin (0.75 -> 1.0), tried for the same lobe, changed nothing either
  way. Attached lobes are common in photos and v4 handles them there
  (Ear_Landmarker), so pp16 is a render-domain miss.
- **Attached lobes are a known miss** (pp16 left; pp72 left in the low view).
  With no free lobe edge, the 2D model continues the rim onto the cheek at
  every angle used, most visibly in view 4 (yaw -30, pitch +30, which is
  from ABOVE: negative pitch is below). The view table already keeps view 4
  out of the lobe's best 8, which it builds mostly from below (pitch -30..-15),
  so excluding views by hand changes nothing, and stronger crease AO did nothing.
  Ear_Landmarker: v4 handles attached lobes in eye-level photos; the gap is
  viewpoint, and fixing it needs render labels neither reference provides
  (v12 is model-derived, v7 sits inside the rim).
- **Hand-corrected held-out reference, and the first 2D model to beat v4 on
  it** (2026-10-22). The user placed all 55 points on the 32 held-out ears
  (`scripts/handfix`, scored by `scripts/eval/score_handfix.py`): the only
  reference no 2D model shaped. Ear_Landmarker trained v5 on 68 hand-corrected
  attached-lobe ears rendered as synthetic_v1. Paired vs v4 (view_table_v13):
  v5syn25 (6% synthetic) -0.21 mm median (25/32, p=0.002), upper helix -0.61,
  >5 mm 4.66% -> 1.53%; v5syn100 (21%) -0.33 (27/32). The lobe gain is not
  significant: the batch was all attached lobes, and the models learned to run
  point 19 down a ridge even on free lobes (pp33L), so a free-lobe batch
  follows. Photo cost (Ear_Landmarker, 909 ears): +1.8% at 6%, +4.1% at 21%.
  Anchoring favoured v4 (corrections started from v12, v4 shown), so the win
  is if anything understated.
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

## Face landmarks (ported from Face_3D_Lifter, 2026-10-25) *(inherited)*

`ear3d/face.py` lifts MediaPipe's 478 points; runbook in `scripts/face/README.md`. Face_3D_Lifter's own
FINDINGS.md is the long form. In brief:
- **Method:** 33 views (yaw -75..75 in 15 deg steps x pitch -30/0/30) about a pivot 60 mm behind the nose.
  The front sun is fixed to the head, skin tone 5 with blotches. FaceLandmarker runs per view, then per-view
  mean 2D offsets are subtracted, then RANSAC at 2 mm, then the points are seated on the surface.
- **Pseudo GT:** MediaPipe's canonical face, Wrap-fitted to each head with the user's 11 named control points.
- **Result:** leave one head out on 50 HUTUBS heads, the median error vs the pseudo GT is 4.21 mm with offsets
  off and 2.34 mm with them on (p90 5.59). Reproduced here bit-for-bit. On BJ002's 12 physical stickers:
  4.0 -> 2.4 mm.
- **Per-view miss table** (`view_table.py`): split-half Spearman 0.91, so it is stable across heads. Views
  from below the face miss by 5+ mm even near face-on. Weighting or choosing views on top of the offsets adds
  little (2.34 -> 2.25 at best).
- **Bunching fixed by a joint ray fit (2026-10-25, now the default).** Per-landmark triangulation lets
  neighbours settle on different views, so they pile up or spread apart. Fitting all 468 points at once to
  every view's rays (Cauchy-weighted) with a Laplacian prior on the canonical mesh fixes it. Leave one head out
  on a fresh lift (v1, today's ear3d, offsets rebuilt): median 2.34 -> 2.18 mm, p90 5.50 -> 4.90. Better on
  39/50 heads, worst +0.23 mm. Edges longer than 2 mm in the GT that collapse to half their length: 0.62% ->
  0.05% (measured on v0). Caveat: the GT is a fit of the same canonical mesh the prior pulls toward, so it
  flatters this. Independent check, BJ002's 12 stickers (1bb9ed0): fit on 2.8 / 3.8 / 9.3 mm (median/mean/max) vs fit off 2.4 / 3.8 / 9.5, so NO accuracy gain; the differences are within noise for 12 points on one scan. What the fit buys is regular spacing (no bunching), not accuracy. What remains: closed-lip corners (183/191,
  407/415, edges ~1.4 mm) and whole-feature shifts (a mouth a few mm off), which smoothing cannot move.
- **Bunch-prone metric (score_loho.py):** spacing and error split into FEATURES (MediaPipe's lips, eyes,
  brows and nose: 116 landmarks, 178 edges) and the REST, so flat-surface points cannot dilute it. On v1,
  features: spacing |log ratio| median 0.095 -> 0.063, p90 0.307 -> 0.187, collapsed 0.70% -> 0.00%
  (ray fit better on 49/50 heads); error 2.25 -> 2.06 mm. Offsets off, for scale: 0.221 / 0.767 / 9.9%.
  Same caveat: the spacing reference is the canonical-mesh GT.
- **Per-view weights in the ray fit (2026-10-25, default):** each ray weighted 1/miss^2, miss = that view's
  median raw ray miss for the landmark over the training heads (miss_mm in the offsets file). Leave one head
  out, error, ray fit -> weighted: features 2.06 -> 1.98, rest 2.22 -> 2.15 (rest p90 5.03 -> 5.19); nostril
  1.69 -> 1.58, lip corners 2.56 -> 2.52. 1/miss and best-8-views were no better.
- **pp22's septum is a 2D error weighting cannot fix:** FaceLandmarker puts the nostrils on the upper lip in
  every face-on and below view (the nostrils sit high); weighting makes it worse (3.4 -> 3.8 mm down). No
  bias across heads (+0.1 mm). A geometric subnasale snap (deepest midline point under the tip-to-labrale
  chord) is worse than the detection at lm 2 (2.26 vs 1.42 mm median); gated to vertical disagreements >3 mm
  it fixes pp22 (3.6 -> 2.0) but helps 4 heads and hurts 3. Not adopted.
- **Dense nose fit fixes the nostrils (2026-10-25, default, step 6):** the 2x-subdivided canonical face is
  fitted point-to-plane to the scan within 35 mm of the subnasale, with the weighted rays at 0.5 and a stiff
  Laplacian (10). The nose's own shape puts back nostrils FaceLandmarker misplaced. Leave one head out, from
  step 5: nose base 1.58 -> 1.07 mm (p90 3.41 -> 2.07; better on 39/50; heads over 3 mm 6 -> 0); features
  1.98 -> 1.71; rest 2.15 -> 1.84; whole face better on 38/50, worst +0.45 (pp31). pp22's septum 5.4 -> 1.2.
  Sweep: the surface term over the whole face drags closed lips (pp44 1.92 -> 2.61), so it is limited to the
  nose. Soft stiffness (lambda <= 3) lets the template slide along the skin and does nothing. Caveat: the
  pseudo GT is the same kind of fit (Wrap of the canonical face), so the HUTUBS score flatters it; judge by
  the renders and BJ002's stickers. ~10 s per head.
  **Then restricted to the nose base (dense_apply_mm 12):** BJ002's stickers showed the surface term slides
  the nose tip (lm 1) 2.5 mm down its lower slope (sticker 1.7 -> 3.9 mm; point-to-plane cannot hold an
  apex) and costs the lateral stickers; region 15/22 did not cure both. HUTUBS liked the tip only because its
  GT is the same canonical fit. Now only the 21 nose-base landmarks take the dense result; the rest keep step
  5 exactly. Leave one head out: nose base 1.58 -> 1.07 mm, features 1.98 -> 1.96, rest 2.15 -> 2.10.
  BJ002 (no sticker in the nose base): stickers identical to dense off; the nose base moves 5.4 mm median
  (8.7 at lm 2), beyond any HUTUBS head (max per-head 4.1). It is a correction: lm 2 lands 0.97 mm from the
  geometric subnasale with the fit, 8.6 mm (out on the columella) without; moves are all back toward the
  face; nostril cavities are shallow (3.5 mm) and no landmark went into one. Unverified: the lateral
  nose-base points (alar base, rim), which rest on the canonical prior.
- **Lip-corner spacing worsened by the view weights:** collapsed edges there 4.1% -> 11.1% (p90 0.36 ->
  0.86); after the dense fit 9.0%. Those edges are ~1.4 mm in the GT (closed lips), so the ratio is noisy,
  but this one is open.
- **Wrapping the canonical face on lifted key points (2026-10-25, scratch 2026-10-25-face-wrapkeys, not
  adopted).** Wrap (ear3d.face_gt.wrap_template) with lifted landmarks as control points, the GT's own method.
  Leave one head out (whole / lip corners / corner collapse): current 2.06 / 2.52 / 11%; 25 keys (C: eyes,
  nose, chin, mouth corners, brow/forehead, cheeks, nose base) 1.66 / 2.21 / 2%; C + 12 face-oval keys (D)
  1.64 / 2.23 / 2% (47/50 heads better); every landmark with LOHO error < 2 mm as a key (~228) 2.06 / 2.15
  / 1%. BJ002's stickers disagree: median 2.9 (default) vs 3.7 (C) vs 3.4 (D). Between the keys the wrap
  swaps the lift for the canonical shape, which the GT shares. What the wrap reliably buys is spacing at the
  lip corners. Four lifted oval points sit 1.4-3.3 mm off the skin before seating (93, 234, 361, 288).
  In-house replacements so far (468-mesh pins + membrane; dense pinned film with stiffness annealing) reach
  1.88-1.95 mm whole face but leave corner collapse at 7-9%: they do not yet reproduce Wrap's spacing.
  BJ002 stickers near 345 / 383 / 347 miss by 6-10 mm in every method; what they mark is unknown.
- **ear3d drift:** v0 (Face_3D_Lifter era) vs v1 normalisation differs. pp1 was pitched in v0 and is
  upright in v1; the other heads moved <5 mm. BJ002 lost one view (stickers 2.4 -> 2.7 mm on the v0 offsets).
- **Error is sliding along the skin** (0.26 mm along the normal), so off-surface checks cannot see it. RANSAC
  residual does not predict error.

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
| Reprojection error chose a 20 deg view cone; it rewards the lateral agreement the fit already optimises and is blind to depth | Judge placement in 3D (distance from the surface or the reference), not by reprojection |
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
- **SAM 2.1 silhouette check** (`archive-final:archive/scripts/check_helix_silhouette.py`):
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
| Frozen experiments | git tag `archive-final` (`git show archive-final:archive/README.md`) |

Nothing under the data paths is committed, in either repository. Licences by
source matter for anything released; the HUTUBS-only `ccby` bases exist for
that.
