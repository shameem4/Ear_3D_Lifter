# Where this stands, 2026-10-16

The lift path, end to end, with every number held out by subject where a
table was involved. Written at a restart; the conversation that produced it
is gone, so anything not here is lost.

## The stack, and what each step is worth

    single good view                      2.78 mm paired median
    RANSAC over 25 views                  2.70      robustness, not accuracy
    + per-landmark view table, best-8     2.24      the largest single gain
    + re-seat onto the head               1.98

Gross errors (>10 mm paired) over the same path: 1.039% for one view,
0.162% for RANSAC, 0.065% at the end. The original failure -- landmarks on
the scalp -- is gone.

## Settled

**The view table.** `scripts/view_table.py`, built by `scripts/cache_rays.py`
(112 ears cached in `runs/raycache/`, reusable, ~40 min to rebuild).
table[view, landmark] = median 2D error in px, learned on half the subjects.
Pick each landmark's best 8 views, triangulate from only those.

- split-half over disjoint subjects: Spearman **+0.937**, per-landmark view
  ordering +0.895, 55/55 above +0.3. It is geometry, not corpus noise.
- beats random-8 (2.24 against 2.86), and random-8 is WORSE than all-25,
  so more views help when chosen blindly -- the gain is selection.
- survives a RANSAC threshold sweep 0.01 to 0.15 at a steady ~0.45 mm.
- SMOOTHING THE TABLE MAKES IT WORSE (2.33 against 2.24). The table is
  spatially smooth and smoothing it still destroys real structure. Dead end.
- oracle-3 (best views for THAT ear, by its own GT) reaches 1.49 mm, so
  ~0.75 mm of per-ear structure the table cannot capture is still there.

**The re-seat.** Project the fused point onto the head mesh.
- gate: **T=0, always re-seat.** Paired error degrades monotonically with
  the gate (1.98 / 2.00 / 2.05 / 2.14 at 0 / 0.5 / 1.0 / 2.0 mm). The 0.5 mm
  I shipped earlier was justified on a metric I later disproved.
- snap TARGET: head wins in the body (2.57 against 2.68), fit wins in the
  tail (p99 7.98 against 8.33). Neither fixes gross errors -- a landmark the
  2D model placed wrongly stays wrong, only on a nearer surface.
- **SCORE A RE-SEAT ON PAIRED DISTANCE ONLY.** Surface distance after a snap
  is largely tautological: a point 10 mm from truth gets pulled onto scalp
  beside the ear and its surface score improves. This was caught after
  reporting it as a headline.

## The production path has no fit and no ground truth in it

    head mesh -> detector-axis render -> BlazeEar box -> chop the head to it
              -> PCA the chop: smallest axis = face-on, largest = up
              -> render the grid -> rays -> table best-8 -> RANSAC
              -> re-seat onto the head

The fit is used ONLY to grade the output (`_surface_mm`, `on_mask`).

**Bootstrap accuracy**, chop + PCA against the GT frame, 112 ears:

    view axis (face-on)   2.4-3.1 deg median, 15.2 max   (grid step is 15)
    up axis               3.71 deg median
    centre offset         6.85 mm        absorbed: detector re-crops per view
    scale ratio           1.076          absorbed: ROI loop drives occupancy

Why not the alternatives:
- **silhouette / depth-cliff flood fill: NO.** Pure geometry, so hair,
  spectacle arms and earrings read as "the thing that sticks out".
- **plane from the 55 predicted landmarks: NO.** Clean it is excellent
  (1.3 deg under isotropic 2.7 mm noise), but pass 1 is a single-view depth
  lift and its scalp-landed points displace ~20 mm ALONG THE NORMAL, the
  worst direction for a plane fit: 5.90 deg median and 48 deg worst at a 10%
  scalp rate. A trimmed (80%) fit is robust at ~2.3 deg if this route is
  ever revisited.
- **plane from the ear SURFACE: NO.** 3.93 deg, and it is BIAS not variance
  -- the surface plane genuinely differs from the landmark plane, so more
  points do not help.
- My trimmed PCA on the chop made the normal WORSE on these bald heads
  (4.61 against 2.41), because it drops the rim points that define the
  plane. It would plausibly help on hairy scans; HUTUBS cannot say.

## Open, in the order I would take them

1. **Per-point heatmap confidence as a view filter.** Deep-MVLM
   (RasmusRPaulsen/Deep-MVLM, arXiv 1910.06007) filters views by the
   per-landmark HEATMAP PEAK before its RANSAC. We tested *detection
   confidence* and found it worthless (r=-0.010) -- a different quantity.
   `predict(crop, with_confidence=True)` already returns per-point
   confidence and we discard it. This is per-ear evidence where the table is
   a per-scheme prior, so it aims straight at the 0.75 mm oracle gap.
2. **View selection killed our only confidence signal.** With 25 views the
   RANSAC inlier count ran 7-25 and flagged bad landmarks; with best-8 it is
   8/8 everywhere. Reprojection spread may survive; unchecked.
3. **Does the bootstrap converge?** The frame sets the render that produces
   the box that sets the frame. `label_two_pass` iterates; confirm it
   settles rather than oscillates.
4. The sun optimum (+1.2) was swept on the OLD checkpoint. The direction is
   confirmed on the shipped one; the optimum is not re-derived.

## Deep-MVLM, for the record

Same architecture as ours, arrived at independently: rays not depth lifts,
RANSAC not least squares (100 iters, 3 lines per sample, inlier radius 10,
requires inliers > n/3, falls back to LSQ on all and logs it -- ours returns
None and the landmark silently becomes NaN, which matters more now that we
select down to 8 views). They render geometry/depth modes and TRAIN on those
renders, which is why lighting is worth 5.84x to us and nothing to them.
