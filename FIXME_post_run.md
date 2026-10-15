# Fix list, queued behind the sun sweep and the converged comparison

Written 2026-10-15, while `runs/sun_sweep` (cell 4 of 5) and `runs/conv` are in
flight. Everything here is a consequence of the two instrument bugs found on
2026-10-14 — the NaN gate and the inverted winding — plus the policy that this
repo inherits Ear_Landmarker's defaults.

Ordered by what breaks if it is skipped, not by effort.

## 1. DONE (251fc8b) — `cfg["run"]` follows their `SHIPPED_RUN`

`ear3d/config.py:28` names `manual_occ_s42`; their shipped run is now
`fixed_conv_s42`. Policy says follow. Held only because changing it mid-sweep
would switch models under a running cell, which is the exact failure
`5e86fdc` was written about.

**Do after both runs land.** Read the converged comparison first — not to gate
the change, but because `runs/conv/fixed_conv.csv` against
`runs/winding_fix/frame_wound.csv` says what adopting it costs on the escape
rate. If it is a regression there, that is a root-repo conversation, not a
local override.

## 2. Decide where fusion lives — MEASURED, now waiting on them

Adopting their defaults end to end costs **+0.584pp** on the escape rate, and
the split says it is not the model: implementation +0.407pp p=1.6e-08,
checkpoint +0.177pp p=0.43. Their fusion mirrors the crop tensor inside
LandmarkPredictor, so the detector and the ROI loop run once on the
un-mirrored frame; ours mirrors the whole frame. Our SINGLE mirrored pass
(2.823%) beats their TWO-arm fusion (3.303%), which rules out averaging model
outputs as the effect.

Asked them for a frame-level option rather than overriding locally, per the
use-their-defaults policy. **If they decline, this becomes a deliberate
divergence to decide with the user** — not a silent local pin.

Original note follows.

If we adopt their checkpoint we inherit `flip_fusion=True`, which mirrors the
normalised tensor after the resize inside one batch-2 forward. Ours re-runs
detect and ROI refinement on a mirrored frame and so reintroduces the
`square_roi_crop` rounding asymmetry that fusion exists to cancel — 73.5 ms
against their +0.3% of wall clock. If theirs is on, `--tta avg` here is both
redundant and worse, and `escape_audit`'s `--tta` becomes a measurement-only
flag. Note `escape_audit` pins `flip_fusion=False` deliberately (`0fbae2b`);
that pin stays.

## 3. DONE (251fc8b) — one escape rate, in `ear3d/escape.py`

`nan-distance-is-an-escape`: `d_snap_mm` is NaN when the landmark pixel is
background, and `d > gate` is False for NaN, so the worst outcome scores as a
success. Fixed in `handedness_vs_illumination.rate()` and `sun_sweep`. Still
outstanding:

- `scripts/tta_bench.py:324` — `(d > 5).mean()` straight over a column that
  can be NaN. This is the script the whole TTA table came from, and the arms
  do not miss equally, so that table is affected the same way the fusion
  result was.
- `scripts/escape_audit.py:489` (`summarise`) and `scripts/escape_report.py:54`
  — both DROP via `ok = np.isfinite(d)`. Defensible for one arm, wrong for
  comparing two. Left alone on purpose in `856652b` rather than changed under
  cover of another commit; decide and apply deliberately, and report the
  off-head rate as its own column wherever a gate is printed.

## 4. PARTLY DONE — presentation multiplier re-derived, rest superseded

The 1.247x multiplier, re-cut on corrected renders from the data we already
have (no new renders needed):

    side       none     flip   ratio
    left     3.800%   3.047%   1.247x
    right    3.169%   2.857%   1.109x
    per-ear median 1.204x against 1.078x, Mann-Whitney p=0.040

**The equality claim does not survive.** The old reading was 1.247x on both
sides at p=0.27, measured with half the corpus inside out. Treat the new
p=0.040 cautiously too: these are ratios on different baselines, left ears
being harder to begin with, so it is weak evidence of a real difference
rather than strong evidence of one.

It is also largely moot. `scripts/mixed_handedness.py` shows the whole
single-arm gain is an artefact of this corpus being handed -- worth -0.005pp
on a mixed one -- so a per-side multiplier is a property of our rendering
convention, not a transferable fact about the model.

The rest of the original item:

All measured with reflected heads shaded inside out, so all void for anything
that mixes or compares sides:

- the 1.247x **presentation multiplier** (the mirror's gain, claimed equal on
  both sides)
- the **one-sided per-ear sun result**, +4.20pp left against -0.67 right
- the `light2x2` cells in `runs/light2x2/`
- `scripts/oracle_2d_lift.py`'s `det_snap` row — pre-gate and pre-winding. The
  CONCLUSION is safe (a worse detector only widens the gap it reports) but the
  numbers in its docstring are stale.

The illumination factor is being re-derived now and is not on this list.

## 5. DONE (a692e4d) — marked at the scripts, not in this list

`scripts/light_direction.py` and `scripts/normal_error_sensitivity.py` import
`escape_audit.load_ear`, so they are correct from `856652b` onward, but any
result produced by them before that date was measured on inside-out right
ears. Re-run before quoting.

## 6. DONE (a692e4d) — brief marked superseded in their copy

`Ear_Landmarker/runs/flip_equivariance/BRIEF.md` still carries the original
lighting and handedness tables and the claim that the sun does one thing to
each side. All of it is pre-winding-fix. Rewrite or mark superseded — it is
their copy of our evidence and they have been reading it.

Carry the release-note wording they corrected us on: **no side name**. On a
single-handed corpus an anti-equivariant offset is a constant whose sign
happens to suit the convention, so it is "the legacy checkpoint carried a
constant offset that a single-handed pipeline may be benefiting from, and
which would reverse for the opposite presentation", not "the fixed model is
worse as-is".

## 7. CLOSED, NOT RE-RUN — the lost cell measured a retired concept

The scratch wipe took the 40-ear 2x2's `opposite` cell. Re-running it would
measure the difference between two sun ANCHORS, and anchors are rules for
mirroring the sun per ear, which this project no longer does -- the mirror
existed to cancel the winding bug. `scripts/sun_sweep.py` supersedes it by
asking the question of the sun's direction instead, over a wider range, on
corrected renders.

So those docstring rows are marked unreproducible rather than re-measured.
Re-running a measurement of a concept we have retired would produce a number
nobody should use.

Original note:

`/tmp/claude-1000` was cleared mid-session and took the 40-ear 2x2 with it.
`handedness_vs_illumination`'s docstring still cites the `opposite` cell,
whose raw data no longer exists. Either re-run it on corrected renders or mark
those rows as unreproducible. Run outputs go to `runs/` from now on, and
long jobs launch under `setsid` — a `nohup` alone died with the session.

## Not on this list, on purpose

- The three filename-ranked checkpoint selectors are **theirs** and they have
  fixed them. Nothing here touches those paths.
- The ROI refinement loop costing 46% of pipeline latency is theirs, reported,
  and on their backlog.
