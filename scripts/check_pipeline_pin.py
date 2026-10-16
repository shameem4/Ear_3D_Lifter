"""Behavioural pin on the whole lift path. Fails if anything moved it.

This repo consumes two sibling repos that ship independently, and twice in
one day a change on their side reached a measurement here without anyone
choosing it: Ear_Landmarker turned flip fusion on by default BETWEEN cells
of a running sweep, and moved SHIPPED_RUN to a new checkpoint. Both were
caught by luck rather than by a guard.

PIN BEHAVIOUR, NOT CONSTRUCTION. Checking that the pipeline builds with the
right flags proves nothing about what it produces. BlazeEar's own near miss
this week was a tensor hash that matched first try while the scores were
wrong by an order of magnitude, because the hash certified the preprocessing
and said nothing about the call site; their fix was to pin behaviourally over
a shared sequence, since "a constants-only pin is exactly what let the demo
drift". This runs the real path end to end -- render, detect, ROI refine,
landmark, lift, snap, score -- and compares the output.

Three ears, nine views, about a minute. It is deliberately not a unit test:
it needs a GPU and the corpus, and `pytest tests/` stays fast and offline.

WHEN IT FAILS, IT HAS DONE ITS JOB AND YOU DECIDE. A deliberate improvement
moves these numbers too; the point is that nothing moves them silently. Re-cut
the expected table with --update and say in the commit what moved and why.

Usage:
    python scripts/check_pipeline_pin.py
    python scripts/check_pipeline_pin.py --update     # after a deliberate change
"""

import argparse
import collections
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

EARS = ("pp1_left", "pp10_right", "pp11_left")
GRID, STEP = 15.0, 15.0

# (landmark-views, median mm, escapes past 10 mm), measured 2026-10-15 at
# sun +1.2, fixed_conv_s42/433, their flip fusion on, detector_frame_tta on.
EXPECTED = {
    "pp1_left":   (495, 0.663047, 14.0),
    "pp10_right": (495, 0.350295,  0.0),
    "pp11_left":  (495, 0.381766,  6.0),
}


def measure():
    from escape_audit import DEFAULTS, audit_ear
    from ear3d.twod import make_pipeline
    cfg = dict(DEFAULTS)
    cfg["flip_fusion"] = None            # inherit theirs, which is the point
    cfg["detector_frame_tta"] = True
    pipe, ck = make_pipeline(cfg)
    angles = [(float(y), float(p))
              for y in np.arange(-GRID, GRID + 1e-9, STEP)
              for p in np.arange(-GRID, GRID + 1e-9, STEP)]
    out = {}
    for nm in EARS:
        rows = audit_ear(nm, cfg, pipe, angles, light="frame")
        rows = rows[0] if isinstance(rows, tuple) else rows
        d = np.array([r["d_snap_mm"] for r in rows
                      if r.get("detected") and r["j"] >= 0], float)
        f = d[np.isfinite(d)]
        out[nm] = (len(d), float(np.median(f)), float(np.sum(~(d <= 10))))
    return out, ck


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--update", action="store_true",
                   help="print a new EXPECTED table after a deliberate change")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="median tolerance in mm; counts must match exactly")
    a = p.parse_args()

    got, ck = measure()
    print(f"checkpoint {ck.parent.name}/{ck.name}\n")
    if a.update:
        print("EXPECTED = {")
        for k in sorted(got):
            n, m, e = got[k]
            print(f'    "{k}":{" " * (12 - len(k))}({n}, {m:.6f}, {e}),')
        print("}")
        return

    bad = []
    print(f"  {'ear':12s} {'n':>5s} {'median mm':>22s} {'escapes':>16s}")
    for k in sorted(EXPECTED):
        en, em, ee = EXPECTED[k]
        if k not in got:
            bad.append(f"{k}: produced nothing"); continue
        gn, gm, ge = got[k]
        ok = gn == en and abs(gm - em) <= a.tol and ge == ee
        if not ok:
            bad.append(f"{k}: ({en}, {em:.6f}, {ee}) -> ({gn}, {gm:.6f}, {ge})")
        print(f"  {k:12s} {gn:5d} {em:10.6f} -> {gm:9.6f} "
              f"{ee:7.0f} -> {ge:6.0f}   {'ok' if ok else 'MOVED'}")
    if bad:
        print("\nTHE LIFT PATH HAS MOVED. That is not automatically wrong -- a")
        print("deliberate change moves it too -- but nothing should move it by")
        print("surprise, and the sibling repos ship on their own schedule:")
        for b in bad:
            print(f"  {b}")
        print("\nIf you made the change, re-cut with --update and say what moved.")
        sys.exit(1)
    print("\nOK: the lift path reproduces exactly.")


if __name__ == "__main__":
    main()
