"""Flag wrap fits that slid along the scan: the reference landmarks moved as a
block, so a single rigid move explains most of the pipeline's disagreement.

Usage: slid_fit_flag.py OUT_CSV RUN_DIR [RUN_DIR...]

Reads per-ear npz files with `pred` and `gt_lm3d_v9` (validate_scans.py,
validate_wrapped_ears.py). For each ear: the median error, then the
prediction rigidly aligned to the reference (Kabsch, no scale): `move` is
the median distance that rigid move carries the points, `resid` what is
left. A slid fit is a big move; a landmarker failure is a big residual.

FULL-PIPELINE PREDICTIONS ONLY. Ear-only runs (validate_wrapped_ears.py)
carry their own systematic 8-10 deg framing rotation, so every ear looks
moved; AudioEar, which has no heads, cannot be screened this way.

Surface distance cannot see this (pp73, P0057_left: references on the scan to
0.05 mm, wrap 0.66 mm from it), which is why it needs the prediction.
"""
import csv
import glob
import pathlib
import sys

import numpy as np

MOVE_MM = 4.0         # rigid move larger than this: the reference slid
GROSS_MM = 50.0       # median error above this is the PIPELINE (wrong head frame), not the fit


def rigid(P, G):
    a, b = P.mean(0), G.mean(0)
    U, _, Vt = np.linalg.svd((P - a).T @ (G - b))
    R = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
    return R, a, b


def score(P, G):
    ok = np.isfinite(P).all(1)
    e = np.linalg.norm(P - G, axis=1)
    R, a, b = rigid(P[ok], G[ok])
    r = np.linalg.norm((P[ok] - a) @ R.T + b - G[ok], axis=1)
    deg = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    mv = float(np.median(np.linalg.norm((P[ok] - a) @ R.T + b - P[ok], axis=1)))
    return dict(err=float(np.nanmedian(e)), move=mv, resid=float(np.median(r)), deg=deg,
                shift=float(np.linalg.norm(b - a)),
                slid=bool(MOVE_MM < mv and np.nanmedian(e) < GROSS_MM), gross=bool(np.nanmedian(e) >= GROSS_MM))


if __name__ == "__main__":
    rows = []
    for d in sys.argv[2:]:
        for f in sorted(glob.glob(str(pathlib.Path(d) / "*.npz"))):
            z = np.load(f)
            if "pred" not in z or not np.isfinite(z["pred"]).any():
                continue
            rows.append(dict(ear=pathlib.Path(f).stem, run=pathlib.Path(d).name, **score(z["pred"], z["gt_lm3d_v9"])))
    with open(sys.argv[1], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    fl = [r for r in rows if r["slid"]]
    print(f"{len(rows)} ears, {len(fl)} flagged slid, {sum(r['gross'] for r in rows)} gross (pipeline head frame)")
    for r in sorted(rows, key=lambda r: -r["move"])[:25]:
        print(f"  {'SLID' if r['slid'] else 'GROSS' if r['gross'] else '    ':5} {r['run']:>22} {r['ear']:<14} err {r['err']:5.2f}  "
              f"move {r['move']:5.2f}  resid {r['resid']:5.2f}  {r['deg']:5.1f} deg  shift {r['shift']:5.1f} mm")
