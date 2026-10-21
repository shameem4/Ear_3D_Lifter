#!/usr/bin/env python3
"""Score a validate_pipeline run: the median AND the tail, per region.

Usage: score_heldout.py RUN_DIR [RUN_DIR...] [--lm3d FILE ...]

For each run (validate_pipeline.py output: per head pred_{L,R} and
gt_{L,R}_<ref>), against the reference it stored, and against any extra
lm3d files (templates/ name or a path) mapped the same way. Prints the
per-ear median over 55 points (median, mean, p95), the helix (0-34) and the
rest, the upper helix (0-8) and the lobe (15-19), and the share of points
beyond 5 mm -- the tail is where the 2D model loses, and a median hides it.

COMPARE ON ONE REFERENCE. Switching lm3d moved the score 0.13 mm with no
change to the model (2026-10-19), more than most model differences. The
baseline for the shipped pipeline is runs/e2e_heldout_lmw: 1.574 mm vs
lm3d_v11, 4.03% beyond 5 mm (fixed_conv_s42 ep433, before Ear_Landmarker's
2026-10-20 inference changes).
"""
import argparse
import glob
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ear3d import corpus                                             # noqa: E402
from ear3d.gt import inv, inv_gt, load_reference                     # noqa: E402


def stats(E, lab):
    E = np.array(E); m = np.nanmedian(E, axis=1)
    print(f"  {lab:34s} ears {len(E):3d}  median {np.median(m):.3f}  mean {m.mean():.3f}  p95 {np.percentile(m, 95):.2f}  "
          f"helix {np.nanmedian(E[:, :35]):.2f}  rest {np.nanmedian(E[:, 35:]):.2f}  "
          f"0-8 {np.nanmedian(E[:, :9]):.2f}  lobe {np.nanmedian(E[:, 15:20]):.2f}  >5mm {np.mean(~(E <= 5)):.2%}")


def main():
    a = argparse.ArgumentParser(); a.add_argument("runs", nargs="+"); a.add_argument("--lm3d", nargs="*", default=[]); a = a.parse_args()
    extra = {pathlib.Path(f).stem.split("_canonical")[0]: load_reference(f) for f in a.lm3d}
    for run in a.runs:
        files = sorted(glob.glob(f"{run}/*.npz"))
        if not files:
            print(run, "-- no results"); continue
        z0 = np.load(files[0]); stored = sorted({k[5:] for k in z0.files if k.startswith("gt_L_")})
        rows = {f"vs {k} (stored)": [] for k in stored} | {f"vs {k}": [] for k in extra}
        for f in files:
            z = np.load(f); hd = pathlib.Path(f).stem
            for sd in "LR":
                P = z[f"pred_{sd}"]; nm = f"{hd}_{'left' if sd == 'L' else 'right'}"
                for k in stored:
                    rows[f"vs {k} (stored)"].append(np.linalg.norm(P - z[f"gt_{sd}_{k}"], axis=1))
                for k, (lmz, names) in extra.items():
                    T = inv(f"{hd}_left", corpus.CURRENT) @ np.linalg.inv(inv_gt(nm))
                    G = lmz["landmarks"][names.index(nm)][:55]; G = (np.c_[G, np.ones(55)] @ T.T)[:, :3] @ z["Rr"].T
                    rows[f"vs {k}"].append(np.linalg.norm(P - G, axis=1))
        ck = str(z0["ckpt"]).split("checkpoints/")[-1] if "ckpt" in z0.files else "ckpt not recorded"
        print(f"{run}: {len(files)} heads, {ck}")
        for lab, E in rows.items():
            stats(E, lab)


if __name__ == "__main__":
    main()
