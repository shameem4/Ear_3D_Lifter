#!/usr/bin/env python3
"""Score a validate_pipeline run: the median AND the tail, per region.

Usage: score_heldout.py RUN_DIR [RUN_DIR...] [--lm3d FILE ...]

For each run (validate_pipeline.py output: per head pred_{L,R} and
gt_{L,R}_<ref>), against the reference it stored, and against any extra
lm3d files (templates/ name or a path) mapped the same way. Prints the
per-ear median over 55 points (median, mean, p95), the helix (0-34) and the
rest, the upper helix (0-8) and the lobe (15-19), and the share of points
beyond 5 mm -- the tail is where the 2D model loses, and a median hides it.

--contour splits each error into ACROSS the reference's strip polyline
(contour following) and ALONG it (spacing). Spacing is a convention, and a
reference derived from one 2D model carries that model's: compare models on
the across part, ideally against hand-drawn curves (lm3d_v7).

COMPARE ON ONE REFERENCE. Switching lm3d moved the score 0.13 mm with no
change to the model (2026-10-19), more than most model differences. The
reference is derived from a 2D model, so it favours that model: v4_fc05_s42
1.561 mm vs lm3d_v12 (its own), 1.721 vs v11; fixed_conv_s42 1.574 vs v11
(its own), 1.679 vs v12. Against the hand-marked v7 they tie (3.080 / 3.099).
Rank models on v7 too, and on repeat runs: the same ear moves ~0.1 mm median
between runs under GPU load.
"""
import argparse
import glob
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ear3d import corpus                                             # noqa: E402
from ear3d.gt import inv, inv_gt, load_reference                     # noqa: E402


STRIPS = ((0, 20), (20, 35), (35, 39), (40, 50), (50, 55))         # 39, the canal, is a point


def across_along(P, G):
    """Per point: distance to the reference's own strip polyline (contour
    following, convention-free) and the arc-length offset along it (spacing,
    a convention). NaN where the prediction is missing or the point is 39."""
    ac, al = np.full(55, np.nan), np.full(55, np.nan)
    for a, b in STRIPS:
        C = G[a:b]; seg = np.diff(C, axis=0); L = np.linalg.norm(seg, axis=1); cum = np.r_[0, np.cumsum(L)]
        for j in range(a, b):
            if not np.isfinite(P[j]).all(): continue
            t = np.clip(((P[j] - C[:-1]) * seg).sum(1) / L ** 2, 0, 1)
            d = np.linalg.norm(P[j] - (C[:-1] + t[:, None] * seg), axis=1); i = int(np.argmin(d))
            ac[j], al[j] = d[i], cum[i] + t[i] * L[i] - cum[j - a]
    return ac, al


def contour_stats(P_G, lab):
    A = np.array([across_along(P, G)[0] for P, G in P_G]); B = np.array([abs(across_along(P, G)[1]) for P, G in P_G])
    print(f"  {'  ' + lab:34s} across: median {np.nanmedian(A):.2f}  0-8 {np.nanmedian(A[:, :9]):.2f}  lobe {np.nanmedian(A[:, 15:20]):.2f}  "
          f">2mm {np.nanmean(A > 2):.1%} | along |.|: median {np.nanmedian(B):.2f}  0-8 {np.nanmedian(B[:, :9]):.2f}")


def stats(E, lab):
    E = np.array(E); m = np.nanmedian(E, axis=1)
    print(f"  {lab:34s} ears {len(E):3d}  median {np.median(m):.3f}  mean {m.mean():.3f}  p95 {np.percentile(m, 95):.2f}  "
          f"helix {np.nanmedian(E[:, :35]):.2f}  rest {np.nanmedian(E[:, 35:]):.2f}  "
          f"0-8 {np.nanmedian(E[:, :9]):.2f}  lobe {np.nanmedian(E[:, 15:20]):.2f}  >5mm {np.mean(~(E <= 5)):.2%}")


def main():
    a = argparse.ArgumentParser(); a.add_argument("runs", nargs="+"); a.add_argument("--lm3d", nargs="*", default=[])
    a.add_argument("--contour", action="store_true", help="also split error across vs along each reference curve"); a = a.parse_args()
    extra = {pathlib.Path(f).stem.split("_canonical")[0]: load_reference(f) for f in a.lm3d}
    for run in a.runs:
        files = sorted(glob.glob(f"{run}/*.npz"))
        if not files:
            print(run, "-- no results"); continue
        z0 = np.load(files[0]); stored = sorted({k[5:] for k in z0.files if k.startswith("gt_L_")})
        rows = {f"vs {k} (stored)": [] for k in stored} | {f"vs {k}": [] for k in extra}
        pairs = {lab: [] for lab in rows}
        for f in files:
            z = np.load(f); hd = pathlib.Path(f).stem
            for sd in "LR":
                P = z[f"pred_{sd}"]; nm = f"{hd}_{'left' if sd == 'L' else 'right'}"
                for k in stored:
                    rows[f"vs {k} (stored)"].append(np.linalg.norm(P - z[f"gt_{sd}_{k}"], axis=1)); pairs[f"vs {k} (stored)"].append((P, z[f"gt_{sd}_{k}"]))
                for k, (lmz, names) in extra.items():
                    T = inv(f"{hd}_left", corpus.CURRENT) @ np.linalg.inv(inv_gt(nm))
                    G = lmz["landmarks"][names.index(nm)][:55]; G = (np.c_[G, np.ones(55)] @ T.T)[:, :3] @ z["Rr"].T
                    rows[f"vs {k}"].append(np.linalg.norm(P - G, axis=1)); pairs[f"vs {k}"].append((P, G))
        ck = str(z0["ckpt"]).split("checkpoints/")[-1] if "ckpt" in z0.files else "ckpt not recorded"
        print(f"{run}: {len(files)} heads, {ck}")
        for lab, E in rows.items():
            stats(E, lab)
            if a.contour:
                contour_stats(pairs[lab], "contour")


if __name__ == "__main__":
    main()
