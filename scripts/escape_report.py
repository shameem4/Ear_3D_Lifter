#!/usr/bin/env python3
"""Read an escape_audit csv and report what it says about view dependence.

Separate from the audit so the measurement can be re-cut without re-rendering
112 ears. Everything here is a RATE or a tail quantile -- the whole point is
that a median cannot see a landmark that flew onto the scalp, and the medians
in this project's history were what hid it.

    python scripts/escape_report.py runs/escape_grid2.csv
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.scheme import GROUP_NAMES                              # noqa: E402

GATE = 10.0        # millimetres; the pinna stands 15-20 mm off the skull


def load(path):
    rows = list(csv.DictReader(open(path)))
    good = [r for r in rows if r["j"] != "-1"]
    for r in good:
        for k in ("yaw", "pitch", "d_raw_mm", "d_snap_mm", "paired_mm", "conf"):
            r[k] = float(r[k]) if r.get(k) not in (None, "") else np.nan
        for k in ("j", "on_mask", "hit_raw", "hit_snap", "snapped"):
            r[k] = int(r[k]) if r.get(k) not in (None, "") else 0
    return rows, good


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--gate", type=float, default=GATE)
    a = p.parse_args()
    rows, r = load(a.csv)
    gate = a.gate

    ears = sorted({x["ear"] for x in r})
    views = sorted({(x["yaw"], x["pitch"]) for x in r})
    miss = [x for x in rows if x["j"] == "-1"]
    d = np.array([x["d_snap_mm"] for x in r])
    draw = np.array([x["d_raw_mm"] for x in r])
    om = np.array([x["on_mask"] for x in r])
    sn = np.array([x["snapped"] for x in r])
    ok = np.isfinite(d)

    print(f"{len(ears)} ears x {len(views)} view angles; {len(r)} landmark-views")
    nav = len(ears) * len(views)
    print(f"no ear detected at all: {len(miss)}/{nav} ear-views "
          f"({100 * len(miss) / max(nav, 1):.1f}%)")
    print()
    print("THE NUMBER A MEDIAN HIDES")
    print(f"  median             {np.median(d[ok]):8.2f} mm")
    print(f"  p90                {np.percentile(d[ok], 90):8.2f} mm")
    print(f"  p99                {np.percentile(d[ok], 99):8.2f} mm")
    print(f"  max                {d[ok].max():8.2f} mm")
    print()
    print("ESCAPE RATE -- landmarks this far from the fitted ear")
    print(f"  {'gate':>8}  {'raw lift':>9}  {'+cliff snap':>12}")
    for t in (2.0, 5.0, 10.0, 20.0, 30.0):
        print(f"  {t:6.0f}mm  {100 * (draw[np.isfinite(draw)] > t).mean():8.2f}%"
              f"  {100 * (d[ok] > t).mean():11.2f}%")
    print()

    far = ok & (d > gate)
    print(f"WHOSE FAULT, at the {gate:.0f} mm gate ({far.sum()} escapes)")
    if far.any():
        a_ = (far & (om == 0)).sum()
        b_ = (far & (om == 1)).sum()
        print(f"  pixel shows something other than the ear  {a_:6d}"
              f"  {100 * a_ / far.sum():6.1f}%   -> 2D placed it off the pinna")
        print(f"  pixel shows the ear, 3D still far away    {b_:6d}"
              f"  {100 * b_ / far.sum():6.1f}%   -> the depth sample itself")
    print()

    # Does the cliff snap reach the gross escapes, or only near-boundary ones?
    # Its search window is 8% of the ear's PIXEL extent, so a landmark far
    # outside the silhouette has no cliff within reach and cannot be moved.
    print("CAN THE CLIFF SNAP REACH THEM?")
    bands = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]
    print(f"  {'raw error':>14}  {'n':>7}  {'snap fired':>11}  {'fixed to <10mm':>15}")
    for lo, hi in bands:
        sel = np.isfinite(draw) & (draw >= lo) & (draw < hi)
        if not sel.any():
            continue
        fired = 100 * sn[sel].mean()
        fixed = 100 * (d[sel & ok] < gate).mean() if (sel & ok).any() else np.nan
        lab = f"{lo:.0f}-{hi:.0f}mm" if hi < 1e9 else f">{lo:.0f}mm"
        print(f"  {lab:>14}  {sel.sum():7d}  {fired:10.1f}%  {fixed:14.1f}%")
    print()

    # The reference is a closed shell with a back surface, and the scalp sits
    # only ~5 mm behind that back where the pinna stands 15-20 mm off the skull.
    # So a landmark that escaped onto the scalp can be NEAR the shell and score
    # small -- an under-count, and one that would flatter every number above.
    # These are the candidates: off the visible pinna, yet close to the shell.
    sus = ok & (om == 0) & (d < gate)
    print("IS THE SHELL'S BACK HIDING ESCAPES?")
    print(f"  off the visible pinna but within {gate:.0f} mm of the fitted shell:"
          f" {sus.sum()} ({100 * sus.sum() / max(ok.sum(), 1):.2f}% of all)")
    if sus.any():
        print(f"    their distances: median {np.median(d[sus]):.2f} mm  "
              f"p90 {np.percentile(d[sus], 90):.2f}  max {d[sus].max():.2f}")
        print(f"    for comparison, ON-pinna landmarks: median "
              f"{np.median(d[ok & (om == 1)]):.2f} mm  "
              f"p90 {np.percentile(d[ok & (om == 1)], 90):.2f}")
        print("    if these two distributions match, the off-pinna flag is firing")
        print("    on silhouette-edge pixels, not on real escapes; if the first")
        print("    is clearly larger, the shell IS absorbing some of them.")
    print()

    print(f"BY SCHEME GROUP -- escape rate past {gate:.0f} mm")
    print(f"  {'group':24s} {'escape':>8} {'off-pinna':>10} {'median':>9} {'p99':>9}")
    for g in GROUP_NAMES:
        sel = np.array([x["group"] == g for x in r])
        if not sel.any():
            continue
        dd = d[sel & ok]
        if not len(dd):
            continue
        print(f"  {g:24s} {100 * (dd > gate).mean():7.2f}% "
              f"{100 * (1 - om[sel].mean()):9.1f}% {np.median(dd):8.2f} "
              f"{np.percentile(dd, 99):8.2f}")
    print()

    # The view grid, as a grid. A column of numbers cannot show that the
    # failure is a direction.
    yaws = sorted({v[0] for v in views})
    pitches = sorted({v[1] for v in views})
    for title, fn in (("escape rate past %.0f mm, by view" % gate,
                       lambda s: 100 * (d[s & ok] > gate).mean() if (s & ok).any() else np.nan),
                      ("ear-views with no detection, by view",
                       None)):
        print(title.upper())
        print("        pitch" + "".join(f"{p:+7.0f}" for p in pitches))
        for y in yaws:
            line = f"  yaw {y:+5.0f}  "
            for p_ in pitches:
                if fn is None:
                    n = sum(1 for x in miss if float(x["yaw"]) == y and float(x["pitch"]) == p_)
                    line += f"{100 * n / max(len(ears), 1):6.0f}%"
                else:
                    sel = np.array([(x["yaw"], x["pitch"]) == (y, p_) for x in r])
                    v = fn(sel) if sel.any() else np.nan
                    line += "     --" if not np.isfinite(v) else f"{v:6.1f}%"
            print(line)
        print()

    print("PER EAR, worst first")
    per = []
    for e in ears:
        sel = np.array([x["ear"] == e for x in r])
        dd = d[sel & ok]
        if len(dd):
            per.append((100 * (dd > gate).mean(), e, len(dd), np.percentile(dd, 99)))
    per.sort(reverse=True)
    for rate, e, n, p99 in per[:10]:
        print(f"  {e:14s} {rate:6.2f}%   n={n:5d}   p99={p99:7.2f} mm")
    print(f"  ... {len(per)} ears, "
          f"{sum(1 for x in per if x[0] == 0)} with no escape at all")


if __name__ == "__main__":
    main()
