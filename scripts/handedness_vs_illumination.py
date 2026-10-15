"""Illumination direction, presented handedness, and ear side, crossed.

Three things were tangled together in this project's gross-error rate and each
was mistaken for the others at some point. This runs the design that separates
them, and cuts the table.

    illumination   --light anatomy | opposite, which anchors the sun to each
                   EAR'S OWN anatomy rather than to the canonical frame, so
                   both sides get the same treatment in every cell
    handedness     --tta pair, which returns the as-rendered arm, the mirrored
                   arm and their coordinate fusion from ONE pair of forward
                   passes on the same render
    side           recorded per row, so it is a stratum rather than a factor
                   that has to be read out of an interaction

MEASURED, 40 HUTUBS ears x 25 views x 55 points per cell, rate past 10 mm:

    light      arm        all     left    right
    anatomy    none      3.24%   3.76%   2.72%
    anatomy    flip      2.05%   2.32%   1.77%
    anatomy    fused     2.30%   2.56%   2.05%
    opposite   none      5.68%   7.97%   3.39%
    opposite   flip      3.00%   3.68%   2.33%
    opposite   fused     3.37%   4.33%   2.42%

WHAT IT SETTLES.

1. Illumination and handedness are BOTH real and INDEPENDENT. Lighting alone is
   1.75x (3.24 against 5.68). The mirror alone, with lighting already correct,
   is 1.58x (2.05 against 3.24, better on 37 of 40 ears, Wilcoxon p=3.6e-07).
   Together 2.8x. Correcting the sun does not shrink the handedness gap, which
   is what you expect of a property of the 2D model rather than of the render.

2. REPLACEMENT STILL BEATS FUSION under corrected lighting: 2.05% against
   2.30%, replacement better on 33 of 40 ears, p=4.1e-04. Ear_Landmarker
   predicted the opposite, on the account that fusion loses only where one arm
   is systematically much better. That account is NOT refuted here -- its
   premise is. The sibling session's light2x2, a different sun parameterisation
   on a different process, agrees in both of its cells: sun +0.3 flip 2.32%
   against avg 2.49% (24/40 ears, p=0.019), sun -0.3 flip 2.72% against avg
   3.19% (27/40, p=1.7e-04). Four cells, two sun parameterisations, two
   processes, replacement in every one -- though the same checkout, checkpoint
   and wrap reference throughout, so this is a replicate of the MEASUREMENT and
   not of the pipeline. Fixing the sun did not move the arms closer together,
   because the thing separating them is the handedness bias and lighting does
   not touch it. The middling-disagreement regime that would test the account
   is still unvisited. Fusion does beat doing nothing (2.30 against 3.24,
   39 of 40).

3. A SIDE RESIDUAL APPEARS IN EVERY CELL BUT IS MARGINAL. At the best
   lighting and with the mirror, left 2.32% against right 1.77%. Paired within
   subject, which is the right unit because the two ears of one head share a
   scan and a subject, it does not reach significance on these 20 subjects:

       anatomy/none   +1.04pp   13/20 subjects   p=0.084
       anatomy/flip   +0.55pp   14/20 subjects   p=0.053
       anatomy/avg    +0.51pp   14/20 subjects   p=0.083

   Consistently signed and consistently 13-14 of 20, so it is suggestive and
   nothing more. Five mechanisms are excluded: lighting (each side is at its
   own optimum), presented handedness (identical across sides -- canal-x minus
   rim-centroid-x is -125.0 px left against -124.9 right), geometric chirality
   and frame orientation (the sibling session: signed tetra volumes agree in
   sign, mean ear frames agree to 0.006), and the ground truth itself.

   THE GROUND-TRUTH TEST IS A CLEAN NULL, and worth the detail because its
   40-ear subset was not. Fit-to-scan and scan-to-fit distance over all 112
   HUTUBS ears, paired within subject: fit->scan median L 0.4909 / R 0.5059
   (29/56, p=0.97), p95 (27/56, p=0.49), scan->fit median (30/56, p=0.81), p95
   (29/56, p=0.71). On the same 40 ears as this table it read right-better by
   11-14% at p~0.2, which looked like the right direction and the right size;
   at full power the direction reverses and the effect vanishes. The sixth
   possibility is that there is nothing to explain.

ONE-SIDED, NOT ANTISYMMETRIC. c243e8f described the lighting effect as a clean
swap, from two ears at one view. Over 40 ears at 25 views it is not: the
per-ear preference for the sun's sign is +4.20 points on left ears (19 of 20
positive) and -0.67 on right ears (8 of 20), Mann-Whitney p=3.4e-07. The side
structure is real and highly significant, but left ears care a great deal and
right ears barely care, which "the sun lights the two sides from opposite
directions" does not predict.

Usage:
    python scripts/handedness_vs_illumination.py                  # run + table
    python scripts/handedness_vs_illumination.py --table-only DIR # re-cut
"""

import argparse
import collections
import csv
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

WRAP = ROOT.parent / "wrap_work"
LIGHTS = ("anatomy", "opposite")
ARMS = ("none", "flip", "avg")


def run_cells(out_dir, ears, grid, step):
    out_dir.mkdir(parents=True, exist_ok=True)
    for light in LIGHTS:
        csv_path = out_dir / f"{light}.csv"
        if csv_path.exists() and csv_path.stat().st_size:
            print(f"skip {csv_path.name}")
            continue
        cmd = [sys.executable, "-u", str(ROOT / "scripts" / "escape_audit.py"),
               "--ears", *ears, "--grid", str(grid), "--grid-step", str(step),
               "--light", light, "--tta", "pair", "--out", str(csv_path)]
        print(f"### {light}")
        subprocess.run(cmd, check=True)


def load(path):
    d = {}
    for r in csv.DictReader(open(path)):
        if r["detected"] != "1" or int(r["j"]) < 0:
            continue
        d[(r["ear"], r["yaw"], r["pitch"], r["j"], r["tta"])] = float(r["d_snap_mm"])
    return d


def side_test(D, k, gate, lights, arms):
    """The side residual, paired WITHIN SUBJECT.

    The two ears of one head share a scan, a wrap fit and a subject, so the
    subject is the independent unit and 112 ears are 56 of them, not 112. Run
    unpaired over ears this reads as a much stronger effect than it is.
    """
    from scipy import stats
    rate = collections.defaultdict(dict)
    for l in lights:
        for arm in arms:
            t = collections.defaultdict(list)
            for x in k:
                if x[4] == arm:
                    t[x[0]].append(D[l][x])
            for e, v in t.items():
                rate[(l, arm)][e] = 100 * np.mean(np.array(v) > gate)
    print("\nside residual, left minus right, paired within subject:")
    for l in lights:
        for arm in arms:
            r = rate[(l, arm)]
            subj = sorted({e.rsplit("_", 1)[0] for e in r})
            pairs = [(r[f"{p}_left"], r[f"{p}_right"]) for p in subj
                     if f"{p}_left" in r and f"{p}_right" in r]
            if len(pairs) < 3:
                continue
            x, y = np.array(pairs).T
            print(f"  {l:9s} {arm:5s} {np.mean(x - y):+6.2f}pp   left worse on "
                  f"{sum(a > b for a, b in pairs)}/{len(pairs)} subjects   "
                  f"p={stats.wilcoxon(x, y).pvalue:.3f}")


def table(out_dir, gate=10.0):
    D = {l: load(out_dir / f"{l}.csv") for l in LIGHTS
         if (out_dir / f"{l}.csv").exists()}
    lights = [l for l in LIGHTS if l in D]
    k = sorted(set.intersection(*[set(v) for v in D.values()]))
    side = lambda x: "left" if x[0].endswith("_left") else "right"
    print(f"\n{len(k) // len(ARMS)} paired landmark-views per arm, "
          f"escape past {gate:.0f} mm\n")
    print(f"{'light':10s} {'arm':6s} {'all':>8s} {'left':>8s} {'right':>8s}")
    cell = {}
    for l in lights:
        for arm in ARMS:
            kk = [x for x in k if x[4] == arm]
            v = {w: np.array([D[l][x] for x in kk
                              if w == "all" or side(x) == w])
                 for w in ("all", "left", "right")}
            cell[(l, arm)] = v
            print(f"{l:10s} {arm:6s} " +
                  " ".join(f"{100 * np.mean(v[w] > gate):7.2f}%"
                           for w in ("all", "left", "right")))
        print()

    a = cell[("anatomy", "none")]["all"]
    f, u = cell[("anatomy", "flip")]["all"], cell[("anatomy", "avg")]["all"]
    if "opposite" in lights:
        o = cell[("opposite", "none")]["all"]
        print(f"illumination alone   {np.mean(o > gate) / np.mean(a > gate):.2f}x")
    print(f"handedness alone     {np.mean(a > gate) / np.mean(f > gate):.2f}x  "
          f"(mirror, with the sun already correct)")
    if "opposite" in lights:
        print(f"both                 {np.mean(o > gate) / np.mean(f > gate):.2f}x")
    print(f"fusion vs replacement under corrected lighting: "
          f"{100 * np.mean(u > gate):.2f}% against {100 * np.mean(f > gate):.2f}%")

    side_test(D, k, gate, lights, ARMS)

    from scipy import stats
    pe = collections.defaultdict(dict)
    for l in lights:
        for arm in ARMS:
            t = collections.defaultdict(list)
            for x in k:
                if x[4] == arm:
                    t[x[0]].append(D[l][x])
            for e, v in t.items():
                pe[e][(l, arm)] = 100 * np.mean(np.array(v) > gate)
    ears = sorted(pe)
    print(f"\nper-ear, paired over the {len(ears)} ears:")
    for p, q in ((("anatomy", "none"), ("anatomy", "flip")),
                 (("anatomy", "none"), ("anatomy", "avg")),
                 (("anatomy", "flip"), ("anatomy", "avg"))):
        x = [pe[e][p] for e in ears]
        y = [pe[e][q] for e in ears]
        print(f"  {p[1]:5s} {np.mean(x):5.2f}%  vs {q[1]:5s} {np.mean(y):5.2f}%   "
              f"second better on {sum(j < i for i, j in zip(x, y))}/{len(ears)}   "
              f"Wilcoxon p={stats.wilcoxon(x, y).pvalue:.2e}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="runs/handedness_vs_illumination")
    p.add_argument("--ears", type=int, default=40)
    p.add_argument("--grid", type=float, default=30.0)
    p.add_argument("--grid-step", type=float, default=15.0)
    p.add_argument("--table-only", metavar="DIR", default=None)
    p.add_argument("--gate", type=float, default=10.0)
    a = p.parse_args()
    out = pathlib.Path(a.table_only or a.out)
    if a.table_only is None:
        ears = sorted(f.stem for f in (WRAP / "corpus/canon2scan").glob("pp*.npz"))[:a.ears]
        run_cells(out, ears, a.grid, a.grid_step)
    table(out, a.gate)


if __name__ == "__main__":
    main()
