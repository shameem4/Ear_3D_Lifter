"""Illumination direction, presented handedness, and ear side, crossed.

Three things were tangled together in this project's gross-error rate and each
was mistaken for the others at some point. This runs the design that separates
them, and cuts the table.

    illumination   --light anatomy | opposite, which anchors the sun to each
                   EAR'S OWN anatomy rather than to the canonical frame, so
                   both sides get the same treatment in every cell
    handedness     --tta pair, which returns the as-rendered arm, the mirrored
                   arm and their COORDINATE fusion from ONE pair of forward
                   passes on the same render. Coordinate, not heatmap: for a
                   soft-argmax decoder averaging the probability maps is
                   algebraically the same operation (77f821c), so the `avg`
                   row below is the only fusion there is to measure
    side           recorded per row, so it is a stratum rather than a factor
                   that has to be read out of an interaction

SETTLED 2026-10-14: THERE IS NO SIDE RESIDUAL TO EXPLAIN. It was two bugs of
this project's own, and the second of them had been flattering the very ears
the first made look worse.

    THE GATE. `np.mean(d > gate)` is False for NaN, and d_snap_mm is NaN when
    the predicted pixel is background -- the landmark missed the head
    entirely. The worst outcome the 2D model can produce was scoring as a
    SUCCESS, and the arms do not miss equally, so it was not a wash. Fixed:
    see rate().

    THE WINDING. canon2scan's `inverse` has det < 0 for every right ear and
    load_ear applied it without reversing the triangle winding, so every
    right-ear render was shaded with normals pointing INTO the head. Fixed in
    escape_audit.load_ear. Re-running the same 112 ears with the winding as
    the only difference: left ears byte-identical, depth buffers and
    silhouettes bit-identical (so the lift was never affected), right ears
    2.89% -> 6.79% past 10 mm on 54 of 56 subjects, and the side gap REVERSED
    from left-worse +0.91pp to left-better -2.99pp.

    THE SUN RULE WAS CANCELLING THE WINDING. Inverting normals is close to
    negating the light, and c243e8f mirrors the sun for reflected geometry, so
    the two cancelled. With the winding right the mirror double-counts and
    front-lights every right ear. Turning it off takes right ears 6.79% ->
    3.17%, worse on 1 of 56. `--light frame` is now the default.

CORRECTED BASELINE. 112 HUTUBS ears (56 subjects) x 25 views x 55 points,
winding fixed, sun never mirrored, an off-head miss counted as an escape:

    arm      all     left    right   off head
    none    3.48%   3.80%   3.17%    0.660%
    flip    2.95%   3.05%   2.86%    1.068%
    avg     2.72%   2.92%   2.52%    0.717%

    side gap, paired within subject:  none +0.63pp (38/56, p=0.018)
                                      flip +0.19pp (33/56, p=0.118)
                                      avg  +0.40pp (40/56, p=0.009)

WHAT IT SETTLES.

1. THE SIDE GAP IS SMALL AND MARGINAL, and the flip arm's does not reach
   significance at 56 subjects. Every mechanism once proposed for it is now
   moot: it was the renderer. The one exclusion that still means something is
   the ground truth, because it never involved a render -- the wrap fits are
   equally good on both sides over all 112 ears paired within subject
   (fit->scan median 29/56 p=0.97, p95 27/56 p=0.49, scan->fit median 30/56
   p=0.81, p95 29/56 p=0.71). Its 40-ear subset read right-better by 11-14%
   at p~0.2, and at full power the direction REVERSES and the effect
   vanishes, which is the best cautionary example this project has produced.

2. COORDINATE FUSION BEATS REPLACEMENT: 2.72% against 2.95%, fusion better on
   69 of 112 ears, p<0.001. This reverses what this file said twice. The
   earlier "replacement wins on 33 of 40 ears at p=4.1e-04" was the gate bug
   forgiving the flip arm's extra off-head misses (1.068% against 0.660%),
   and Ear_Landmarker predicted the reversal from a mechanism: their model
   carries an ANTI-EQUIVARIANT x offset from a sub-pixel defect in its
   flip/translation augmentation, which a coordinate average cancels exactly
   and a replacement only moves to the other sign. Their matched A/B
   collapses it and leaves accuracy unchanged in either direction.

   Coordinate fusion, not heatmap: for a soft-argmax decoder the expectation
   is linear, so probability-space fusion is algebraically the same operation
   (77f821c). Only logit-space differs, and it sharpens.

3. THE MIRROR IS WORTH 1.18x, not the 1.58x first reported. Each correction
   took a bite: 1.58 -> 1.31 (gate) -> 1.25 (56 subjects) -> 1.18 (winding).

NOT RE-DERIVED, AND SO NOT TO BE QUOTED. The illumination factor (1.75x, then
1.68x), the 1.247x presentation multiplier, the one-sided per-ear sun result
(+4.20pp left against -0.67 right) and everything in the light2x2 cells were
all measured with half the corpus shaded inside out. The mechanism behind
them survives -- a front-lit occluding contour has no edge, and that is
visible in a render -- but the numbers do not.

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


def rate(d, gate):
    """Fraction of landmarks past the gate, COUNTING the ones that missed.

    d_snap_mm is nan when the predicted pixel is background: the ray hit
    nothing, so there is no 3D point and no distance to the ear. That is the
    worst thing the 2D model can do, not a missing observation, and at any
    gate "not within `gate` mm of the ear" is true of it. So it counts.

    `~(d <= gate)` counts it; `d > gate` is False for nan and scores it as a
    SUCCESS, which is what this script did until f65c713 and what changed the
    fusion result -- see the docstring. Everything else in the repository drops
    these rows instead (`escape_audit.summarise`, `escape_report`), which is
    defensible for a single arm but not for comparing two, because the arms do
    not miss equally: over the anatomy cell flip misses 0.744% of the time
    against fusion's 0.493%, so dropping hands the flip arm 138 extra free
    passes.
    """
    return np.mean(~(np.asarray(d, float) <= gate))


def off_head(d):
    """Fraction whose pixel was background, so the lift had nothing to hit."""
    return np.mean(~np.isfinite(np.asarray(d, float)))


def side_test(D, k, gate, lights, arms):
    """The side residual, paired WITHIN SUBJECT.

    The two ears of one head share a scan, a wrap fit and a subject, so the
    subject is the independent unit and 112 ears are 56 of them, not 112. Run
    unpaired over ears this reads as a much stronger effect than it is.
    """
    from scipy import stats
    per = collections.defaultdict(dict)
    for l in lights:
        for arm in arms:
            t = collections.defaultdict(list)
            for x in k:
                if x[4] == arm:
                    t[x[0]].append(D[l][x])
            for e, v in t.items():
                per[(l, arm)][e] = 100 * rate(v, gate)
    print("\nside residual, left minus right, paired within subject:")
    for l in lights:
        for arm in arms:
            r = per[(l, arm)]
            subj = sorted({e.rsplit("_", 1)[0] for e in r})
            pairs = [(r[f"{p}_left"], r[f"{p}_right"]) for p in subj
                     if f"{p}_left" in r and f"{p}_right" in r]
            if len(pairs) < 3:
                continue
            x, y = np.array(pairs).T
            print(f"  {l:9s} {arm:5s} {np.mean(x - y):+6.2f}pp   left worse on "
                  f"{sum(a > b for a, b in pairs)}/{len(pairs)} subjects   "
                  f"p={stats.wilcoxon(x, y).pvalue:.3f}")


def compare(before, after, gate=10.0):
    """Two runs of the same cell, paired per ear. For the winding fix.

    THE LEFT EARS ARE THE CONTROL. Only reflected geometry was re-wound, so
    every left-ear render is byte-identical between the two runs and their
    rates must agree. If they do not, something other than the winding moved
    and nothing below can be read as the fix's doing.
    """
    from scipy import stats
    A, B = load(before), load(after)
    k = sorted(set(A) & set(B))
    if not k:
        sys.exit("no shared landmark-views between the two runs")
    arms = sorted({x[4] for x in k})
    ears = sorted({x[0] for x in k})
    L = lambda e: e.endswith("_left")
    per = lambda D, arm, e: 100 * rate(
        [D[x] for x in k if x[4] == arm and x[0] == e], gate)

    print(f"\n{len(k) // max(len(arms), 1)} shared landmark-views per arm, "
          f"{len(ears)} ears\n")
    print(f"{'arm':6s} {'side':6s} {'before':>8s} {'after':>8s} {'change':>9s} "
          f"{'worse on':>10s} {'p':>9s}")
    for arm in arms:
        for lab, sel in (("left", L), ("right", lambda e: not L(e))):
            es = [e for e in ears if sel(e)]
            x = np.array([per(A, arm, e) for e in es])
            y = np.array([per(B, arm, e) for e in es])
            p = stats.wilcoxon(x, y).pvalue if np.any(x != y) else float("nan")
            print(f"{arm:6s} {lab:6s} {x.mean():7.3f}% {y.mean():7.3f}% "
                  f"{y.mean() - x.mean():+8.3f}pp {int((y > x).sum()):6d}/{len(es)} "
                  f"{p:9.4f}")
        print()

    ident = all(abs(per(A, arm, e) - per(B, arm, e)) < 1e-9
                for arm in arms for e in ears if L(e))
    print("CONTROL: left-ear rates are "
          + ("IDENTICAL between the runs, as they must be.\n"
             if ident else
             "NOT identical -- something other than the winding moved, and the\n"
             "         right-ear change below cannot be attributed to the fix.\n"))

    print("side gap, left minus right, paired within subject:")
    for lab, D in (("before", A), ("after", B)):
        for arm in arms:
            r = {e: per(D, arm, e) for e in ears}
            subj = sorted({e.rsplit("_", 1)[0] for e in ears})
            pr = [(r[f"{t}_left"], r[f"{t}_right"]) for t in subj
                  if f"{t}_left" in r and f"{t}_right" in r]
            if len(pr) < 3:
                continue
            x, y = np.array(pr).T
            print(f"  {lab:7s} {arm:5s} {np.mean(x - y):+6.2f}pp   left worse on "
                  f"{sum(u > v for u, v in pr)}/{len(pr)} subjects   "
                  f"p={stats.wilcoxon(x, y).pvalue:.4f}")


def table(out_dir, gate=10.0):
    D = {l: load(out_dir / f"{l}.csv") for l in LIGHTS
         if (out_dir / f"{l}.csv").exists()}
    lights = [l for l in LIGHTS if l in D]
    k = sorted(set.intersection(*[set(v) for v in D.values()]))
    side = lambda x: "left" if x[0].endswith("_left") else "right"
    print(f"\n{len(k) // len(ARMS)} paired landmark-views per arm, "
          f"escape past {gate:.0f} mm\n")
    print(f"{'light':10s} {'arm':6s} {'all':>8s} {'left':>8s} {'right':>8s}"
          f" {'off head':>9s}")
    cell = {}
    for l in lights:
        for arm in ARMS:
            kk = [x for x in k if x[4] == arm]
            v = {w: np.array([D[l][x] for x in kk
                              if w == "all" or side(x) == w])
                 for w in ("all", "left", "right")}
            cell[(l, arm)] = v
            print(f"{l:10s} {arm:6s} " +
                  " ".join(f"{100 * rate(v[w], gate):7.2f}%"
                           for w in ("all", "left", "right")) +
                  f" {100 * off_head(v['all']):8.3f}%")
        print()

    a = cell[("anatomy", "none")]["all"]
    f, u = cell[("anatomy", "flip")]["all"], cell[("anatomy", "avg")]["all"]
    if "opposite" in lights:
        o = cell[("opposite", "none")]["all"]
        print(f"illumination alone   {rate(o, gate) / rate(a, gate):.2f}x")
    print(f"handedness alone     {rate(a, gate) / rate(f, gate):.2f}x  "
          f"(mirror, with the sun already correct)")
    if "opposite" in lights:
        print(f"both                 {rate(o, gate) / rate(f, gate):.2f}x")
    print(f"coordinate fusion vs replacement under corrected lighting: "
          f"{100 * rate(u, gate):.2f}% against {100 * rate(f, gate):.2f}%")

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
                pe[e][(l, arm)] = 100 * rate(v, gate)
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
    p.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), default=None,
                   help="two CSVs of the same cell, paired per ear; for the "
                        "winding fix, where left ears are the control")
    a = p.parse_args()
    if a.compare:
        compare(pathlib.Path(a.compare[0]), pathlib.Path(a.compare[1]), a.gate)
        return
    out = pathlib.Path(a.table_only or a.out)
    if a.table_only is None:
        ears = sorted(f.stem for f in (WRAP / "corpus/canon2scan").glob("pp*.npz"))[:a.ears]
        run_cells(out, ears, a.grid, a.grid_step)
    table(out, a.gate)


if __name__ == "__main__":
    main()
