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

TWO INSTRUMENT BUGS WERE FOUND ON 2026-10-14 AND EVERY NUMBER BELOW MOVED.
Read this section before quoting anything from this file.

    THE GATE. `np.mean(d > gate)` is False for NaN, and d_snap_mm is NaN when
    the predicted pixel is background -- the landmark missed the head entirely.
    The worst outcome the 2D model can produce was scoring as a SUCCESS. The
    arms do not miss equally (flip 0.744%, fusion 0.493%, none 0.422%), so this
    was not a wash: it manufactured the fusion result. Fixed -- see rate().

    THE WINDING. canon2scan's `inverse` has det < 0 for every right ear, and
    load_ear applied it without reversing the triangle winding, so every
    right-ear render was shaded with normals pointing INTO the head (+0.720
    outward on pp1_left, -0.720 on pp1_right). Fixed in escape_audit.load_ear.
    The depth buffer never used normals, so distances are sound -- but EVERY
    LEFT-AGAINST-RIGHT NUMBER BELOW WAS MEASURED ON THOSE RENDERS and is
    confounded. A re-wound run is what settles the side rows.

MEASURED POST-GATE-FIX, PRE-WINDING-FIX. 112 HUTUBS ears (56 subjects) x 25
views x 55 points, anatomy cell, rate past 10 mm with a miss counted:

    arm      all     left    right   off head
    none    3.35%   3.80%   2.89%    0.492%
    flip    2.68%   3.05%   2.32%    0.701%
    avg     2.65%   2.92%   2.38%    0.503%

WHAT IT SETTLES.

1. ILLUMINATION AND HANDEDNESS ARE BOTH REAL AND INDEPENDENT, but smaller than
   first reported. On the 40-ear 2x2 with the gate fixed: illumination alone
   1.68x (was 1.75x), the mirror alone 1.31x (was 1.58x), together 2.21x (was
   2.78x). At 112 ears the mirror is 1.25x, better on 84 of 112, p=9.8e-10.
   Correcting the sun does not shrink the mirror's gain.

2. REPLACEMENT AND COORDINATE FUSION ARE TIED under corrected lighting: 2.68%
   against 2.65%, fusion better on 48 of 112, p=0.89. The earlier claim that
   replacement wins, on 33 of 40 ears at p=4.1e-04, was the gate bug -- it
   forgave the flip arm's extra off-head misses. Re-cut across four cells with
   misses counted, replacement wins clearly only in the WORST-lit cell
   (sun -0.3, p=0.007), weakly in the next (opposite, p=0.045), and the two
   best-lit cells are dead heats. That is a gradient in the direction of
   Ear_Landmarker's magnitude account, which I had reported as refuted.

   Coordinate fusion, not heatmap: for a soft-argmax decoder, averaging the
   probability maps is algebraically the same operation (77f821c).

3. THE SIDE GAP IS SIGNIFICANT AND CONFOUNDED. Paired within subject over 56
   subjects, left minus right: none +0.91pp (38/56, p=0.004), flip +0.73pp
   (41/56, p=0.000), avg +0.54pp (37/56, p=0.011). The 40-ear version read
   p=0.05-0.08 and I downgraded it on that; counting the off-head misses turns
   it significant, because that is exactly where the asymmetry lives (left
   0.662% against right 0.182% in the none arm). But the right ears in this
   table were rendered inside out, so this does not yet say whether the 2D
   model prefers a side or my renderer does.

   Four mechanisms were excluded on the old renders and those exclusions are
   now suspect too: lighting, presented handedness (ear_frame puts EVERY ear in
   left form, so presentation is constant by construction), chirality and frame
   orientation. One exclusion survives untouched, because it never involved a
   render: the wrap fits themselves are equally good on both sides, over all
   112 ears paired within subject (fit->scan median 29/56 p=0.97, p95 27/56
   p=0.49, scan->fit median 30/56 p=0.81, p95 29/56 p=0.71). Its 40-ear subset
   read right-better by 11-14% at p~0.2 -- right direction, right size -- and
   at full power the direction REVERSES and the effect vanishes.

4. THE MIRROR'S GAIN IS A PRESENTATION EFFECT. Because ear_frame presents every
   ear in left form, the mirror presents it in right form. The gain is then the
   same multiplier on both sides and the side gap passes through it unchanged:
   left 3.800 -> 3.047 and right 2.892 -> 2.319, both 1.247x. Per-ear medians
   1.204x and 1.285x, Mann-Whitney p=0.27, so read this as "no detectable side
   difference in the multiplier". Ear_Landmarker measures a right-favoured bias
   on photographs across nine models, three architectures and both
   initialisations; this is the same sign in a setup where presentation is
   controlled rather than stratified. Caveat: mirroring the image mirrors the
   shading too, and lighting is worth 1.68x, so this does not separate
   presentation from "the sun now comes from the other side".

ONE-SIDED, NOT ANTISYMMETRIC -- AND NOW SUSPECT. c243e8f described the lighting
effect as a clean swap, from two ears at one view. Over 40 ears at 25 views the
per-ear preference for the sun's sign was +4.20 points on left ears (19 of 20)
and -0.67 on right (8 of 20), Mann-Whitney p=3.4e-07. That is a left-against-
right comparison on inside-out right-ear renders, so it is pending the re-wound
run like the rest of section 3.

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
    a = p.parse_args()
    out = pathlib.Path(a.table_only or a.out)
    if a.table_only is None:
        ears = sorted(f.stem for f in (WRAP / "corpus/canon2scan").glob("pp*.npz"))[:a.ears]
        run_cells(out, ears, a.grid, a.grid_step)
    table(out, a.gate)


if __name__ == "__main__":
    main()
