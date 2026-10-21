"""Which views are worth lifting from, and can the pipeline tell without GT?

Escape past 10 mm varies 40x across a 25-view grid -- 0.13% face-on against
5.19% at the worst corner -- so view choice is worth more than anything in
the TTA stack. The question is whether a rule can pick the good ones.

THE RULE THAT LOOKED BEST IS A TAUTOLOGY, and that is the main thing this
script exists to stop anyone re-discovering. "Fraction of rays that hit
nothing" is read off the render's own depth buffer, needs no ground truth,
and correlates with a view's escape rate at Spearman +0.558 (p=1.8e-228)
where camera angle manages +0.326 and detection confidence +/-0.01.
Dropping views containing a missed ray appears to take 1.290% to 0.855%
while keeping 88% of views -- more than the entire TTA stack.

It is worth nothing. A missed ray IS an escape under this project's
convention (see ear3d/escape.py), so dropping those views removes escapes by
construction. Scored on the landmarks that DID hit, the dropped views are
0.610% against the kept views' 0.855% -- they are BETTER. Of the -0.435pp
headline, -0.028pp survives, in the wrong direction.

CAMERA ANGLE IS REAL. It is independent of the metric, so it cannot fail the
same way, and it survives the same test:

    rule                 landmarks   escape   excluding misses
    all 25 views            154000   1.290%             0.827%
    angle <= 30 deg          80080   0.763%             0.585%
    angle <= 21.3 deg        55440   0.483%             0.410%
    angle <= 15 deg          30800   0.373%             0.335%

Both components rise with angle -- off-head 0.039% -> 0.779% from the inner
ring to the outer, and genuine escape 0.335% -> 1.091% -- which is also why
the missed-ray rule looked predictive: it was re-reading the angle through
the metric.

So view restriction works and costs coverage: half the grid for a third of
the error. Whether that trade is available depends on the consumer, and
nothing here makes it free.

Usage:
    python scripts/view_quality.py runs/newsun/ship.csv
"""

import argparse
import collections
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--gate", type=float, default=10.0)
    a = p.parse_args()

    from scipy import stats
    from ear3d.escape import escape_rate

    R = [r for r in csv.DictReader(open(a.csv))
         if r["detected"] == "1" and int(r["j"]) >= 0]
    d = np.array([float(r["d_snap_mm"]) for r in R])
    ang = np.array([(float(r["yaw"]) ** 2 + float(r["pitch"]) ** 2) ** 0.5 for r in R])
    fin = np.isfinite(d)

    V = collections.defaultdict(list)
    for r, x in zip(R, d):
        V[(float(r["yaw"]), float(r["pitch"]))].append(x)
    ys = sorted({k[0] for k in V}); ps = sorted({k[1] for k in V})
    print("escape by view\n  yaw  " + "".join(f"{q:>8.0f}" for q in ps))
    for y in ys:
        print(f"  {y:5.0f}" + "".join(
            f"{100 * escape_rate(V[(y, q)], a.gate):7.2f}%" for q in ps))

    pv = collections.defaultdict(lambda: collections.defaultdict(list))
    for r, x in zip(R, d):
        k = pv[(r["ear"], r["yaw"], r["pitch"])]
        k["d"].append(x); k["hit"].append(int(r["hit_snap"]))
        k["c"].append(float(r["conf"]))
        k["ang"] = (float(r["yaw"]) ** 2 + float(r["pitch"]) ** 2) ** 0.5
    rows = [(100 * escape_rate(v["d"], a.gate), v["ang"], np.mean(v["c"]),
             1 - np.mean(v["hit"])) for v in pv.values()]
    e, g, c, miss = (np.array(x) for x in zip(*rows))
    print(f"\npredictors over {len(rows)} ear-views, none using ground truth:")
    for nm, x in (("camera angle", g), ("detection confidence", c),
                  ("rays that hit nothing", miss)):
        s = stats.spearmanr(x, e)
        print(f"  {nm:24s} Spearman r={s.statistic:+.3f}  p={s.pvalue:.1e}")

    keep = [k for k in pv if all(pv[k]["hit"])]
    drop = [k for k in pv if not all(pv[k]["hit"])]
    print("\nTHE TAUTOLOGY CHECK -- a missed ray is itself an escape, so score the")
    print("landmarks that DID hit and see whether the dropped views were worse:")
    for nm, ks in (("kept  (no ray missed)", keep), ("dropped (some missed)", drop)):
        v = np.array([x for k in ks for x in pv[k]["d"]])
        f = v[np.isfinite(v)]
        print(f"  {nm:24s} escape {100 * escape_rate(v, a.gate):6.3f}%   "
              f"excluding misses {100 * np.mean(f > a.gate):6.3f}%")

    print("\nangle restriction, which is independent of the metric:")
    print(f"  {'rule':22s} {'landmarks':>10s} {'escape':>8s} {'excl. misses':>13s}")
    for nm, m in (("all views", np.ones(len(d), bool)),
                  ("angle <= 30", ang <= 30 + 1e-6),
                  ("angle <= 21.3", ang <= 21.3 + 1e-6),
                  ("angle <= 15", ang <= 15 + 1e-6)):
        print(f"  {nm:22s} {int(m.sum()):10d} "
              f"{100 * escape_rate(d[m], a.gate):7.3f}% "
              f"{100 * np.mean(d[m & fin] > a.gate):12.3f}%")


if __name__ == "__main__":
    main()
