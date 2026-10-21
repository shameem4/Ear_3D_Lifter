"""The sun's direction against the gross-error rate, on corrected renders.

WHY THIS HAD TO BE REDONE. The illumination factor on file -- 1.75x, then
1.68x after the gate fix -- was measured as `anatomy` against `opposite`,
which are ANCHORS: rules for whether to mirror the sun per ear. That
comparison is meaningless now. The mirror existed to compensate for reflected
heads being shaded with their normals pointing inward (see
escape_audit.load_ear), and with the winding fixed it double-counts; `frame`,
which never mirrors, is the default. So the question "how much does lighting
direction cost" has to be asked of the DIRECTION itself again.

It is also a different question than it was. Before, the sun was thought to
do one thing to each side, because the two sides supposedly presented
mirrored lighting geometry. They do not: ear_frame puts every ear in left
form and right ears reach it through a reflection, so with correct normals
both sides face the same camera and the same sun. The prediction this sweep
tests is therefore that the two sides move TOGETHER, and a side split would
mean something is still handed.

The sweep is in x only. The sun is (x, -0.4, -0.9) and the helix's occluding
contour runs roughly vertically in frame, so x is the component that decides
whether that contour is grazed or front-lit. y and z are not swept and the
optimum found here is an optimum along one line, not in the sphere.

Usage:
    python scripts/sun_sweep.py --ears 40                  # run + table
    python scripts/sun_sweep.py --table-only runs/sun_sweep
"""

import argparse
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

WRAP = ROOT.parent / "wrap_work"
# THE FIRST SWEEP RAN -0.6..+0.6 AND CAME OUT MONOTONE WITH THE BEST AT THE
# EDGE, so the optimum is outside it and the range had to grow. Past about
# +1.5 the sun is nearly side-on and the ear loses its fill, so a turning
# point is expected in here somewhere.
XS = (-0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2, 1.8)


def tag(x):
    return f"sun{x:+.1f}".replace(".", "p")


def run_cells(out_dir, ears, grid, step, xs):
    out_dir.mkdir(parents=True, exist_ok=True)
    for x in xs:
        p = out_dir / f"{tag(x)}.csv"
        if p.exists() and p.stat().st_size:
            print(f"skip {p.name}")
            continue
        print(f"### sun x = {x:+.1f}")
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "escape_audit.py"),
                        "--ears", *ears, "--grid", str(grid), "--grid-step", str(step),
                        "--sun-x", str(x), "--tta", "pair", "--out", str(p)], check=True)


def table(out_dir, gate=10.0, arm="none"):
    from scipy import stats
    from handedness_vs_illumination import load, rate, off_head
    cells = {}
    for p in sorted(out_dir.glob("sun*.csv")):
        if not p.stat().st_size:
            continue
        x = float(p.stem.replace("sun", "").replace("p", "."))
        cells[x] = load(p)
    if not cells:
        sys.exit(f"no cells in {out_dir}")
    xs = sorted(cells)
    k = sorted(set.intersection(*[{y for y in c if y[4] == arm} for c in cells.values()]))
    ears = sorted({y[0] for y in k})
    L = lambda e: e.endswith("_left")
    print(f"\n{len(k)} paired landmark-views, {len(ears)} ears, arm '{arm}', "
          f"escape past {gate:.0f} mm\n")
    print(f"{'sun x':>7s} {'all':>8s} {'left':>8s} {'right':>8s} {'off head':>9s}"
          f" {'L-R':>8s}")
    per = {}
    for x in xs:
        D = cells[x]
        sel = lambda w: [D[y] for y in k if w == "all" or L(y[0]) == (w == "left")]
        per[x] = np.array([100 * rate([D[y] for y in k if y[0] == e], gate)
                           for e in ears])
        r = {w: 100 * rate(sel(w), gate) for w in ("all", "left", "right")}
        print(f"{x:+7.1f} {r['all']:7.2f}% {r['left']:7.2f}% {r['right']:7.2f}% "
              f"{100*off_head(sel('all')):8.3f}% {r['left']-r['right']:+7.2f}pp")

    best = min(xs, key=lambda x: per[x].mean())
    worst = max(xs, key=lambda x: per[x].mean())
    print(f"\nillumination factor {per[worst].mean() / per[best].mean():.2f}x  "
          f"(worst {worst:+.1f} at {per[worst].mean():.2f}%, "
          f"best {best:+.1f} at {per[best].mean():.2f}%)")
    print(f"  paired over ears: best better on "
          f"{int((per[best] < per[worst]).sum())}/{len(ears)}, "
          f"p={stats.wilcoxon(per[worst], per[best]).pvalue:.2e}")
    cur = 0.3
    if cur in per and best != cur:
        print(f"  the shipped sun ({cur:+.1f}) is {per[cur].mean():.2f}%, "
              f"{per[cur].mean() - per[best].mean():+.2f}pp off the best, "
              f"p={stats.wilcoxon(per[cur], per[best]).pvalue:.2e}")
    elif cur in per:
        print(f"  the shipped sun ({cur:+.1f}) IS the best of those swept")

    print("\nDO THE TWO SIDES MOVE TOGETHER? With the winding fixed they should:")
    print("ear_frame puts every ear in left form, so both face the same sun.")
    for x in xs:
        a = np.array([per[x][i] for i, e in enumerate(ears) if L(e)])
        b = np.array([per[x][i] for i, e in enumerate(ears) if not L(e)])
        u = stats.mannwhitneyu(a, b).pvalue
        print(f"  {x:+.1f}  left {a.mean():6.2f}%  right {b.mean():6.2f}%  "
              f"Mann-Whitney p={u:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="runs/sun_sweep")
    p.add_argument("--ears", type=int, default=40)
    p.add_argument("--grid", type=float, default=30.0)
    p.add_argument("--grid-step", type=float, default=15.0)
    p.add_argument("--gate", type=float, default=10.0)
    p.add_argument("--arm", default="none")
    p.add_argument("--xs", type=float, nargs="*", default=None, metavar="X",
                   help="sun x values to run; default is the full sweep")
    p.add_argument("--table-only", metavar="DIR", default=None)
    a = p.parse_args()
    out = pathlib.Path(a.table_only or a.out)
    if a.table_only is None:
        ears = sorted(f.stem for f in (WRAP / __import__("ear3d.corpus", fromlist=["get"]).get("canon2scan")).glob("pp*.npz"))[:a.ears]
        run_cells(out, ears, a.grid, a.grid_step, a.xs or XS)
    table(out, a.gate, a.arm)


if __name__ == "__main__":
    main()
