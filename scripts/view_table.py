"""A per-(landmark, view) accuracy table, used to choose whose rays to trust.

RANSAC is handed all 25 views and left to vote. But which views see a given
landmark well is not random -- the concha is unoccluded face-on while the
descending helix is a silhouette edge that only resolves from one side -- and
that is a property of the SCHEME and the viewing angle, not of the subject.
So it can be learned once from a corpus and applied to a new scan, where the
camera angle is known and nothing else need be.

    table[j, v] = how accurately view v places landmark j, over train ears

and at fit time landmark j is triangulated from only its best K views.

THREE CONTROLS, because "fewer views" and "better views" are easy to confuse
and the first is free:

    random-K    the same number of views, chosen at random per landmark. If
                best-K does not beat this, the table is not doing anything.
    all         every view, which is what ships today.
    oracle-K    the best K views for THIS ear judged by its OWN ground truth.
                Unreachable, and the ceiling any table is working against.

SPLIT BY SUBJECT, NOT BY EAR. The two ears of one head share a scan and a
scanner; putting one in train and the other in test would let the table
memorise the subject rather than the scheme.

Usage:
    python scripts/view_table.py --cache runs/raycache
"""

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def fit(o, d, use, method, thresh):
    """Fit one point from the chosen rays. Returns None if too few."""
    from ear3d.triangulate import lsq_point, ransac_point
    sel = [i for i in use if np.isfinite(o[i]).all() and np.isfinite(d[i]).all()]
    if len(sel) < 2:
        return None
    O, D = o[sel], d[sel]
    if method == "ransac":
        q = ransac_point(O, D, thresh)
        q = q[0] if isinstance(q, tuple) else q
        return q
    return lsq_point(O, D)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="runs/raycache")
    p.add_argument("--thresh", type=float, default=0.03)
    p.add_argument("--ks", type=int, nargs="*", default=[3, 5, 8, 12, 25])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    files = sorted(pathlib.Path(a.cache).glob("*.npz"))
    if len(files) < 10:
        sys.exit(f"only {len(files)} cached ears in {a.cache}")
    D = {f.stem: np.load(f, allow_pickle=True) for f in files}
    subj = sorted({k.rsplit("_", 1)[0] for k in D})
    rng = np.random.default_rng(a.seed)
    rng.shuffle(subj)
    tr = set(subj[: len(subj) // 2]); te = set(subj[len(subj) // 2:])
    TR = [k for k in D if k.rsplit("_", 1)[0] in tr]
    TE = [k for k in D if k.rsplit("_", 1)[0] in te]
    V = D[TR[0]]["uv"].shape[0]
    print(f"{len(D)} ears, {len(subj)} subjects -> {len(TR)} train / {len(TE)} test "
          f"ears, {V} views\n")

    # the table: median 2D error in pixels, per landmark per view, over train
    err = np.full((len(TR), V, 55), np.nan)
    for i, k in enumerate(TR):
        z = D[k]
        err[i] = np.linalg.norm(z["uv"] - z["gt_uv"], axis=2)
    table = np.nanmedian(err, axis=0)              # (V, 55)
    order = np.argsort(table, axis=0)              # best views first, per landmark

    ang = D[TR[0]]["angles"]
    print("the table, as median 2D error in px -- a few landmarks:")
    from ear3d.scheme import group_of
    for j in (0, 10, 25, 40, 52):
        b = order[0, j]; w = order[-1, j]
        print(f"  j={j:2d} {group_of(j):22s} best view "
              f"({ang[b][0]:+.0f},{ang[b][1]:+.0f}) {table[b, j]:5.1f} px   "
              f"worst ({ang[w][0]:+.0f},{ang[w][1]:+.0f}) {table[w, j]:5.1f} px")

    def run(pick, method, K):
        out = []
        for k in TE:
            z = D[k]
            o, d, lm_f, s = z["o"], z["d"], z["lm_f"], float(z["scale"])
            gt_err = np.linalg.norm(z["uv"] - z["gt_uv"], axis=2)
            for j in range(55):
                use = pick(j, K, gt_err)
                q = fit(o[:, j], d[:, j], use, method, a.thresh)
                out.append(np.linalg.norm(q - lm_f[j]) * s if q is not None else np.nan)
        return np.array(out)

    best = lambda j, K, g: order[:K, j]
    rand = lambda j, K, g: rng.choice(V, size=min(K, V), replace=False)
    allv = lambda j, K, g: np.arange(V)
    orac = lambda j, K, g: np.argsort(np.nan_to_num(g[:, j], nan=np.inf))[:K]

    print(f"\npaired error on {len(TE)} HELD-OUT ears, mm")
    print(f"  {'rule':16s} {'K':>3s} {'p50':>7s} {'p90':>7s} {'>5mm':>8s} {'>10mm':>8s} {'nan':>6s}")
    for method in ("ransac",):
        for K in a.ks:
            for nm, fn in (("best-K", best), ("random-K", rand), ("oracle-K", orac)):
                if K >= V and nm != "best-K":
                    continue
                if K >= V:
                    nm, fn = "all views", allv
                x = run(fn, method, K)
                f = x[np.isfinite(x)]
                print(f"  {nm:16s} {K:3d} {np.percentile(f,50):7.2f} "
                      f"{np.percentile(f,90):7.2f} {100*np.mean(f>5):7.2f}% "
                      f"{100*np.mean(f>10):7.3f}% {100*np.mean(~np.isfinite(x)):5.1f}%")


if __name__ == "__main__":
    main()
