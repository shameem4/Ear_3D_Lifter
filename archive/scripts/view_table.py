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


def soft_point(o, d, w, irls=3):
    """Weighted fit over ALL views, then robustify on the residuals.

    best-K spends the table twice over as a hard include/exclude: it drops
    view K+1 even when it is barely worse than view K, and then treats views 1
    and K as equals. Weighting by 1/table^2 keeps every view and lets the
    table say how much each is worth.

    The reason that needs IRLS rather than a plain weighted solve: the table
    is a PRIOR, learned over other subjects, and it cannot know that on THIS
    ear a good view happens to have been occluded by hair or that the detector
    put the point on the scalp. A gross 2D error in a high-weight view would
    otherwise dominate the fit, which is the robustness RANSAC was providing.
    Tukey biweight on the ray residuals, so the prior decides who to trust and
    the data can still overrule it.
    """
    from ear3d.triangulate import lsq_point, ray_distance
    w = np.asarray(w, float)
    ok = np.isfinite(o).all(1) & np.isfinite(d).all(1) & np.isfinite(w) & (w > 0)
    if ok.sum() < 2:
        return None
    O, D_, W = o[ok], d[ok], w[ok]
    q = lsq_point(O, D_, W)
    if q is None or not irls:
        return q
    for _ in range(irls):
        r = np.asarray(ray_distance(q, O, D_), float)
        sig = np.median(r) / 0.6745
        if not np.isfinite(sig) or sig <= 0:
            break
        u = r / (4.685 * sig)
        rw = np.where(u < 1.0, (1.0 - u ** 2) ** 2, 0.0)
        if (W * rw > 0).sum() < 2:
            break
        q2 = lsq_point(O, D_, W * rw)
        if q2 is None:
            break
        q = q2
    return q


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="runs/raycache")
    p.add_argument("--thresh", type=float, default=0.03)
    p.add_argument("--ks", type=int, nargs="*", default=[3, 5, 8, 12, 25])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--irls", type=int, default=3,
                   help="IRLS iterations for the soft arms (0 = plain "
                        "weighted least squares, no robustness)")
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

    # 1/err^2, with a never-detected (landmark, view) treated as worthless
    # rather than as NaN, which would poison the whole weight vector.
    tw = 1.0 / (np.nan_to_num(table, nan=1e6) ** 2 + 1e-6)

    def run_soft(K, irls):
        out = []
        for k in TE:
            z = D[k]
            o, d, lm_f, s = z["o"], z["d"], z["lm_f"], float(z["scale"])
            for j in range(55):
                w = tw[:, j].copy()
                if K < V:
                    m = np.zeros(V)
                    m[order[:K, j]] = 1.0
                    w = w * m
                q = soft_point(o[:, j], d[:, j], w, irls)
                out.append(np.linalg.norm(q - lm_f[j]) * s if q is not None else np.nan)
        return np.array(out)

    def show(nm, K, x):
        f = x[np.isfinite(x)]
        print(f"  {nm:16s} {K:3d} {np.percentile(f,50):7.2f} "
              f"{np.percentile(f,90):7.2f} {100*np.mean(f>5):7.2f}% "
              f"{100*np.mean(f>10):7.3f}% {100*np.mean(~np.isfinite(x)):5.1f}%")

    print(f"\npaired error on {len(TE)} HELD-OUT ears, mm")
    print(f"  {'rule':16s} {'K':>3s} {'p50':>7s} {'p90':>7s} {'>5mm':>8s} {'>10mm':>8s} {'nan':>6s}")
    for method in ("ransac",):
        for K in a.ks:
            for nm, fn in (("best-K", best), ("random-K", rand), ("oracle-K", orac)):
                if K >= V and nm != "best-K":
                    continue
                if K >= V:
                    nm, fn = "all views", allv
                show(nm, K, run(fn, method, K))

    # THE SOFT ARMS. Same split, same table, same metric, so the only thing
    # that differs from best-K is how the table is spent.
    print(f"\n  soft weighting, w = 1/table^2 (irls={a.irls})")
    for K in a.ks:
        show("soft-K+irls" if a.irls else "soft-K", K, run_soft(K, a.irls))
    show("soft-all+irls" if a.irls else "soft-all", V, run_soft(V, a.irls))
    show("soft-all no irls", V, run_soft(V, 0))


if __name__ == "__main__":
    main()
