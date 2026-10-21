#!/usr/bin/env python3
"""Rank 2D checkpoints by how well their views AGREE in 3D -- no labels needed.

Usage: cross_view_agreement.py CACHE_DIR [CACHE_DIR...] [--thresh 1.0]
                                [--table ear3d/view_table_v12.npz --k 8]

Each CACHE_DIR is a runs/mvcache/<tag>/ produced by
scripts/caches/3_multiview_collect_rays.py: per ear, the 2D model's
prediction for 55 landmarks in 25 orbit views, as rays (o, d) in head mm.
For each (ear, landmark) the 25 rays are triangulated by ransac_point (the
pipeline's own robust intersection, 1 mm by default), and every view is
scored by the perpendicular distance of its ray from that consensus point.
That residual is in head mm, which is what the lift consumes: a view whose
ray misses the consensus is a view the 2D model placed inconsistently.

Why this and not single-photo NME: a 3D point is built from several views, and
it fails when they disagree. v4_fc05_s42 and fixed_conv_s42 tie end to end on
the hand-marked lm3d_v7 (3.080 vs 3.099 mm, view_table_v11), despite the newer
model's lower validation NME. This metric is the proxy that can rank checkpoints
without running the pipeline.

--table restricts each landmark to its best K views by that view table,
exactly as ear3d/pipeline.py does (right ears mirrored onto the anatomical
grid first; where the best K give no consensus, all views, as the pipeline
falls back). A landmark with no consensus even then is counted as a FAILURE
and reported, never dropped: a missing point is not an agreeing one. Without it all 25 views are scored, including extreme views the
pipeline never uses, which overstates what disagreement reaches the lift.

What it CANNOT see: a mistake that every view makes the same way (all 25 rays
agree on the wrong ridge) has a small residual. Agreement is necessary for a
good lift, not sufficient; the labelled check stays score_heldout.py on v7.

Reported, per cache and as paired differences between the first cache and each
other (same ears required, Wilcoxon over ears):
  - median and p90 residual, all points and per region (upper helix 0-8,
    lobe 15-19, antihelix/crura 26-28, the rest);
  - the share of scored (view, point) residuals over 5 mm, the tail;
  - failed landmarks (no consensus point at all).
The cache's `gt` is deliberately NOT used: each cache stores the reference
that was current when it was written (fixed_conv under lm3d_v11, v4fc05 under
v12; they differ by a median 0.76 mm), so it cannot be compared across caches.
"""
import argparse
import pathlib
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ear3d.triangulate import ransac_point, ray_distance             # noqa: E402

REGIONS = (("all 0-54", slice(0, 55)), ("upper helix 0-8", slice(0, 9)),
           ("lobe 15-19", slice(15, 20)), ("antihelix 26-28", slice(26, 29)))
GROSS = 5.0


def score_ear(f, thresh, sel=None):
    z = np.load(f, allow_pickle=True)
    o, d = z["o"], z["d"]                                   # (V, 55, 3)
    if sel is not None:
        table, mirror, K = sel
        if str(z["side"]) == "R":                           # anatomical view order
            o, d = o[mirror], d[mirror]
    V, L, _ = o.shape
    res = np.full((V, L), np.nan)
    pts = np.full((L, 3), np.nan)
    for k in range(L):
        vv = np.arange(V) if sel is None else np.argsort(table[:, k])[:K]
        p, _ = ransac_point(o[vv, k], d[vv, k], thresh, rng=np.random.default_rng(k))
        if p is None and sel is not None:                   # the pipeline's fallback
            vv = np.arange(V)
            p, _ = ransac_point(o[vv, k], d[vv, k], thresh, rng=np.random.default_rng(k))
        if p is None:
            continue
        pts[k] = p
        dn = d[vv, k] / np.linalg.norm(d[vv, k], axis=1, keepdims=True)
        res[vv, k] = ray_distance(p, o[vv, k], dn)
    return res, ~np.isfinite(pts).all(1)                   # residuals, failed landmarks


def summarise(caches, thresh, sel=None):
    out = {}
    for c in caches:
        files = sorted(pathlib.Path(c).glob("*.npz"))
        out[c] = {f.stem: score_ear(f, thresh, sel) for f in files}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("caches", nargs="+")
    ap.add_argument("--thresh", type=float, default=1.0,
                    help="RANSAC inlier distance in mm (pipeline default 1)")
    ap.add_argument("--table", help="view table npz: score only each landmark's best K views")
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    sel = None
    if a.table:
        t = np.load(a.table, allow_pickle=True)
        key = {tuple(x): i for i, x in enumerate(t["angles"])}
        sel = (t["table"], np.array([key[(-x[0], x[1])] for x in t["angles"]]), a.k)
    S = summarise(a.caches, a.thresh, sel)
    base = a.caches[0]
    common = sorted(set.intersection(*(set(S[c]) for c in a.caches)))
    print(f"{len(common)} ears common to all caches; RANSAC thresh {a.thresh} mm; "
          f"views: {'best %d by %s' % (a.k, a.table) if a.table else 'all 25'}\n")

    print("CROSS-VIEW RESIDUAL, mm (ray distance from the consensus point)")
    print(f"  {'cache':44s} {'region':16s} {'median':>7s} {'p90':>7s} {'>5mm':>7s}")
    for c in a.caches:
        for name, sl in REGIONS:
            r = np.concatenate([S[c][e][0][:, sl].ravel() for e in common])
            r = r[np.isfinite(r)]
            print(f"  {pathlib.Path(c).name[-44:]:44s} {name:16s} "
                  f"{np.median(r):7.3f} {np.percentile(r, 90):7.3f} "
                  f"{(r > GROSS).mean()*100:6.2f}%")
        fail = np.array([S[c][e][1] for e in common])
        print(f"  failed landmarks (no consensus): {fail.sum()} of {fail.size}\n")

    def gross(r):                                           # share of SCORED residuals; NaN = view not used
        r = r[np.isfinite(r)]
        return (r > GROSS).mean() * 100 if r.size else np.nan

    for c in a.caches[1:]:
        print(f"PAIRED over ears: {pathlib.Path(c).name}  minus  {pathlib.Path(base).name}")
        for name, sl in REGIONS:
            def per_ear(cc, fn):
                return np.array([fn(S[cc][e][0][:, sl]) for e in common])
            for lab, fn in (("mean residual", np.nanmean),
                            (">5mm share  ", gross)):
                x, y = per_ear(base, fn), per_ear(c, fn)
                ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
                dd = y - x
                p = wilcoxon(y, x).pvalue if np.any(dd != 0) else 1.0
                print(f"  {name:16s} {lab}  {x.mean():7.3f} -> {y.mean():7.3f}  "
                      f"delta {dd.mean():+.3f}  better on {(dd < 0).mean()*100:5.1f}% "
                      f"of ears  p={p:.1e}")
        print()


if __name__ == "__main__":
    main()
