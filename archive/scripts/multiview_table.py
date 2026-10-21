"""Multiview accuracy table from ray caches (multiview_collect_rays.py).

Usage: multiview_table.py CACHE_DIR [EAR_SUBSET_DIR]
Table views are learned on the other subject set (v4_s0 heads vs v4_s1 heads:
disjoint subjects), indexed by ANATOMICAL view (right-ear yaw mirrored).
"""
import sys, glob, pathlib, numpy as np
R0 = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_3D_Lifter"); sys.path.insert(0, str(R0))
import open3d as o3d
from ear3d.triangulate import lsq_point, ransac_point
cd = pathlib.Path(sys.argv[1])
fs = sorted(cd.glob("*.npz"))
if len(sys.argv) > 2: keep = {p.stem for p in pathlib.Path(sys.argv[2]).glob("*.npz")}; fs = [f for f in fs if f.stem in keep]
E = [dict(np.load(f, allow_pickle=True)) for f in fs]; names = [f.stem for f in fs]
# Ground truth from config's reference (gt_lm3d), not the copy stored in the
# ray cache, which is whatever reference was current when it was collected.
_g = {}; exec(open(R0 / "scripts/gt_check_ear_views.py").read().split("out = sys.argv[1]")[0], _g)
for nm_, e in zip(names, E):
    hd_, sd_ = nm_.rsplit("_", 1); e["gt"] = _g["gt_in_cache"](hd_, sd_, np.load(str(e["head"])))
grp = np.array([0 if "v4_s0" in str(e["head"]) else 1 for e in E])
ANG = E[0]["angles"]; V = len(ANG)
def anat(e):                       # view permutation so index = anatomical view
    if str(e["side"]) == "L": return np.arange(V)
    key = {tuple(a): i for i, a in enumerate(ANG)}
    return np.array([key[(-a[0], a[1])] for a in ANG])
for e in E:
    p = anat(e); e["o"], e["d"], e["uv"] = e["o"][p], e["d"][p], e["uv"][p]
def raydist(e):                    # (V,55) distance from GT to each ray, mm
    v = e["gt"][None] - e["o"]; return np.linalg.norm(v - (v * e["d"]).sum(-1, keepdims=True) * e["d"], axis=-1)
RD = np.array([raydist(e) for e in E])                      # (N,V,55)
tables = {g: np.nanmedian(RD[grp == g], axis=0) for g in (0, 1)}   # (V,55)
rng = np.random.default_rng(0)
def fuse(e, j, views, method, th=1.0):
    o, d = e["o"][views, j], e["d"][views, j]; ok = np.isfinite(o).all(1) & np.isfinite(d).all(1); o, d = o[ok], d[ok]
    if len(o) < 2: return np.full(3, np.nan)
    if method == "lsq": q = lsq_point(o, d)
    else: q = ransac_point(o, d, th); q = q[0] if isinstance(q, tuple) else q
    return np.full(3, np.nan) if q is None else q
scenes = {}
def reseat(e, P):
    h = str(e["head"])
    if h not in scenes:
        z = np.load(h); m = o3d.t.geometry.TriangleMesh(); sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.core.Tensor(z["V"].astype(np.float32)), o3d.core.Tensor(z["F"].astype(np.uint32))); scenes.clear(); scenes[h] = sc
    ok = np.isfinite(P).all(1); Q = P.copy()
    if ok.any(): Q[ok] = scenes[h].compute_closest_points(o3d.core.Tensor(P[ok].astype(np.float32)))["points"].numpy()
    return Q
rows = {k: [] for k in ("single face-on view (snapped)", "LSQ, all 25", "RANSAC, all 25", "RANSAC, random-8",
                        "RANSAC, table best-8", "  + re-seat onto head", "RANSAC, oracle best-8 (ceiling)")}
for n, e in enumerate(E):
    T = tables[1 - grp[n]]; out = {k: np.full((55, 3), np.nan) for k in rows}
    out["single face-on view (snapped)"] = e["p0"]
    for j in range(55):
        allv = np.arange(V)
        out["LSQ, all 25"][j] = fuse(e, j, allv, "lsq")
        out["RANSAC, all 25"][j] = fuse(e, j, allv, "ransac")
        out["RANSAC, random-8"][j] = fuse(e, j, rng.choice(V, 8, replace=False), "ransac")
        out["RANSAC, table best-8"][j] = fuse(e, j, np.argsort(T[:, j])[:8], "ransac")
        out["RANSAC, oracle best-8 (ceiling)"][j] = fuse(e, j, np.argsort(RD[n][:, j])[:8], "ransac")
    out["  + re-seat onto head"] = reseat(e, out["RANSAC, table best-8"])
    for k in rows: rows[k].append(np.linalg.norm(out[k] - e["gt"], axis=1))
print(f"GT reference: {_g['DEFAULTS']['gt_lm3d']}")
print(f"{cd.name}: {len(E)} ears ({(grp == 0).sum()} / {(grp == 1).sum()} by subject set), {V} views, RANSAC thresh 1.0 mm")
print(f"{'method':34s} {'median':>7s} {'mean':>6s} {'>5mm':>6s} {'>10mm':>6s} {'>20mm':>6s} {'fail':>5s}")
base = None
for k, D in rows.items():
    D = np.array(D); fail = ~np.isfinite(D); Dn = np.where(fail, np.nan, D); Dx = np.where(fail, np.inf, D)
    print(f"{k:34s} {np.nanmedian(Dn):7.2f} {np.nanmean(Dn):6.2f} {np.mean(Dx > 5) * 100:5.1f}% {np.mean(Dx > 10) * 100:5.1f}% {np.mean(Dx > 20) * 100:5.1f}% {fail.sum():5d}")
np.savez(cd / "../" / f"table_{cd.name}{'_sub' + str(len(E)) if len(sys.argv) > 2 else ''}.npz", **{k.strip(): np.array(v) for k, v in rows.items()}, names=names)
# split-half table agreement
from scipy.stats import spearmanr
print(f"table split-half Spearman (all cells): {spearmanr(tables[0].ravel(), tables[1].ravel()).correlation:+.3f}")
