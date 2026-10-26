import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
from concurrent.futures import ProcessPoolExecutor
from lap import *
from lap import _similarity

def rayfit(UV, cams, off, V, F, init, lam, c_mm, iters=6):
    """mesh of 468 points fit to every view's rays at once; Cauchy weights per (view, landmark) ray; Laplacian prior."""
    nv = len(cams); O = np.full((nv, 468, 3), np.nan); Dd = np.full((nv, 468, 3), np.nan)
    for k, cam in enumerate(cams):
        ok = np.isfinite(UV[k, :, 0])
        if ok.any():
            o, d = cam.rays(UV[k, ok] - off[k, ok]); O[k, ok] = o * s; Dd[k, ok] = d
    has = np.isfinite(O[..., 0])
    ok = np.isfinite(init).all(1); ref = _similarity(V0[ok], init[ok])(V0)
    LL = sp.kron(lam * (L.T @ L), sp.eye(3)).tocsr(); X = np.where(ok[:, None], init, ref)
    for _ in range(iters):
        v = X[None] - O; r = np.linalg.norm(v - (v * Dd).sum(2, keepdims=True) * Dd, axis=2)
        w = np.where(has, 1 / (1 + (np.nan_to_num(r) / c_mm) ** 2), 0)
        Pm = np.eye(3)[None, None] - np.nan_to_num(Dd)[..., :, None] * np.nan_to_num(Dd)[..., None, :]
        A = (w[..., None, None] * Pm).sum(0)                                    # 468 x 3 x 3
        b = (w[..., None] * (Pm @ np.nan_to_num(O)[..., None])[..., 0]).sum(0)
        M = sp.block_diag(list(A)).tocsr() + LL
        X = sla.spsolve(M.tocsc(), b.ravel() + LL @ ref.ravel()).reshape(468, 3)
        X, _ = seat(X, V, F)
    return X

def one3(j):
    z = np.load(RUN / f"{H[j]}.npz"); G = np.load(RUN / f"{H[j]}_gtd.npz")["G"]
    off = np.nan_to_num(np.nanmean(OFF[[i for i in range(len(H)) if i != j]], 0))
    cams = cameras(z["E"]); UV = z["UV"][:, :468]
    P, _, _ = triangulate(UV, cams, off, FACE["ransac_mm"] / s)
    Ps, _ = seat(P * s, z["V"], z["F"])
    out = {"current": Ps}
    for lam in (0.3, 1, 3):
        for c in (1.0, 2.0):
            out[f"ray l{lam} c{c}"] = rayfit(UV, cams, off, z["V"], z["F"], Ps, lam, c)
    return {k: (np.linalg.norm(X - G, axis=1), edge_ratio(X, G)) for k, X in out.items()}, (H[j], out, G)

if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex: R = list(ex.map(one3, range(len(H))))
    for k in R[0][0]:
        d = np.array([r[0][k][0] for r in R]); e = np.array([r[0][k][1] for r in R]); lr = np.abs(np.log(e))
        print(f"{k:16s} err med {np.nanmedian(d):.2f} p90 {np.nanpercentile(d,90):.2f} | edge med {np.nanmedian(lr):.3f} p90 {np.nanpercentile(lr,90):.3f} collapsed {np.nanmean(e<0.5):.1%}")
    import pickle; pickle.dump([r[1] for r in R], open("lifts3.pkl", "wb"))
