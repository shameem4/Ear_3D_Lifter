import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
from concurrent.futures import ProcessPoolExecutor
from lap import *
from lap import _similarity

def robust(P, V, F, lam, c_mm, iters=5):
    ok = np.isfinite(P).all(1); Pz = np.nan_to_num(P)
    ref = _similarity(V0[ok], P[ok])(V0); D = L @ ref; w = ok.astype(float); LL = lam * L.T @ L
    for _ in range(iters):
        X = sla.splu((sp.diags(w) + LL).tocsc()).solve(w[:, None] * Pz + LL @ ref)
        X, _ = seat(X, V, F)
        r = np.linalg.norm(X - Pz, axis=1)
        w = ok / (1 + (r / c_mm) ** 2)            # Cauchy
    return X

def one2(j):
    z = np.load(RUN / f"{H[j]}.npz"); G = np.load(RUN / f"{H[j]}_gtd.npz")["G"]
    off = np.nan_to_num(np.nanmean(OFF[[i for i in range(len(H)) if i != j]], 0))
    P, _, _ = triangulate(z["UV"][:, :468], cameras(z["E"]), off, FACE["ransac_mm"] / s)
    Ps, _ = seat(P * s, z["V"], z["F"])
    out = {"lap 3": lapfit(Ps, z["V"], z["F"], 3)}
    for lam in (1, 3):
        for c in (1.0, 2.0):
            out[f"irls l{lam} c{c}"] = robust(Ps, z["V"], z["F"], lam, c)
    return {k: (np.linalg.norm(X - G, axis=1), edge_ratio(X, G)) for k, X in out.items()}, (H[j], out, G)

if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex: R = list(ex.map(one2, range(len(H))))
    for k in R[0][0]:
        d = np.array([r[0][k][0] for r in R]); e = np.array([r[0][k][1] for r in R]); lr = np.abs(np.log(e))
        print(f"{k:16s} err med {np.nanmedian(d):.2f} p90 {np.nanpercentile(d,90):.2f} | edge med {np.nanmedian(lr):.3f} p90 {np.nanpercentile(lr,90):.3f} collapsed {np.nanmean(e<0.5):.1%}")
    import pickle; pickle.dump([r[1] for r in R], open("lifts2.pkl", "wb"))
