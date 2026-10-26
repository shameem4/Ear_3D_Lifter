import sys, pathlib, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
R0 = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_3D_Lifter"); sys.path.insert(0, str(R0))
from concurrent.futures import ProcessPoolExecutor
from ear3d.face import FACE, cameras, triangulate, seat
from ear3d.face_gt import canonical_raw
from ear3d.faceform import _similarity
RUN = R0 / "runs/face/v0"; s = FACE["scale_mm"]
T = np.load(R0 / "runs/face/view_table.npz"); H, OFF = list(T["heads"]), T["off_px"]
V0, F0 = canonical_raw(); V0, F0 = V0[:468], F0
E_ = np.unique(np.sort(np.r_[F0[:, [0, 1]], F0[:, [1, 2]], F0[:, [2, 0]]], 1), axis=0)
A = sp.coo_matrix((np.ones(2 * len(E_)), (np.r_[E_[:, 0], E_[:, 1]], np.r_[E_[:, 1], E_[:, 0]])), (468, 468)).tocsr()
L = sp.diags(np.asarray(A.sum(1)).ravel()) - A           # umbrella Laplacian

def edge_ratio(X, G):   # per-edge length ratio vs GT: bunching = spread of this
    return np.linalg.norm(X[E_[:, 0]] - X[E_[:, 1]], axis=1) / np.linalg.norm(G[E_[:, 0]] - G[E_[:, 1]], axis=1)

def lapfit(P, V, F, lam, iters=3):
    ok = np.isfinite(P).all(1)
    f = _similarity(V0[ok], P[ok]); ref = f(V0)                  # canonical posed onto the lift
    D = L @ ref; W = sp.diags(ok.astype(float)); Pz = np.nan_to_num(P)
    M = (W + lam * L.T @ L).tocsc(); X = ref
    tgt = Pz
    for _ in range(iters):
        X = sla.splu(M).solve(W @ tgt + lam * L.T @ D)
        X, _ = seat(X, V, F)
        tgt = np.where(ok[:, None], Pz, 0)
    return X

def one(j):
    z = np.load(RUN / f"{H[j]}.npz"); G = np.load(RUN / f"{H[j]}_gtd.npz")["G"]
    off = np.nan_to_num(np.nanmean(OFF[[i for i in range(len(H)) if i != j]], 0))
    P, _, _ = triangulate(z["UV"][:, :468], cameras(z["E"]), off, FACE["ransac_mm"] / s)
    Ps, _ = seat(P * s, z["V"], z["F"])
    out = {"current": Ps}
    for lam in (0.3, 1, 3, 10):
        out[f"lap {lam}"] = lapfit(Ps, z["V"], z["F"], lam)
    return {k: (np.linalg.norm(X - G, axis=1), edge_ratio(X, G)) for k, X in out.items()}, (H[j], out, G)

if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex: R = list(ex.map(one, range(len(H))))
    for k in R[0][0]:
        d = np.array([r[0][k][0] for r in R]); e = np.array([r[0][k][1] for r in R])
        lr = np.abs(np.log(e))
        print(f"{k:10s} err med {np.nanmedian(d):.2f} p90 {np.nanpercentile(d,90):.2f} | edge |log ratio| med {np.nanmedian(lr):.3f} p90 {np.nanpercentile(lr,90):.3f}  collapsed(<0.5) {np.nanmean(e<0.5):.1%}")
    import pickle; pickle.dump([r[1] for r in R], open("lifts.pkl", "wb"))
