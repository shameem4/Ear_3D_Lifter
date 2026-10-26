"""Dense pinned film: subdivided canonical face, whole-face surface term (point-to-point + point-to-plane), reliable
landmarks pinned, stiffness annealed. All linear solves: portable."""
import sys, os, glob, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, open3d as o3d
from concurrent.futures import ProcessPoolExecutor
from pf import R0, W0, ERR, EH, REG, seat, _similarity
from ear3d.face import dense_template, FACE
Vd, LL3, _ = dense_template(); N = len(Vd)

def dense_pin(X, V, F, w, sched=(100, 30, 10, 3), its=3, mu=0.2, trim=8.0):
    """w: per-landmark pin weight (468). mu: point-to-point share of the surface term."""
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(np.asarray(V, np.float32)), o3d.core.Tensor(np.asarray(F, np.uint32)))
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)); m.compute_triangle_normals(); TN = np.asarray(m.triangle_normals)
    ok = w >= np.quantile(w, 0.5); Y = _similarity(Vd[:468][ok], X[ok])(Vd)
    Pw = np.zeros(N); Pw[:468] = w; PX = np.zeros((N, 3)); PX[:468] = X
    for lam in sched:
        for _ in range(its):
            q = sc.compute_closest_points(o3d.core.Tensor(Y.astype(np.float32)))
            C = q["points"].numpy().astype(float); n = TN[q["primitive_ids"].numpy()]
            wd = (np.linalg.norm(C - Y, axis=1) < trim).astype(float)
            B = wd[:, None, None] * ((1 - mu) * n[:, :, None] * n[:, None, :] + mu * np.eye(3)[None]) + Pw[:, None, None] * np.eye(3)[None]
            b = (wd[:, None, None] * ((1 - mu) * n[:, :, None] * n[:, None, :] + mu * np.eye(3)[None]) @ C[..., None])[..., 0] + Pw[:, None] * PX
            ref = _similarity(Vd, Y)(Vd)
            Y = sla.spsolve((sp.block_diag(list(B)).tocsr() + lam * LL3).tocsc(), b.ravel() + lam * (LL3 @ ref.ravel())).reshape(N, 3)
    return seat(Y[:468], V, F)[0]

def one(h):
    z = np.load(f"{R0}/runs/face/v1/{h}.npz"); G = np.load(f"{R0}/runs/face/v1/{h}_gtd.npz")["G"]
    X = np.load(f"{W0}/out/{h}.npz")["current"]; e = np.median(ERR["E"][[i for i, x in enumerate(EH) if x != h]], 0)
    arms = {"current": X,
            "soft mu0.2": dense_pin(X, z["V"], z["F"], (1.5 / e) ** 2),
            "soft mu0.5": dense_pin(X, z["V"], z["F"], (1.5 / e) ** 2, mu=0.5),
            "hard<2 mu0.2": dense_pin(X, z["V"], z["F"], np.where(e < 2.0, 10.0, 0.0)),
            "soft x0.3 mu0.2": dense_pin(X, z["V"], z["F"], 0.3 * (1.5 / e) ** 2)}
    out = {}
    for k, Y in arms.items():
        m, E = REG["lip corners"]; g = np.linalg.norm(G[E[:, 0]] - G[E[:, 1]], axis=1); x = np.linalg.norm(Y[E[:, 0]] - Y[E[:, 1]], axis=1)
        out[k] = (np.linalg.norm(Y - G, axis=1), (x / g < 0.5)[g > 1])
    return h, out

if __name__ == "__main__":
    H = [os.path.basename(f)[:-4] for f in sorted(glob.glob(f"{W0}/out/*.npz"))][: int(sys.argv[1]) if len(sys.argv) > 1 else 50]
    with ProcessPoolExecutor(8) as ex: R = list(ex.map(one, H))
    print(f"{len(H)} heads " + f"{'':12s}" + "".join(f"{n:>13s}" for n in REG) + "    all  corner-collapse  heads-better(all)")
    cur = np.array([np.median(r[1]["current"][0]) for r in R])
    for k in R[0][1]:
        d = np.array([r[1][k][0] for r in R]); c = np.concatenate([r[1][k][1] for r in R]); ph = np.median(d, 1)
        print(f"{k:20s}" + "".join(f"{np.median(d[:, m]):13.2f}" for m, _ in REG.values()) + f"  {np.median(d):5.2f}  {c.mean():8.1%}  {np.sum(ph < cur - 1e-9):3d}/{len(H)} worst {np.max(ph - cur):+.2f}")
