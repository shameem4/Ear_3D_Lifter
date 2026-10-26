"""In-house pins + film: reliable landmarks stay where the lift put them; the rest are filled by a Laplacian
membrane on the 468-vertex canonical mesh, seated on the scan, repeated. Score leave one head out."""
import sys, os, glob, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
R0 = "/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_3D_Lifter"; sys.path[:0] = [R0, R0 + "/scripts/face"]; os.environ["FACE_RUN"] = "v1"
from concurrent.futures import ProcessPoolExecutor
from score_loho import REG
from ear3d.face import canonical_laplacian, seat, canonical_raw
import open3d as o3d
FC = canonical_raw()[1]
from ear3d.faceform import _similarity
L, Vc = canonical_laplacian(); Vc = Vc[:468]; LL = (L.T @ L).tocsr()
W0 = "/var/tmp/claude/ear3d/2026-10-25-face-wrapkeys"
_e = np.load(f"{W0}/lm_err_current.npz"); ERR = {"E": _e["E"].copy()}; EH = list(_e["heads"])

def seat_normal(Y, V, F, sc=None):
    """Move each point along the film's own vertex normal onto the scan (nearest hit either way); closest
    point where no hit within 10 mm. Closest-point seating pulls neighbours together into creases."""
    if sc is None:
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.core.Tensor(np.asarray(V, np.float32)), o3d.core.Tensor(np.asarray(F, np.uint32)))
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Y), o3d.utility.Vector3iVector(FC)); m.compute_vertex_normals()
    n = np.asarray(m.vertex_normals)
    best = np.full(len(Y), np.inf); out = seat(Y, V, F)[0]
    for sgn in (1, -1):
        r = np.c_[Y, sgn * n].astype(np.float32); t = sc.cast_rays(o3d.core.Tensor(r))["t_hit"].numpy()
        use = (t < 10) & (t < best); best[use] = t[use]; out[use] = Y[use] + sgn * n[use] * t[use, None]
    return out

def pinfill(X, V, F, w, lam=1.0, iters=4, normal=False):
    """w: per-landmark data weight (468). Minimise sum w_i |Y_i - X_i|^2 + lam |L (Y - ref)|^2, ref = canonical
    posed on Y; seat; repeat."""
    Y = X.copy()
    for _ in range(iters):
        ok = w >= np.quantile(w, 0.5)                             # pose the reference on the better half
        ref = _similarity(Vc[ok], Y[ok])(Vc)
        M = (sp.diags(w) + lam * LL).tocsc()
        Y = sla.splu(M).solve(w[:, None] * X + lam * (LL @ ref))
        Y = seat_normal(Y, V, F) if normal else seat(Y, V, F)[0]
    return Y

def one(h):
    z = np.load(f"{R0}/runs/face/v1/{h}.npz"); G = np.load(f"{R0}/runs/face/v1/{h}_gtd.npz")["G"]
    X = np.load(f"{W0}/out/{h}.npz")["current"]
    e = np.median(ERR["E"][[i for i, x in enumerate(EH) if x != h]], 0)        # reliability from the other heads
    arms = {"current": X}
    for nm in (False, True):
        for lam in (0.1, 1.0, 10.0):
            arms[f"soft 1/e^2 l{lam}{' normal' if nm else ''}"] = pinfill(X, z["V"], z["F"], (1.5 / e) ** 2, lam, normal=nm)
        arms[f"hard<2.0 l1.0{' normal' if nm else ''}"] = pinfill(X, z["V"], z["F"], np.where(e < 2.0, 100.0, 0.0), 1.0, normal=nm)
    out = {}
    for k, Y in arms.items():
        m, E = REG["lip corners"]; g = np.linalg.norm(G[E[:, 0]] - G[E[:, 1]], axis=1); x = np.linalg.norm(Y[E[:, 0]] - Y[E[:, 1]], axis=1)
        out[k] = (np.linalg.norm(Y - G, axis=1), (x / g < 0.5)[g > 1])
    return h, out

if __name__ == "__main__":
    H = [os.path.basename(f)[:-4] for f in sorted(glob.glob(f"{W0}/out/*.npz"))]
    with ProcessPoolExecutor(8) as ex: R = list(ex.map(one, H))
    print(f"{'':20s}" + "".join(f"{n:>13s}" for n in REG) + "    all  corner-collapse  heads-better(all)")
    cur = np.array([np.median(r[1]["current"][0]) for r in R])
    for k in R[0][1]:
        d = np.array([r[1][k][0] for r in R]); c = np.concatenate([r[1][k][1] for r in R]); ph = np.median(d, 1)
        print(f"{k:20s}" + "".join(f"{np.median(d[:, m]):13.2f}" for m, _ in REG.values()) + f"  {np.median(d):5.2f}  {c.mean():8.1%}  {np.sum(ph < cur - 1e-9):3d}/50 worst {np.max(ph - cur):+.2f}")
