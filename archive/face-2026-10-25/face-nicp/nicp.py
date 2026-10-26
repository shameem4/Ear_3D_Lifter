"""Dense non-rigid fit of the subdivided canonical face to the scan: point-to-plane + rays (lm 0-467) + Laplacian."""
import sys, os, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, open3d as o3d
R0 = "/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_3D_Lifter"; sys.path[:0] = [R0, R0 + "/scripts/face"]
os.environ["FACE_RUN"] = "v1"
from ear3d.face import FACE, cameras, triangulate, seat, fit_rays, canonical_laplacian, canonical_raw, view_weights
from ear3d.face_gt import canonical
from ear3d.faceform import _similarity

Vd, Fd = canonical(2)                                  # dense; vertices 0-467 are the landmarks
N = len(Vd)
E_ = np.unique(np.sort(np.r_[Fd[:, [0, 1]], Fd[:, [1, 2]], Fd[:, [2, 0]]], 1), axis=0)
A = sp.coo_matrix((np.ones(2 * len(E_)), (np.r_[E_[:, 0], E_[:, 1]], np.r_[E_[:, 1], E_[:, 0]])), (N, N)).tocsr()
L = (sp.diags(np.asarray(A.sum(1)).ravel()) - A).tocsr()
LL3 = sp.kron(L.T @ L, sp.eye(3)).tocsr()

def scene(V, F):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(np.asarray(V, np.float32)), o3d.core.Tensor(np.asarray(F, np.uint32)))
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)); m.compute_triangle_normals()
    return sc, np.asarray(m.triangle_normals)

def nicp(UV, cams, off, w_view, V, F, X0, lam=1.0, alpha=1.0, beta=1.0, iters=8, trim_mm=5.0, s=100.0, region_mm=None):
    """X0: 468 x 3 current lift (mm, frame of V). Returns 468 x 3 and the dense fit."""
    sc, TN = scene(V, F)
    ok = np.isfinite(X0).all(1); f = _similarity(Vd[:468][ok], X0[ok]); ref = f(Vd)
    Y = ref.copy()
    # ray projectors for lm 0-467
    n = 468; O = np.zeros((len(cams), n, 3)); D = np.zeros((len(cams), n, 3)); has = np.isfinite(UV[:, :n, 0])
    for k, cam in enumerate(cams):
        if has[k].any():
            o, d = cam.rays(UV[k, :n][has[k]] - off[k, :n][has[k]]); O[k, has[k]] = o * s; D[k, has[k]] = d
    Pm = np.eye(3)[None, None] - D[..., :, None] * D[..., None, :]; PmO = (Pm @ O[..., None])[..., 0]
    for it in range(iters):
        q = sc.compute_closest_points(o3d.core.Tensor(Y.astype(np.float32)))
        C = q["points"].numpy().astype(float); nrm = TN[q["primitive_ids"].numpy()]
        dist = np.linalg.norm(C - Y, axis=1); wd = alpha * (dist < trim_mm)
        if region_mm is not None: wd = wd * (np.linalg.norm(Vd - Vd[2], axis=1) < region_mm)
        # point-to-plane: (n.(Y - C))^2 -> 3x3 blocks n n^T
        Bp = wd[:, None, None] * nrm[:, :, None] * nrm[:, None, :]; bp = (Bp @ C[..., None])[..., 0]
        v = Y[None, :n] - O; miss = np.linalg.norm(v - (v * D).sum(2, keepdims=True) * D, axis=2)
        w = np.where(has, 1 / (1 + (miss / 2.0) ** 2), 0.0) * (w_view[:, :n] if w_view is not None else 1)
        Br = np.zeros((N, 3, 3)); br = np.zeros((N, 3))
        Br[:n] = beta * (w[..., None, None] * Pm).sum(0); br[:n] = beta * (w[..., None] * PmO).sum(0)
        # re-pose the reference each iteration so the prior is on shape, not pose
        ref = _similarity(Vd, Y)(Vd)
        M = sp.block_diag(list(Bp + Br)).tocsr() + lam * LL3
        Y = sla.spsolve(M.tocsc(), (bp + br).ravel() + lam * (LL3 @ ref.ravel())).reshape(N, 3)
    Xs, _ = seat(Y[:468], V, F)
    return Xs, Y
