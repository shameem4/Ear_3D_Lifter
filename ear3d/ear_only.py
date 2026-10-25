"""Ear-only scans: no head, so no face to orient by.

The pipeline's flow-gate sends a mesh here when it is too small to be a head
(longest side < HEAD_MIN_MM). The ear's pose comes from its shape alone: the
mean ear template is aligned to it from each principal-axis sign choice, in
both handednesses (a right ear is a mirrored left ear in the canonical frame),
refined by ICP, and the best fit wins. The aligned ear is then viewed as the
pipeline views a left ear on a normalised head (CANON_TO_HEAD) and landmarked
by the same 25-view step.

Before this, AudioEar3D was oriented from our own fit of each ear, which a
scan from the wild does not have (2026-10-24).
"""
import numpy as np
import open3d as o3d

from . import corpus
from .corpus import WRAP
from .ear_view import PAD, side_frame

HEAD_MIN_MM = 100.0
# canonical (left ear) -> normalised-head frame: the rotation mean of R_head @ Rr
# over the 40 cached lmw heads (runs/cache/v4_s0, v4_s1), as 03_rewrap_audioear uses
CANON_TO_HEAD = np.array([[-0.979385, 0.125842, 0.158014],
                          [0.12899, 0.991598, 0.009783],
                          [-0.155455, 0.029964, -0.987388]])
_TPL = None


def _template():
    global _TPL
    if _TPL is None:
        m = o3d.io.read_triangle_mesh(str(WRAP / "templates" / corpus.get("mean")))
        _TPL = np.asarray(m.vertices)
    return _TPL


def _mirror_z(V, F):
    return V * [1.0, 1.0, -1.0], F[:, ::-1]                          # flip winding too, or it renders inside out


def orient(V):
    """(R, t, s, mirror, residual): canonical-left-ear X ~ s * (V' @ R.T) + t,
    where V' is V mirrored about z if mirror (a right ear)."""
    from .config import DEFAULTS
    if DEFAULTS.get("core") == "rust":                               # ear3d_core.orient: stride subsample, own ICP
        import ear3d_core
        o = np.array(ear3d_core.orient(np.ravel(V).astype(float).tolist(), np.ravel(_template()).astype(float).tolist()))
        return o[:9].reshape(3, 3), o[9:12], float(o[12]), bool(o[13]), float(o[14])
    T = _template(); ct = T.mean(0); Et = np.linalg.svd(T - ct, full_matrices=False)[2]
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(T))
    best = None
    for mirror in (False, True):
        W = V * [1.0, 1.0, -1.0] if mirror else V
        cm = W.mean(0); Em = np.linalg.svd(W - cm, full_matrices=False)[2]
        s = float(np.ptp((T - ct) @ Et[0])) / float(np.ptp((W - cm) @ Em[0]))
        sub = W[np.random.default_rng(0).choice(len(W), min(len(W), 6000), replace=False)]
        for sg in np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3):
            R = Et.T @ np.diag(sg) @ Em
            if np.linalg.det(R) < 0:
                continue
            X = s * (sub - cm) @ R.T + ct
            src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))
            reg = o3d.pipelines.registration.registration_icp(
                src, tgt, 5.0, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True))
            A = reg.transformation; Xf = X @ A[:3, :3].T + A[:3, 3]
            d = np.asarray(tgt.compute_point_cloud_distance(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Xf))))
            back = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Xf)).compute_point_cloud_distance(tgt))
            res = float(np.median(d) + np.median(back))
            if best is None or res < best[-1]:
                k = float(np.cbrt(np.linalg.det(A[:3, :3])))              # ICP's own scale
                Rf = A[:3, :3] / k @ R; sf = s * k
                tf = A[:3, :3] @ (ct - s * cm @ R.T) + A[:3, 3]            # X = sf * W @ Rf.T + tf
                best = (Rf, tf, sf, mirror, res)
    return best


def frame(pipe, mesh):
    """The ear-only path's view of a mesh: (skinned mesh in the normalised-head
    frame, view rotation, view centre, view scale, to_head) where to_head maps
    points of the input mesh's frame into that skinned mesh's frame. Shared by
    landmarks() and the pinna-only training renders (scripts/handfix/5)."""
    V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.triangles)
    R, t, s, mirror, res = orient(V)
    W, Fw = _mirror_z(V, F) if mirror else (V, F)
    Xc = (W @ R.T) * s + t                                            # canonical left ear (template units ~ mm)
    c0 = Xc.mean(0); Xh = (Xc - c0) @ CANON_TO_HEAD.T                 # as a left ear on a normalised head
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Xh), o3d.utility.Vector3iVector(Fw)); m.compute_vertex_normals()
    c = pipe.cfg
    ear_m = pipe._skinned(m, c["ear_tone"], c["ear_skin_noise"])
    Rv = side_frame("L"); Wv = Xh @ Rv.T
    lo, hi = Wv[:, :2].min(0), Wv[:, :2].max(0); mid = 0.5 * (lo + hi)
    near = Wv[np.linalg.norm(Wv[:, :2] - mid, axis=1) < 3.0]
    p0 = np.r_[mid, near[:, 2].max() if len(near) else Wv[:, 2].max()]
    half = c["eye_z"] * np.tan(np.radians(c["vfov"]) / 2)

    def to_head(P):
        P = np.asarray(P, float) * ([1.0, 1.0, -1.0] if mirror else 1.0)
        return ((P @ R.T) * s + t - c0) @ CANON_TO_HEAD.T

    def from_head(P):
        X = ((P @ CANON_TO_HEAD + c0) - t) / s @ R
        return X * [1.0, 1.0, -1.0] if mirror else X
    return dict(ear=ear_m, mesh=m, R=Rv, centre=p0 @ Rv, scale=0.5 * float((hi - lo).max()) * (1 + 2 * PAD) / half,
                to_head=to_head, from_head=from_head, side="R" if mirror else "L", residual=res)


def landmarks(pipe, mesh):
    """55 landmarks on an ear-only mesh (mm), in the mesh's own frame, and its side ("L"/"R")."""
    f = frame(pipe, mesh)
    P = pipe._multiview(f["ear"], "L", f["R"], f["centre"], f["scale"])
    P = pipe._snap(P, f["mesh"])
    return f["from_head"](P), f["side"], f["residual"]
