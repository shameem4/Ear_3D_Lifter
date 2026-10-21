"""Check and re-wrap AudioEar3D ears: landmarks from the cloud, pose from landmarks.

Usage: rewrap_audioear.py OUT_DIR EAR [EAR...]        (EAR like 004_right)

AudioEar3D has no heads, so the face stage cannot run. Per ear:
  1. Mesh the cloud (Poisson, trimmed to where the cloud has points). This
     surface, not the old fit, is what gets landmarked and wrapped.
  2. Orientation only from the old fit (ae3d_v2 -> canonical_esc, then the
     mean canonical -> head rotation over the cached HUTUBS heads), so the
     ear can be viewed like the pipeline views one. Framing, the 25 views,
     best-8 RANSAC and the re-seat all run on the cloud mesh.
  3. wrap_ear(..., cut=False): RigidAlignment on the 55 landmarks, then the
     automatic Wrap.
  4. Score both fits by how much of the cloud they leave uncovered (points
     farther than 5% of the ear's size from the fit surface).

Right ears are mirrored in z first, as the old fits were; outputs are in
that mirrored cloud frame, like corpus/ae3d_v2.
"""
import glob
import pathlib
import sys

import numpy as np
import open3d as o3d
import trimesh

R0 = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R0))
from ear3d.ear_view import PAD, side_frame                         # noqa: E402
from ear3d.pipeline import Pipeline                                # noqa: E402
from ear3d.faceform import wrap_ear, robust_inliers, TEMPLATE                      # noqa: E402

A3D = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/clean_3d_data/AudioEar3D")
CORP = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work/corpus")
UNCOVERED = 0.05
MAX_FACES = 80000
import os as _os
ORIENT = _os.environ.get("AE_ORIENT", "oldfit")     # "pca": ear's own principal axes vs the mean template
SMOOTH = int(_os.environ.get("AE_SMOOTH", 10))      # Taubin iterations on the cloud mesh (0 = off)
SHRINK = 0.85        # new fit smaller than this x the old: reject (every PCA-orientation failure)
WORSE_PP = 1.0       # new fit may leave at most this many more % of the cloud uncovered
COLLAPSE = 0.82       # landmark span / cloud-mesh size below this = the landmarker collapsed
                      # (005_left 0.76, 037_left 0.75: all 55 points in a band down the
                      # middle; good ears 0.85-0.99)
PASSES = int(__import__('os').environ.get('AE_PASSES', 1))   # 2 = re-landmark after re-orienting: A/B on 20 ears, no help, broke 005_right/014_left

M = sum(h["R_head"] @ h["Rr"] for h in (np.load(f) for f in sorted(glob.glob(str(R0 / "runs/cache/v4_s[01]/*.npz")))))
U, _, Vt = np.linalg.svd(M); AROT = U @ Vt
Ft = np.asarray(trimesh.load(TEMPLATE, process=False).faces)


def sim(A, B):
    a, b = A.mean(0), B.mean(0); A0, B0 = A - a, B - b
    U_, S, Vt_ = np.linalg.svd(A0.T @ B0); R = U_ @ Vt_; s = S.sum() / (A0 ** 2).sum()
    return (lambda X: s * (X - a) @ R + b), (lambda Y: (Y - b) @ R.T / s + a)


MEAN_V = np.asarray(trimesh.load("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work/templates/ear_mean_v4.obj", process=False).vertices)


def pca_to_canonical(Vm):
    """Similarity cloud -> canonical from principal axes alone (no old fit).
    Axis signs are ambiguous: of the 4 proper rotations, keep the one whose
    placed mean template sits closest to the mesh (both ways, symmetric)."""
    from scipy.spatial import cKDTree
    cm, ct = Vm.mean(0), MEAN_V.mean(0)
    Em = np.linalg.svd(Vm - cm, full_matrices=False)[2]; Et = np.linalg.svd(MEAN_V - ct, full_matrices=False)[2]
    s = float(np.ptp((MEAN_V - ct) @ Et[0])) / float(np.ptp((Vm - cm) @ Em[0]))
    tm, tt = cKDTree(Vm), cKDTree(MEAN_V)
    best = None
    for sg in np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3):   # 8; half are rotations
        R = Et.T @ np.diag(sg) @ Em                        # cloud axes -> template axes
        if np.linalg.det(R) < 0:
            continue
        X = s * (Vm - cm) @ R.T + ct
        d = np.median(tt.query(X[::20])[0]) + np.median(tm.query((MEAN_V - ct) @ R / s + cm)[0]) * s
        if best is None or d < best[0]:
            best = (d, R)
    R = best[1]
    return (lambda X: s * (X - cm) @ R.T + ct), (lambda Y: (Y - ct) @ R / s + cm)


def cloud_mesh(C, N):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(C)); pc.normals = o3d.utility.Vector3dVector(N)
    m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=9)
    dens = np.asarray(dens); V = np.asarray(m.vertices)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(C).query(V)
    spacing = float(np.median(cKDTree(C).query(C, k=2)[0][:, 1]))
    m.remove_vertices_by_mask((dens < np.quantile(dens, 0.05)) | (d > 4 * spacing))
    m.remove_unreferenced_vertices()
    # ~head-cut density: the raw Poisson mesh (~170k vertices) got Wrap
    # killed under a 4 GB guard with no ERROR line, just no output
    if len(m.triangles) > MAX_FACES:
        m = m.simplify_quadric_decimation(MAX_FACES)
    if SMOOTH:
        # speckled clouds (037_left) render as a texture the landmarker never
        # saw and it collapses; Taubin smooths without shrinking the ear
        # clean first: Taubin turns degenerate / non-manifold vertices to NaN
        # (8 on 004_right) and the skin's SVD then fails to converge
        m.remove_degenerate_triangles(); m.remove_duplicated_triangles()
        m.remove_non_manifold_edges(); m.remove_unreferenced_vertices()
        V0 = np.asarray(m.vertices).copy()
        m = m.filter_smooth_taubin(number_of_iterations=SMOOTH)
        V1 = np.asarray(m.vertices); bad = ~np.isfinite(V1).all(1)
        if bad.any():
            V1[bad] = V0[bad]; m.vertices = o3d.utility.Vector3dVector(V1)
    return np.asarray(m.vertices), np.asarray(m.triangles)


def uncovered(C, V, F):
    e = float(np.ptp(V, axis=0).max())
    _, d, _ = trimesh.proximity.closest_point(trimesh.Trimesh(V, F, process=False), C[::3])
    return float(np.mean(d > UNCOVERED * e))


out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
pipe = Pipeline(); c = pipe.cfg
CANON_LM = np.load(pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work/templates/lm3d_v9_canonical_mm.npz"),
                   allow_pickle=True)["landmarks"][:, :55].mean(0)          # mean ear, canonical mm


def landmarks(Vm, Fm, to_can, from_can):
    """55 landmarks on the cloud mesh, viewed as the pipeline views a left ear
    once to_can has put it in the canonical frame. Returned in the cloud frame."""
    Xc = to_can(Vm); c0 = Xc.mean(0); Xh = (Xc - c0) @ AROT.T
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Xh), o3d.utility.Vector3iVector(Fm)); mesh.compute_vertex_normals()
    ear_m = pipe._skinned(mesh, c["ear_tone"], c["ear_skin_noise"])
    R = side_frame("L"); W = Xh @ R.T
    lo, hi = W[:, :2].min(0), W[:, :2].max(0); mid = 0.5 * (lo + hi)
    near = W[np.linalg.norm(W[:, :2] - mid, axis=1) < 3.0]
    p0 = np.r_[mid, near[:, 2].max() if len(near) else W[:, 2].max()]
    half = c["eye_z"] * np.tan(np.radians(c["vfov"]) / 2)
    P = pipe._multiview(ear_m, "L", R, p0 @ R, 0.5 * float((hi - lo).max()) * (1 + 2 * PAD) / half)
    ok = np.isfinite(P).all(1)
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    P[ok] = sc.compute_closest_points(o3d.core.Tensor(P[ok].astype(np.float32)))["points"].numpy()
    return from_can(P @ AROT + c0)
for ear in sys.argv[2:]:
    sid, side = ear.split("_")
    pl = trimesh.load(A3D / sid / f"{side}.ply", process=False)
    raw = pl.metadata["_ply_raw"]["vertex"]["data"]
    C = np.asarray(pl.vertices, float); N = np.asarray(raw[["nx", "ny", "nz"]].tolist(), float)
    if side == "right":
        C, N = C * [1, 1, -1.0], N * [1, 1, -1.0]
    Vm, Fm = cloud_mesh(C, N)
    old = np.asarray(trimesh.load(CORP / f"ae3d_v2/{ear}.obj", process=False).vertices)
    if ORIENT == "pca":
        to_can, from_can = pca_to_canonical(Vm)
    else:
        to_can, from_can = sim(old[:1188], np.asarray(trimesh.load(CORP / f"canonical_esc/audioear/{ear}.obj", process=False).vertices)[:1188])
    # PASS 1 orients by the old fit; PASS 2 re-orients by pass 1's own
    # landmarks against the mean canonical landmarks and reads them again.
    # 011_left's bad old fit left the ear 15 deg tilted in the views (others
    # 3-4 deg) and the points bunched on the rim; pass 2 does not depend on
    # the old fit at all.
    P = landmarks(Vm, Fm, to_can, from_can)
    if PASSES >= 2:
        ok = np.isfinite(P).all(1)
        to_can2, from_can2 = sim(P[ok], CANON_LM[ok])
        P = landmarks(Vm, Fm, to_can2, from_can2)
    ok = np.isfinite(P).all(1)
    inl, res = robust_inliers(P)
    span = float(np.ptp(P[ok], axis=0).max()) / float(np.ptp(Vm, axis=0).max())
    if span < COLLAPSE:
        print(f"{ear}: COLLAPSED landmarks (span {span:.2f}); keep the old fit", flush=True)
        np.savez(out / f"{ear}.npz", V=old[:1188], P=P, old=old, C=C, Vm=Vm, Fm=Fm, inliers=inl, resid=res, span=span, kept_old=True)
        continue
    try:
        Vf = wrap_ear(Vm, Fm, P, "L", out / "work", ear, cut=False, use=inl)
    except Exception as e:
        print(ear, "FAILED", e, flush=True); continue
    Fo = np.asarray(trimesh.load(CORP / f"ae3d_v2/{ear}.obj", process=False).faces)
    u_old, u_new = uncovered(C, old, Fo), uncovered(C, Vf, Ft)
    size_r = float(np.ptp(Vf, axis=0).max() / np.ptp(old, axis=0).max())
    reason = ("shrunk %.2f" % size_r if size_r < SHRINK else
              "coverage worse" if 100 * (u_new - u_old) > WORSE_PP else "")
    if reason:
        print(f"{ear}: new fit rejected ({reason}; uncovered {u_old*100:.1f}% -> {u_new*100:.1f}%); keep the old fit", flush=True)
        np.savez(out / f"{ear}.npz", V=old[:1188], Vnew=Vf, P=P, old=old, C=C, Vm=Vm, Fm=Fm, inliers=inl, resid=res,
                 span=span, kept_old=True, reason=reason)
        continue
    np.savez(out / f"{ear}.npz", V=Vf, P=P, old=old, C=C, Vm=Vm, Fm=Fm, inliers=inl, resid=res, span=span, kept_old=False)
    print(f"{ear}: span {span:.2f}, landmarks {ok.sum()}/55, inliers {inl.sum()} (resid {res:.2f} mm) | uncovered cloud old {u_old*100:.1f}% -> new {u_new*100:.1f}% | "
          f"size old {np.ptp(old,0).max():.1f} new {np.ptp(Vf,0).max():.1f} mm", flush=True)
