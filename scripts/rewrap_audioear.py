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

R0 = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R0), str(R0 / "scripts")]
from ear3d.ear_view import PAD, side_frame                         # noqa: E402
from ear3d.pipeline import Pipeline                                # noqa: E402
from wrap_from_landmarks import wrap_ear, TEMPLATE                 # noqa: E402

A3D = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/clean_3d_data/AudioEar3D")
CORP = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work/corpus")
UNCOVERED = 0.05
MAX_FACES = 80000

M = sum(h["R_head"] @ h["Rr"] for h in (np.load(f) for f in sorted(glob.glob(str(R0 / "runs/cache/v4_s[01]/*.npz")))))
U, _, Vt = np.linalg.svd(M); AROT = U @ Vt
Ft = np.asarray(trimesh.load(TEMPLATE, process=False).faces)


def sim(A, B):
    a, b = A.mean(0), B.mean(0); A0, B0 = A - a, B - b
    U_, S, Vt_ = np.linalg.svd(A0.T @ B0); R = U_ @ Vt_; s = S.sum() / (A0 ** 2).sum()
    return (lambda X: s * (X - a) @ R + b), (lambda Y: (Y - b) @ R.T / s + a)


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
    return np.asarray(m.vertices), np.asarray(m.triangles)


def uncovered(C, V, F):
    e = float(np.ptp(V, axis=0).max())
    _, d, _ = trimesh.proximity.closest_point(trimesh.Trimesh(V, F, process=False), C[::3])
    return float(np.mean(d > UNCOVERED * e))


out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
pipe = Pipeline(); c = pipe.cfg
for ear in sys.argv[2:]:
    sid, side = ear.split("_")
    pl = trimesh.load(A3D / sid / f"{side}.ply", process=False)
    raw = pl.metadata["_ply_raw"]["vertex"]["data"]
    C = np.asarray(pl.vertices, float); N = np.asarray(raw[["nx", "ny", "nz"]].tolist(), float)
    if side == "right":
        C, N = C * [1, 1, -1.0], N * [1, 1, -1.0]
    Vm, Fm = cloud_mesh(C, N)
    old = np.asarray(trimesh.load(CORP / f"ae3d_v2/{ear}.obj", process=False).vertices)
    to_can, from_can = sim(old[:1188], np.asarray(trimesh.load(CORP / f"canonical_esc/audioear/{ear}.obj", process=False).vertices)[:1188])
    # into the normalised-head frame, the way the pipeline sees a left ear
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
    P = from_can(P @ AROT + c0)                                     # back to the (mirrored) cloud frame
    try:
        Vf = wrap_ear(Vm, Fm, P, "L", out / "work", ear, cut=False)
    except Exception as e:
        print(ear, "FAILED", e, flush=True); continue
    Fo = np.asarray(trimesh.load(CORP / f"ae3d_v2/{ear}.obj", process=False).faces)
    u_old, u_new = uncovered(C, old, Fo), uncovered(C, Vf, Ft)
    np.savez(out / f"{ear}.npz", V=Vf, P=P, old=old, C=C, Vm=Vm, Fm=Fm)
    print(f"{ear}: landmarks {ok.sum()}/55 | uncovered cloud old {u_old*100:.1f}% -> new {u_new*100:.1f}% | "
          f"size old {np.ptp(old,0).max():.1f} new {np.ptp(Vf,0).max():.1f} mm", flush=True)
