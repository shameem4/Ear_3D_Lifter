"""Head mesh (mm, any orientation) -> MediaPipe's 478 face landmarks in 3D, in the mesh's coordinates.

The face counterpart of Pipeline, ported from Face_3D_Lifter (2026-10-25):
  1. face skin -> frames.normalize_head (FaceMesh frame + checks)
  2. orbit skin (FACE orbit_*) -> 33 views on a yaw x pitch grid about a point behind the nose
  3. FaceLandmarker per view (all 478 points, or none)
  4. per landmark: subtract each view's learned mean 2D offset (face_view_offsets_v0.npz,
     scripts/face/build_view_offsets.py), then RANSAC ray intersection over every view that found the face
  5. re-seat onto the head surface (closest point); the off-surface distance is kept
Leave one head out on 50 HUTUBS heads: 2.34 mm median against the canonical-fit pseudo GT (FINDINGS).

    from ear3d.face import FaceLifter
    out, dbg = FaceLifter().run(mesh_mm)
"""
import pathlib
import numpy as np, open3d as o3d
from .config import DEFAULTS
from .frames import normalize_head
from .render import render
from .camera import orbit_extrinsic, camera_ke, Camera
from .skin import skinned
from .triangulate import ransac_point, ray_distance

N_LM = 478

# Orbit renders chosen 2026-10-20 by a sweep of 6 suns x tones {3,5,10} x blotch on/off, 6 HUTUBS heads x 33
# views: world-fixed sun from the front, tone 5, blotchy found the face in 71% of views at 1.53 mm median ray
# residual; ~30 arms tie within 63-71% / 1.5-1.8 mm. The ear's tone 10 was among the worst (47%, 2.69 mm).
# Face FINDING (normalize_head) keeps the ear pipeline's own skin.
FACE = dict(
    DEFAULTS,
    orbit_tone=5,
    orbit_skin_noise=True,
    orbit_colours="skin",        # "skin": synthetic (tone, noise above); "scan": the mesh's own vertex colours
    view_offsets=str(pathlib.Path(__file__).with_name("face_view_offsets_v0.npz")),   # None = off
    orbit_light_dir=(0.0, -0.5, -1.0),   # direction the light TRAVELS, head frame (+z nose)
    orbit_light_frame="world",
    yaws=(-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75),
    pitches=(-30, 0, 30),
    scale_mm=100.0,                      # mm per render unit (the camera is unit-scale)
    pivot_behind_nose_mm=60.0,           # orbit centre, behind the nose tip along -z
    ransac_mm=2.0,
)


def canonical_obj(cfg=FACE):
    """MediaPipe's canonical face model (Apache-2.0), kept beside face_landmarker.task."""
    return pathlib.Path(cfg["face_task"]).parent / "canonical_face_model.obj"


def views(cfg=FACE):
    return [(y, p) for p in cfg["pitches"] for y in cfg["yaws"]]


def cameras(E, cfg=FACE):
    """Cameras from stored 4x4 extrinsics."""
    K, _ = camera_ke(cfg["size"], cfg)
    return [Camera(K, e, (cfg["size"],) * 2, cfg["size"]) for e in E]


def load_offsets(cfg=FACE):
    """view x 478 x 2 px, checked against the view grid; zeros when view_offsets is None."""
    vs = views(cfg)
    if not cfg["view_offsets"]:
        return np.zeros((len(vs), N_LM, 2))
    t = np.load(cfg["view_offsets"])
    if t["views"].shape != (len(vs), 2) or not np.allclose(t["views"], vs):
        raise ValueError(f"{cfg['view_offsets']} was learned on a different view grid")
    return t["off_px"]


def triangulate(UV, cams, off, thresh):
    """Per landmark RANSAC over the views that detected it, after subtracting off (view x lm x 2 px).
    -> P (lm x 3, render units, NaN if unplaced), residual (render units), inliers (lm x view)."""
    n = UV.shape[1]
    P = np.full((n, 3), np.nan); res = np.full(n, np.nan); inliers = np.zeros((n, len(cams)), bool)
    for i in range(n):
        ks = np.where(np.isfinite(UV[:, i, 0]))[0]
        if len(ks) < 2:
            continue
        o, d = map(np.vstack, zip(*[cams[k].rays(UV[k, i] - off[k, i]) for k in ks]))
        p, inl = ransac_point(o, d, thresh)
        if p is None:
            continue
        P[i] = p; inliers[i, ks] = inl
        res[i] = np.median(ray_distance(p, o[inl], d[inl]))
    return P, res, inliers


def seat(P, V, F):
    """Closest surface point for each finite row of P. -> (seated, distance)."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(V, np.float32)), o3d.core.Tensor(np.asarray(F, np.uint32)))
    ok = np.isfinite(P).all(1); Ps = P.copy(); dist = np.full(len(P), np.nan)
    if ok.any():
        Ps[ok] = scene.compute_closest_points(o3d.core.Tensor(P[ok].astype(np.float32)))["points"].numpy()
        dist[ok] = np.linalg.norm(Ps[ok] - P[ok], axis=1)
    return Ps, dist


class FaceLifter:
    def __init__(self, cfg=None):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        self.mp = mp
        self.cfg = dict(FACE, **(cfg or {}))
        self.fl = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=self.cfg["face_task"]), num_faces=1))
        self.off = load_offsets(self.cfg)

    def run(self, mesh, keep_images=False):
        c = self.cfg
        ext = float(np.ptp(np.asarray(mesh.vertices), axis=0).max())
        if not 100.0 <= ext <= 600.0:
            raise ValueError(f"mesh spans {ext:.3g}; need a head in MILLIMETRES")
        nh = normalize_head(skinned(mesh, c, c["tone"], c["skin_noise"]), c)
        if nh is None:
            return None
        mc, R_head, centre, nose = nh
        piv = nose - np.array([0.0, 0.0, c["pivot_behind_nose_mm"]])
        s = c["scale_mm"]
        geo = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector((np.asarray(mc.vertices) - piv) / s),
                                        mc.triangles)
        if c["orbit_colours"] == "scan":                 # the scan's own vertex colours (normalize_head keeps vertex order)
            if not mesh.has_vertex_colors() or len(mesh.vertex_colors) != len(geo.vertices):
                raise ValueError("orbit_colours='scan' needs a mesh with vertex colours")
            sk = o3d.geometry.TriangleMesh(geo); sk.vertex_colors = mesh.vertex_colors; sk.compute_vertex_normals()
        else:
            sk = skinned(geo, c, c["orbit_tone"], c["orbit_skin_noise"])
        rc = dict(c, light_dir=c["orbit_light_dir"], light_frame=c["orbit_light_frame"], light_mirror=False)
        vs = views(c)
        UV = np.full((len(vs), N_LM, 2), np.nan); cams, imgs = [], []
        for k, (y, p) in enumerate(vs):
            img, _, cam = render(sk, rc, E=orbit_extrinsic(y, p, c))
            cams.append(cam)
            r = self.fl.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
            if r.face_landmarks:
                UV[k] = [(q.x * img.shape[1], q.y * img.shape[0]) for q in r.face_landmarks[0]]
            if keep_images:
                imgs.append(img)
        P, res, inliers = triangulate(UV, cams, self.off, c["ransac_mm"] / s)
        Ps, dist = seat(P, geo.vertices, geo.triangles)
        to_input = lambda X: (X * s + piv) @ R_head + centre
        out = dict(P=to_input(Ps), P_raw=to_input(P), off_surface_mm=dist * s, residual_mm=res * s,
                   n_inliers=inliers.sum(1), n_views=np.isfinite(UV[:, 0, 0]).sum())
        E = np.array([np.r_[np.c_[cm.R, cm.t], [[0, 0, 0, 1]]] for cm in cams])
        dbg = dict(UV=UV, inliers=inliers, views=np.array(vs), E=E, cams=cams, piv=piv, R_head=R_head, centre=centre,
                   V=np.asarray(geo.vertices) * s, F=np.asarray(geo.triangles), P_norm=Ps * s, P_raw_norm=P * s, imgs=imgs)
        return out, dbg


def show(V, F, sets, out, yaws=(-45, 0, 45), pitch=0, centre=None, scale=100.0, crop=True, cfg=FACE):
    """Render V (mm, facing +z) with landmark sets [(P, (b,g,r)), ...] as dots where visible (depth within 1.5 mm)."""
    import cv2
    centre = V.mean(0) if centre is None else centre
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector((V - centre) / scale), o3d.utility.Vector3iVector(F))
    m.compute_vertex_normals(); m.paint_uniform_color([0.75, 0.72, 0.68])
    tiles = []
    for y in yaws:
        img, depth, cam = render(m, dict(cfg, light_dir=cfg["orbit_light_dir"], light_frame="world"), E=orbit_extrinsic(y, pitch, cfg))
        img = img[:, :, ::-1].copy()
        for P, col in sets:
            Q = (np.asarray(P) - centre) / scale; ok = np.isfinite(Q).all(1)
            uv = cam.project(Q[ok]); zc = (Q[ok] @ cam.R.T + cam.t)[:, 2]
            for (u, v), z in zip(uv, zc):
                ui, vi = int(round(u)), int(round(v))
                if 0 <= vi < depth.shape[0] and 0 <= ui < depth.shape[1] and z - depth[vi, ui] < 1.5 / scale:
                    cv2.circle(img, (ui, vi), 2, col, -1)
        tiles.append(img[60:560, 50:550] if crop else img)
    cv2.imwrite(str(out), np.hstack(tiles))
