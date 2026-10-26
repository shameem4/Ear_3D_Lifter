"""Head mesh (mm, any orientation) -> MediaPipe's 478 face landmarks in 3D, in the mesh's coordinates.

The face counterpart of Pipeline, ported from Face_3D_Lifter (2026-10-25):
  1. face skin -> frames.normalize_head (FaceMesh frame + checks)
  2. orbit skin (FACE orbit_*) -> 33 views on a yaw x pitch grid about a point behind the nose
  3. FaceLandmarker per view (all 478 points, or none)
  4. subtract each view's learned mean 2D offset (face_view_offsets_*.npz, scripts/face/build_view_offsets.py),
     then RANSAC ray intersection per landmark over every view that found the face, seated on the surface
  5. joint ray fit (fit_rays): all 468 points at once to every view's rays, Cauchy-weighted per ray, with a
     Laplacian prior on MediaPipe's canonical mesh, re-seated each iteration. Per-landmark triangulation lets
     neighbours settle on different views and bunch; the joint fit keeps the canonical spacing.
  6. dense fit (fit_dense): the 2x-subdivided canonical face, posed on step 5, fitted point-to-plane to the
     scan around the nose, with step 5's rays and a stiff Laplacian; only the nose base (21 landmarks) takes
     its result. FaceLandmarker misplaces hidden nostrils on the upper lip in every view (pp22: 5.4 mm); the
     nose's own shape puts them back.
  7. spacing (fit_spacing): one-sided edge springs on the landmark mesh push apart any neighbours closer than
     spacing_rho x their typical distance (learned from the training heads' GT, scaled to the face), seated on
     the scan each sweep. Mouth-corner collapsed edges 11.1% -> 0.2%; error unchanged (2.06 -> 2.05 mm).
Leave one head out on 50 HUTUBS heads vs the canonical-fit pseudo GT: 2.34 mm after step 4, 2.16 (p90 5.59
-> 4.99, collapsed edges 0.62 -> 0.05%) after step 5; 2.10 with per-view weights (FINDINGS). Iris (468-477)
keeps step 4's points.

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
from .faceform import _similarity

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
    view_offsets=str(pathlib.Path(__file__).with_name("face_view_offsets_v1.npz")),   # None = off; v1 = runs/face/v1
    orbit_light_dir=(0.0, -0.5, -1.0),   # direction the light TRAVELS, head frame (+z nose)
    orbit_light_frame="world",
    yaws=(-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75),
    pitches=(-30, 0, 30),
    scale_mm=100.0,                      # mm per render unit (the camera is unit-scale)
    pivot_behind_nose_mm=60.0,           # orbit centre, behind the nose tip along -z
    ransac_mm=2.0,
    fit_rays=True,                       # step 5; False returns step 4's points
    fit_lambda=0.3,                      # Laplacian prior weight (swept 0.3-3: all within 0.02 mm)
    fit_cauchy_mm=2.0,                   # ray residual scale for the Cauchy weights
    fit_iters=6,
    fit_dense=True,                      # step 6: dense canonical-face fit to the scan's nose geometry
    dense_lambda=10.0,                   # stiffness (Laplacian weight) of the dense template
    dense_beta=0.5,                      # weight of the rays against the surface term
    dense_region_mm=35.0,                # surface term only within this of the subnasale (canonical frame):
                                         # the lips fit the canonical shape badly and get dragged
    dense_apply_mm=12.0,                 # ...but only landmarks this close to the subnasale take its result:
                                         # the surface term slides the nose tip (17 mm away) down its slope
                                         # (BJ002 sticker 1.7 -> 3.9 mm) and costs the lateral stickers
    dense_trim_mm=5.0,                   # ignore template vertices further than this from the scan
    dense_iters=8,
    fit_spacing=True,                    # step 7: anti-bunching edge springs
    spacing_rho=0.85,                    # push apart edges shorter than this x their typical length
    spacing_sweeps=30,
    spacing_anchor=0.05,                 # per sweep, pull back toward step 6's points by this fraction
    fit_view_weights=True,               # weight each ray by 1/(its view's typical miss for that landmark)^2,
                                         # miss_mm in the view-offset file; leave one head out 2.18 -> 2.10 mm
)


def canonical_obj(cfg=FACE):
    """MediaPipe's canonical face model (Apache-2.0), kept beside face_landmarker.task."""
    return pathlib.Path(cfg["face_task"]).parent / "canonical_face_model.obj"


def canonical_raw(cfg=FACE):
    """The canonical face (cm -> mm) in FILE order, which is the landmark numbering and Wrap's numbering.
    Parsed by hand: Open3D's OBJ reader REORDERS this file's vertices (it has texture coordinates)."""
    V, F = [], []
    for l in open(canonical_obj(cfg)):
        if l.startswith("v "): V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith("f "): F.append([int(x.split("/")[0]) - 1 for x in l.split()[1:4]])
    return np.array(V) * 10.0, np.array(F)


def canonical_dense(cfg=FACE, levels=2):
    """canonical_raw, midpoint-subdivided (mm); vertices 0-467 stay the landmarks."""
    V, F = canonical_raw(cfg)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)).subdivide_midpoint(levels)
    return np.asarray(m.vertices, float), np.asarray(m.triangles)


def views(cfg=FACE):
    return [(y, p) for p in cfg["pitches"] for y in cfg["yaws"]]


def cameras(E, cfg=FACE):
    """Cameras from stored 4x4 extrinsics."""
    K, _ = camera_ke(cfg["size"], cfg)
    return [Camera(K, e, (cfg["size"],) * 2, cfg["size"]) for e in E]


def load_offsets(cfg=FACE, what="off_px"):
    """From the view-offset file, checked against the view grid: "off_px" (view x 478 x 2, zeros when
    view_offsets is None), "miss_mm" (view x 468, each view's median ray miss per landmark, raw detections) or
    "edge_rest" (per canonical_edges edge, its typical length / face_size, over the GT heads)."""
    vs = views(cfg)
    if not cfg["view_offsets"]:
        return np.zeros((len(vs), N_LM, 2)) if what == "off_px" else None
    t = np.load(cfg["view_offsets"])
    if t["views"].shape != (len(vs), 2) or not np.allclose(t["views"], vs):
        raise ValueError(f"{cfg['view_offsets']} was learned on a different view grid")
    return t[what]


def view_weights(miss_mm, floor_mm=0.5, unseen_mm=50.0):
    """1 / miss^2 per (view, landmark); a view that never detected a landmark gets unseen_mm."""
    m = np.where(np.isfinite(miss_mm), np.maximum(miss_mm, floor_mm), unseen_mm)
    return 1.0 / m ** 2


def ray_misses(UV, cams, off, G, scale_mm):
    """How far each view's ray through its (offset-corrected) detection passes from G (n x 3 mm): view x n mm."""
    n = len(G); miss = np.full((len(cams), n), np.nan)
    for k, cam in enumerate(cams):
        ok = np.isfinite(UV[k, :n, 0])
        if ok.any():
            o, d = cam.rays(UV[k, :n][ok] - off[k, :n][ok]); v = G[ok] / scale_mm - o
            miss[k, ok] = np.linalg.norm(v - (v * d).sum(1)[:, None] * d, axis=1) * scale_mm
    return miss


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


def canonical_laplacian(cfg=FACE):
    """Umbrella Laplacian of the canonical mesh's 468 vertices, and those vertices (mm)."""
    import scipy.sparse as sp
    V, F = canonical_raw(cfg)
    E = np.unique(np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1), axis=0)
    A = sp.coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), (len(V), len(V))).tocsr()
    return (sp.diags(np.asarray(A.sum(1)).ravel()) - A).tocsr(), V


def fit_rays(UV, cams, off, V, F, init, cfg=FACE, lap=None, view_w=None):
    """Step 5: the 468 face points fitted jointly to every (view, landmark) ray, mm in the frame of V.
    UV, off: view x >=468 x 2 px; cams in render units (scale_mm); init: 468 x 3 seated triangulation (NaN ok).
    Minimises sum w_kj |ray_kj miss|^2 + lambda |L (X - canonical posed onto init)|^2, w Cauchy on the miss,
    times view_w (view x landmark prior weight) when given."""
    import scipy.sparse as sp, scipy.sparse.linalg as sla
    L, Vc = canonical_laplacian(cfg) if lap is None else lap
    n, s, c = len(Vc), cfg["scale_mm"], cfg["fit_cauchy_mm"]
    O = np.zeros((len(cams), n, 3)); D = np.zeros((len(cams), n, 3))
    has = np.isfinite(UV[:, :n, 0])
    for k, cam in enumerate(cams):
        if has[k].any():
            o, d = cam.rays(UV[k, :n][has[k]] - off[k, :n][has[k]]); O[k, has[k]] = o * s; D[k, has[k]] = d
    ok = np.isfinite(init).all(1)
    if ok.sum() < 4:
        return init
    ref = _similarity(Vc[ok], init[ok])(Vc)
    LL = sp.kron(cfg["fit_lambda"] * (L.T @ L), sp.eye(3)).tocsr()
    Pm = np.eye(3)[None, None] - D[..., :, None] * D[..., None, :]          # projector off each ray
    PmO = (Pm @ O[..., None])[..., 0]
    X = np.where(ok[:, None], init, ref)
    for _ in range(cfg["fit_iters"]):
        v = X[None] - O; miss = np.linalg.norm(v - (v * D).sum(2, keepdims=True) * D, axis=2)
        w = np.where(has, 1 / (1 + (miss / c) ** 2), 0.0)
        if view_w is not None:
            w = w * view_w[:, :n]
        A = sp.block_diag(list((w[..., None, None] * Pm).sum(0))).tocsr()
        b = (w[..., None] * PmO).sum(0)
        X = sla.spsolve((A + LL).tocsc(), b.ravel() + LL @ ref.ravel()).reshape(n, 3)
        X, _ = seat(X, V, F)
    return X


def dense_template(cfg=FACE):
    """What fit_dense needs from the canonical face, built once: vertices, Laplacian^T Laplacian (3N), and
    the per-vertex mask of where the surface term applies."""
    import scipy.sparse as sp
    Vd, Fd = canonical_dense(cfg); n = len(Vd)
    E = np.unique(np.sort(np.r_[Fd[:, [0, 1]], Fd[:, [1, 2]], Fd[:, [2, 0]]], 1), axis=0)
    A = sp.coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), (n, n)).tocsr()
    L = (sp.diags(np.asarray(A.sum(1)).ravel()) - A).tocsr()
    return Vd, sp.kron(L.T @ L, sp.eye(3)).tocsr(), np.linalg.norm(Vd - Vd[2], axis=1) < cfg["dense_region_mm"]


def dense_applies(cfg=FACE):
    """Landmarks (468) that take fit_dense's result: the nose base."""
    V = canonical_raw(cfg)[0][:468]
    return np.linalg.norm(V - V[2], axis=1) < cfg["dense_apply_mm"]


def fit_dense(UV, cams, off, V, F, init, cfg=FACE, tpl=None, view_w=None):
    """Step 6: the dense canonical face fitted to the scan (V, F mm) from init (468 x 3, step 5), minimising
    point-to-plane distance over tpl's region + dense_beta x step 5's weighted ray term on vertices 0-467 +
    dense_lambda |L (Y - canonical posed on Y)|^2. Returns 468 x 3, seated: the fit's points for the nose base
    (dense_applies), init's elsewhere."""
    import scipy.sparse as sp, scipy.sparse.linalg as sla
    Vd, LL3, region = dense_template(cfg) if tpl is None else tpl
    N, n, s = len(Vd), 468, cfg["scale_mm"]
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(np.asarray(V, np.float32)), o3d.core.Tensor(np.asarray(F, np.uint32)))
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)); m.compute_triangle_normals()
    TN = np.asarray(m.triangle_normals)
    ok = np.isfinite(init).all(1)
    if ok.sum() < 4:
        return init
    Y = _similarity(Vd[:n][ok], init[ok])(Vd)
    O = np.zeros((len(cams), n, 3)); D = np.zeros((len(cams), n, 3)); has = np.isfinite(UV[:, :n, 0])
    for k, cam in enumerate(cams):
        if has[k].any():
            o, d = cam.rays(UV[k, :n][has[k]] - off[k, :n][has[k]]); O[k, has[k]] = o * s; D[k, has[k]] = d
    Pm = np.eye(3)[None, None] - D[..., :, None] * D[..., None, :]; PmO = (Pm @ O[..., None])[..., 0]
    lam, beta, c = cfg["dense_lambda"], cfg["dense_beta"], cfg["fit_cauchy_mm"]
    for _ in range(cfg["dense_iters"]):
        q = sc.compute_closest_points(o3d.core.Tensor(Y.astype(np.float32)))
        C = q["points"].numpy().astype(float); nrm = TN[q["primitive_ids"].numpy()]
        wd = region * (np.linalg.norm(C - Y, axis=1) < cfg["dense_trim_mm"])
        B = wd[:, None, None] * nrm[:, :, None] * nrm[:, None, :]; b = (B @ C[..., None])[..., 0]
        v = Y[None, :n] - O; miss = np.linalg.norm(v - (v * D).sum(2, keepdims=True) * D, axis=2)
        w = np.where(has, 1 / (1 + (miss / c) ** 2), 0.0) * (1 if view_w is None else view_w[:, :n])
        B[:n] += beta * (w[..., None, None] * Pm).sum(0); b[:n] += beta * (w[..., None] * PmO).sum(0)
        ref = _similarity(Vd, Y)(Vd)                      # re-posed each pass: the prior is on shape, not pose
        Y = sla.spsolve((sp.block_diag(list(B)).tocsr() + lam * LL3).tocsc(),
                        b.ravel() + lam * (LL3 @ ref.ravel())).reshape(N, 3)
    out = np.array(init, float).copy(); keep = dense_applies(cfg)
    out[keep] = seat(Y[:n], V, F)[0][keep]
    return out


def canonical_edges(cfg=FACE):
    """The canonical mesh's edges among the 468 landmarks (sorted pairs)."""
    F = canonical_raw(cfg)[1]
    return np.unique(np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1), axis=0)


def face_size(X):
    """RMS distance of the points from their centroid (mm)."""
    return float(np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean()))


def spread(X, V, F, edge_rest, cfg=FACE, E=None):
    """Step 7: push apart every landmark-mesh edge shorter than spacing_rho x its typical length
    (edge_rest x face_size(X)), averaging the pushes per point, pulling back toward X by spacing_anchor, and
    seating on the scan (V, F mm), for up to spacing_sweeps sweeps. Edges at normal length are untouched."""
    E = canonical_edges(cfg) if E is None else E
    rho, L = cfg["spacing_rho"], edge_rest * face_size(X)
    Y = X.copy()
    for _ in range(cfg["spacing_sweeps"]):
        d = Y[E[:, 1]] - Y[E[:, 0]]; l = np.linalg.norm(d, axis=1); short = l < rho * L
        if not short.any():
            break
        k = 0.5 * (rho * L[short] - l[short])[:, None] * d[short] / np.maximum(l[short], 1e-9)[:, None]
        c = np.zeros_like(Y); n = np.zeros(len(Y))
        np.add.at(c, E[short, 0], -k); np.add.at(c, E[short, 1], k); np.add.at(n, E[short].ravel(), 1)
        Y = Y + c / np.maximum(n, 1)[:, None]
        Y = seat(Y + cfg["spacing_anchor"] * (X - Y), V, F)[0]
    return Y


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
        self.lap = canonical_laplacian(self.cfg) if self.cfg["fit_rays"] else None
        self.tpl = dense_template(self.cfg) if self.cfg["fit_dense"] else None
        self.edge_rest = load_offsets(self.cfg, "edge_rest") if self.cfg["fit_spacing"] else None
        miss = load_offsets(self.cfg, "miss_mm") if self.cfg["fit_view_weights"] else None
        self.view_w = None if miss is None else view_weights(miss)

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
        Pt = Ps.copy()                                      # step 4, kept for comparison
        if c["fit_rays"]:
            n = len(self.lap[1])
            Ps[:n] = fit_rays(UV, cams, self.off, np.asarray(geo.vertices) * s, np.asarray(geo.triangles),
                              Ps[:n] * s, c, self.lap, self.view_w) / s
        if c["fit_dense"]:
            n = len(self.lap[1]) if self.lap is not None else 468
            Ps[:n] = fit_dense(UV, cams, self.off, np.asarray(geo.vertices) * s, np.asarray(geo.triangles),
                               Ps[:n] * s, c, self.tpl, self.view_w) / s
        if c["fit_spacing"]:
            Ps[:468] = spread(Ps[:468] * s, np.asarray(geo.vertices) * s, np.asarray(geo.triangles), self.edge_rest, c) / s
        to_input = lambda X: (X * s + piv) @ R_head + centre
        out = dict(P=to_input(Ps), P_raw=to_input(P), P_tri=to_input(Pt), off_surface_mm=dist * s, residual_mm=res * s,
                   n_inliers=inliers.sum(1), n_views=np.isfinite(UV[:, 0, 0]).sum())
        E = np.array([np.r_[np.c_[cm.R, cm.t], [[0, 0, 0, 1]]] for cm in cams])
        dbg = dict(UV=UV, inliers=inliers, views=np.array(vs), E=E, cams=cams, piv=piv, R_head=R_head, centre=centre,
                   V=np.asarray(geo.vertices) * s, F=np.asarray(geo.triangles), P_norm=Ps * s, P_tri_norm=Pt * s,
                   P_raw_norm=P * s, imgs=imgs)
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
