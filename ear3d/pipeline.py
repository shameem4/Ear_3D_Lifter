"""End to end: a head mesh in, 55 3D landmarks per ear out. No fit, no GT.

    from ear3d.pipeline import Pipeline
    out = Pipeline().run(mesh)          # {"L": (55, 3), "R": (55, 3)} or None per side

The mesh is a head scan in MILLIMETRES, in any orientation; it is not
modified. Landmarks come back in the mesh's own coordinates. "L" is the
subject's left ear (+x of the normalised head frame).

Stages, each the one measured best on 2026-10-17 (80 HUTUBS ears):

  0. TEXTURE GATE: a mesh that brings its own colour (vertex colours with real
     variation, or a UV texture, which is baked to vertices) is rendered with
     it, and both synthetic skins below are skipped. OFF by default
     (cfg use_mesh_texture); synthetic skin looked better on a real scan.
  1. face skin (blotchy, config tone) -> frames.normalize_head: head frame
     from FaceMesh, upside-down guard. Clean skin turned 2/40 heads over.
  2. ear skin (clean, config ear_tone) -> ear_view: side view, ear box
     (config ear_detector: BlazeEar by default, Apache-2.0; YOLO optional),
     pad, one zoom. Fixes each ear's frame (R, centre, scale).
  3. 25 views (+-30 deg, 15 deg steps) orbited about that frame, sun fixed to
     the head (config multiview_light_dir, x mirrored for the right ear),
     shipped 2D landmarker (bare LandmarkPredictor, no detector or ROI) on
     the whole frame, flip fusion = its default.
  4. per landmark, RANSAC over its best 8 views by view_table_v9.npz
     (scripts/caches/4_build_view_table.py), 1 mm threshold.
  5. re-seat each point onto the head surface.

Against lm3d_v9 this is 1.46 mm median on those 80 ears (2.29 vs v7).
"""
from __future__ import annotations

import pathlib

import numpy as np
import open3d as o3d

from .camera import orbit_extrinsic
from .config import DEFAULTS
from .ear_view import ear_score, ear_view, load_detector, occupancy_scale, side_frame
from .frames import in_frame, normalize_head
from .render import render, side_light
from .triangulate import ransac_point

from . import corpus as _corpus
TABLE = pathlib.Path(__file__).with_name(_corpus.get("view_table"))   # EAR3D_CORPUS selects
TEXTURE_MIN_SD = 0.02
FACE_RETRIES = 4             # extra face searches on the mesh turned by a fixed rotation
# FRAME CHECKS, independent of FaceMesh. On SONICOM (cut at neck and crown)
# FaceMesh sometimes "finds" a face on the back of the head or on an ear and
# the frame comes out ~150-170 deg off. Measured on 34 SONICOM heads (31
# right frames, 3 wrong):
SYM_P90_MM = 7.0             # face mirror-symmetric about the nose: right 1.6-5.9, wrong 9.6-10.5
NOSE_EAR_MIN_MM = 110.0      # nose far from both ears: right 127-165, wrong 31-87
NOSE_EAR_BALANCE = 0.7       # nearer / farther ear distance: right 0.77-1.00, wrong 0.20-0.57
# Upside-down frames pass both checks above (they are symmetric under the
# flip) and swap the ears. The skull is wider than the neck/jaw: girth 60 mm
# above vs below the nose, right 1.23-1.91 (10 heads), upside-down 1.02-1.07.
GIRTH_OFFSET_MM = 60.0
GIRTH_RATIO_MIN = 1.15
_FLIP = np.diag([-1.0, -1.0, 1.0])  # 180 deg about the nose axis


def _retry_rotation(k):
    """A fixed, reproducible 3D rotation for face-search retry k."""
    q = np.random.default_rng(1000 + k).normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])        # colour s.d. below this is treated as untextured
K_BEST = 8
RANSAC_MM = 1.0


def _apply_skin():
    from .skin import apply_skin
    return apply_skin


def mesh_texture(mesh):
    """Per-vertex colours the mesh brings with it, or None.

    Vertex colours count if they vary (a flat or default colour is not a
    texture). Otherwise a UV texture is baked: each vertex takes the mean of
    the texel colours at its triangle corners.
    """
    if mesh.has_vertex_colors():
        C = np.asarray(mesh.vertex_colors)
        if len(C) == len(mesh.vertices) and C.std(0).max() > TEXTURE_MIN_SD:
            return C.copy()
    if mesh.has_triangle_uvs() and len(mesh.textures) > 0:
        F = np.asarray(mesh.triangles); uv = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)
        ids = (np.asarray(mesh.triangle_material_ids) if mesh.has_triangle_material_ids()
               else np.zeros(len(F), int))
        acc = np.zeros((len(mesh.vertices), 3)); n = np.zeros(len(mesh.vertices))
        for t in np.unique(ids):
            img = np.asarray(mesh.textures[int(t)])
            if img.size == 0:
                continue
            img = img[..., :3].astype(float) / (255.0 if img.dtype == np.uint8 else 1.0)
            h, w = img.shape[:2]; sel = ids == t
            u = np.clip((uv[sel, :, 0] * (w - 1)).round().astype(int), 0, w - 1)
            v = np.clip(((1 - uv[sel, :, 1]) * (h - 1)).round().astype(int), 0, h - 1)
            np.add.at(acc, F[sel].ravel(), img[v.ravel(), u.ravel()]); np.add.at(n, F[sel].ravel(), 1)
        if n.any() and (acc[n > 0] / n[n > 0, None]).std(0).max() > TEXTURE_MIN_SD:
            return acc / np.maximum(n, 1)[:, None]
    return None


def _girth_ratio(mesh_c, nose):
    """Longest closed-ish section perimeter GIRTH_OFFSET_MM above the nose
    over the one below. Skull over neck/jaw: > 1 when the frame is upright."""
    import trimesh
    tm = trimesh.Trimesh(np.asarray(mesh_c.vertices), np.asarray(mesh_c.triangles), process=False)
    def girth(y):
        sec = tm.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
        if sec is None:
            return 0.0
        return max((float(np.linalg.norm(np.diff(np.asarray(d), axis=0), axis=1).sum()) for d in sec.discrete), default=0.0)
    return girth(nose[1] + GIRTH_OFFSET_MM) / max(girth(nose[1] - GIRTH_OFFSET_MM), 1.0)


class Pipeline:
    def __init__(self, cfg=None, device="cuda"):
        from .twod import LandmarkPredictor, render_checkpoint
        self.cfg = dict(DEFAULTS, **(cfg or {}))
        self.yolo = load_detector(self.cfg["ear_detector"])
        self.device = device
        self.ckpt = str(render_checkpoint(self.cfg))          # record it with any result
        self.pred = self.pred_render = LandmarkPredictor(self.ckpt, device)
        self.pred_texture = None                             # the regular model, loaded on first textured mesh
        t = np.load(TABLE, allow_pickle=True)
        self.table, self.angles = t["table"], t["angles"]
        key = {tuple(a): i for i, a in enumerate(self.angles)}
        self.mirror = np.array([key[(-a[0], a[1])] for a in self.angles])
        self.apply_skin = _apply_skin()

    def _skinned(self, mesh, tone, noise):
        """A coloured copy. The skin's blotches and AO rays are laid out in
        world axes, so it is computed on a copy in the mesh's own PCA frame
        (signs fixed by the third moment) and the per-vertex colours copied
        back: the same head in any orientation gets the same skin."""
        V = np.asarray(mesh.vertices); c0 = V.mean(0)
        _, _, Vt = np.linalg.svd(V - c0, full_matrices=False)
        A = (V - c0) @ Vt.T
        Vt = Vt * np.sign(np.sign((A ** 3).sum(0)) + 0.5)[:, None]        # 0 -> +1
        if np.linalg.det(Vt) < 0:
            Vt[2] = -Vt[2]
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector((V - c0) @ Vt.T),
                                      o3d.utility.Vector3iVector(np.asarray(mesh.triangles)))
        m.compute_vertex_normals()
        c = self.cfg
        self.apply_skin(m, seed=c["skin_seed"], tone=tone, ao_strength=c["ao_strength"],
                        n_rays=c["ao_rays"], noise=noise)
        out = o3d.geometry.TriangleMesh(mesh)
        out.vertex_colors = m.vertex_colors
        out.compute_vertex_normals()
        return out

    def run(self, mesh, return_debug=False):
        c = self.cfg
        ext = float(np.ptp(np.asarray(mesh.vertices), axis=0).max())
        if not 100.0 <= ext <= 600.0:
            raise ValueError(f"mesh spans {ext:.3g} along its longest axis; the pipeline needs a head "
                             f"in MILLIMETRES (expected ~150-350). Metres? scale by 1000.")
        tex = mesh_texture(mesh) if c.get("use_mesh_texture", False) else None
        from .twod import LandmarkPredictor, landmarker_checkpoint, render_checkpoint
        if tex is not None:                                      # own texture: the regular model
            if self.pred_texture is None:
                self.ckpt_texture = str(landmarker_checkpoint(c))
                self.pred_texture = LandmarkPredictor(self.ckpt_texture, self.device)
            self.pred, self.ckpt = self.pred_texture, self.ckpt_texture
        else:                                                    # synthetic skin: the render-only model
            self.pred, self.ckpt = self.pred_render, str(render_checkpoint(c))
        if tex is not None:                                      # the mesh's own appearance
            face = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
                                             o3d.utility.Vector3iVector(np.asarray(mesh.triangles)))
            face.vertex_colors = o3d.utility.Vector3dVector(tex); face.compute_vertex_normals()
        else:
            face = self._skinned(mesh, c["tone"], c["skin_noise"])
        nh = self._find_face(face)
        if nh is None:
            return None
        mesh_c, R_head, centre, _ = nh
        # Same vertices, re-coloured with the ear skin (colour is per vertex,
        # so the face-skinned normalised mesh can simply be re-skinned).
        ear = mesh_c if tex is not None else self._skinned(mesh_c, c["ear_tone"], c["ear_skin_noise"])
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(np.asarray(mesh_c.vertices, np.float32)),
                            o3d.core.Tensor(np.asarray(mesh_c.triangles, np.uint32)))
        out, dbg = {}, {"R_head": R_head, "centre": centre, "textured": tex is not None}
        for side in "LR":
            v = ear_view(ear, side, c, self.yolo)
            if v is None:
                out[side] = None
                continue
            P = self._multiview(ear, side, v["R"], v["centre"], v["scale"])
            ok = np.isfinite(P).all(1)                           # a failed landmark stays NaN:
            if ok.any():                                         # the snap maps NaN to ~origin
                P[ok] = scene.compute_closest_points(o3d.core.Tensor(P[ok].astype(np.float32)))["points"].numpy()
            out[side] = P @ R_head + centre                      # back to the input frame
            dbg[side] = dict(R=v["R"], centre=v["centre"], scale=v["scale"], P_norm=P)
        return (out, dbg) if return_debug else out

    def _find_face(self, face):
        """normalize_head, retried on the mesh turned by fixed rotations.

        The face search renders 26 directions x 4 rolls fixed to the INPUT's
        axes, so an unlucky orientation can leave every candidate view short
        of the gate. Turning the mesh moves the views relative to the head;
        the frame found is mapped back exactly (V' = V Q^T, so the head frame
        of the original is R' Q and the centre c' Q). Skin is per vertex, so
        the colours travel with the vertices unchanged.
        """
        V = np.asarray(face.vertices)
        best = (np.inf, None)    # least-bad frame, when no search passes every check
        for k in range(FACE_RETRIES + 1):
            Q = np.eye(3) if k == 0 else _retry_rotation(k)
            m = o3d.geometry.TriangleMesh(face)
            m.vertices = o3d.utility.Vector3dVector(V @ Q.T); m.compute_vertex_normals()
            nh = normalize_head(m, dict(self.cfg, upright_judge=self._ear_judge(m)))
            if nh is not None:
                mesh_c, Rh, cen, nose = nh
                g = _girth_ratio(mesh_c, nose); print(f"pipeline: girth ratio {g:.2f}")
                if g < GIRTH_RATIO_MIN:                      # try it the other way up
                    mf = o3d.geometry.TriangleMesh(mesh_c)
                    mf.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_c.vertices) @ _FLIP)
                    mf.compute_vertex_normals(); gf = _girth_ratio(mf, nose @ _FLIP)
                    print(f"pipeline: face search {k} girth ratio {g:.2f}, flipped {gf:.2f}")
                    if gf < GIRTH_RATIO_MIN:
                        print(f"pipeline: face search {k} rejected (no clear skull above neck); retrying turned")
                        if GIRTH_RATIO_MIN / g < best[0]:
                            best = (GIRTH_RATIO_MIN / g, (mesh_c, Rh @ Q, cen @ Q, nose))
                        continue
                    mesh_c, Rh, nose = mf, _FLIP @ Rh, nose @ _FLIP
                ok, why, bad = self._frame_ok(mesh_c, nose)
                if ok:
                    return mesh_c, Rh @ Q, cen @ Q, nose
                print(f"pipeline: face search {k} rejected ({why}); retrying turned")
                if bad < best[0]:
                    best = (bad, (mesh_c, Rh @ Q, cen @ Q, nose))
        if best[1] is not None:
            print(f"pipeline: no face search passed; keeping the least-bad (x{best[0]:.2f} past its gate)")
        return best[1]

    def _frame_ok(self, mesh_c, nose):
        """(ok, reason, badness): is the normalised frame a real face? Symmetry about
        the nose, and the nose well away from, and between, the two ears."""
        from scipy.spatial import cKDTree
        Vn = np.asarray(mesh_c.vertices); L = Vn - nose
        near = L[np.hypot(L[:, 0], L[:, 1]) < 30]
        if not len(near):
            return False, "no surface at the nose", np.inf
        t = near[np.argmax(near[:, 2])] + nose                   # geometric nose tip
        reg = Vn[np.linalg.norm(Vn - t, axis=1) < 40]
        reg = reg[reg[:, 2] > t[2] - 40]
        mir = reg.copy(); mir[:, 0] = 2 * t[0] - mir[:, 0]
        sym = float(np.percentile(cKDTree(Vn).query(mir)[0], 90))
        if sym > SYM_P90_MM:
            return False, f"asymmetric about the nose, p90 {sym:.1f} mm", sym / SYM_P90_MM
        ear = self._skinned(mesh_c, self.cfg["ear_tone"], self.cfg["ear_skin_noise"])
        d = {}
        for side in "LR":
            v = ear_view(ear, side, self.cfg, self.yolo)
            if v is not None:
                d[side] = float(np.linalg.norm(v["centre"] - t))
        if not d:
            return False, "no ear on either side", np.inf
        if len(d) == 1:
            # One ear unseen (a coarse or occluded side; SONICOM's graded meshes)
            # in a frame that passed the face checks: still a rejection, but the
            # mildest one. At inf it lost the fallback to an asymmetric frame,
            # which put P0022_right's head on its side (2026-10-24).
            (s, x), = d.items()
            if x < NOSE_EAR_MIN_MM:
                return False, f"nose-ear {x:.0f} mm (side {s})", NOSE_EAR_MIN_MM / x
            return False, f"no ear on side {'R' if s == 'L' else 'L'}", 1.0
        d = [d["L"], d["R"]]
        if min(d) < NOSE_EAR_MIN_MM or min(d) / max(d) < NOSE_EAR_BALANCE:
            return False, f"nose-ear {d[0]:.0f}/{d[1]:.0f} mm", max(NOSE_EAR_MIN_MM / min(d), NOSE_EAR_BALANCE * max(d) / min(d))
        return True, "", 0.0

    def _ear_judge(self, face):
        """Score a candidate head frame by how readily the ear detector
        finds an ear in each side view (top confidence, L + R). An ear seen
        upside down scores lower. Used only on close upside-down calls."""
        V = np.asarray(face.vertices); c0 = V.mean(0)

        def judge(r):
            lat, vert = np.asarray(r[0], float), np.asarray(r[1], float)
            vert = vert - lat * (vert @ lat); vert /= np.linalg.norm(vert)
            Rh = np.stack([lat, vert, np.cross(lat, vert)])
            g = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector((V - c0) @ Rh.T), face.triangles)
            g = self._skinned(g, self.cfg["ear_tone"], self.cfg["ear_skin_noise"])
            tot = 0.0
            for side in "LR":
                R = side_frame(side); sc = float(np.abs(np.asarray(g.vertices) @ R.T).max())
                img, _, _ = render(in_frame(g, R, np.zeros(3), sc), side_light(self.cfg, side))
                tot += ear_score(self.yolo, img)
            return tot
        return judge

    def _multiview(self, ear, side, R, c0, s):
        c = self.cfg
        d0 = np.asarray(c["multiview_light_dir"], float) * ([-1.0, 1, 1] if side == "R" else 1.0)
        cfg = dict(c, light_dir=tuple(d0), light_mirror=False, light_frame="world")
        if c.get("target_occupancy"):                            # frame from the ear, not the box
            s = occupancy_scale(ear, R, c0, s, c, cfg, self.pred, c["target_occupancy"])
        g = in_frame(ear, R, c0, s)
        V = len(self.angles)
        O = np.full((V, 55, 3), np.nan); D = O.copy()
        for i, (yw, pt) in enumerate(self.angles):
            img, _, cam = render(g, cfg, E=orbit_extrinsic(float(yw), float(pt), c))
            o, d = cam.rays(self.pred.predict(img))
            O[i], D[i] = o * s @ R + c0, d @ R
        if side == "R":                                          # anatomical view order
            O, D = O[self.mirror], D[self.mirror]
        P = np.full((55, 3), np.nan)
        for j in range(55):
            vv = np.argsort(self.table[:, j])[:K_BEST]
            q = ransac_point(O[vv, j], D[vv, j], RANSAC_MM)
            q = q[0] if isinstance(q, tuple) else q
            if q is not None:
                P[j] = q
        bad = ~np.isfinite(P).all(1)                             # fall back to all views
        for j in np.where(bad)[0]:
            q = ransac_point(O[:, j], D[:, j], RANSAC_MM)
            q = q[0] if isinstance(q, tuple) else q
            if q is not None:
                P[j] = q
        return P
