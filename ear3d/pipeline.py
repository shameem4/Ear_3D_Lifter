"""End to end: a head mesh in, 55 3D landmarks per ear out. No fit, no GT.

    from ear3d.pipeline import Pipeline
    out = Pipeline().run(mesh)          # {"L": (55, 3), "R": (55, 3)} or None per side

The mesh is a head scan in MILLIMETRES, in any orientation; it is not
modified. Landmarks come back in the mesh's own coordinates. "L" is the
subject's left ear (+x of the normalised head frame).

Stages, each the one measured best on 2026-10-17 (80 HUTUBS ears):

  1. face skin (blotchy, config tone) -> frames.normalize_head: head frame
     from FaceMesh, upside-down guard. Clean skin turned 2/40 heads over.
  2. ear skin (clean, config ear_tone) -> ear_view: side view, YOLO box,
     pad, one zoom. Fixes each ear's frame (R, centre, scale).
  3. 25 views (+-30 deg, 15 deg steps) orbited about that frame, sun fixed to
     the head (config multiview_light_dir, x mirrored for the right ear),
     shipped 2D landmarker on the whole frame (flip fusion = its default).
  4. per landmark, RANSAC over its best 8 views by view_table_v9.npz
     (scripts/build_view_table.py), 1 mm threshold.
  5. re-seat each point onto the head surface.

Against lm3d_v9 this is 1.46 mm median on those 80 ears (2.29 vs v7).
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import open3d as o3d

from .camera import orbit_extrinsic
from .config import DEFAULTS, ROOT
from .ear_view import ear_view, load_yolo
from .frames import in_frame, normalize_head
from .render import render
from .triangulate import ransac_point

TABLE = pathlib.Path(__file__).with_name("view_table_v9.npz")
K_BEST = 8
RANSAC_MM = 1.0


def _apply_skin():
    # skin.py lives in scripts/ and is shared with the training-image
    # generators; load it by path rather than move it.
    spec = importlib.util.spec_from_file_location("skin", ROOT / "scripts" / "skin.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.apply_skin


class Pipeline:
    def __init__(self, cfg=None, device="cuda"):
        from .twod import LandmarkPredictor, find_best_checkpoint
        self.cfg = dict(DEFAULTS, **(cfg or {}))
        self.yolo = load_yolo()
        self.pred = LandmarkPredictor(find_best_checkpoint(), device)
        t = np.load(TABLE, allow_pickle=True)
        self.table, self.angles = t["table"], t["angles"]
        key = {tuple(a): i for i, a in enumerate(self.angles)}
        self.mirror = np.array([key[(-a[0], a[1])] for a in self.angles])
        self.apply_skin = _apply_skin()

    def _skinned(self, mesh, tone, noise):
        m = o3d.geometry.TriangleMesh(mesh)
        m.compute_vertex_normals()
        c = self.cfg
        return self.apply_skin(m, seed=c["skin_seed"], tone=tone, ao_strength=c["ao_strength"],
                               n_rays=c["ao_rays"], noise=noise)

    def run(self, mesh, return_debug=False):
        c = self.cfg
        face = self._skinned(mesh, c["tone"], c["skin_noise"])
        nh = normalize_head(face, c)
        if nh is None:
            return None
        mesh_c, R_head, centre, _ = nh
        # Same vertices, re-coloured with the ear skin (colour is per vertex,
        # so the face-skinned normalised mesh can simply be re-skinned).
        ear = self._skinned(mesh_c, c["ear_tone"], c["ear_skin_noise"])
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(np.asarray(mesh_c.vertices, np.float32)),
                            o3d.core.Tensor(np.asarray(mesh_c.triangles, np.uint32)))
        out, dbg = {}, {"R_head": R_head, "centre": centre}
        for side in "LR":
            v = ear_view(ear, side, c, self.yolo)
            if v is None:
                out[side] = None
                continue
            P = self._multiview(ear, side, v["R"], v["centre"], v["scale"])
            P = scene.compute_closest_points(o3d.core.Tensor(P.astype(np.float32)))["points"].numpy()
            out[side] = P @ R_head + centre                      # back to the input frame
            dbg[side] = dict(R=v["R"], centre=v["centre"], scale=v["scale"], P_norm=P)
        return (out, dbg) if return_debug else out

    def _multiview(self, ear, side, R, c0, s):
        c = self.cfg
        g = in_frame(ear, R, c0, s)
        d0 = np.asarray(c["multiview_light_dir"], float) * ([-1.0, 1, 1] if side == "R" else 1.0)
        V = len(self.angles)
        O = np.full((V, 55, 3), np.nan); D = O.copy()
        for i, (yw, pt) in enumerate(self.angles):
            cfg = dict(c, light_dir=tuple(d0), light_mirror=False, light_frame="world")
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
