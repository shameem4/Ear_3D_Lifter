"""The Python host for ear3d_core's pipeline (core/src/pipeline.rs).

The Rust pipeline makes every decision and asks its host for four things:
hold a mesh and its vertex colours, render views of it (returning the
pictures, or FaceMesh's points on them), and evaluate the bare 2D graphs.
Pre- and post-processing around those graphs is the crate's (detect.rs). This
is that host on Filament (open3d) and onnxruntime, with the graphs exported
by demo/export_models.py; the browser demo implements the same interface on
three.js and onnxruntime-web.

Arrays cross as little-endian bytes: f64 for geometry, f32 depth, u32 faces.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d

from .render import MATERIAL, SUN_LUX

BACKGROUND = (0.5, 0.5, 0.5)
MODELS = Path(__file__).resolve().parents[1] / "demo" / "models"
GRAPHS = {"r2j": ("render_landmarker.onnx", 192), "blazeear": ("blazeear.onnx", 256)}


def anchors():
    return np.fromfile(MODELS / "blazeear_anchors.bin", "<f8")


class FilamentHost:
    def __init__(self, face_task, verbose=True):
        import onnxruntime as ort
        self.verbose = verbose
        self.meshes, self.colors, self.events = {}, {}, []
        self.graphs = {k: ort.InferenceSession(str(MODELS / f), providers=["CPUExecutionProvider"])
                       for k, (f, _) in GRAPHS.items()}
        self._face = None
        self._face_task = face_task

    # ---- the interface
    def set_mesh(self, id, v, f):
        V = np.frombuffer(v, "<f8").reshape(-1, 3).copy()
        F = np.frombuffer(f, "<u4").reshape(-1, 3).astype(np.int32)
        self.meshes[id] = (V, F)

    def set_colors(self, id, mesh, c):
        self.colors[id] = (mesh, np.frombuffer(c, "<f8").reshape(-1, 3).copy())

    def render(self, req):
        out = []
        for img, depth in self._draw(req):
            o = {}
            if req["depth"]:
                o["depth"] = np.asarray(depth, "<f4").tobytes()
            if req["image"]:
                o["image"] = np.ascontiguousarray(img, np.uint8).tobytes()
            if req["detect"] == "face":
                uv = self._face_points(img)
                o["face"] = None if uv is None else uv.astype("<f8").tobytes()
            out.append(o)
        return out

    def _draw(self, req):
        """(image, depth or None) per view: Filament, exactly as ear3d.render draws."""
        size = int(req["size"])
        K = np.array([[req["f"], 0, req["c"]], [0, req["f"], req["c"]], [0, 0, 1.0]])
        V, F = self.meshes[req["mesh"]]
        cols = self.colors[req["colors"]][1]
        out = []
        for v in req["views"]:
            # in_frame: the mesh moved by the view's model transform, normals recomputed
            R, C = np.reshape(v["R"], (3, 3)), np.asarray(v["C"])
            m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(((V - C) @ R.T) / v["s"]), o3d.utility.Vector3iVector(F))
            m.vertex_colors = o3d.utility.Vector3dVector(cols)
            m.compute_vertex_normals()
            # A FRESH renderer per view, as ear3d.render does: a reused one carries
            # state between frames (~0.4 grey levels), and FaceMesh flips on that.
            # Never two alive at once -- that segfaults Filament.
            r = o3d.visualization.rendering.OffscreenRenderer(size, size)
            r.scene.add_geometry("m", m, MATERIAL)
            E = np.array(v["E"], float).reshape(4, 4)
            r.scene.set_background([*BACKGROUND, 1.0])
            r.scene.scene.set_sun_light([float(x) for x in v["sun"]], [1.0, 1.0, 1.0], SUN_LUX)
            r.scene.scene.enable_sun_light(True)
            r.setup_camera(K, E, size, size)
            img = np.asarray(r.render_to_image())[:, :, :3]
            depth = np.asarray(r.render_to_depth_image(z_in_view_space=True)) if req["depth"] else None
            del r
            out.append((img, depth))
        return out

    def infer(self, model, x):
        g, n = self.graphs[model], GRAPHS[model][1]
        x = np.frombuffer(x, "<f4").reshape(1, 3, n, n)
        return [np.ascontiguousarray(y, "<f4").tobytes() for y in g.run(None, {g.get_inputs()[0].name: x})]

    def event(self, kind, payload):
        if kind == "log":
            if self.verbose:
                print(json.loads(payload))
        else:
            self.events.append((kind, payload))

    # ---- helpers
    def _face_points(self, img):
        import mediapipe as mp
        if self._face is None:
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
            self._face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(self._face_task)), num_faces=1))
        res = self._face.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
        if not res.face_landmarks:
            return None
        h, w = img.shape[:2]
        return np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])


CFG_KEYS = ("size", "vfov", "eye_z", "light_dir", "face_light_dir", "multiview_light_dir", "tone",
            "skin_noise", "ear_tone", "ear_skin_noise", "skin_seed", "ao_strength", "ao_rays",
            "target_occupancy")


def canonical_face(cfg):
    model = Path(cfg.get("face_model") or Path(cfg["face_task"]).parent / "canonical_face_model.obj")
    return np.array([[float(x) for x in l.split()[1:4]] for l in open(model) if l.startswith("v ")])


def run(mesh, cfg, table, angles, template, verbose=True, host=None):
    """The Rust pipeline on an open3d mesh. Returns ({"L", "R"}, debug)."""
    import ear3d_core
    host = host or FilamentHost(cfg["face_task"], verbose)
    b = lambda a: np.ascontiguousarray(a, "<f8").tobytes()
    out = ear3d_core.run(host, b(np.asarray(mesh.vertices)), np.asarray(mesh.triangles, "<u4").tobytes(),
                         json.dumps({k: cfg[k] for k in CFG_KEYS}), b(table), b(angles), b(template),
                         b(canonical_face(cfg)), b(anchors()))
    pts = lambda x: None if x is None else np.frombuffer(x, "<f8").reshape(-1, 3).copy()
    res = {"L": pts(out["L"]), "R": pts(out["R"])}
    dbg = {"ear_only": out["ear_only"], "fit_residual": out["fit_residual"], "events": host.events,
           "R_head": None if out["R_head"] is None else np.reshape(out["R_head"], (3, 3)),
           "centre": None if out["centre"] is None else np.asarray(out["centre"])}
    return res, dbg
