"""Ear framing review on cached normalised heads (cache_head_frames.py).

Usage: review_ear_view.py CACHE_DIR OUT_DIR WORKER N_WORKERS
Writes OUT_DIR/<head>_<side>.png per ear, crosshair at frame centre, and
prints the ear height as a fraction of frame (YOLO re-run on the zoom).
Run N_WORKERS copies; then tile the PNGs to look at them.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import open3d as o3d
from PIL import Image, ImageDraw

from escape_audit import DEFAULTS
from ear3d.ear_view import ear_box, ear_view, load_yolo

cache, out, WK, NW = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
yolo = load_yolo()
for f in sorted(cache.glob("*.npz"))[WK::NW]:
    z = np.load(f)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(z["V"]), o3d.utility.Vector3iVector(z["F"]))
    m.vertex_colors = o3d.utility.Vector3dVector(z["C"])
    m.compute_vertex_normals()
    for side in "LR":
        v = ear_view(m, side, dict(DEFAULTS), yolo)
        if v is None:
            print(f.stem, side, "NO EAR", flush=True)
            continue
        img = v["img"]; H, W = img.shape[:2]
        # Measured on the CLEAN render, before anything is drawn: a crosshair
        # through the ear stops YOLO finding it (14 of 80 saved PNGs).
        b = ear_box(yolo, img)
        fill = (b[3] - b[1]) / H if b is not None else float("nan")
        r = yolo.predict(np.ascontiguousarray(img[:, :, ::-1]), conf=0.05, verbose=False)[0]
        t = r.boxes.xyxy.cpu().numpy()[int(np.argmax(r.boxes.conf.cpu().numpy()))] if len(r.boxes) else None
        fill_top = (t[3] - t[1]) / H if t is not None else float("nan")
        a = Image.fromarray(img); d = ImageDraw.Draw(a)
        d.line([(W // 2, 0), (W // 2, H)], fill=(0, 255, 0)); d.line([(0, H // 2), (W, H // 2)], fill=(0, 255, 0))
        d.text((4, 4), f"{f.stem} {side}  box {v['box_mm']:.0f} mm  ear {fill * 100:.0f}% of frame", fill=(255, 255, 0))
        a.save(out / f"{f.stem}_{side}.png")
        print(f.stem, side, f"box_mm {v['box_mm']:.0f} fill {fill:.3f} fill_top {fill_top:.3f}", flush=True)
