"""2D ear landmarks on the framed side view (ear_view), one view per ear.

Usage: review_ear_landmarks.py CACHE_DIR OUT_DIR WORKER N_WORKERS
The ear_view frame is already a padded ear crop, so the shipped
LandmarkPredictor runs on the whole frame; no 2D-repo detector involved.
Writes OUT_DIR/<head>_<side>.png (points + index) and <head>_<side>.npy (55x2 px).
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import open3d as o3d
from PIL import Image, ImageDraw

from escape_audit import DEFAULTS
from ear3d.ear_view import ear_view, load_yolo
from ear3d.twod import LandmarkPredictor, find_best_checkpoint

cache, out, WK, NW = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
yolo = load_yolo()
ck = find_best_checkpoint()
print("checkpoint", ck, flush=True)
pred = LandmarkPredictor(ck, "cuda")
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
        lm, conf = pred.predict(v["img"], with_confidence=True)
        np.save(out / f"{f.stem}_{side}.npy", lm)
        a = Image.fromarray(v["img"]); d = ImageDraw.Draw(a)
        for j, (x, y) in enumerate(lm):
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 0, 0))
            d.text((x + 3, y - 5), str(j), fill=(255, 255, 0))
        c = "" if conf is None else f"  conf {float(np.mean(conf)):.2f}"
        d.text((4, 4), f"{f.stem} {side}{c}", fill=(0, 255, 0))
        a.save(out / f"{f.stem}_{side}.png")
        print(f.stem, side, "ok" + c, flush=True)
