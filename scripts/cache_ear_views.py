"""Run ear framing (ear_view) once on cached heads and cache its output, so
landmark and lift checks can start from the framed ear.

Usage: cache_ear_views.py HEAD_CACHE OUTDIR WORKER N_WORKERS
Per ear writes OUTDIR/<head>_<side>.npz: img, depth, K, E, size (rebuild with
ear3d.Camera(K, E, img.shape[:2], size)), R, centre, scale, box_mm, and
head = the HEAD_CACHE npz it came from (the mesh lives there, in the same
normalised frame: the view shows in_frame(mesh, R, centre, scale)).
Lives under runs/ (gitignored): it holds scan renders.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import open3d as o3d

from escape_audit import DEFAULTS
from ear3d.ear_view import ear_view, load_yolo

cache, out, WK, NW = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
yolo = load_yolo()
for f in sorted(cache.glob("*.npz"))[WK::NW]:
    z = np.load(f)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(z["V"]), o3d.utility.Vector3iVector(z["F"]))
    m.vertex_colors = o3d.utility.Vector3dVector(z["C"])
    m.compute_vertex_normals()
    for side in "LR":
        dst = out / f"{f.stem}_{side}.npz"
        if dst.exists():
            continue
        v = ear_view(m, side, dict(DEFAULTS), yolo)
        if v is None:
            print(f.stem, side, "NO EAR", flush=True)
            continue
        cam = v["cam"]
        K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1.0]])
        E = np.eye(4); E[:3, :3], E[:3, 3] = cam.R, cam.t
        np.savez_compressed(dst, img=v["img"], depth=v["depth"], K=K, E=E, size=cam.w,
                            R=v["R"], centre=v["centre"], scale=v["scale"], box_mm=v["box_mm"],
                            head=str(f.resolve()))
        print(f.stem, side, f"box_mm {v['box_mm']:.0f}", flush=True)
