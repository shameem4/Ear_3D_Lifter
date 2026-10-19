"""Head mesh -> 3D ear landmarks, end to end (ear3d.pipeline).

Usage: run_pipeline.py MESH [OUT.npz]
MESH: a head scan in millimetres (.obj/.ply/...), any orientation.
OUT:  L, R = (55, 3) landmarks in the mesh's own coordinates (NaN-filled if
      an ear was not found), and ckpt = the 2D checkpoint used. Defaults to
      MESH with _ear_landmarks.npz. Env EAR3D_CKPT (path or run) pins the 2D model.
"""
import pathlib
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ear3d.pipeline import Pipeline

mp = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else mp.with_name(mp.stem + "_ear_landmarks.npz")
mesh = o3d.io.read_triangle_mesh(str(mp))
if len(mesh.triangles) == 0:
    sys.exit(f"{mp}: no triangles")
pipe = Pipeline(); res = pipe.run(mesh)
if res is None:
    sys.exit("no face found: could not orient the head")
nan = np.full((55, 3), np.nan)
np.savez(out, L=res["L"] if res["L"] is not None else nan, R=res["R"] if res["R"] is not None else nan,
         ckpt=pipe.ckpt)
print(f"wrote {out}  (L {'ok' if res['L'] is not None else 'NOT FOUND'}, R {'ok' if res['R'] is not None else 'NOT FOUND'})")
