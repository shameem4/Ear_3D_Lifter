"""Ear stage only, on wrapped ears with no head (AudioEar3D).

Usage: validate_wrapped_ears.py OUT_DIR DATASET WORKER N_WORKERS

There is no face to find, so the head frame is supplied: each wrapped ear
(corpus/canonical_esc, canonical mm, right ears already mirrored to left) is
turned into the normalised-head frame by the MEAN canonical-left -> head
rotation over the 40 cached HUTUBS heads (R_head @ Rr), and run as side "L".
NO DETECTOR: the ear's pose is already known, so the frame comes from the
wrap itself -- side view, the wrap's own extent in that view as the box
(same PAD as ear_view), pivot on the outermost surface point at the box
centre. Then the pipeline: ear skin, 25 views,
best-8 RANSAC, re-seat (on the ear itself). Scored in canonical mm against
config gt_lm3d and its alternates.

What this does NOT test: face finding, and an ear with a head around it
(the detector and the view table were built with one).
"""
import glob
import os
import pathlib
import sys

import numpy as np
import open3d as o3d

R0 = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R0))
from ear3d.config import DEFAULTS
from ear3d.corpus import WRAP
from ear3d.ear_view import PAD, side_frame
from ear3d.pipeline import Pipeline

out, ds, WK, NW = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)

# mean canonical-left -> normalised-head rotation (projected back onto SO(3))
M = sum(h["R_head"] @ h["Rr"] for h in (np.load(f) for f in sorted(glob.glob(str(R0 / "runs/cache/v4_s[01]/*.npz")))))
U, _, Vt = np.linalg.svd(M); A = U @ Vt
A = A if np.linalg.det(A) > 0 else U @ np.diag([1, 1, -1]) @ Vt

refs = {}
for fn in (DEFAULTS["gt_lm3d"], *DEFAULTS.get("gt_lm3d_also", ())):
    z = np.load(WRAP / "templates" / fn, allow_pickle=True)
    refs[fn.split("_canonical")[0]] = (z["landmarks"], [str(x) for x in z["names"]])

pipe = Pipeline(cfg={"ear_detector": os.environ.get("EAR_DETECTOR", DEFAULTS["ear_detector"])})
c = pipe.cfg
from ear3d import corpus as _corpus
K0 = DEFAULTS["gt_lm3d"].split("_canonical")[0]
files = sorted(glob.glob(str(WRAP / _corpus.get("canonical") / ds / "*.obj")))
for f in files[WK::NW]:
    nm = pathlib.Path(f).stem; dst = out / f"{nm}.npz"
    if dst.exists() or nm not in refs[K0][1]:
        continue
    m = o3d.io.read_triangle_mesh(f); V = np.asarray(m.vertices); c0 = V.mean(0)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector((V - c0) @ A.T), m.triangles)
    m.compute_vertex_normals()
    ear = pipe._skinned(m, c["ear_tone"], c["ear_skin_noise"])
    R = side_frame("L"); W = np.asarray(m.vertices) @ R.T          # x, y in view, z toward camera
    lo, hi = W[:, :2].min(0), W[:, :2].max(0); mid = 0.5 * (lo + hi)
    near = W[np.linalg.norm(W[:, :2] - mid, axis=1) < 3.0]
    p0 = np.r_[mid, near[:, 2].max() if len(near) else W[:, 2].max()]
    half = c["eye_z"] * np.tan(np.radians(c["vfov"]) / 2)
    v = dict(R=R, centre=p0 @ R, scale=0.5 * float((hi - lo).max()) * (1 + 2 * PAD) / half)
    P = np.full((55, 3), np.nan)
    if True:
        P = pipe._multiview(ear, "L", v["R"], v["centre"], v["scale"])
        ok = np.isfinite(P).all(1)
        if ok.any():
            sc = o3d.t.geometry.RaycastingScene()
            sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
            P[ok] = sc.compute_closest_points(o3d.core.Tensor(P[ok].astype(np.float32)))["points"].numpy()
    P = P @ A + c0                                              # back to canonical mm
    rec = {"pred": P, "found": v is not None}
    for k, (L, names) in refs.items():
        rec[f"gt_{k}"] = L[names.index(nm)][:55]
    np.savez(dst, **rec)
    e = np.nanmedian(np.linalg.norm(P - rec[f"gt_{K0}"], axis=1)) if v is not None else np.nan
    print(nm, "ear" if v is not None else "NO EAR", f"{e:.2f} mm vs {K0}", flush=True)
