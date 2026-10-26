"""Each view's mean 2D offset of FaceLandmarker's detections from the pseudo GT, per landmark, and the
median ray miss of the raw detections (as validated, scripts/face/score_loho.py): ear3d/face_view_offsets_<tag>.npz (views, off_px: view x 478 x 2,
miss_mm: view x 468, edge_rest: typical canonical_edges length / face_size over the GT heads, n_heads, heads),
for FACE view_offsets, fit_view_weights and fit_spacing.

python scripts/face/build_view_offsets.py v0
Needs runs/face/$FACE_RUN/<head>.npz and <head>_gtd.npz. Iris (468-477) has no GT, so its offset is 0.
Valid only for the render settings and view grid it was learned with (FaceLifter checks the grid).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import sys, glob, os, pathlib, numpy as np
from ear3d.face import FACE, cameras, ray_misses, canonical_edges, face_size

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN_TAG = __import__("os").environ.get("FACE_RUN", "v1")       # runs/face/<tag>: one lift generation + its GT
RUN = ROOT / "runs" / "face" / RUN_TAG; s = FACE["scale_mm"]
tag = sys.argv[1]
H = sorted(os.path.basename(f)[:-8] for f in glob.glob(str(RUN / "*_gtd.npz")))
O, views = [], None
for h in H:
    z = np.load(RUN / f"{h}.npz"); G = np.load(RUN / f"{h}_gtd.npz")["G"]
    views = z["views"] if views is None else views
    assert (z["views"] == views).all(), f"{h}: different view grid"
    off = np.full((len(views), 478, 2), np.nan)
    for k, cam in enumerate(cameras(z["E"])):
        off[k, :468] = z["UV"][k, :468] - cam.project(G / s)
    O.append(off)
off = np.nan_to_num(np.nanmean(np.array(O), 0))         # no detection on any head -> no correction
M = [ray_misses(np.load(RUN / f"{h}.npz")["UV"], cameras(np.load(RUN / f"{h}.npz")["E"]), 0 * off,
                np.load(RUN / f"{h}_gtd.npz")["G"], s) for h in H]
miss = np.nanmedian(np.array(M), 0)                      # NaN: no head detected it in that view
dst = ROOT / "ear3d" / f"face_view_offsets_{tag}.npz"
CE = canonical_edges()
Gs = [np.load(RUN / f"{h}_gtd.npz")["G"] for h in H]
edge_rest = np.median([np.linalg.norm(G[CE[:, 0]] - G[CE[:, 1]], axis=1) / face_size(G) for G in Gs], 0)
np.savez(dst, views=views, off_px=off, miss_mm=miss, edge_rest=edge_rest, n_heads=len(H), heads=H)
print(f"{len(H)} heads -> {dst}; median |offset| {np.median(np.linalg.norm(off[:, :468], axis=2)):.1f} px")
