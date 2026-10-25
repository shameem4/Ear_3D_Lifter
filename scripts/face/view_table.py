"""Landmark accuracy vs view: how far each view's ray misses the pseudo GT, per view x landmark, over the
lifted HUTUBS heads (runs/face/$FACE_RUN/<head>.npz + <head>_gtd.npz).

python scripts/face/view_table.py  -> runs/face/$FACE_RUN/view_table.npz (ray_mm, off_px: head x view x 468) + .csv

  ray_mm: distance from GT landmark i to the ray view k casts through its detection (what triangulation sees)
  off_px: 2D offset, detection minus GT projection (its mean over heads is the view-offset table)
Also prints the split-half Spearman of the view x landmark median miss: how stable the table is across heads.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import glob, os, pathlib, numpy as np
from scipy.stats import spearmanr
from ear3d.face import FACE, cameras

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN_TAG = __import__("os").environ.get("FACE_RUN", "v1")       # runs/face/<tag>: one lift generation + its GT
RUN = ROOT / "runs" / "face" / RUN_TAG; s = FACE["scale_mm"]


def rays_table(z, G):
    cams = cameras(z["E"]); nv = len(cams)
    ray = np.full((nv, 468), np.nan); off = np.full((nv, 468, 2), np.nan)
    for k, cam in enumerate(cams):
        uv = z["UV"][k, :468]; ok = np.isfinite(uv[:, 0])
        if not ok.any(): continue
        o, d = cam.rays(uv[ok]); g = G[ok] / s
        v = g - o; ray[k, ok] = np.linalg.norm(v - (v * d).sum(1)[:, None] * d, axis=1) * s
        off[k, ok] = uv[ok] - cam.project(g)
    return ray, off


if __name__ == "__main__":
    H = sorted(os.path.basename(f)[:-8] for f in glob.glob(str(RUN / "*_gtd.npz")))
    R, O = map(np.array, zip(*[rays_table(np.load(RUN / f"{h}.npz"), np.load(RUN / f"{h}_gtd.npz")["G"]) for h in H]))
    views = np.load(RUN / f"{H[0]}.npz")["views"]
    np.savez(RUN / "view_table.npz", heads=H, views=views, ray_mm=R, off_px=O)
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as fc
    REG = {k.lower(): sorted({i for e in getattr(fc, "FACE_LANDMARKS_" + k) for i in (e.start, e.end)})
           for k in ("FACE_OVAL", "LIPS", "LEFT_EYE", "RIGHT_EYE", "LEFT_EYEBROW", "RIGHT_EYEBROW", "NOSE")}
    REG = {"all": list(range(468)), **REG}
    rows = []
    for k, (y, p) in enumerate(views):
        det = np.isfinite(R[:, k, 0]).mean()
        rows.append([int(y), int(p), f"{det:.0%}"] + [f"{np.nanmedian(R[:, k, v]):.1f}" if det else "-" for v in REG.values()])
    hdr = ["yaw", "pitch", "face found"] + list(REG)
    with open(RUN / "view_table.csv", "w") as f:
        f.write(",".join(hdr) + "\n"); [f.write(",".join(map(str, r)) + "\n") for r in rows]
    print(f"median ray miss (mm) per view and region over {len(H)} heads; 'face found' = share of heads detected")
    print(" ".join(f"{h:>10s}" for h in hdr))
    for r in rows: print(" ".join(f"{str(x):>10s}" for x in r))
    a, b = np.nanmedian(R[0::2], 0), np.nanmedian(R[1::2], 0); ok = np.isfinite(a) & np.isfinite(b)
    print("split-half consistency of the view x landmark median miss: spearman %.2f" % spearmanr(a[ok], b[ok])[0])
