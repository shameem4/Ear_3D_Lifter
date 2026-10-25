"""Leave one head out: re-triangulate each head's stored detections with the view offsets learned from the
OTHER heads, seat on the surface, and score against its pseudo GT (468 landmarks, mm).

python scripts/face/score_loho.py   (needs runs/face/$FACE_RUN/view_table.npz from view_table.py)
On v0 (50 HUTUBS heads, 2026-10-21 lift): offsets off 4.21, on 2.34, + ray fit 2.16 mm median (seated).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import pathlib, numpy as np
from concurrent.futures import ProcessPoolExecutor
from ear3d.face import FACE, cameras, triangulate, seat, fit_rays, canonical_laplacian

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN_TAG = __import__("os").environ.get("FACE_RUN", "v1")       # runs/face/<tag>: one lift generation + its GT
RUN = ROOT / "runs" / "face" / RUN_TAG; s = FACE["scale_mm"]
LAP = canonical_laplacian()
T = np.load(RUN / "view_table.npz"); H, OFF = list(T["heads"]), T["off_px"]


def one(j):
    z = np.load(RUN / f"{H[j]}.npz"); G = np.load(RUN / f"{H[j]}_gtd.npz")["G"]
    off = np.nan_to_num(np.nanmean(OFF[[i for i in range(len(H)) if i != j]], 0))
    cams, UV = cameras(z["E"]), z["UV"][:, :468]
    out = {}
    for arm, o in (("offsets off", np.zeros_like(off)), ("offsets on", off)):
        P, _, _ = triangulate(UV, cams, o, FACE["ransac_mm"] / s)
        Ps, _ = seat(P * s, z["V"], z["F"])
        out[arm] = np.linalg.norm(Ps - G, axis=1)
    X = fit_rays(z["UV"], cams, off, z["V"], z["F"], Ps, FACE, LAP)
    out["offsets on + ray fit"] = np.linalg.norm(X - G, axis=1)
    return out


if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex: R = list(ex.map(one, range(len(H))))
    for arm in R[0]:
        d = np.array([r[arm] for r in R])
        print(f"{arm:12s} median {np.nanmedian(d):.2f}  p90 {np.nanpercentile(d, 90):.2f}  worst head {np.nanmedian(d, 1).max():.2f} mm  ({len(H)} heads)")
    a, b = (np.array([np.nanmedian(r[k]) for r in R]) for k in ("offsets on", "offsets on + ray fit"))
    print(f"ray fit better on {(b < a).sum()}/{len(H)} heads; per-head change median {np.median(b - a):+.2f} mm, worst {np.max(b - a):+.2f}")
