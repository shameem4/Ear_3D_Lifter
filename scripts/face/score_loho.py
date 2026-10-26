"""Leave one head out: re-triangulate each head's stored detections with the view offsets learned from the
OTHER heads, seat on the surface, and score against its pseudo GT (468 landmarks, mm).

Two scores per arm, each split into FEATURES (MediaPipe's lips, eyes, eyebrows and nose landmarks: where
points are dense and bunch) and the REST (cheeks, forehead, jaw: flat, sparse, would dilute it), plus two
hard spots: the NOSE BASE (within 12 mm of subnasale, canonical) and the LIP CORNERS (8 mm of 61 / 291):
  error    point distance to the GT
  spacing  over canonical-mesh edges with both ends in the region: |log(lifted length / GT length)|, median
           and p90, and "collapsed" = share of edges under half their GT length (GT length > 1 mm)

python scripts/face/score_loho.py   (needs runs/face/$FACE_RUN/view_table.npz from view_table.py)
On v0 (50 HUTUBS heads, 2026-10-21 lift): offsets off 4.21, on 2.34, + ray fit 2.16 mm median (seated).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import pathlib, numpy as np
from concurrent.futures import ProcessPoolExecutor
from ear3d.face import FACE, cameras, triangulate, seat, fit_rays, canonical_laplacian, canonical_raw, view_weights, fit_dense, dense_template, spread, canonical_edges, face_size

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN_TAG = __import__("os").environ.get("FACE_RUN", "v1")       # runs/face/<tag>: one lift generation + its GT
RUN = ROOT / "runs" / "face" / RUN_TAG; s = FACE["scale_mm"]
LAP = canonical_laplacian(); TPL = dense_template(); CE = canonical_edges()


def regions():
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as fc
    feat = sorted({i for k in ("LIPS", "LEFT_EYE", "RIGHT_EYE", "LEFT_EYEBROW", "RIGHT_EYEBROW", "NOSE")
                   for e in getattr(fc, "FACE_LANDMARKS_" + k) for i in (e.start, e.end)})
    m = np.zeros(468, bool); m[feat] = True
    _, F = canonical_raw()
    E = np.unique(np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1), axis=0)
    Vc = canonical_raw()[0][:468]
    nose = np.linalg.norm(Vc - Vc[2], axis=1) < 12                 # nostrils, columella, subnasale (21 lm)
    corners = (np.linalg.norm(Vc - Vc[61], axis=1) < 8) | (np.linalg.norm(Vc - Vc[291], axis=1) < 8)
    return {"features": (m, E[m[E].all(1)]), "rest": (~m, E[(~m[E]).all(1)]),
            "nose base": (nose, E[nose[E].all(1)]), "lip corners": (corners, E[corners[E].all(1)])}


REG = regions()
T = np.load(RUN / "view_table.npz"); H, OFF, RAY = list(T["heads"]), T["off_px"], T["ray_mm"]
GT = {h: np.load(RUN / f"{h}_gtd.npz")["G"] for h in H}


def one(j):
    z = np.load(RUN / f"{H[j]}.npz"); G = np.load(RUN / f"{H[j]}_gtd.npz")["G"]
    off = np.nan_to_num(np.nanmean(OFF[[i for i in range(len(H)) if i != j]], 0))
    cams, UV = cameras(z["E"]), z["UV"][:, :468]
    out = {}
    for arm, o in (("offsets off", np.zeros_like(off)), ("offsets on", off)):
        P, _, _ = triangulate(UV, cams, o, FACE["ransac_mm"] / s)
        Ps, _ = seat(P * s, z["V"], z["F"])
        out[arm] = Ps
    out["offsets on + ray fit"] = fit_rays(z["UV"], cams, off, z["V"], z["F"], out["offsets on"], FACE, LAP)
    w = view_weights(np.nanmedian(RAY[[i for i in range(len(H)) if i != j]], 0))
    out["+ view weights"] = fit_rays(z["UV"], cams, off, z["V"], z["F"], out["offsets on"], FACE, LAP, w)
    out["+ dense nose fit"] = fit_dense(z["UV"], cams, off, z["V"], z["F"], out["+ view weights"], FACE, TPL, w)
    tr = [H[i] for i in range(len(H)) if i != j]
    rest = np.median([np.linalg.norm(GT[h][CE[:, 0]] - GT[h][CE[:, 1]], axis=1) / face_size(GT[h]) for h in tr], 0)
    out["+ spacing springs"] = spread(out["+ dense nose fit"], z["V"], z["F"], rest, FACE, CE)
    res = {}
    for arm, X in out.items():
        for r, (m, E) in REG.items():
            g = np.linalg.norm(G[E[:, 0]] - G[E[:, 1]], axis=1); x = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
            res[arm, r] = (np.linalg.norm(X - G, axis=1)[m], np.abs(np.log(x / g)), (x / g < 0.5)[g > 1])
    return res


if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex: R = list(ex.map(one, range(len(H))))
    print(f"{len(H)} heads, leave one head out | features: {REG['features'][0].sum()} lm, {len(REG['features'][1])} edges;"
          f" rest: {REG['rest'][0].sum()} lm, {len(REG['rest'][1])} edges")
    print(f"{'':22s} {'region':9s} error med  p90 | spacing med  p90  collapsed")
    for key in R[0]:
        e, sp_, c = (np.concatenate([r[key][i] for r in R]) for i in range(3))
        print(f"{key[0]:22s} {key[1]:9s} {np.nanmedian(e):8.2f} {np.nanpercentile(e, 90):5.2f} |"
              f" {np.nanmedian(sp_):10.3f} {np.nanpercentile(sp_, 90):5.3f}  {np.nanmean(c):8.2%}")
    pa, pb = ([np.nanmedian(r["offsets on", "features"][1]) for r in R], [np.nanmedian(r["offsets on + ray fit", "features"][1]) for r in R])
    print(f"features spacing: ray fit better on {(np.array(pb) < np.array(pa)).sum()}/{len(H)} heads")
