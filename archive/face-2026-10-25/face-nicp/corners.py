import sys, os, numpy as np
R0 = "/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_3D_Lifter"; sys.path[:0] = [R0, R0 + "/scripts/face"]; os.environ["FACE_RUN"] = "v1"
from concurrent.futures import ProcessPoolExecutor
from score_loho import RUN, H, OFF, RAY, LAP, s, REG
import ear3d.face as FM
from ear3d.face import FACE, cameras, triangulate, seat, fit_rays, view_weights, fit_dense, dense_template, canonical_raw
Vd = FM.canonical_dense()[0]; Vc = canonical_raw()[0][:468]
CORN = REG["lip corners"][0]
def tpl(r, lam):
    Vd_, LL3, _ = dense_template(dict(FACE, dense_lambda=lam))
    reg = (np.linalg.norm(Vd - Vd[61], axis=1) < r) | (np.linalg.norm(Vd - Vd[291], axis=1) < r)
    return Vd_, LL3, reg
ARMS = {f"r{r} l{l} b{b}": (r, l, b) for r, l, b in [(10, 10, 0.5), (15, 10, 0.5), (20, 10, 0.5), (15, 30, 0.5), (15, 10, 0.2)]}
T_ = {k: tpl(r, l) for k, (r, l, b) in ARMS.items()}
FM.dense_applies = lambda cfg=FACE: CORN                     # take the result for the lip-corner set only
def one(j):
    tr = [i for i in range(len(H)) if i != j]; z = np.load(RUN / f"{H[j]}.npz"); G = np.load(RUN / f"{H[j]}_gtd.npz")["G"]
    off = np.nan_to_num(np.nanmean(OFF[tr], 0)); w = view_weights(np.nanmedian(RAY[tr], 0)); cams = cameras(z["E"])
    P, _, _ = triangulate(z["UV"][:, :468], cams, off, FACE["ransac_mm"] / s); T, _ = seat(P * s, z["V"], z["F"])
    X = fit_rays(z["UV"], cams, off, z["V"], z["F"], T, FACE, LAP, w); out = {"current": X}
    for k, (r, l, b) in ARMS.items():
        out[k] = fit_dense(z["UV"], cams, off, z["V"], z["F"], X, dict(FACE, dense_lambda=l, dense_beta=b), T_[k], w)
    E = REG["lip corners"][1]; g = np.linalg.norm(G[E[:, 0]] - G[E[:, 1]], axis=1)
    return H[j], {k: (np.linalg.norm(Y - G, axis=1)[CORN], (np.linalg.norm(Y[E[:, 0]] - Y[E[:, 1]], axis=1) / g < 0.5)[g > 1]) for k, Y in out.items()}
if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex: R = list(ex.map(one, range(len(H))))
    for k in R[0][1]:
        e = np.concatenate([r[1][k][0] for r in R]); ph = np.array([np.median(r[1][k][0]) for r in R]); p0 = np.array([np.median(r[1]["current"][0]) for r in R])
        c = np.concatenate([r[1][k][1] for r in R])
        print(f"{k:14s} lip corners {np.median(e):.2f} p90 {np.percentile(e, 90):.2f} | collapsed {c.mean():.1%} | better {np.sum(ph < p0)}/50, worst {np.max(ph - p0):+.2f} ({R[int(np.argmax(ph - p0))][0]})")
