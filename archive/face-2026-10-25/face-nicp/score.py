import sys, numpy as np
from concurrent.futures import ProcessPoolExecutor
from nicp import *
from score_loho import RUN, H, OFF, RAY, LAP, s, REG
Vc, _ = canonical_raw(); Vc = Vc[:468]
SUB = {"nostril": np.linalg.norm(Vc - Vc[2], axis=1) < 12,
       "lip corners": (np.linalg.norm(Vc - Vc[61], axis=1) < 8) | (np.linalg.norm(Vc - Vc[291], axis=1) < 8),
       "features": REG["features"][0], "all": np.ones(468, bool)}
CFG = [dict(lam=10, alpha=1, beta=0.5), dict(lam=10, alpha=1, beta=0.5, region_mm=25), dict(lam=10, alpha=1, beta=0.5, region_mm=35), dict(lam=10, alpha=1, beta=1, region_mm=25)]
HEADS = sys.argv[1:] or H

def one(h):
    j = H.index(h); tr = [i for i in range(len(H)) if i != j]
    z = np.load(RUN / f"{h}.npz"); G = np.load(RUN / f"{h}_gtd.npz")["G"]
    off = np.nan_to_num(np.nanmean(OFF[tr], 0)); w = view_weights(np.nanmedian(RAY[tr], 0)); cams = cameras(z["E"])
    P, _, _ = triangulate(z["UV"][:, :468], cams, off, FACE["ransac_mm"] / s); T, _ = seat(P * s, z["V"], z["F"])
    X = fit_rays(z["UV"], cams, off, z["V"], z["F"], T, FACE, LAP, w)
    out = {"current default": X}
    for c in CFG:
        out[f"nicp l{c['lam']} b{c['beta']} r{c.get('region_mm')}"] = nicp(z["UV"], cams, off, w, z["V"], z["F"], X, **c)[0]
    return h, {k: np.linalg.norm(Y - G, axis=1) for k, Y in out.items()}, out, G

if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex: R = list(ex.map(one, HEADS))
    import pickle; pickle.dump(R, open("nicp.pkl", "wb"))
    print(f"{'':22s}" + "".join(f"{r:>18s}" for r in SUB) + "   pp22 nostril")
    for k in R[0][1]:
        cells = [f"{np.nanmedian(np.concatenate([x[1][k][m] for x in R])):8.2f} p90 {np.nanpercentile(np.concatenate([x[1][k][m] for x in R]), 90):5.2f}" for m in SUB.values()]
        p22 = [np.median(x[1][k][SUB['nostril']]) for x in R if x[0] == "pp22"]
        print(f"{k:22s}" + "".join(f"{c:>18s}" for c in cells) + (f"   {p22[0]:.2f}" if p22 else ""))
