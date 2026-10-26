import sys, os, numpy as np
R0 = "/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_3D_Lifter"; sys.path[:0] = [R0, R0 + "/scripts/face"]
os.environ["FACE_RUN"] = "v1"
from concurrent.futures import ProcessPoolExecutor
from ear3d.face import FACE, cameras, triangulate, seat, fit_rays, canonical_laplacian, canonical_raw
from score_loho import REG, RUN, H, OFF, s, LAP
RAY = np.load(RUN / "view_table.npz")["ray_mm"]          # head x view x 468
Vc, _ = canonical_raw(); Vc = Vc[:468]
SUB = {"nostril": np.linalg.norm(Vc - Vc[2], axis=1) < 12,
       "lip corners": (np.linalg.norm(Vc - Vc[61], axis=1) < 8) | (np.linalg.norm(Vc - Vc[291], axis=1) < 8),
       "features": REG["features"][0], "all": np.ones(468, bool)}

def one(j):
    tr = [i for i in range(len(H)) if i != j]
    z = np.load(RUN / f"{H[j]}.npz"); G = np.load(RUN / f"{H[j]}_gtd.npz")["G"]
    off = np.nan_to_num(np.nanmean(OFF[tr], 0)); m = np.nanmedian(RAY[tr], 0)      # view x 468, mm
    m = np.where(np.isfinite(m), np.maximum(m, 0.5), 50.0)
    cams = cameras(z["E"])
    P, _, _ = triangulate(z["UV"][:, :468], cams, off, FACE["ransac_mm"] / s)
    T, _ = seat(P * s, z["V"], z["F"])
    rank = np.argsort(np.argsort(m, 0), 0)
    arms = {"fit": None, "fit w=1/m^2": 1 / m ** 2, "fit w=1/m": 1 / m, "fit best 8": (rank < 8).astype(float)}
    out = {"tri": T}
    for k, w in arms.items():
        out[k] = fit_rays(z["UV"], cams, off, z["V"], z["F"], T, FACE, LAP, w)
    return H[j], {k: np.linalg.norm(X - G, axis=1) for k, X in out.items()}, out, G

if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex: R = list(ex.map(one, range(len(H))))
    import pickle; pickle.dump(R, open("vw.pkl", "wb"))
    print(f"{'':14s}" + "".join(f"{r:>20s}" for r in SUB))
    for k in R[0][1]:
        cells = []
        for r, msk in SUB.items():
            d = np.concatenate([x[1][k][msk] for x in R]); cells.append(f"{np.nanmedian(d):9.2f} p90 {np.nanpercentile(d, 90):5.2f}")
        print(f"{k:14s}" + "".join(f"{c:>20s}" for c in cells))
    print({r: int(m.sum()) for r, m in SUB.items()})
