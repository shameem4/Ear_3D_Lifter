import sys, os, numpy as np
R0 = "/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_3D_Lifter"; sys.path[:0] = [R0, R0 + "/scripts/face"]; os.environ["FACE_RUN"] = "v1"
from concurrent.futures import ProcessPoolExecutor
from score_loho import RUN, H, OFF, RAY, LAP, s, REG
from ear3d.face import FACE, cameras, triangulate, seat, fit_rays, view_weights, fit_dense, dense_template
TPL = {r: dense_template(dict(FACE, dense_region_mm=r)) for r in (15, 22, 35)}
def one(j):
    tr = [i for i in range(len(H)) if i != j]; z = np.load(RUN / f"{H[j]}.npz"); G = np.load(RUN / f"{H[j]}_gtd.npz")["G"]
    off = np.nan_to_num(np.nanmean(OFF[tr], 0)); w = view_weights(np.nanmedian(RAY[tr], 0)); cams = cameras(z["E"])
    P, _, _ = triangulate(z["UV"][:, :468], cams, off, FACE["ransac_mm"] / s); T, _ = seat(P * s, z["V"], z["F"])
    X = fit_rays(z["UV"], cams, off, z["V"], z["F"], T, FACE, LAP, w); out = {"no dense": X}
    for r, t in TPL.items(): out[f"region {r}"] = fit_dense(z["UV"], cams, off, z["V"], z["F"], X, dict(FACE, dense_region_mm=r), t, w)
    return {k: np.linalg.norm(Y - G, axis=1) for k, Y in out.items()}
if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex: R = list(ex.map(one, range(len(H))))
    for k in R[0]:
        d = np.array([r[k] for r in R])
        print(f"{k:10s} lm1 {np.median(d[:,1]):.2f} | nose base {np.median(d[:, REG['nose base'][0]]):.2f} | features {np.median(d[:, REG['features'][0]]):.2f} | rest {np.median(d[:, REG['rest'][0]]):.2f}")
