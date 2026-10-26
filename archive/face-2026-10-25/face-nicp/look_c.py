import sys, numpy as np, cv2
from corners import *
from ear3d.face import show
k = "r15 l30 b0.5"; r, l, b = ARMS[k]
for h in ("pp73", "pp47", "pp44"):
    j = H.index(h); tr = [i for i in range(len(H)) if i != j]; z = np.load(RUN / f"{h}.npz"); G = np.load(RUN / f"{h}_gtd.npz")["G"]
    off = np.nan_to_num(np.nanmean(OFF[tr], 0)); w = view_weights(np.nanmedian(RAY[tr], 0)); cams = cameras(z["E"])
    P, _, _ = triangulate(z["UV"][:, :468], cams, off, FACE["ransac_mm"] / s); T, _ = seat(P * s, z["V"], z["F"])
    X = fit_rays(z["UV"], cams, off, z["V"], z["F"], T, FACE, LAP, w)
    Y = fit_dense(z["UV"], cams, off, z["V"], z["F"], X, dict(FACE, dense_lambda=l, dense_beta=b), T_[k], w)
    rows = []
    for lab, Q in (("current", X), ("corner dense fit", Y)):
        show(z["V"], z["F"], [(G, (0, 255, 0)), (Q, (0, 0, 255))], "t.jpg", yaws=(-40, 0, 40), centre=(G[61] + G[291]) / 2 - np.array([0, 0, 50]), scale=35)
        im = cv2.imread("t.jpg"); cv2.putText(im, f"{h} {lab} corners {np.median(np.linalg.norm(Q - G, axis=1)[CORN]):.2f} mm", (12, 34), 0, 0.85, (255, 255, 255), 2); rows.append(im)
    cv2.imwrite(f"corner_{h}.jpg", cv2.resize(np.vstack(rows), None, fx=0.7, fy=0.7))
