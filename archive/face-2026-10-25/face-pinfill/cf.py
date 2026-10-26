"""Corner fix: pin every landmark except the lip-corner region exactly; refill that region with the canonical
membrane (Laplacian) from its pinned surroundings; seat along the film normal. Leave one head out."""
import sys, os, glob, numpy as np
from concurrent.futures import ProcessPoolExecutor
from pf import R0, W0, REG, pinfill, Vc
def region(r):
    return (np.linalg.norm(Vc - Vc[61], axis=1) < r) | (np.linalg.norm(Vc - Vc[291], axis=1) < r)
def one(h):
    z = np.load(f"{R0}/runs/face/v1/{h}.npz"); G = np.load(f"{R0}/runs/face/v1/{h}_gtd.npz")["G"]
    X = np.load(f"{W0}/out/{h}.npz")["current"]; arms = {"current": X}
    for r in (6, 8, 12):
        free = region(r); w = np.where(free, 0.0, 1e6)
        for nm in (False, True):
            arms[f"free<{r}mm{' normal' if nm else ''}"] = pinfill(X, z["V"], z["F"], w, lam=1.0, iters=3, normal=nm)
    out = {}
    m, E = REG["lip corners"]; g = np.linalg.norm(G[E[:, 0]] - G[E[:, 1]], axis=1)
    for k, Y in arms.items():
        x = np.linalg.norm(Y[E[:, 0]] - Y[E[:, 1]], axis=1); lr = np.abs(np.log(x / g))
        out[k] = (np.linalg.norm(Y - G, axis=1), (x / g < 0.5)[g > 1], lr, np.abs(Y - X).max())
    return h, out
if __name__ == "__main__":
    H = [os.path.basename(f)[:-4] for f in sorted(glob.glob(f"{W0}/out/*.npz"))]
    with ProcessPoolExecutor(8) as ex: R = list(ex.map(one, H))
    m = REG["lip corners"][0]; cur = np.array([np.median(r[1]["current"][0][m]) for r in R])
    print(f"{'':20s} corners err med/p90 | collapse | spacing p90 | all med | heads better (corners) | max move outside? ")
    for k in R[0][1]:
        d = np.array([r[1][k][0] for r in R]); c = np.concatenate([r[1][k][1] for r in R]); lr = np.concatenate([r[1][k][2] for r in R])
        ph = np.median(d[:, m], 1)
        print(f"{k:20s} {np.median(d[:, m]):.2f}/{np.percentile(d[:, m], 90):.2f} | {c.mean():6.1%} | {np.percentile(lr, 90):.3f} | {np.median(d):.2f} | {np.sum(ph < cur - 1e-9)}/50 worst {np.max(ph - cur):+.2f}")
