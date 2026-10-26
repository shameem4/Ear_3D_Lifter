"""Wrap the canonical face onto each head with key points taken from the lift (not the user's hand points)."""
import sys, os, numpy as np
R0 = "/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_3D_Lifter"; sys.path[:0] = [R0, R0 + "/scripts/face"]; os.environ["FACE_RUN"] = "v1"
from score_loho import RUN, H, OFF, RAY, LAP, s
from ear3d.face import FACE, cameras, triangulate, seat, fit_rays, view_weights, fit_dense, dense_template
from ear3d.face_gt import canonical, wrap_template, bary
TPL = dense_template(); Vc, Fc = canonical()
B = [33, 133, 362, 263, 1, 168, 2, 152, 129, 358, 61, 291]          # eye corners, tip, nasion, subnasale, chin, alar, mouth corners
KEYS = {"B": B,
        "C": B + [10, 151, 67, 297,                                    # forehead: top, centre, left/right
                  50, 280, 205, 425, 123, 352,                         # cheeks: upper, mid, outer
                  94, 97, 326]}                                        # nose base (dense-fit fixed)
OUT = "/var/tmp/claude/ear3d/2026-10-25-face-wrapkeys/out"
def lift(h):
    j = H.index(h); tr = [i for i in range(len(H)) if i != j]; z = np.load(RUN / f"{h}.npz")
    off = np.nan_to_num(np.nanmean(OFF[tr], 0)); w = view_weights(np.nanmedian(RAY[tr], 0)); cams = cameras(z["E"])
    P, _, _ = triangulate(z["UV"][:, :468], cams, off, FACE["ransac_mm"] / s); T, _ = seat(P * s, z["V"], z["F"])
    X = fit_rays(z["UV"], cams, off, z["V"], z["F"], T, FACE, LAP, w)
    return z, fit_dense(z["UV"], cams, off, z["V"], z["F"], X, FACE, TPL, w)
for h in sys.argv[1:]:
    z, X = lift(h); res = {"current": X}
    for name, K in KEYS.items():
        K = np.array(K); tid, bb = bary(Vc, Fc, Vc[K]); sid, sb = bary(z["V"], z["F"], X[K])
        ctrl = ([[int(t), float(b[0]), float(b[1])] for t, b in zip(tid, bb)], [[int(t), float(b[0]), float(b[1])] for t, b in zip(sid, sb)])
        try:
            Vf, _ = wrap_template(Vc, Fc, Vc[K], z["V"], z["F"], X[K], f"{OUT}/wrapd/{h}_{name}", name=h, ctrl=ctrl)
            res[f"wrap {name}"] = seat(Vf[:468], z["V"], z["F"])[0]
        except Exception as e:
            print(h, name, "FAILED", e, flush=True)
    np.savez(f"{OUT}/{h}.npz", **res); print(h, "done", flush=True)
