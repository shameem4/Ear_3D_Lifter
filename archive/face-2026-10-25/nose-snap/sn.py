import sys, glob, os, numpy as np
sys.path.insert(0, "/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_3D_Lifter")
RUN = "/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_3D_Lifter/runs/face/v1"

def subnasale(V, X, band=1.5):
    """deepest midline point behind the chord pronasale -> labrale superius (normalised frame: +z out, +y up)"""
    near = np.flatnonzero(np.linalg.norm(V - X[1], axis=1) < 15); tip = V[near[np.argmax(V[near, 2])]]
    ls = X[0]
    sel = (np.abs(V[:, 0] - tip[0]) < band) & (V[:, 1] < tip[1]) & (V[:, 1] > ls[1]) & (V[:, 2] > ls[2] - 15)
    Q = V[sel]
    if len(Q) < 5: return None, tip
    a, b = tip[1:], ls[1:]; t = b - a; nrm = np.array([t[1], -t[0]]) / np.linalg.norm(t)   # in (y, z)
    if nrm[1] < 0: nrm = -nrm                                      # +out of the face
    d = (Q[:, 1:] - a) @ nrm
    return Q[np.argmin(d)], tip

if __name__ == "__main__":
    rows = []
    for f in sorted(glob.glob(RUN + "/*_gtd.npz")):
        h = os.path.basename(f)[:-8]; z = np.load(f"{RUN}/{h}.npz"); G = np.load(f)["G"]; X = z["P_norm"]
        S, tip = subnasale(z["V"], X)
        rows.append((h, np.linalg.norm(S - G[2]), np.linalg.norm(X[2] - G[2]), (S - G[2])[1], (X[2] - G[2])[1], np.linalg.norm(tip - G[1])))
    r = np.array([x[1:] for x in rows])
    print(f"lm 2 vs GT: geometric median {np.median(r[:,0]):.2f} p90 {np.percentile(r[:,0],90):.2f} max {r[:,0].max():.2f} | lift median {np.median(r[:,1]):.2f} p90 {np.percentile(r[:,1],90):.2f} max {r[:,1].max():.2f}")
    print(f"geometric better on {(r[:,0] < r[:,1]).sum()}/{len(r)}; nose tip geom vs GT[1] median {np.median(r[:,4]):.2f}")
    for x in sorted(rows, key=lambda x: -x[2])[:6]: print(x[0], "geom %.2f (dy %+.1f) | lift %.2f (dy %+.1f)" % (x[1], x[3], x[2], x[4]))
    for x in sorted(rows, key=lambda x: -x[1])[:4]: print("worst geom", x[0], "%.2f" % x[1])
