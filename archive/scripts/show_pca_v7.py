"""Look at the new basis before trusting it.

Two questions a variance table cannot answer:
  - do the leading modes look like SHAPE (size, helix roll, lobe, concha
    depth), or like correspondence jitter sprayed over the surface?
  - do the 55 landmarks of ear_lmv_v3 actually sit on the anatomy?

Compactness against v6 is reported too, because cleaner correspondence
should need fewer modes for the same variance.
"""
import sys, pathlib, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from ear3d.wrap import load_mesh
W = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work")
OUT = pathlib.Path(sys.argv[1])

print(f"{'basis':26s}{'n':>5}{'90%':>6}{'95%':>6}{'99%':>6}")
for nm in ("ear_pca_v6_ccby_mm", "ear_pca_v7_ccby_mm",
           "ear_pca_v6_all_mm", "ear_pca_v7_all_mm"):
    p = W/"templates"/f"{nm}.npz"
    if not p.exists():
        print(f"{nm:26s}  (absent)"); continue
    z = np.load(p, allow_pickle=True)
    v = np.asarray(z["variance"], float); c = np.cumsum(v)/v.sum()
    print(f"{nm:26s}{int(z['n_ears']):>5}"
          + "".join(f"{int(np.searchsorted(c,q))+1:>6}" for q in (.90, .95, .99)))

z = np.load(W/"templates/ear_pca_v7_ccby_mm.npz", allow_pickle=True)
mean, comp, var = z["mean"], z["components"], np.asarray(z["variance"], float)
F = np.asarray(z["faces"])
lmv = np.load(W/"templates/ear_lmv_v3.npz", allow_pickle=True)
L = load_mesh(W/"templates/ear_lmv_v3.obj")[0][:int(lmv["n_landmarks"])]


def frame(V, R=None):
    c = V.mean(0)
    if R is None:
        _, _, R = np.linalg.svd(V - c, full_matrices=False)
    return (V - c) @ R.T, R


_, RR = frame(mean)


def panel(ax, V, t_, pts=None):
    P = (V - mean.mean(0)) @ RR.T
    tri = P[F]
    n = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True); n = n/np.where(ln == 0, 1, ln)
    sh = np.clip(np.abs(n @ np.array([.25, .25, 1.])/1.06), .22, 1.)
    col = np.tile(np.array([.80, .78, .75]), (len(F), 1))*sh[:, None]
    o = tri[:, :, 2].mean(1).argsort()
    ax.add_collection3d(Poly3DCollection(tri[o], facecolors=col[o],
                        edgecolors=(0, 0, 0, .12), linewidths=.12))
    if pts is not None:
        Q = (pts - mean.mean(0)) @ RR.T
        ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=9, c="#d1332e", depthshade=False)
    r = float(np.ptp(P, axis=0).max())*0.58; c = P.mean(0)
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_axis_off(); ax.view_init(elev=90, azim=-90); ax.set_title(t_, fontsize=8.5)
    try: ax.set_box_aspect((1, 1, 1), zoom=1.5)
    except TypeError: pass


sd = np.sqrt(var)
fig = plt.figure(figsize=(13.5, 10.6), dpi=130)
for k in range(4):
    C = comp[k].reshape(-1, 3)
    for j, s in enumerate((-2, 0, +2)):
        panel(fig.add_subplot(4, 4, 4*k+j+1, projection="3d"),
              mean + s*sd[k]*C,
              f"mode {k+1}  {s:+d} sd" if s else f"mode {k+1}  mean"
              f"  ({100*var[k]/var.sum():.1f}% var)")
    ax = fig.add_subplot(4, 4, 4*k+4, projection="3d")
    d = np.linalg.norm(2*sd[k]*C, axis=1)
    P = (mean - mean.mean(0)) @ RR.T; tri = P[F]
    fv = d[F].mean(1)
    cm = plt.cm.inferno(np.clip(fv/max(fv.max(), 1e-9), 0, 1))[:, :3]
    o = tri[:, :, 2].mean(1).argsort()
    ax.add_collection3d(Poly3DCollection(tri[o], facecolors=cm[o],
                        edgecolors=(0, 0, 0, .10), linewidths=.1))
    r = float(np.ptp(P, axis=0).max())*0.58; c = P.mean(0)
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_axis_off(); ax.view_init(elev=90, azim=-90)
    ax.set_title(f"mode {k+1} displacement (max {d.max():.1f} mm)", fontsize=8.5)
    try: ax.set_box_aspect((1, 1, 1), zoom=1.5)
    except TypeError: pass
fig.suptitle("ear_pca_v7_ccby: leading modes at +/-2 sd, and where each moves the surface",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(OUT/"pca_v7_modes.png")

fig2 = plt.figure(figsize=(13.5, 4.6), dpi=140)
for j, (az, el, nm) in enumerate(((-90, 90, "lateral"), (-90, 40, "oblique"),
                                  (-90, -90, "medial"))):
    ax = fig2.add_subplot(1, 3, j+1, projection="3d")
    panel(ax, mean, f"ear_lmv_v3 landmarks, {nm}", pts=L)
    ax.view_init(elev=el, azim=az)
fig2.suptitle("The 55 landmarks of ear_lmv_v3 on the new mean", fontsize=12)
fig2.tight_layout(rect=(0, 0, 1, .92)); fig2.savefig(OUT/"lmv_v3.png")
print("\nwrote", OUT/"pca_v7_modes.png", "and", OUT/"lmv_v3.png")
