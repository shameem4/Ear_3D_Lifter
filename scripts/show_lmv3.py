"""ear_lmv_v3 drawn on ITS OWN mesh, with both barycentric recipes checked.

The recipe carries two: insert_tri/insert_bary index the ORIGINAL 2296
faces of ear_mean_v4, triangle/barycentric index the NEW 2406 faces of
the lmv mesh. Mixing them up puts a landmark 55 mm away, which is how
the first check read.
"""
import sys, pathlib, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from ear3d.wrap import load_mesh
W = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work")
OUT = pathlib.Path(sys.argv[1])
Vm, Fm = load_mesh(W/"templates/ear_mean_v4.obj")
Vl, Fl = load_mesh(W/"templates/ear_lmv_v3.obj")
z = np.load(W/"templates/ear_lmv_v3.npz", allow_pickle=True)
k = int(z["n_landmarks"]); L = Vl[:k]
a = np.einsum("ij,ijk->ik", z["insert_bary"], Vm[np.asarray(z["F_orig"])[z["insert_tri"]]])
b = np.einsum("ij,ijk->ik", z["barycentric"], Vl[Fl[z["triangle"]]])
print(f"landmarks vs insert_* on the ORIGINAL faces : max {np.abs(L-a).max():.4f} mm")
print(f"landmarks vs triangle/barycentric on the NEW: max {np.abs(L-b).max():.2e} mm")

c = Vl.mean(0); _, _, RR = np.linalg.svd(Vl - c, full_matrices=False)
P = (Vl - c) @ RR.T; Q = P[:k]
fig = plt.figure(figsize=(13.5, 4.8), dpi=145)
for j, (az, el, nm) in enumerate(((-90, 90, "lateral"), (-90, 35, "oblique"),
                                  (-90, -90, "medial"))):
    ax = fig.add_subplot(1, 3, j+1, projection="3d")
    tri = P[Fl]
    n = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True); n = n/np.where(ln == 0, 1, ln)
    sh = np.clip(np.abs(n @ np.array([.25, .25, 1.])/1.06), .22, 1.)
    col = np.tile(np.array([.82, .80, .77]), (len(Fl), 1))*sh[:, None]
    o = tri[:, :, 2].mean(1).argsort()
    ax.add_collection3d(Poly3DCollection(tri[o], facecolors=col[o],
                        edgecolors=(0, 0, 0, .14), linewidths=.13))
    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=22, c="#d1332e", depthshade=False,
               edgecolors="k", linewidths=.3)
    r = float(np.ptp(P, axis=0).max())*0.56; m = P.mean(0)
    ax.set_xlim(m[0]-r, m[0]+r); ax.set_ylim(m[1]-r, m[1]+r); ax.set_zlim(m[2]-r, m[2]+r)
    ax.set_axis_off(); ax.view_init(elev=el, azim=az)
    ax.set_title(f"ear_lmv_v3, {nm}", fontsize=9)
    try: ax.set_box_aspect((1, 1, 1), zoom=1.5)
    except TypeError: pass
fig.suptitle("The 55 landmark vertices of ear_lmv_v3, on ear_lmv_v3", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(OUT/"lmv_v3.png")
print("wrote", OUT/"lmv_v3.png")
