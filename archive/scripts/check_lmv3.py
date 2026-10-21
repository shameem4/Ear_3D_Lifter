"""Do ear_lmv_v3's 55 landmark vertices actually lie on ear_mean_v4?

The first render drew them against the PCA mean, which is a different
frame and scale, so they floated. This checks the artifact itself:
distance from each landmark vertex to the nearest point of the surface
it was built from, and that the remaining 1188 carry that surface
unchanged.
"""
import sys, pathlib, numpy as np, trimesh
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from ear3d.wrap import load_mesh
W = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work")
Vm, Fm = load_mesh(W/"templates/ear_mean_v4.obj")
Vl, Fl = load_mesh(W/"templates/ear_lmv_v3.obj")
z = np.load(W/"templates/ear_lmv_v3.npz", allow_pickle=True)
k = int(z["n_landmarks"]); perm = z["perm"]
print(f"lmv_v3: {len(Vl)} v, {len(Fl)} f;  mean_v4: {len(Vm)} v, {len(Fm)} f")
print(f"landmarks = first {k} vertices;  perm[:{k}] all -1: "
      f"{bool((perm[:k] == -1).all())}")
tail = Vl[k:]
print(f"the other {len(tail)} vertices equal ear_mean_v4 exactly: "
      f"{np.allclose(tail, Vm, atol=1e-9)}  (max diff "
      f"{np.abs(tail-Vm).max():.2e} mm)")
d = trimesh.proximity.closest_point(
    trimesh.Trimesh(vertices=Vm, faces=Fm, process=False), Vl[:k])[1]
print(f"landmark distance to the ear_mean_v4 SURFACE: max {d.max():.2e} mm, "
      f"mean {d.mean():.2e} mm")
L0 = np.einsum("ij,ijk->ik", z["barycentric"], Vm[Fm[z["triangle"]]])
print(f"landmarks vs their barycentric placement on mean_v4: max "
      f"{np.abs(Vl[:k]-L0).max():.4f} mm   (7 were snapped to edges, "
      f"budget 0.10)")
