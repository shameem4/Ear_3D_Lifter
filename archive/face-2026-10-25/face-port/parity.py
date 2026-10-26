import sys, numpy as np, open3d as o3d
which = sys.argv[1]
m = o3d.io.read_triangle_mesh("/mnt/14BE47C2BE479ADE/Code/ear_stuff/clean_3d_data/3D head meshes/pp1_3DheadMesh.ply")
m.vertices = o3d.utility.Vector3dVector(np.asarray(m.vertices) * 1000.0)
if which == "old":
    sys.path.insert(0, "/mnt/14BE47C2BE479ADE/Code/landmarking_stuff/Face_3D_Lifter")
    from face3d.lift import Lifter; out, _ = Lifter().run(m)
else:
    sys.path.insert(0, "/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_3D_Lifter")
    from ear3d.face import FaceLifter; out, _ = FaceLifter().run(m)
np.save(f"P_{which}.npy", out["P"]); print(which, out["n_views"], np.nanmedian(out["residual_mm"]))
