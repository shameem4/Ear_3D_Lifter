"""Face pipeline on held-out HUTUBS heads, each in two random orientations: repeatability, plus renders."""
import sys, time, numpy as np, open3d as o3d
sys.path.insert(0, '.')
from ear3d.config import DEFAULTS
from ear3d.data import load_ear
from ear3d.face_pipeline import FacePipeline
def rot(seed):
    q = np.random.default_rng(seed).normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
fp = FacePipeline(); res = {}
for hd in sys.argv[1:]:
    head, _, _ = load_ear(f"{hd}_left", "hutubs", dict(DEFAULTS, skin=False)); V0 = np.asarray(head.vertices)
    Ps = []
    for k in (0, 1):
        Rr = rot((11, k, hash(hd) % 1000)); m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V0 @ Rr.T), head.triangles)
        t = time.time(); P, dbg = fp.run(m, return_debug=True); P = P @ Rr          # back to the unturned head
        Ps.append(P); print(f"{hd} turn {k}: {time.time()-t:.0f} s, FaceMesh found a face in {dbg['faces_found']}/25 views, "
                            f"views per point median {np.median(dbg['n_views']):.0f} min {dbg['n_views'].min()}, NaN {int(np.isnan(P[:,0]).sum())}", flush=True)
    d = np.linalg.norm(Ps[0] - Ps[1], axis=1)
    print(f"{hd} REPEAT median {np.nanmedian(d):.2f} mm p95 {np.nanpercentile(d,95):.2f} max {np.nanmax(d):.1f}", flush=True)
    res[hd] = dict(P=Ps, V=V0, F=np.asarray(head.triangles))
np.save(sys.argv[0].replace('try.py', 'try.npy'), res, allow_pickle=True)
