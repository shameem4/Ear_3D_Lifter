"""Run the orientation stage once and cache its output, so later stages
(ear detection, zoom, chop) can be iterated without re-finding the face.

Usage: cache_head_frames.py OUTDIR ROT_SEED N_HEADS WORKER N_WORKERS [SAMPLE_SEED]

SAMPLE_SEED 0 (default) is the original sample; any other seed draws
N_HEADS new heads that exclude the seed-0 sample, for held-out checks.

Per head writes OUTDIR/<head>.npz: the normalised mesh (V, F, C colours),
R_head, centre, nose (all in the normalised frame) and the random rotation
Rr that was applied. Same head sample and rotations as review_head_sides.
Lives under runs/ (gitignored): it holds scan geometry.
"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from escape_audit import DEFAULTS, WRAP, load_ear, light_mirror_for
from ear3d.frames import normalize_head
import open3d as o3d

out, rot_seed, n_heads, WK, NW = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
out.mkdir(parents=True, exist_ok=True)
names = sorted(f.stem for f in (WRAP / "corpus/canon2scan").glob("*.npz") if str(np.load(f, allow_pickle=True)["dataset"]) == "hutubs")
heads = sorted({n.rsplit("_", 1)[0] for n in names})
ss = int(sys.argv[6]) if len(sys.argv) > 6 else 0
ALLS = sorted(np.random.default_rng(0).choice(heads, n_heads, replace=False).tolist())
if ss:
    pool = sorted(set(heads) - set(ALLS))
    ALLS = sorted(np.random.default_rng(ss).choice(pool, n_heads, replace=False).tolist())

def random_rotation(seed):
    q = np.random.default_rng(seed).normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

for hd in ALLS[WK::NW]:
    if (out / f"{hd}.npz").exists(): continue
    i = ALLS.index(hd)
    c = dict(DEFAULTS, light_mirror=light_mirror_for(f"{hd}_left", "frame", "hutubs"))
    head, _, _ = load_ear(f"{hd}_left", "hutubs", c)
    Rr = random_rotation((rot_seed, i))
    head.vertices = o3d.utility.Vector3dVector(np.asarray(head.vertices) @ Rr.T); head.compute_vertex_normals()
    nh = normalize_head(head, c)
    if nh is None: print(hd, "NO FACE", flush=True); continue
    m, R_head, centre, nose = nh
    np.savez(out / f"{hd}.npz", V=np.asarray(m.vertices), F=np.asarray(m.triangles), C=np.asarray(m.vertex_colors),
             R_head=R_head, centre=centre, nose=nose, Rr=Rr, light_mirror=c["light_mirror"])
    print(hd, "cached", flush=True)
