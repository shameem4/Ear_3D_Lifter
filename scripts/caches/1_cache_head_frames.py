"""Run the orientation stage once and cache its output, so later stages
(ear detection, zoom, chop) can be iterated without re-finding the face.

Usage: cache_head_frames.py OUTDIR ROT_SEED N_HEADS WORKER N_WORKERS [SAMPLE_SEED [TONE SKIN_NOISE]]

TONE and SKIN_NOISE (0/1) override config's tone and skin_noise.

SAMPLE_SEED 0 (default) is the original sample; any other seed draws
N_HEADS new heads that exclude the seed-0 sample, for held-out checks.

Per head writes OUTDIR/<head>.npz: the normalised mesh (V, F, C colours),
R_head, centre, nose (all in the normalised frame) and the random rotation
Rr that was applied. Same head sample and rotations as review_head_sides.
Lives under runs/ (gitignored): it holds scan geometry.
"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ear3d.config import DEFAULTS
from ear3d.corpus import WRAP
from ear3d.data import load_ear, light_mirror_for
from ear3d.frames import normalize_head
import open3d as o3d

out, rot_seed, n_heads, WK, NW = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
out.mkdir(parents=True, exist_ok=True)
from ear3d import corpus as _corpus
names = sorted(f.stem for f in (WRAP / _corpus.get("canon2scan")).glob("*.npz") if str(np.load(f, allow_pickle=True)["dataset"]) == "hutubs")
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
    if len(sys.argv) > 8:
        c.update(tone=int(sys.argv[7]), skin_noise=bool(int(sys.argv[8])))
    head, _, _ = load_ear(f"{hd}_left", "hutubs", c)
    V0 = np.asarray(head.vertices).copy()
    for k in range(4):   # same plain search on a fresh random rotation; k=0 is the original draw,
        # so a head that passes first time caches exactly as before (pp95 needed one more draw)
        Rr = random_rotation((rot_seed, i) if k == 0 else (rot_seed, i, k))
        head.vertices = o3d.utility.Vector3dVector(V0 @ Rr.T); head.compute_vertex_normals()
        nh = normalize_head(head, c)
        if nh is not None: break
        print(hd, f"no face on rotation draw {k}", flush=True)
    if nh is None: print(hd, "NO FACE", flush=True); continue
    m, R_head, centre, nose = nh
    np.savez(out / f"{hd}.npz", V=np.asarray(m.vertices), F=np.asarray(m.triangles), C=np.asarray(m.vertex_colors),
             R_head=R_head, centre=centre, nose=nose, Rr=Rr, light_mirror=c["light_mirror"],
             corpus=_corpus.CURRENT)
    print(hd, "cached", flush=True)
