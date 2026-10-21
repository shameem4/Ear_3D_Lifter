"""Orientation-stage review, side views.

Usage: review_head_sides.py OUT.png ROT_SEED WORKER N_WORKERS
Writes one tile per head and side next to OUT; run N_WORKERS copies.

Normalised head (164->9 frame) seen from +x (subject's left ear) and -x
(right ear): the head turned -90 / +90 deg about y. Whole head, centred on
the head centroid, crosshair at frame centre. Output only."""
import sys, numpy as np
sys.path.insert(0, "/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_3D_Lifter"); sys.path.insert(0, "/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_3D_Lifter/scripts")
from escape_audit import DEFAULTS, WRAP, load_ear, light_mirror_for
from ear3d.frames import frame_from, in_frame, normalize_head
from ear3d.render import render, side_light
from PIL import Image, ImageDraw
import open3d as o3d
out, rot_seed, WK, NW = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
cfg = dict(DEFAULTS)
names = sorted(f.stem for f in (WRAP / __import__("ear3d.corpus", fromlist=["get"]).get("canon2scan")).glob("*.npz") if str(np.load(f, allow_pickle=True)["dataset"]) == "hutubs")
heads = sorted({n.rsplit("_", 1)[0] for n in names})
ALLS = sorted(np.random.default_rng(0).choice(heads, 20, replace=False).tolist())
def random_rotation(seed):
    q = np.random.default_rng(seed).normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
for hd in ALLS[WK::NW]:
    i = ALLS.index(hd)
    c = dict(cfg, light_mirror=light_mirror_for(f"{hd}_left", "frame", "hutubs"))
    head, _, _ = load_ear(f"{hd}_left", "hutubs", c)
    Rr = random_rotation((rot_seed, i))
    head.vertices = o3d.utility.Vector3dVector(np.asarray(head.vertices) @ Rr.T); head.compute_vertex_normals()
    nh = normalize_head(head, c)
    if nh is None: print(hd, "NO FACE"); continue
    mesh_c = nh[0]; V = np.asarray(mesh_c.vertices); C = np.zeros(3)
    for tag, sd in (("L", np.array([1.0, 0, 0])), ("R", np.array([-1.0, 0, 0]))):
        R = frame_from(sd, np.array([0, 1.0, 0]), front=sd)
        sc = float(np.abs((V - C) @ R.T).max())
        img, _, _ = render(in_frame(mesh_c, R, C, sc), side_light(c, tag))
        im = Image.fromarray(img); d = ImageDraw.Draw(im); W, H = im.size
        d.line([(W//2, 0), (W//2, H)], fill=(0, 255, 0)); d.line([(0, H//2), (W, H//2)], fill=(0, 255, 0))
        d.text((4, 4), f"{hd} {'LEFT ear (+x)' if tag == 'L' else 'RIGHT ear (-x)'}", fill=(255, 255, 0))
        im.save(out.replace(".png", f"_{hd}_{tag}.png"))
    print(hd, "done", flush=True)
