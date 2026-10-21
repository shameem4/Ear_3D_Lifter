"""Multiview ray collection on cached ear views (current pipeline conditions).

Usage: multiview_collect_rays.py EAR_TAG OUT LIGHT WORKER N_WORKERS [EAR_LIMIT] [SUN]
  LIGHT: head  -- sun fixed to the head (same world direction in every view)
         cam   -- sun fixed to the camera (rotates with the orbit; equals
                  'head' at the face-on view)
  SUN: optional "x,y,z" override of config multiview_light_dir (before side mirroring).
Per ear writes OUT/<ear>.npz: angles (V,2), uv (V,55,2), o/d (V,55,3) rays in
head mm, gt (55,3) head mm, p0 (55,3) face-on snapped back-projection, side.
"""
import sys, glob, pathlib, numpy as np
R0 = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(R0))
import open3d as o3d
from ear3d.config import DEFAULTS
from ear3d.skin import apply_skin
from ear3d.camera import orbit_extrinsic
from ear3d.ear_view import occupancy_scale
from ear3d.frames import in_frame
from ear3d.render import render
from ear3d.backproject import backproject_snapped
from ear3d.twod import LandmarkPredictor, landmarker_checkpoint
from ear3d.gt import gt_in_cache

tag, out, LIGHT, WK, NW = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
LIMIT = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] != "-" else None
SUN = np.array([float(x) for x in sys.argv[7].split(",")]) if len(sys.argv) > 7 else np.asarray(DEFAULTS["multiview_light_dir"], float)
out.mkdir(parents=True, exist_ok=True)
ANG = [(float(y), float(p)) for y in range(-30, 31, 15) for p in range(-30, 31, 15)]
CKPT = str(landmarker_checkpoint(DEFAULTS)); pred = LandmarkPredictor(CKPT, "cuda")
fs = sorted(glob.glob(str(R0 / f"runs/cache/{tag}_s*/*_[LR].npz")))
if LIMIT: fs = fs[::max(1, len(fs) // LIMIT)][:LIMIT]
skinned = {}
for f in fs[WK::NW]:
    stem = pathlib.Path(f).stem; dst = out / f"{stem}.npz"
    if dst.exists(): continue
    z = np.load(f); hp = str(z["head"]); head = np.load(hp); hd, side = stem.rsplit("_", 1)
    if hp not in skinned:
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(head["V"]), o3d.utility.Vector3iVector(head["F"]))
        m.compute_vertex_normals()
        apply_skin(m, seed=DEFAULTS["skin_seed"], tone=DEFAULTS["ear_tone"], ao_strength=DEFAULTS["ao_strength"],
                   n_rays=DEFAULTS["ao_rays"], noise=DEFAULTS["ear_skin_noise"])
        skinned = {hp: m}
    m = skinned[hp]
    R, c0, s = z["R"], z["centre"], float(z["scale"])
    d0 = SUN * ([-1.0, 1, 1] if side == "R" else 1.0)          # side_light, applied by hand
    if DEFAULTS.get("target_occupancy"):                       # as pipeline._multiview
        s = occupancy_scale(m, R, c0, s, DEFAULTS, dict(DEFAULTS, light_dir=tuple(d0), light_mirror=False,
                            light_frame="world"), pred, DEFAULTS["target_occupancy"])
    gf = in_frame(m, R, c0, s)
    E0 = orbit_extrinsic(0, 0, DEFAULTS)
    uv = np.full((len(ANG), 55, 2), np.nan); O = np.full((len(ANG), 55, 3), np.nan); D = O.copy(); p0 = None
    for i, (yw, pt) in enumerate(ANG):
        E = orbit_extrinsic(yw, pt, DEFAULTS)
        d = d0 if LIGHT == "head" else E[:3, :3].T @ E0[:3, :3] @ d0
        cfg = dict(DEFAULTS, light_dir=tuple(d), light_mirror=False, light_frame="world")
        img, depth, cam = render(gf, cfg, E=E)
        lm = pred.predict(img); uv[i] = lm
        o, dd = cam.rays(lm)
        O[i] = o * s @ R + c0; D[i] = dd @ R
        if (yw, pt) == (0.0, 0.0):
            P, hit, _ = backproject_snapped(depth, lm, cam, DEFAULTS)
            p0 = np.where(hit[:, None], P * s @ R + c0, np.nan)
    gt = gt_in_cache(hd, side, head)
    np.savez(dst, angles=np.array(ANG), uv=uv, o=O, d=D, gt=gt, p0=p0, side=side, head=hp, ckpt=CKPT)
    print(stem, "ok", flush=True)
