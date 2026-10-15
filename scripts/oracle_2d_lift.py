"""Is the gross-error problem the LANDMARKER or the LIFTER?

MEASURED, 40 ears x 25 views x 55 points, paired error to the GT landmark:

    arm              median      p90     >5mm    >10mm    >20mm
    oracle_raw         0.34     3.09    3.56%    0.60%    0.23%
    oracle_snap        0.36     3.27    4.21%    0.25%    0.00%
    det_snap           3.73    10.86   33.88%   11.06%    6.15%
    flip_snap          3.48     8.82   29.14%    8.13%    3.61%

The lifter is not the problem. Handed the right pixels it reproduces the
ground truth to 0.36 mm and NEVER misses by 20 mm, over 55,000 points, even
though 15.2% of those points are occluded by more than 2 mm of nearer
surface -- the cliff snap covers that case. Every 20 mm error in the real
pipeline is the 2D model putting the point somewhere else.

This is the oracle-2D ablation that the pose-lifting literature uses for the
same question (Martinez et al. 2017 report 37.1 mm on ground-truth 2D against
62.9 mm on detected). The gap is far wider here because this lifter READS
depth from a Z-buffer instead of inferring it, so it has no depth ambiguity
to lose to -- only occlusion and sampling across a discontinuity.

Note the snap trades small errors for the tail: it costs 3.56% -> 4.21% past
5 mm and buys 0.23% -> 0.00% past 20 mm.

THE TWO det_* ROWS PREDATE the lighting fix in config.py and will now read
better; the two oracle_* rows do not use the 2D model at all and are unchanged
by it. The conclusion is unaffected -- the gap was two orders of magnitude and
the fix closes well under one.

This script takes ~40 minutes and has no argparse; it runs on import.

The standard decomposition: run the lift on ORACLE 2D -- the ground-truth 3D
landmarks projected into the same image -- so the 2D model is out of the loop
entirely. Whatever error survives is the lifter's own: self-occlusion, grazing
surfaces, depth quantisation. Whatever the detected-2D run has on top of that
is the landmarker's.

The escape metric (distance to the fitted ear) is NEARLY TAUTOLOGICAL for this
question -- if the pixel shows the ear, the depth sampled there is the ear's,
so the point lands on the ear by construction. So this scores PAIRED error:
distance from the lifted point to the GT landmark it is supposed to be.
"""
import sys, pathlib, tempfile, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from escape_audit import load_ear, ear_frame, landmark_view
from ear3d.config import DEFAULTS
from ear3d.camera import orbit_extrinsic
from ear3d.frames import in_frame
from ear3d.render import render
from ear3d.backproject import backproject, backproject_snapped
from ear3d.twod import make_pipeline

cfg = dict(DEFAULTS)
pipe, ck = make_pipeline(cfg)
WRAP = ROOT.parent / "wrap_work"
ears = sorted(f.stem for f in (WRAP / "corpus/canon2scan").glob("pp*.npz"))[:40]
ANG = [(y, p) for y in (-30, -15, 0, 15, 30) for p in (-30, -15, 0, 15, 30)]

acc = {k: [] for k in ("oracle_raw", "oracle_snap", "det_snap", "flip_snap")}
vis = []
for n, name in enumerate(ears):
    try:
        head, ear, lm = load_ear(name, "hutubs", cfg)
    except Exception as e:
        print(f"{name}: {e}"); continue
    R, C, scale = ear_frame(head, lm, cfg)
    g_head = in_frame(head, R, C, scale)
    G = ((lm - C) @ R.T) / scale          # GT landmarks in the render's frame
    for yaw, pitch in ANG:
        img, depth, cam = render(g_head, cfg, E=orbit_extrinsic(yaw, pitch, cfg))
        uv_gt = cam.project(G)            # ORACLE 2D: the GT points' own pixels
        for tag, fn in (("oracle_raw",  lambda u: backproject(depth, u, cam)[0]),
                        ("oracle_snap", lambda u: backproject_snapped(depth, u, cam, cfg)[0])):
            P = fn(uv_gt)
            acc[tag].append(np.linalg.norm(P - G, axis=1) * scale)
        # how many GT points does the depth buffer even SEE?  compare the depth
        # it holds at the GT pixel against the GT point's own depth
        zc = (G @ cam.R.T + cam.t)[:, 2]
        ij = np.round(uv_gt[:, ::-1]).astype(int)
        okp = ((ij[:, 0] >= 0) & (ij[:, 0] < depth.shape[0]) &
               (ij[:, 1] >= 0) & (ij[:, 1] < depth.shape[1]))
        dz = np.full(55, np.nan)
        dz[okp] = depth[ij[okp, 0], ij[okp, 1]] - zc[okp]
        vis.append(dz * scale)
        for tag, mode in (("det_snap", "none"), ("flip_snap", "flip")):
            best = landmark_view(img, cfg, pipe, mode, depth.shape)
            if best is None:
                acc[tag].append(np.full(55, np.nan)); continue
            uv = np.asarray(best["landmarks"], float)
            P = backproject_snapped(depth, uv, cam, cfg)[0]
            acc[tag].append(np.linalg.norm(P - G, axis=1) * scale)
    print(f"[{n+1}/{len(ears)}] {name}", flush=True)

print("\nPAIRED error to the GT landmark, mm -- 40 ears x 25 views x 55 points")
print(f"{'arm':14s} {'median':>8s} {'p90':>8s} {'>5mm':>8s} {'>10mm':>8s} {'>20mm':>8s}")
for k in ("oracle_raw", "oracle_snap", "det_snap", "flip_snap"):
    d = np.concatenate(acc[k]); d = d[np.isfinite(d)]
    print(f"{k:14s} {np.median(d):8.2f} {np.percentile(d,90):8.2f} "
          f"{100*(d>5).mean():7.2f}% {100*(d>10).mean():7.2f}% {100*(d>20).mean():7.2f}%")
V = np.concatenate(vis); V = V[np.isfinite(V)]
print(f"\ndepth at the GT pixel minus the GT point's own depth (mm): "
      f"median {np.median(V):+.2f}, p10 {np.percentile(V,10):+.2f}, "
      f"p90 {np.percentile(V,90):+.2f}")
print(f"  GT landmark occluded by more than 2 mm of nearer surface: {100*(V<-2).mean():.2f}%")
np.savez(pathlib.Path(tempfile.gettempdir()) / "oracle2d.npz", **{k: np.concatenate(v) for k, v in acc.items()},
         vis=V)
