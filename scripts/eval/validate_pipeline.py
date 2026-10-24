"""Score the end-to-end pipeline on heads it has never seen.

Usage: validate_pipeline.py OUT_DIR SAMPLE_SEED N_HEADS WORKER N_WORKERS
Heads: SAMPLE_SEED 0 / 1 are the two cached 20-head sets; any other seed
draws from the heads in neither (16 of 56 HUTUBS heads), given a random 3-axis rotation, and handed to
the pipeline UNSKINNED -- exactly what a user would pass in. Output per head:
OUT_DIR/<head>.npz with pred_L/R, gt_L/R under config gt_lm3d and its
alternates, all in the rotated input frame.

EAR3D_JEWELLERY=<seed> adds synthetic jewellery (ear3d.jewellery) to both
ears of every head before it is turned, placed from the reference landmarks;
the landmarks are unchanged. Held-out heads are never rendered for training,
so this is a test condition only.
"""
import pathlib
import sys

import numpy as np
import open3d as o3d

R0 = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R0))
from ear3d import corpus, gt
from ear3d.config import DEFAULTS
from ear3d.corpus import WRAP
from ear3d.data import load_ear
from ear3d.jewellery import add_jewellery
from ear3d.pipeline import Pipeline

out, ss, n, WK, NW = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
out.mkdir(parents=True, exist_ok=True)
names = sorted(f.stem for f in (WRAP / __import__("ear3d.corpus", fromlist=["get"]).get("canon2scan")).glob("*.npz") if str(np.load(f, allow_pickle=True)["dataset"]) == "hutubs")
heads = sorted({x.rsplit("_", 1)[0] for x in names})
seed0 = sorted(np.random.default_rng(0).choice(heads, 20, replace=False).tolist())
seed1 = sorted(np.random.default_rng(1).choice(sorted(set(heads) - set(seed0)), 20, replace=False).tolist())
if ss == 0:   sample = seed0
elif ss == 1: sample = seed1
else:         # heads used by NEITHER cached set: the held-out pool (16 of 56 HUTUBS heads)
    pool = sorted(set(heads) - set(seed0) - set(seed1))
    sample = sorted(np.random.default_rng(ss).choice(pool, min(n, len(pool)), replace=False).tolist())

def rot(seed):
    q = np.random.default_rng(seed).normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

refs = {DEFAULTS["gt_lm3d"]: (gt.LM, gt.NAMES), **gt.ALSO}
import os
pipe = Pipeline(cfg={"ear_detector": os.environ.get("EAR_DETECTOR", DEFAULTS["ear_detector"])})
for i, hd in enumerate(sample[WK::NW]):
    dst = out / f"{hd}.npz"
    if dst.exists(): continue
    head, _, _ = load_ear(f"{hd}_left", "hutubs", dict(DEFAULTS, skin=False))
    jew = {}
    if os.environ.get("EAR3D_JEWELLERY"):
        rj = np.random.default_rng([int(os.environ["EAR3D_JEWELLERY"]), 1000 + sample.index(hd)])
        for sd in "LR":
            nm = f"{hd}_{'left' if sd == 'L' else 'right'}"; T = gt.inv(f"{hd}_left", corpus.CURRENT) @ np.linalg.inv(gt.inv_gt(nm))
            P = (np.c_[gt.LM["landmarks"][gt.NAMES.index(nm)][:55], np.ones(55)] @ T.T)[:, :3]
            head, jew[sd] = add_jewellery(head, P, np.asarray(head.vertices).mean(0), rj)
    Rr = rot((ss, 1000 + sample.index(hd)))
    head.vertices = o3d.utility.Vector3dVector(np.asarray(head.vertices) @ Rr.T); head.vertex_colors = o3d.utility.Vector3dVector()
    res = pipe.run(head)
    rec = {"Rr": Rr, "ckpt": pipe.ckpt, "jewellery": str(jew)}
    for sd in "LR":
        rec[f"pred_{sd}"] = res[sd] if res and res[sd] is not None else np.full((55, 3), np.nan)
        nm = f"{hd}_{'left' if sd == 'L' else 'right'}"; T = gt.inv(f"{hd}_left", corpus.CURRENT) @ np.linalg.inv(gt.inv_gt(nm))   # load_ear built the head in CURRENT
        for fn, (lmz, nms) in refs.items():
            P = lmz["landmarks"][nms.index(nm)][:55]; P = (np.c_[P, np.ones(55)] @ T.T)[:, :3]
            rec[f"gt_{sd}_{fn.split('_canonical')[0]}"] = P @ Rr.T
    np.savez(dst, **rec)
    k0 = DEFAULTS["gt_lm3d"].split("_canonical")[0]
    e = {sd: np.nanmedian(np.linalg.norm(rec[f"pred_{sd}"] - rec[f"gt_{sd}_{k0}"], axis=1)) for sd in "LR"}
    print(hd, "face" if res else "NO FACE", f"L {e['L']:.2f} R {e['R']:.2f} mm vs {k0}", flush=True)
