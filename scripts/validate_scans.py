"""Score the end-to-end pipeline on raw scans of a dataset (sonicom, hutubs).

Usage: validate_scans.py OUT_DIR DATASET N_SUBJECTS SEED WORKER N_WORKERS

Each ear's scan is the file its canon2scan entry was built against, loaded
raw (metres -> mm by extent), turned by a random rotation, handed to the
pipeline UNSKINNED. SONICOM ships one graded mesh per ear, so each ear is
its own pipeline run and only that side is scored. The reference landmarks
(config gt_lm3d and its alternates) go canonical -> scan through the inverse
of canon2scan's `inverse`, then through the same rotation.

Nothing here was used to build the pipeline: the view table and v9's spacing
come from HUTUBS only.
"""
import os
import pathlib
import sys

import numpy as np
import open3d as o3d

R0 = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R0), str(R0 / "scripts")]
from escape_audit import DEFAULTS, HUTUBS_TO_MM, WRAP
from ear3d.pipeline import Pipeline

out, ds, n, seed, WK, NW = (pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]),
                            int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
out.mkdir(parents=True, exist_ok=True)
refs = {}
for fn in (DEFAULTS["gt_lm3d"], *DEFAULTS.get("gt_lm3d_also", ())):
    z = np.load(WRAP / "templates" / fn, allow_pickle=True)
    refs[fn.split("_canonical")[0]] = (z["landmarks"], [str(x) for x in z["names"]])

from ear3d import corpus as _corpus
C2S = WRAP / _corpus.get("canon2scan")      # pairs with DEFAULTS["gt_lm3d"]
ears = sorted(f.stem for f in C2S.glob("*.npz")
              if str(np.load(f, allow_pickle=True)["dataset"]) == ds)
subjects = sorted({e.rsplit("_", 1)[0] for e in ears})
sample = sorted(np.random.default_rng(seed).choice(subjects, min(n, len(subjects)), replace=False).tolist())


def rot(s):
    q = np.random.default_rng(s).normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


pipe = Pipeline(cfg={"ear_detector": os.environ.get("EAR_DETECTOR", DEFAULTS["ear_detector"])})
cache = {}
for sub in sample[WK::NW]:
    for side, word in (("L", "left"), ("R", "right")):
        nm = f"{sub}_{word}"; dst = out / f"{nm}.npz"
        if dst.exists() or nm not in ears:
            continue
        z = np.load(C2S / f"{nm}.npz", allow_pickle=True)
        scan = str(z["scan"])
        if scan not in cache:                      # HUTUBS: both ears share one head
            cache.clear()
            m = o3d.io.read_triangle_mesh(scan)
            V = np.asarray(m.vertices); V = V * (HUTUBS_TO_MM if np.ptp(V, 0).max() < 10 else 1.0)
            Rr = rot((seed, sample.index(sub), len(scan)))
            m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V @ Rr.T), m.triangles)
            cache[scan] = (Rr, pipe.run(m))
        Rr, res = cache[scan]
        rec = {"Rr": Rr, "scan": scan, "face": res is not None}
        rec["pred"] = res[side] if res and res[side] is not None else np.full((55, 3), np.nan)
        T = np.linalg.inv(z["inverse"])
        for k, (L, names) in refs.items():
            P = L[names.index(nm)][:55]
            rec[f"gt_{k}"] = (np.c_[P, np.ones(55)] @ T.T)[:, :3] @ Rr.T
        np.savez(dst, **rec)
        k0 = DEFAULTS["gt_lm3d"].split("_canonical")[0]
        e = np.nanmedian(np.linalg.norm(rec["pred"] - rec[f"gt_{k0}"], axis=1)) if np.isfinite(rec["pred"]).any() else np.nan
        print(nm, "face" if res else "NO FACE", f"{e:.2f} mm vs {k0}", flush=True)
