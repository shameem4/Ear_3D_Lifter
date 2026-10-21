"""Freeze the per-(view, landmark) accuracy table the pipeline selects rays with.

Usage: build_view_table.py RAY_CACHE_DIR OUT.npz

table[v, j] = median over ears of the distance (mm) from the reference 3D
landmark j (config gt_lm3d) to the ray view v asserts for it. Views are
indexed ANATOMICALLY: a right ear's yaw is mirrored onto the left-ear grid,
so the pipeline mirrors back at lookup. RAY_CACHE_DIR is a
multiview_collect_rays.py cache collected with the pipeline's own settings.
"""
import pathlib
import sys

import numpy as np

R0 = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R0))
from ear3d.gt import GT_C2S, GT_FILE, gt_in_cache                                               # noqa: E402

cd, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
fs = sorted(cd.glob("*.npz"))
E = [dict(np.load(f, allow_pickle=True)) for f in fs]
ANG = E[0]["angles"]; key = {tuple(a): i for i, a in enumerate(ANG)}
mir = np.array([key[(-a[0], a[1])] for a in ANG])
RD = []
for f, e in zip(fs, E):
    hd, sd = f.stem.rsplit("_", 1)
    gt = gt_in_cache(hd, sd, np.load(str(e["head"])))
    p = np.arange(len(ANG)) if sd == "L" else mir
    o, d = e["o"][p], e["d"][p]; v = gt[None] - o
    RD.append(np.linalg.norm(v - (v * d).sum(-1, keepdims=True) * d, axis=-1))
T = np.nanmedian(RD, axis=0)
np.savez(out, table=T, angles=ANG, n_ears=len(E), gt=GT_FILE, gt_canon2scan=GT_C2S, source=cd.name)
print(f"{out}: {T.shape} from {len(E)} ears, GT {GT_FILE} via {GT_C2S}; median cell {np.median(T):.2f} mm")
