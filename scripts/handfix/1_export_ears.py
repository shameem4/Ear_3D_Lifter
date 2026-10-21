#!/usr/bin/env python3
"""Step 1: export the ears to hand-correct, for the browser editor (serve.py).

Usage: 1_export_ears.py [--preds DIR] [--lobe-mm 5] [--out runs/handfix] [EAR ...]

Picks every corpus ear whose worst lobe point (15-19) is more than --lobe-mm
from the reference (lm3d_v12) in the 2D model's corpus predictions, plus any
EAR named. Per ear writes OUT/<ear>/:
  surface.ply   the scan around the ear, in the ear's canonical mm (right ears
                mirrored, as stored), cropped to a sphere around the fit
  points.json   start (55x3, the reference), pred (55x3, the model, read-only),
                dataset, held_out, error per point
The editor saves OUT/<ear>/adjusted.json beside them; this script never
overwrites an adjusted.json.

OUT holds scan geometry: it lives under runs/ (gitignored) and stays local.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ear3d.config import DEFAULTS                                    # noqa: E402
from ear3d.corpus import WRAP                                        # noqa: E402
from ear3d.data import load_ear                                      # noqa: E402
from ear3d.gt import inv_gt, load_reference                          # noqa: E402

HELD_OUT = {f"pp{n}" for n in (16, 19, 21, 22, 33, 40, 46, 47, 49, 57, 69, 70, 71, 72, 77, 78)}
LOBE = slice(15, 20)
MARGIN_MM = 12.0


def main():
    a = argparse.ArgumentParser()
    a.add_argument("ears", nargs="*")
    a.add_argument("--preds", default=str(WRAP / "corpus/lmwrap_v1/preds_v4fc05"))
    a.add_argument("--lobe-mm", type=float, default=5.0)
    a.add_argument("--out", default="runs/handfix")
    a = a.parse_args()
    L, N = load_reference("lm3d_v12_canonical_mm.npz")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    picked = {}
    for f in sorted(pathlib.Path(a.preds).glob("*.npz")):
        nm = f.stem; P = np.load(f)["P"][:55]
        if nm not in N or not np.isfinite(P).all():
            continue
        P = (np.c_[P, np.ones(55)] @ inv_gt(nm).T)[:, :3]           # scan mm -> canonical
        G = L["landmarks"][N.index(nm)][:55]
        e = np.linalg.norm(P - G, axis=1)
        if e[LOBE].max() > a.lobe_mm or nm in a.ears:
            picked[nm] = (G, P, e)
    cfg = dict(DEFAULTS, skin=False)
    for nm, (G, P, e) in picked.items():
        d = out / nm; d.mkdir(exist_ok=True)
        ds = "sonicom" if nm.startswith("P0") else "hutubs"
        head, fit, _ = load_ear(nm, ds, cfg)
        F = np.asarray(fit.vertices); c = F.mean(0); r = np.linalg.norm(F - c, axis=1).max() + MARGIN_MM
        V = np.asarray(head.vertices); T = np.asarray(head.triangles)
        keep = np.linalg.norm(V - c, axis=1) < r
        T = T[keep[T].all(1)]; used = np.unique(T); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V[used]), o3d.utility.Vector3iVector(remap[T]))
        o3d.io.write_triangle_mesh(str(d / "surface.ply"), m, write_ascii=False)
        json.dump(dict(ear=nm, dataset=ds, held_out=nm.rsplit("_", 1)[0] in HELD_OUT,
                       frame="canonical mm (right ears mirrored about z)", reference="lm3d_v12_canonical_mm.npz",
                       start=np.round(G, 4).tolist(), pred=np.round(P, 4).tolist(),
                       error=np.round(e, 3).tolist(), lobe_max=round(float(e[LOBE].max()), 2)),
                  open(d / "points.json", "w"))
    print(f"{len(picked)} ears -> {out}  (held out: {sum(n.rsplit('_', 1)[0] in HELD_OUT for n in picked)})")


if __name__ == "__main__":
    main()
