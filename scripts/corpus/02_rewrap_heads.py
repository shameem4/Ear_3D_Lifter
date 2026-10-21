#!/usr/bin/env python3
"""Step 2: fit the ear template to each head ear, posed by its predicted landmarks.

Usage: 02_rewrap_heads.py OUT_DIR PRED_DIR [--lobe EAR,EAR...] [EAR...]

For each ear (default: every PRED_DIR/<ear>.npz from step 1): robust-inlier
landmarks pose the template (Faceform RigidAlignment), then the automatic
Wrap fits it -- see ear3d/faceform.py. Ears named in --lobe also get three
lobe anchors (points 15, 17, 19) as Wrap control points: the rescue for fits
that clip the lobe. lmwrap_v1 used it on P0082_left, P0088_left, P0027_left
(P0066_right kept its plain fit).

Writes OUT_DIR/<ear>.npz: V (1188, 3) fit in scan mm, P, inliers, resid,
scan, anchors. Faceform work files are deleted per ear. No GPU path; WrapCmd
uses ~7 cores per fit, so run about 4 workers.

Ported from scratch rewrap_heads.py + lobe_rewrap.py (2026-10-17), unchanged
in method.
"""
import argparse
import pathlib
import shutil
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ear3d.faceform import robust_inliers, wrap_ear                  # noqa: E402

LOBE_ANCHORS = (15, 17, 19)


def main():
    a = argparse.ArgumentParser(); a.add_argument("out"); a.add_argument("preds"); a.add_argument("ears", nargs="*")
    a.add_argument("--lobe", default="", help="comma-separated ears to refit with lobe anchors"); a = a.parse_args()
    out, preds = pathlib.Path(a.out), pathlib.Path(a.preds); out.mkdir(parents=True, exist_ok=True)
    lobe = set(filter(None, a.lobe.split(",")))
    for ear in a.ears or sorted(f.stem for f in preds.glob("*.npz")):
        if (out / f"{ear}.npz").exists():
            continue
        z = np.load(preds / f"{ear}.npz"); P = z["P"]
        h = trimesh.load(str(z["scan"]), process=False)
        V = np.asarray(h.vertices); V = V * (1000.0 if np.ptp(V, 0).max() < 10 else 1.0)
        side = "L" if ear.endswith("left") else "R"
        anchors = LOBE_ANCHORS if ear in lobe else ()
        wd = out / "work" / ear
        try:
            inl, res = robust_inliers(P)
            Vf = wrap_ear(V, np.asarray(h.faces), P, side, wd, ear, use=inl, anchors=anchors)
        except Exception as e:
            print(ear, "FAILED", e, flush=True); continue
        finally:
            shutil.rmtree(wd, ignore_errors=True)
        np.savez(out / f"{ear}.npz", V=Vf, P=P, inliers=inl, resid=res, scan=str(z["scan"]), anchors=np.array(anchors, int))
        print(f"{ear}: inliers {inl.sum()}/55 resid {res:.2f} | fit extent {np.ptp(Vf, 0).max():.1f} mm"
              f"{' | lobe anchors' if anchors else ''}", flush=True)


if __name__ == "__main__":
    main()
