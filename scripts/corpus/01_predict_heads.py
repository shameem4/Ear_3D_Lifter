#!/usr/bin/env python3
"""Step 1: run the pipeline on every corpus head scan; keep each ear's 55 landmarks.

Usage: 01_predict_heads.py OUT_DIR WORKER N_WORKERS [--generation esc|lmw]

The scans are the ones a corpus generation's canon2scan was built against
(HUTUBS: one head per subject, both ears; SONICOM: one graded mesh per ear),
so `--generation` only chooses the scan list. Each scan is turned by a
rotation seeded from its file name and handed to the pipeline unskinned, as a
user would; results go back to the scan's own frame.

Writes OUT_DIR/<ear>.npz: P (55, 3) in scan millimetres (NaN = not found),
scan, Rr, ckpt. These are the input to step 2 (rewrap) and step 11 (marks).

PROVENANCE. lmwrap_v1 was built from two earlier passes of this step (scratch
judge_heads.py / judge_all.py, 2026-10-17), with rotations taken from a
validate_scans run (runs/corpus_pred) and a pipeline build just before the
face-frame checks were committed. A rebuild reproduces the method, not the
exact landmarks.
"""
import argparse
import collections
import pathlib
import sys
import zlib

import numpy as np
import open3d as o3d

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ear3d import corpus                                             # noqa: E402
from ear3d.corpus import WRAP                                        # noqa: E402
from ear3d.pipeline import Pipeline                                  # noqa: E402


def rotation(seed):
    q = np.random.default_rng(seed).normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def main():
    a = argparse.ArgumentParser(); a.add_argument("out"); a.add_argument("worker", type=int); a.add_argument("n_workers", type=int)
    a.add_argument("--generation", default=corpus.CURRENT); a = a.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    by_scan = collections.defaultdict(list)                 # one pipeline run per scan
    for f in sorted((WRAP / corpus.get("canon2scan", a.generation)).glob("*.npz")):
        by_scan[str(np.load(f, allow_pickle=True)["scan"])].append(f.stem)
    pipe = Pipeline()
    for scan in sorted(by_scan)[a.worker::a.n_workers]:
        ears = by_scan[scan]
        if all((out / f"{e}.npz").exists() for e in ears):
            continue
        m = o3d.io.read_triangle_mesh(scan); V = np.asarray(m.vertices)
        V = V * (1000.0 if np.ptp(V, 0).max() < 10 else 1.0)         # HUTUBS is metres
        R = rotation(zlib.crc32(pathlib.Path(scan).name.encode()))
        res = pipe.run(o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V @ R.T), m.triangles))
        for e in ears:
            sd = "L" if e.endswith("left") else "R"
            P = res[sd] @ R if res and res[sd] is not None else np.full((55, 3), np.nan)
            np.savez(out / f"{e}.npz", P=P, scan=scan, Rr=R, ckpt=pipe.ckpt)
            print(e, "ok" if np.isfinite(P).all() else "MISSING", flush=True)


if __name__ == "__main__":
    main()
