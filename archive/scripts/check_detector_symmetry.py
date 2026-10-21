"""Regression check: is the chosen detector box the same under a mirror?

Ear_Landmarker's `detector_frame_tta` runs the detector on the mirrored frame
too and merges both presentations' boxes through NMS. A union is SYMMETRIC --
the merged set is the same whichever presentation you start from -- and that
is not a nicety, it is the property that puts the feature in the class a
handed corpus cannot flatter (see scripts/mixed_handedness.py, where an
asymmetric rule worth 0.53pp on this corpus is worth -0.005pp on a mixed one).

MEASURED, 6 ears x 5 yaws, the box the PIPELINE chose:

    detector_frame_tta=True    median |box diff|   0.00 px   max   0.00
    detector_frame_tta=False   median |box diff|  53.66 px   max 134.88

With the union on it is bit-identical on every view. With it off, the box
moves by a median of 54 px on the SAME frame and the SAME ear, decided by
nothing but which way the image was presented -- a different crop, not
jitter.

TWO WAYS TO GET THIS TEST WRONG, both of which this file was written after
making. `detector_frame_tta` lives in the pipeline's __call__, NOT in
EarDetector.detect, so calling detect() directly bypasses the feature and
measures nothing. And detect() returns every box above cfg["detector_conf"],
which this project sets to 0.05 for renders -- about a hundred of them,
before the pipeline's own NMS. Comparing those two sets reports a ~581 px
disagreement that looks exactly like a refutation of the symmetry claim.
Go through __call__ and compare the box that was CHOSEN.

Exits non-zero if the union's symmetry has broken, so it can be run after
any change to square_roi_crop, the NMS thresholds or the merge.

Usage:
    python scripts/check_detector_symmetry.py --ears 6
"""

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ears", type=int, default=6)
    p.add_argument("--yaws", type=float, nargs="*", default=[-30, -15, 0, 15, 30])
    p.add_argument("--tol", type=float, default=0.5,
                   help="px of slack allowed with the union ON (default 0.5)")
    a = p.parse_args()

    import importlib
    import open3d as o3d
    from escape_audit import (DEFAULTS, WRAP, load_ear, ear_frame, pick,
                              light_mirror_for)
    from ear3d.camera import orbit_extrinsic
    from ear3d.twod import make_pipeline
    R3 = importlib.import_module("ear3d.render")

    names = sorted(f.stem for f in (WRAP / __import__("ear3d.corpus", fromlist=["get"]).get("canon2scan")).glob("pp*.npz"))[:a.ears]
    rows = {}
    for tta in (True, False):
        cfg = dict(DEFAULTS, flip_fusion=True, detector_frame_tta=tta)
        pipe, ck = make_pipeline(cfg)
        d = []
        for nm in names:
            head, ear, lm = load_ear(nm, "hutubs", cfg)
            Rm, C, s = ear_frame(head, lm, cfg)
            h = o3d.geometry.TriangleMesh(head)
            h.vertices = o3d.utility.Vector3dVector(
                (np.asarray(head.vertices) - C) @ Rm.T / s)
            h.compute_vertex_normals()
            c = dict(cfg, light_mirror=light_mirror_for(nm, "frame"))
            for yaw in a.yaws:
                img, _, _ = R3.render(h, c, E=orbit_extrinsic(float(yaw), 0.0, c))
                W, hw = img.shape[1], img.shape[:2]
                A = pick(pipe(img, timestamp=0.0), c, hw)
                raw = pipe(np.ascontiguousarray(img[:, ::-1]), timestamp=0.0)
                for r in raw:                      # un-mirror x: W - x, exact
                    b = list(r["bbox"])
                    r["bbox"] = np.array([b[0], W - b[3], b[2], W - b[1]])
                B = pick(raw, c, hw)
                if A is None or B is None:
                    continue
                d.append(float(np.abs(np.asarray(A["bbox"], float) -
                                      np.asarray(B["bbox"], float)).max()))
        rows[tta] = np.array(d)
        print(f"  detector_frame_tta={str(tta):5s} {len(d):3d} views   "
              f"median {np.median(d):7.2f} px   p90 {np.percentile(d, 90):7.2f}   "
              f"max {max(d):7.2f}")

    on, off = rows[True], rows[False]
    if not len(on):
        sys.exit("no views produced a detection in both presentations")
    if on.max() > a.tol:
        sys.exit(f"\nFAIL: with the union ON the chosen box differs by up to "
                 f"{on.max():.2f} px between presentations. The merge is no "
                 f"longer symmetric, and the feature has lost the property it "
                 f"was adopted for.")
    if len(off) and np.median(off) <= a.tol:
        print(f"\nNOTE: the union is symmetric, but so is the OFF arm "
              f"(median {np.median(off):.2f} px). Either the detector stopped "
              f"being presentation-sensitive or this test stopped exercising "
              f"it -- check before trusting the pass.")
    print(f"\nOK: symmetric to {on.max():.2f} px with the union on, against a "
          f"median {np.median(off):.2f} px shift with it off.")


if __name__ == "__main__":
    main()
