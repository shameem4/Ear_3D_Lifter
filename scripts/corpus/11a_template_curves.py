#!/usr/bin/env python3
"""Step 11a: the reference curve SHAPE -- the 2D model, multiview, on the mean template.

Usage: 11a_template_curves.py TEMPLATE OUT.npz        (paths under wrap_work unless absolute)

Renders the skinned mean template face-on (side away from the absent head)
on the same 25-view orbit, lighting and framing the pipeline uses, runs the
landmarker in every view, and triangulates each landmark with RANSAC over all
25 rays, snapped to the template surface. Writes `all_25` (55, 3) on TEMPLATE,
plus `v7` (the hand marks on TEMPLATE) and `ckpt`.

Step 11 re-spaces these curves to the corpus's median spacing. Together they
make the reference follow the CURRENT 2D model's contour convention: re-run
both when the model changes how it traces contours. lm3d_v11 used the old
model's curves (v9_provenance/tpl_marks.npz on ear_mean_v4); lm3d_v12 these,
with v4_fc05_s42 on ear_mean_v5 (2026-10-21).

Ported from wrap_work/templates/v9_provenance/tpl_mv.py; its "look" mode, the
best-8 variant (unused downstream) and the corpus-mark comparison are dropped.
"""
import pathlib
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ear3d.camera import orbit_extrinsic                             # noqa: E402
from ear3d.config import DEFAULTS                                    # noqa: E402
from ear3d.corpus import WRAP                                        # noqa: E402
from ear3d.frames import frame_from, in_frame, plane_normal          # noqa: E402
from ear3d.render import render                                      # noqa: E402
from ear3d.skin import apply_skin                                    # noqa: E402
from ear3d.triangulate import ransac_point                           # noqa: E402
from ear3d.twod import LandmarkPredictor, landmarker_checkpoint      # noqa: E402
from ear3d.wrap import load_mesh                                     # noqa: E402

ANG = [(float(y), float(p)) for y in range(-30, 31, 15) for p in range(-30, 31, 15)]   # the pipeline's grid


def main():
    res = lambda s: pathlib.Path(s) if pathlib.Path(s).is_absolute() else WRAP / s
    tpl, out = res(sys.argv[1]), res(sys.argv[2])
    r = load_mesh(str(tpl)); TV, TF = np.asarray(r[0]), np.asarray(r[1])
    v7 = np.load(WRAP / "templates/lm3d_v7_canonical_mm.npz", allow_pickle=True)
    A7 = np.einsum("ij,ijk->ik", v7["barycentric"][:55], TV[TF[v7["triangle"][:55]]])   # hand marks, for framing and a check

    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(TV), o3d.utility.Vector3iVector(TF)); m.compute_vertex_normals()
    apply_skin(m, seed=0, tone=DEFAULTS["ear_tone"], noise=DEFAULTS["ear_skin_noise"])
    n = plane_normal(A7); c = A7.mean(0)
    if n @ (TV.mean(0) - c) > 0: n = -n                # face away from the (absent) head side
    R = frame_from(n, np.array([0, 1.0, 0]))
    ext = np.ptp((A7 - c) @ R.T, axis=0)[:2].max()
    s = 0.5 * ext * 1.5 / (DEFAULTS["eye_z"] * np.tan(np.radians(DEFAULTS["vfov"]) / 2))
    g = in_frame(m, R, c, s)
    light = dict(DEFAULTS, light_dir=tuple(np.asarray(DEFAULTS["multiview_light_dir"], float)), light_mirror=False, light_frame="world")

    ck = str(landmarker_checkpoint(DEFAULTS)); pred = LandmarkPredictor(ck, "cuda")
    O = np.full((len(ANG), 55, 3), np.nan); D = O.copy()
    for i, (yw, pt) in enumerate(ANG):
        img, _, cam = render(g, light, E=orbit_extrinsic(yw, pt, DEFAULTS))
        o, dd = cam.rays(pred.predict(img))
        O[i], D[i] = o * s @ R + c, dd @ R                 # back to template mm
    P = np.full((55, 3), np.nan)
    for j in range(55):
        q = ransac_point(O[:, j], D[:, j], 1.0); q = q[0] if isinstance(q, tuple) else q
        if q is not None: P[j] = q
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.core.Tensor(TV.astype(np.float32)), o3d.core.Tensor(TF.astype(np.uint32)))
    ok = np.isfinite(P).all(1)
    P[ok] = sc.compute_closest_points(o3d.core.Tensor(P[ok].astype(np.float32)))["points"].numpy()
    d7 = np.linalg.norm(P - A7, axis=1)
    print(f"{tpl.name}: {ok.sum()}/55 triangulated; vs v7 hand marks median {np.nanmedian(d7):.2f} max {np.nanmax(d7):.2f} mm; ckpt {ck.split('checkpoints/')[-1]}")
    np.savez(out, all_25=P, v7=A7, template=str(tpl), ckpt=ck)


if __name__ == "__main__":
    main()
