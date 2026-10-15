#!/usr/bin/env python3
"""

PRE-2026-10-14 RESULTS FROM THIS SCRIPT ARE VOID. It renders through
escape_audit.load_ear, which until 856652b applied canon2scan's transform
to the head without reversing the triangle winding, so every right-ear
render was shaded with its normals pointing inward. The script is correct
from that commit on; anything measured with it before then mixed correct
left ears with inside-out right ones. Re-run before quoting.
How much gross error does a change in CONTOUR-PERPENDICULAR 2D error buy?

THE QUESTION THIS ANSWERS. Ear_Landmarker measured a handedness bias of +8.53%
of nme_normal -- normal-direction, contour-perpendicular placement error --
between its as-is and mirrored arms. Separately, this repository measured a
0.62-point gross-error residual it could not attribute to lighting. Those are
two numbers in two different units and nothing yet connects them. This script
is the transfer function between them: it takes the 2D error this pipeline
actually makes, scales ONLY its normal component, and re-lifts, so the output
is d(escape rate) / d(normal error) measured on this pipeline's own data.

WHY SCALE RATHER THAN RE-PREDICT. The model cannot be asked for "8.53% more
normal error" -- so the perturbation is applied to the error it already makes,
holding the tangential component fixed. The decomposition is against the
ground-truth 2D, projected from the stored landmarks through the same camera,
so the baseline gain of 1.0 reproduces the model's own predictions exactly and
is checked for that.

TWO PERTURBATIONS, because "a bias" can mean either and they do not behave the
same way:

  gain      scale the existing normal error by g. Symmetric: it makes points
            that were outside the contour further outside and points that were
            inside further inside. This is what a difference in nme_normal
            between two arms literally is, since NME is unsigned.
  outward   add a systematic push along the OUTWARD contour normal, as a
            fraction of the mean normal error. Signed, and it is the one that
            can actually manufacture escapes, because a landmark pushed off the
            pinna's silhouette sends its ray past the ear onto the scalp.

Reporting both is the point. If the unsigned reading is far too small to
explain 0.62 points and the signed one is about right, then the residual is a
DIRECTIONAL drift and not a symmetric widening -- which is a different claim
about the model and a testable one.

THE NORMAL'S SIGN is read off the landmark centroid, not a strip convention:
outward means away from the ear's own centre, which on every strip is the
direction that leaves the silhouette.

    python scripts/normal_error_sensitivity.py --ears pp10_left --views 9
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np
import open3d as o3d

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ear3d import DEFAULTS                                          # noqa: E402
from ear3d.backproject import backproject_snapped                   # noqa: E402
from ear3d.camera import cone_angles, orbit_extrinsic               # noqa: E402
from ear3d.frames import in_frame                                   # noqa: E402
from ear3d.render import render                                     # noqa: E402
from ear3d.scheme import STRIPS                                     # noqa: E402

from escape_audit import (WRAP, load_ear, ear_frame, landmark_view,  # noqa: E402
                          light_mirror_for, _surface_mm)


def contour_frame(uv):
    """Per-point unit tangent and OUTWARD unit normal, from the GT 2D.

    The tangent is the central difference along the point's own strip, so a
    group boundary inside a strip is smoothed over rather than treated as a
    corner -- the strips are the continuous chains, the 13 groups are not.
    Endpoints use the one-sided difference. The normal is the tangent turned 90
    degrees, with its sign chosen to point away from the landmark centroid.
    """
    T = np.zeros_like(uv)
    for lo, hi in STRIPS:
        s = uv[lo:hi]
        d = np.empty_like(s)
        d[1:-1] = s[2:] - s[:-2]
        d[0], d[-1] = s[1] - s[0], s[-1] - s[-2]
        T[lo:hi] = d
    n = np.linalg.norm(T, axis=1, keepdims=True)
    T = T / np.maximum(n, 1e-9)
    N = np.stack([-T[:, 1], T[:, 0]], axis=1)
    out = uv - uv.mean(0)
    flip = (N * out).sum(1) < 0
    N[flip] *= -1.0
    return T, N


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ears", nargs="*", default=["pp10_right"])
    p.add_argument("--all", action="store_true")
    p.add_argument("--views", type=int, default=9)
    p.add_argument("--cone", type=float, default=45.0)
    p.add_argument("--grid", type=float, default=None)
    p.add_argument("--grid-step", type=float, default=15.0)
    p.add_argument("--gains", nargs="*", type=float,
                   default=[0.9147, 1.0, 1.0853, 1.25, 1.5, 2.0])
    p.add_argument("--outward", nargs="*", type=float,
                   default=[0.0, 0.0853, 0.25, 0.5, 1.0])
    p.add_argument("--light", default="anatomy")
    p.add_argument("--sun-x", type=float, default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    cfg = dict(DEFAULTS)
    cfg["interactive"] = False
    if a.sun_x is not None:
        d = list(cfg["light_dir"])
        cfg["light_dir"] = (float(a.sun_x), d[1], d[2])
        a.light = "frame"

    ears = a.ears
    if a.all:
        ears = sorted(f.stem for f in (WRAP / "corpus/canon2scan").glob("pp*.npz"))

    from ear3d.twod import make_pipeline                             # noqa
    pipe, ck = make_pipeline(cfg)
    print(f"landmarker {ck.name}   sun {tuple(cfg['light_dir'])} anchor {a.light}")

    if a.grid is not None:
        ax = np.arange(-a.grid, a.grid + 1e-6, a.grid_step)
        angles = [(float(y), float(q)) for q in ax for y in ax]
    else:
        angles = list(cone_angles(a.views, a.cone))
    print(f"{len(angles)} views per ear, {len(ears)} ears")

    rows = []
    for i, name in enumerate(ears):
        try:
            c = dict(cfg, light_mirror=light_mirror_for(name, a.light))
            head, ear, lm = load_ear(name, "hutubs", c)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"[{i+1}/{len(ears)}] {name}: skipped -- {e}")
            continue
        R, C, scale = ear_frame(head, lm, c)
        g_head, g_ear = in_frame(head, R, C, scale), in_frame(ear, R, C, scale)
        lm_f = ((lm - C) @ R.T) / scale
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(g_ear))

        for yaw, pitch in angles:
            E = orbit_extrinsic(yaw, pitch, c)
            img, depth, cam = render(g_head, c, E=E)
            best = landmark_view(img, c, pipe, "none", depth.shape)
            if best is None:
                continue
            uv = np.asarray(best["landmarks"], float)
            gt = cam.project(lm_f)
            T, N = contour_frame(gt)
            e = uv - gt
            et, en = (e * T).sum(1), (e * N).sum(1)
            mean_n = float(np.abs(en).mean())

            def score(u, kind, amt):
                P, _, _ = backproject_snapped(depth, u, cam, c)
                d = _surface_mm(scene, P, scale)
                pr = np.linalg.norm(P - lm_f, axis=1) * scale
                for j in range(55):
                    rows.append(dict(ear=name, yaw=yaw, pitch=pitch, j=j,
                                     kind=kind, amt=amt, d_mm=d[j],
                                     paired_mm=pr[j]))

            for g in a.gains:
                score(gt + et[:, None] * T + (g * en)[:, None] * N, "gain", g)
            for f in a.outward:
                score(gt + et[:, None] * T + (en + f * mean_n)[:, None] * N,
                      "outward", f)
        print(f"[{i+1}/{len(ears)}] {name}: done")

    print()
    d = {}
    for r in rows:
        if np.isfinite(r["d_mm"]):
            d.setdefault((r["kind"], r["amt"]), []).append(r["d_mm"])
    print(f"{'kind':9s} {'amount':>8s} {'>10mm':>8s} {'>5mm':>8s} {'median':>8s}  n")
    for (kind, amt), v in sorted(d.items()):
        v = np.array(v)
        print(f"{kind:9s} {amt:8.4f} {100*(v>10).mean():7.2f}% "
              f"{100*(v>5).mean():7.2f}% {np.median(v):7.2f}  {len(v)}")

    base = d.get(("gain", 1.0))
    if base is not None:
        b = 100 * (np.array(base) > 10).mean()
        print(f"\nbaseline (gain 1.0) = the model's own predictions: {b:.2f}%")
        for key in (("gain", 1.0853), ("outward", 0.0853)):
            if key in d:
                v = np.array(d[key])
                print(f"  {key[0]} {key[1]}: {100*(v>10).mean():.2f}%  "
                      f"delta {100*(v>10).mean()-b:+.2f} points")

    if a.out:
        op = pathlib.Path(a.out); op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for x in rows for k in x}))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {op}")


if __name__ == "__main__":
    main()
