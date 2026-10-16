"""Score the EXISTING multi-view triangulator with the escape metric.

The infrastructure is already here -- `ear3d/triangulate.py` and
`ear3d.label.collect_rays` / `fit_rays` -- and it was measured once, by
leave-one-view-out REPROJECTION: 8.9 px for a face-on lift, 6.7 px least
squares, 7.1 px RANSAC, ear about 430 px across, 8 heads at 7 views.

That is a mean-like metric in pixels, and this whole project turns on the
fact that means and far-tail gates disagree. Nobody has scored the
triangulator on gross errors, which is the thing the pipeline is judged by
and the thing rays are supposed to fix: a 2D detection asserts a RAY, and
what that ray later hits is a separate question, so the depth-sampling
failure that puts a landmark on the scalp cannot occur.

WHAT THE REDUNDANCY IS WORTH, from the per-view audit of the same grid.
Over 6160 (ear, landmark) pairs at 25 views, 89.06% are clean in EVERY view
and only 0.130% fail in a majority. So a combine that survives a minority of
bad views has a ceiling of about 0.13% against the 1.290% this pipeline
scores per view -- a factor of ten, and the largest remaining win measured
anywhere in this project.

This script asks whether lsq and ransac reach it. Same ears, same grid, same
renderer and the shipped 2D configuration, scored the same way: distance from
the fused point to the wrap-fitted ear, in millimetres, rate past 10 mm.

Usage:
    python scripts/triangulate_audit.py --ears 20
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
    p.add_argument("--ears", type=int, default=20)
    p.add_argument("--dataset", default="hutubs", choices=["hutubs", "sonicom"])
    p.add_argument("--grid", type=float, default=30.0)
    p.add_argument("--grid-step", type=float, default=15.0)
    p.add_argument("--gate", type=float, default=10.0)
    p.add_argument("--thresh", type=float, default=0.03)
    p.add_argument("--out", default="runs/triangulate.csv")
    a = p.parse_args()

    import csv
    import open3d as o3d
    from escape_audit import (WRAP, DEFAULTS, load_ear, ear_frame, _surface_mm,
                              light_mirror_for)
    from ear3d.escape import escape_rate, off_head
    from ear3d.frames import in_frame
    from ear3d.label import collect_rays, fit_rays
    from ear3d.twod import make_pipeline

    cfg = dict(DEFAULTS)
    pipe, ck = make_pipeline(cfg)
    g = a.grid
    angles = [(float(y), float(q))
              for y in np.arange(-g, g + 1e-9, a.grid_step)
              for q in np.arange(-g, g + 1e-9, a.grid_step)]
    names = sorted(f.stem for f in (WRAP / "corpus/canon2scan").glob("*.npz")
                   if str(np.load(f, allow_pickle=True)["dataset"]) == a.dataset)[:a.ears]
    print(f"{ck.parent.name}/{ck.name}   {len(names)} ears x {len(angles)} views "
          f"-> one fused point per landmark\n")

    rows = []
    for i, nm in enumerate(names):
        c = dict(cfg, light_mirror=light_mirror_for(nm, "frame", a.dataset))
        try:
            head, ear, lm = load_ear(nm, a.dataset, c)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"[{i+1}/{len(names)}] {nm}: skipped -- {e}")
            continue
        R, C, scale = ear_frame(head, lm, c)
        g_head, g_ear = in_frame(head, R, C, scale), in_frame(ear, R, C, scale)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(g_ear))
        lm_f = ((lm - C) @ R.T) / scale

        uv, cams = collect_rays(c, g_head, pipe, angles=angles, verbose=False)
        for meth in ("lsq", "ransac"):
            P, n_in, n_view = fit_rays(uv, cams, meth, a.thresh)
            d = _surface_mm(scene, P, scale)
            paired = np.linalg.norm(P - lm_f, axis=1) * scale
            for k in range(len(d)):
                rows.append(dict(ear=nm, j=k, method=meth, d_mm=d[k],
                                 paired_mm=paired[k], n_view=int(n_view[k]),
                                 n_in=int(n_in[k])))
        esc = {m: 100 * escape_rate([r["d_mm"] for r in rows
                                     if r["ear"] == nm and r["method"] == m], a.gate)
               for m in ("lsq", "ransac")}
        print(f"[{i+1}/{len(names)}] {nm}: lsq {esc['lsq']:5.2f}%  "
              f"ransac {esc['ransac']:5.2f}%", flush=True)

    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print(f"\n{'method':10s} {'escape':>8s} {'no point':>9s} {'median mm':>10s} "
          f"{'paired med':>11s}")
    for m in ("lsq", "ransac"):
        d = np.array([r["d_mm"] for r in rows if r["method"] == m])
        q = np.array([r["paired_mm"] for r in rows if r["method"] == m])
        print(f"{m:10s} {100 * escape_rate(d, a.gate):7.3f}% "
              f"{100 * off_head(d):8.3f}% {np.nanmedian(d):9.3f} "
              f"{np.nanmedian(q):10.3f}")
    print(f"\nfor comparison, the same grid scored PER VIEW is 1.290%, and the")
    print(f"majority-vote ceiling over those views is 0.130%.")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
