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

THE BASELINE MATTERS MORE THAN THE METHOD, and the first version of this
docstring got it wrong. Comparing against the per-view rate pools 25 answers
per landmark and scores each -- a DIAGNOSTIC, not a product. Nobody ships
that. Against the fair alternative, one good view used alone:

    paired mm                        p50     p90    >5mm    >10mm
    best single view (-15,-15)      2.78    5.40   13.06%   1.039%
    face-on alone                   2.83    5.55   13.20%   0.860%
    all 25 pooled (the diagnostic)  3.31    6.92   24.13%   3.679%
    RANSAC over 25 views            2.71    4.99    9.97%   0.162%

So triangulation is worth 2.5% on the median and 6.4x on gross errors.
It buys ROBUSTNESS, NOT ACCURACY -- which is what RANSAC is for, and the
conclusion still stands because gross errors are what this project exists to
fix. But the "26x" quoted earlier compared a product against a diagnostic
and is withdrawn.

WHAT THE REDUNDANCY IS WORTH, from the per-view audit of the same grid.
Over 6160 (ear, landmark) pairs at 25 views, 89.06% are clean in EVERY view
and only 0.130% fail in a majority. So a combine that survives a minority of
bad views has a ceiling of about 0.13% against the 1.290% this pipeline
scores per view -- a factor of ten, and the largest remaining win measured
anywhere in this project.

This script asks whether lsq and ransac reach it. Same ears, same grid, same
renderer and the shipped 2D configuration, scored the same way: distance from
the fused point to the wrap-fitted ear, in millimetres, rate past 10 mm.

RE-SEATING, AND A TRAP IN SCORING IT. A triangulated point is a free ray
intersection with nothing holding it to a surface, so unlike a depth lift it
can float: 19.0% of raw points sit more than 2 mm off the ear. Projecting
each point onto the nearest surface fixes that, and the fix is real -- paired
distance to the ground-truth landmark, which no snap consults, improves from
2.89 to 2.57 mm median and gross paired halves, 0.182% to 0.091%.

BUT THE SURFACE METRIC AFTER SNAPPING IS LARGELY TAUTOLOGICAL. Snapping to a
surface guarantees distance to that surface is zero, and makes distance to a
nearby one small. A landmark that the 2D model put 10 mm from truth gets
pulled onto the scalp BESIDE the ear and its surface score improves while it
stays 10 mm from truth -- pp11_right's j=19 did exactly that, reading 4.32 mm
from the fit and 9.90 mm from its own ground-truth point. Score a re-seat on
PAIRED distance only, and keep the surface distance as a PRE-snap diagnostic.

WHICH SURFACE, measured on paired distance over 20 ears:

    snap to    p50    p75    p90    p99    >5mm     >10mm
    raw       2.89   4.08   5.40   8.33   14.09%   0.182%
    head      2.57   3.79   5.21   8.33   11.18%   0.091%
    fit       2.68   3.86   5.12   7.98   10.82%   0.091%

Neither dominates. The head wins in the body, because for a typical point the
nearest head surface IS the ear and the fit carries its own ~0.6 mm residual;
the fit wins in the tail, because the head offers scalp as a landing site and
the fit does not. Both halve the gross rate and then stop, which is the honest
headline: snapping cannot fix a landmark the 2D model placed wrongly, only
move it to a nearby surface.

GATE IT. The points a snap harms are the ones that were already seated -- raw
surface median 0.64 mm for the harmed against 1.14 mm for the helped. Only
moving points beyond a threshold keeps the gain and cuts the disturbance:

    T (mm)  moved   paired p50   paired>10mm   harmed
    0.0     100%        2.57        0.091%      26.7%
    0.5      71%        2.60        0.091%      15.1%
    2.0      19%        2.75        0.091%       3.4%
    never     0%        2.89        0.182%       0.0%

The whole gross benefit survives down to T=5, where under 1% of points move.
"Harmed" counts any worsening however small, so it overstates the damage;
the aggregate that matters is paired p50, which prefers T=0 by 0.03 mm.

Usage:
    python scripts/triangulate_audit.py --ears 20
    python scripts/triangulate_audit.py --ears 20 --reseat head --reseat-gate 0.5
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
    p.add_argument("--reseat", choices=["none", "head", "fit"], default="none",
                   help="project each fused point onto a surface. Scored on "
                        "PAIRED distance only -- the surface score after a "
                        "snap is tautological.")
    p.add_argument("--reseat-gate", type=float, default=0.5, metavar="MM",
                   help="only re-seat points further than this from the ear, "
                        "because the ones a snap harms are those already "
                        "seated (0 re-seats everything)")
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

        if a.reseat != "none":
            tgt = o3d.t.geometry.RaycastingScene()
            tgt.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
                g_head if a.reseat == "head" else g_ear))

        uv, cams = collect_rays(c, g_head, pipe, angles=angles, verbose=False)
        for meth in ("lsq", "ransac"):
            P, n_in, n_view = fit_rays(uv, cams, meth, a.thresh)
            d = _surface_mm(scene, P, scale)
            if a.reseat != "none":
                m = np.isfinite(P).all(1) & (d > a.reseat_gate)
                if m.any():
                    P = P.copy()
                    P[m] = tgt.compute_closest_points(o3d.core.Tensor(
                        P[m].astype(np.float32)))["points"].numpy()
                # keep the PRE-snap surface distance: the post-snap one is
                # tautological, and this is the number worth reporting.
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
