"""Collect rays and projected ground truth once, so view selection is offline.

Building a per-(landmark, view) accuracy table and then testing selection
rules needs many passes over the same data. The renders and the 2D model are
what cost; the ray algebra is free. So do the expensive part once and cache:

    uv      (V, 55, 2)  what the detector said, per view
    gt_uv   (V, 55, 2)  where the ground-truth landmark projects, same view
    o, d    (V, 55, 3)  the ray each detection asserts, in the framed space
    lm_f    (55, 3)     the ground-truth point, framed
    scale               framed units -> millimetres

Everything downstream -- the table, the selection rule, the re-fit -- is then
numpy on these arrays.

Usage:
    python scripts/cache_rays.py --ears 112
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
    p.add_argument("--ears", type=int, default=112)
    p.add_argument("--dataset", default="hutubs")
    p.add_argument("--grid", type=float, default=30.0)
    p.add_argument("--grid-step", type=float, default=15.0)
    p.add_argument("--out", default="runs/raycache")
    p.add_argument("--conf", type=float, default=None,
                   help="detector confidence. Default: whatever config says "
                        "(0.05). A higher gate yields fewer rays per point, "
                        "so the question it answers is the VOTE's gross-error "
                        "rate, never the per-view yield.")
    a = p.parse_args()

    from escape_audit import (DEFAULTS, WRAP, load_ear, ear_frame,
                              light_mirror_for)
    from ear3d.frames import in_frame
    from ear3d.label import collect_rays
    from ear3d.twod import make_pipeline

    cfg = dict(DEFAULTS)
    if a.conf is not None:
        cfg["detector_conf"] = float(a.conf)
    pipe, ck = make_pipeline(cfg)
    g = a.grid
    angles = [(float(y), float(q))
              for y in np.arange(-g, g + 1e-9, a.grid_step)
              for q in np.arange(-g, g + 1e-9, a.grid_step)]
    names = sorted(f.stem for f in (WRAP / "corpus/canon2scan").glob("*.npz")
                   if str(np.load(f, allow_pickle=True)["dataset"]) == a.dataset)[:a.ears]
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"{ck.parent.name}/{ck.name}  {len(names)} ears x {len(angles)} views"
          f"  detector_conf={cfg['detector_conf']}")

    for i, nm in enumerate(names):
        dst = out / f"{nm}.npz"
        if dst.exists() and dst.stat().st_size:
            print(f"[{i+1}/{len(names)}] {nm}: cached"); continue
        c = dict(cfg, light_mirror=light_mirror_for(nm, "frame", a.dataset))
        try:
            head, ear, lm = load_ear(nm, a.dataset, c)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"[{i+1}/{len(names)}] {nm}: skipped -- {e}"); continue
        R, C, s = ear_frame(head, lm, c)
        gh = in_frame(head, R, C, s)
        lm_f = ((lm - C) @ R.T) / s
        uv, cams = collect_rays(c, gh, pipe, angles=angles, verbose=False)
        V = len(angles)
        o = np.full((V, 55, 3), np.nan); d = np.full((V, 55, 3), np.nan)
        gt = np.full((V, 55, 2), np.nan)
        for v in range(V):
            gt[v] = cams[v].project(lm_f)
            ok = np.isfinite(uv[v]).all(1)
            if ok.any():
                oo, dd = cams[v].rays(uv[v][ok])
                o[v, ok] = oo; d[v, ok] = dd
        np.savez_compressed(dst, uv=uv, gt_uv=gt, o=o, d=d, lm_f=lm_f,
                            scale=s, angles=np.array(angles), ear=nm)
        bad = 100 * np.mean(~np.isfinite(uv).all(2))
        print(f"[{i+1}/{len(names)}] {nm}: {bad:.1f}% of landmark-views had no "
              f"detection", flush=True)


if __name__ == "__main__":
    main()
