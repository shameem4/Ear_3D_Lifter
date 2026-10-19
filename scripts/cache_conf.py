"""Cache per-landmark HEATMAP confidence alongside the rays.

Deep-MVLM (arXiv 1910.06007) filters views per landmark by the heatmap peak
value before its RANSAC. This project tested DETECTION confidence -- the
detector's box score -- found it worthless as a view-quality signal
(Spearman -0.010) and wrongly generalised that to "confidence does not
work". The two are different quantities and only the second was measured.

The per-point value is computed inside the pipeline
(`self.landmarker.predict(crop, with_confidence=True)`) and then used ONLY
for tracker smoothing, which this project disables, so it never reaches the
caller. Rather than change Ear_Landmarker, wrap `predict` and keep what the
last call produced -- the last call is the one whose landmarks are returned,
including through the ROI refinement loop.

Why it is worth the renders: the learned view table is a per-SCHEME prior
and cannot adapt to an individual ear, which is why best-8 reaches 2.24 mm
where an oracle picking per ear reaches 1.49. Heatmap confidence is per-ear
evidence, so it is the natural candidate for that gap.

MEASURED, AND IT FAILS. 20 ears, 27,500 landmark-views:

    overall                        Spearman -0.066
    within each landmark (median)  Spearman -0.018,  2/55 below -0.1
    by quintile, 2D error p50      19.39 px lowest -> 16.67 px highest

The WITHIN-LANDMARK figure is the one that matters, because selecting views
means choosing among views for a FIXED landmark; the overall -0.066 is
mostly between-landmark structure the view table already has.

NOT A DEGENERATE SIGNAL, which was the obvious alternative explanation and
is ruled out: confidence spans 35x from p1 to p99, and varies MORE across
views for one landmark (CV 0.590) than between landmarks (CV 0.237) -- so
it moves along exactly the axis selection would use. It simply does not
know which view placed the point well.

BOTH ARCHITECTURE AND DOMAIN MATTER -- an earlier version of this
docstring said "architectural, NOT domain", which was a false dichotomy
with nothing forcing it. Ear_Landmarker then measured the same quantity on
1,054 photographs with the SAME checkpoint and it works there:

    within-landmark rho    mine (renders) -0.018     theirs (photos) -0.486
    landmarks below -0.1          2/55                       55/55
    2D error p50                19.39 px                    1.92 px

REGIME IS REFUTED FROM BOTH ENDS. They could not reach my error level
(their worst 50 images only get to 6.32 px, still rho -0.314). I can reach
theirs: selecting on the view TABLE is selection on a PRIOR, so unlike an
error floor it does not truncate the outcome --

    all views            27500 views, p50 17.38 px, rho -0.018
    table best 8 views    8800 views, p50 12.02 px, rho -0.014
    table best 3 views    3300 views, p50 10.87 px, rho -0.037

Cutting the median error 40% moves nothing. So the signal is absent out of
domain at every error level reachable here, and present in domain at every
error level reachable there.

THE ARCHITECTURE CLAIM STILL STANDS, and is about a different quantity.

`heatmap_confidence` measures the spatial CONCENTRATION of a softmax.
Deep-MVLM reads the AMPLITUDE of an hourglass heatmap trained with MSE
against Gaussian targets. Only one of those can report absence:

  - softmax(x + c) == softmax(x), so under a coordinate loss through
    soft_argmax the absolute amplitude is UNIDENTIFIABLE. Ear_Landmarker
    trains with WingLoss on coordinates, so nothing constrains it.
  - a softmax must put its mass somewhere, so it can say "unsure WHERE"
    and never "nothing is HERE" -- and "nothing is here" is the case view
    selection needs flagged.
  - MSE on Gaussian targets pins the amplitude, so a flat near-zero map is
    a trained way to report absence. That is the channel Deep-MVLM filters
    on and this architecture does not have.

So the spread of the logits is the only information present, and
heatmap_confidence already extracts it. There is no second channel that
was missed. Their docstring says as much -- "heuristic scale, not a
calibrated probability", written for temporal smoothing -- and it does what
it claims; this test asked it for something it never offered.

For Ear_Landmarker, if a usable per-point confidence is ever wanted it
needs an auxiliary term constraining amplitude (MSE or BCE on the heatmap
against a Gaussian) alongside the coordinate loss. Not proposed here, just
noted: the current one cannot be fixed by reading it better, because the
quantity is not in the model.

WHAT THIS MEANS FOR VIEW SELECTION: closed on renders. Ear_Landmarker
distinguishes DETECTION (separating good from bad, where their signal is
strong) from RANKING (ordering among bad points, where it decays to -0.125).
That distinction does not reach this case -- there is nothing to rank with
and the flag does not fire in the right places. Do not revisit without a
retrain changing the input.

STILL UNTESTED BY ANYONE: whether a softmax can report ABSENCE. All 57,970
of their points are visible, so their data cannot speak to it, and the claim
rests on the code alone. Keep it as reasoning, not measurement.

The cache is kept: conf is stored per (view, landmark) alongside the rays,
so any later idea that wants it needs no re-render.

Usage:
    python scripts/cache_conf.py --ears 20        # signal check
    python scripts/cache_conf.py --ears 112
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
    p.add_argument("--dataset", default="hutubs")
    p.add_argument("--grid", type=float, default=30.0)
    p.add_argument("--grid-step", type=float, default=15.0)
    p.add_argument("--out", default="runs/confcache")
    a = p.parse_args()

    import open3d as o3d
    from escape_audit import (DEFAULTS, WRAP, load_ear, ear_frame,
                              light_mirror_for, pick)
    from ear3d.camera import orbit_extrinsic
    from ear3d.frames import in_frame
    from ear3d.render import render
    from ear3d.twod import make_pipeline

    cfg = dict(DEFAULTS)
    pipe, ck = make_pipeline(cfg)

    # The side channel. predict() returns (lm, conf) when asked for
    # confidence; keep the last, which is the call whose landmarks survive
    # the refinement loop.
    stash = {}
    orig = pipe.landmarker.predict

    def wrapped(*args, **kw):
        out = orig(*args, **kw)
        if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
            stash["conf"] = np.asarray(out[1], float).reshape(-1)
        return out
    pipe.landmarker.predict = wrapped

    g = a.grid
    angles = [(float(y), float(q))
              for y in np.arange(-g, g + 1e-9, a.grid_step)
              for q in np.arange(-g, g + 1e-9, a.grid_step)]
    names = sorted(f.stem for f in (WRAP / __import__("ear3d.corpus", fromlist=["get"]).get("canon2scan")).glob("*.npz")
                   if str(np.load(f, allow_pickle=True)["dataset"]) == a.dataset)[:a.ears]
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"{ck.parent.name}/{ck.name}  {len(names)} ears x {len(angles)} views")

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
        V = len(angles)
        uv = np.full((V, 55, 2), np.nan); gt = np.full((V, 55, 2), np.nan)
        cf = np.full((V, 55), np.nan)
        o = np.full((V, 55, 3), np.nan); d = np.full((V, 55, 3), np.nan)
        for v, (yaw, pitch) in enumerate(angles):
            E = orbit_extrinsic(yaw, pitch, c)
            img, depth, cam = render(gh, c, E=E)
            gt[v] = cam.project(lm_f)
            stash.pop("conf", None)
            best = pick(pipe(img, timestamp=0.0), c, img.shape[:2])
            if best is None:
                continue
            uv[v] = np.asarray(best["landmarks"], float)
            if "conf" in stash and len(stash["conf"]) == 55:
                cf[v] = stash["conf"]
            ok = np.isfinite(uv[v]).all(1)
            if ok.any():
                oo, dd = cam.rays(uv[v][ok]); o[v, ok] = oo; d[v, ok] = dd
        np.savez_compressed(dst, uv=uv, gt_uv=gt, conf=cf, o=o, d=d,
                            lm_f=lm_f, scale=s, angles=np.array(angles), ear=nm)
        got = 100 * np.mean(np.isfinite(cf))
        print(f"[{i+1}/{len(names)}] {nm}: confidence captured on {got:.0f}% "
              f"of landmark-views", flush=True)


if __name__ == "__main__":
    main()
