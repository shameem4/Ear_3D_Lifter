"""Which part of the render-to-photograph gap costs us accuracy?

The 2D model is 2.7 mm on renders and the oracle-2D ablation says the lift
is 0.36 mm, so everything left is the landmarker being out of domain. "Out
of domain" is not actionable until it names a cause, and several are
conflated: appearance (contrast, noise, sharpness, texture), occlusion
(hair, jewellery), framing, and viewpoint.

`data/photo_aug.py` separates one of those cleanly. It is PHOTOMETRIC AND
OCCLUSIVE ONLY, never geometric, with knobs fitted to measured photo
statistics -- so landmark coordinates are untouched and ground truth does
not move. Applying one knob at a time asks whether appearance is the gap.

TWO READOUTS, because each alone is ambiguous:

  2D error          the thing we care about. But an augmentation can cut
                    error by accident, or raise it simply by making the
                    image harder, which is degradation and not domain.
  conf-error rho    Ear_Landmarker's heatmap confidence tracks error at
                    -0.486 on photographs and -0.018 on our renders
                    (scripts/cache_conf.py). That collapse is a readout of
                    being out of domain that does not depend on reading the
                    error correctly.

So the signature of closing the gap is error DOWN and rho STRENGTHENING
together. Error down with rho flat is a lucky filter; rho up with error up
is not worth having.

Each view is rendered ONCE and every variant applied to that same image, so
the comparison is paired per landmark-view and costs one render.

Usage:
    python scripts/domain_probe.py --ears 10
"""

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

VARIANTS = {
    "render as-is":   dict(strength=0.0),
    "shading only":   dict(shading=1.0, texture=0.0, noise=0.0, occlude=0.0),
    "texture only":   dict(shading=0.0, texture=1.0, noise=0.0, occlude=0.0),
    "noise only":     dict(shading=0.0, texture=0.0, noise=1.0, occlude=0.0),
    "occlude only":   dict(shading=0.0, texture=0.0, noise=0.0, occlude=1.0),
    "all stages":     dict(),
    # CONTROL, and it earns its place. Stages 6 and 7 -- defocus, JPEG, and
    # the 8% greyscale draw -- are gated by no knob, so they fire in EVERY
    # arm below "render as-is". Measured on a flat test image they move
    # pixels by 72 (median seed, max-abs) where the named stages separate
    # the arms by only 33-59, and they are a no-op on just 6 seeds in 40.
    # Without this row an arm's score is mostly its tail draw and the
    # attribution to the named stage is unsupported.
    "tail only (ctrl)": dict(shading=0.0, texture=0.0, noise=0.0, occlude=0.0),
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ears", type=int, default=10)
    p.add_argument("--grid", type=float, default=30.0)
    p.add_argument("--grid-step", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reps", type=int, default=4,
                   help="draws per (view, variant). ONE IS NOT ENOUGH: the "
                        "ungated defocus/JPEG tail is the same magnitude as "
                        "the stage being attributed, so a single draw scores "
                        "the tail. Reps average it out within each arm.")
    a = p.parse_args()

    from PIL import Image
    from scipy import stats
    from escape_audit import (DEFAULTS, WRAP, load_ear, ear_frame, pick,
                              light_mirror_for)
    from ear3d.camera import orbit_extrinsic
    from ear3d.frames import in_frame
    from ear3d.render import render
    from ear3d.twod import make_pipeline
    sys.path.insert(0, str(ROOT / "data"))
    from photo_aug import photo_realism

    cfg = dict(DEFAULTS)
    pipe, ck = make_pipeline(cfg)
    stash = {}
    orig = pipe.landmarker.predict

    def wrapped(*ar, **kw):
        out = orig(*ar, **kw)
        if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
            stash["conf"] = np.asarray(out[1], float).reshape(-1)
        return out
    pipe.landmarker.predict = wrapped

    g = a.grid
    angles = [(float(y), float(q))
              for y in np.arange(-g, g + 1e-9, a.grid_step)
              for q in np.arange(-g, g + 1e-9, a.grid_step)]
    names = sorted(f.stem for f in (WRAP / "corpus/canon2scan").glob("pp*.npz"))[:a.ears]
    acc = {k: {"e": [], "c": [], "j": []} for k in VARIANTS}

    for i, nm in enumerate(names):
        c = dict(cfg, light_mirror=light_mirror_for(nm, "frame"))
        try:
            head, ear, lm = load_ear(nm, "hutubs", c)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"[{i+1}/{len(names)}] {nm}: skipped -- {e}"); continue
        R, C, s = ear_frame(head, lm, c)
        gh = in_frame(head, R, C, s)
        lm_f = ((lm - C) @ R.T) / s
        for yaw, pitch in angles:
            img, _, cam = render(gh, c, E=orbit_extrinsic(yaw, pitch, c))
            gt = cam.project(lm_f)
            base = Image.fromarray(img)
            for key, kw in VARIANTS.items():
              raw = kw.get("strength") == 0.0
              for rep in range(1 if raw else a.reps):
                # Seeded on the REP, not reset to a.seed per variant. The old
                # line reused one seed for every arm to make them paired, but
                # the occluder gate's probability is 0 when occlude=0, so that
                # gate consumes no draw and every later stage -- including the
                # ungated tail -- reads from a different stream position than
                # in an arm where it fired. The arms were never paired; they
                # were misaligned in a way that looked like pairing.
                rng = np.random.default_rng((a.seed, rep))
                im = base if raw else photo_realism(base, rng, **kw)
                stash.pop("conf", None)
                best = pick(pipe(np.asarray(im), timestamp=0.0), c, img.shape[:2])
                if best is None:
                    continue
                uv = np.asarray(best["landmarks"], float)
                cf = stash.get("conf")
                for j in range(55):
                    if cf is None or len(cf) != 55:
                        continue
                    acc[key]["e"].append(float(np.linalg.norm(uv[j] - gt[j])))
                    acc[key]["c"].append(float(cf[j])); acc[key]["j"].append(j)
        print(f"[{i+1}/{len(names)}] {nm}", flush=True)

    print(f"\nIS THE GAP PHOTOMETRIC? {len(names)} ears x {len(angles)} views")
    print("Closing it looks like error DOWN and rho STRENGTHENING together.")
    print("For scale: photographs read -0.486; plain renders -0.018.\n")
    print(f"  {'variant':16s} {'n':>7s} {'err p50':>8s} {'err p90':>8s} {'rho':>8s}")
    for key in VARIANTS:
        e = np.array(acc[key]["e"]); cc = np.array(acc[key]["c"])
        jj = np.array(acc[key]["j"])
        if not len(e):
            print(f"  {key:16s} {'no detections':>26s}"); continue
        r = [stats.spearmanr(cc[jj == j], e[jj == j]).statistic
             for j in range(55) if (jj == j).sum() > 20]
        print(f"  {key:16s} {len(e):7d} {np.median(e):8.2f} "
              f"{np.percentile(e,90):8.2f} {np.nanmedian(r):+8.3f}")


if __name__ == "__main__":
    main()
