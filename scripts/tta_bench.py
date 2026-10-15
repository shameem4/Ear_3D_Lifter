"""

EVERY NUMBER THIS SCRIPT HAS PRODUCED IS STALE, AND IT IS LEFT STALE ON
PURPOSE. The table was measured at sun x = -0.3, two sun changes ago, with
the NaN gate scoring off-head landmarks as successes and with reflected
heads rendered inside out. Three separate reasons, each worth more than the
differences between most of the arms.

The RANKING is not in danger -- rotation, scale and heatmap averaging lost
by margins far larger than any of those effects, and cost 2 to 6 forward
passes to do it -- and nothing since would revive them. It also no longer
decides anything: what this pipeline runs is settled by ear3d/config.py,
which takes Ear_Landmarker's in-pipeline flip fusion and adds their
detector_frame_tta, measured at the current sun on 112 ears.

So re-running nine arms to restate a ranking that already holds and drives
nothing was judged not worth the renders. If you need a number from here,
re-run it; do not quote the docstring.
Every test-time augmentation the pose literature documents, measured here.

The 2D model is the whole problem -- scripts/oracle_2d_lift.py shows the lift
reproduces ground truth to 0.36 mm when handed the right pixels, and never
misses by 20 mm. So the question is what can be bought at inference time
WITHOUT retraining, and the pose-estimation literature has a standard kit:

    flip TTA            mirror the input, map the answer back
    heatmap averaging   average the two arms' HEATMAPS, not their coordinates
    the one-pixel shift MMPose's shift_heatmap, to realign a flipped grid
    multi-scale TTA     several crop scales, merged
    rotation TTA        several in-plane rotations, merged

This runs each of them against the same grid and scores them the only way that
answers the question that matters here: the rate of GROSS errors, not a median.

TTA IS APPLIED INSIDE THE LANDMARKER, by patching LandmarkPredictor.predict,
which is where the literature applies it. That matters: the pipeline detects,
crops, and then REFINES the crop from its own landmark output, so a flip
applied to the whole frame changes the box as well as the points. Patching
predict leaves detection alone and isolates the landmarker, and the refinement
loop then runs on TTA'd landmarks like it would in production.

"hm_avg" IS LOGIT-SPACE, AND THAT IS NOT A DETAIL. `heatmaps()` returns
`to_heatmap(_decode(x))`, the raw conv output, and `soft_argmax` applies its own
softmax -- so averaging those tensors is a normalised GEOMETRIC mean of the two
distributions, which sharpens: a mode only one arm supports is suppressed rather
than averaged in.

The literature's prescription, taken literally -- average the probability maps
-- is a NO-OP here. soft_argmax returns the expectation of the spatial softmax,
expectation is linear, so decoding each map and averaging the coordinates gives
exactly the same answer as averaging the maps and decoding once. Ear_Landmarker
measured both on 861 photographs: 0.015139 and 0.015139, identical to every
digit printed. So "heatmap averaging beats coordinate averaging" cannot be true
for this decoder in the sense the sources mean it; only the logit-space variant
is a different operation at all.

Whether that variant is BETTER depends on which end of the error distribution
you are paid to care about. Paired over the same 55,000 renders, logit against
coordinate fusion:

    2D off-mask     26.23% vs 26.62%   -0.39pp   McNemar p=6.3e-15
    3D  >5mm         5.42% vs  5.45%             p=0.25
    3D >10mm         3.97% vs  3.95%             p=0.64
    3D >20mm         2.26% vs  2.20%             p=0.006   <- logit WORSE

Logit fusion puts significantly more landmarks on ear pixels and none of that
reaches gross 3D error; at the deepest tail it reverses. A mechanism that would
produce exactly this, and which Ear_Landmarker's corpus also shows the mean half
of: sharpening COMMITS to one mode where the two arms disagree about where the
rim ends, instead of landing between them. Committing wins a pixel-level on-ear
test and loses by the full inter-mode distance when it commits wrong.

So: logit fusion for a mean metric, coordinate averaging for a tail metric,
which is what this repository has. Measured under the stock sun, the regime
where the arms disagree most, so it needs re-measuring post-c243e8f.

THE ONE-PIXEL SHIFT IS NOT NEEDED HERE, and that is derivable rather than
measurable. This decoder is a soft-argmax over pixel CENTRES,
xs = (i + 0.5) / W, so flipping index i -> W-1-i sends the coordinate to
exactly 1 - x. MMPose needs shift_heatmap because its decoders use i / (W - 1)
or a hard argmax, where a flip lands half a pixel off. The 'shift' arm is here
to confirm that it hurts rather than helps.

Usage:
    python scripts/tta_bench.py --ears 40 --grid 15 --grid-step 15
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from escape_audit import WRAP, audit_ear, summarise                  # noqa: E402
from ear3d.config import DEFAULTS
from ear3d.escape import escape_rate, off_head                                    # noqa: E402
from ear3d.twod import BLAZEEAR_DIR, make_pipeline                   # noqa: E402

SIZE = 192                      # LANDMARKER_INPUT_SIZE
ARMS = ["none", "flip", "coord_avg", "hm_avg", "hm_avg_shift",
        "scale", "rot", "hm_avg_scale"]

# The DETECTOR's own TTA, which BlazeEar already ships for its two-stage path
# (utils/tta.py) and which this pipeline never reaches, because config.py runs
# the ear model on the whole frame instead. Its VIEWS are (face-crop expand,
# flip); the full-frame analogue of an expand is how much of the frame to keep,
# so the structure is preserved -- the default view first, the flip second, two
# scale views after -- and the fusion is BlazeEar's own merge_views, including
# its measured `max` score rule.
DET_VIEWS = {
    "none":   ((1.00, False),),
    "flip":   ((1.00, False), (1.00, True)),
    "views4": ((1.00, False), (1.00, True), (0.85, False), (1.15, False)),
}


def _prep(crop_rgb, device):
    """Exactly LandmarkPredictor.predict's preprocessing, kept in one place."""
    t = torch.from_numpy(np.ascontiguousarray(crop_rgb)).float().permute(2, 0, 1) / 255.0
    t = F.interpolate(t.unsqueeze(0), size=(SIZE, SIZE), mode="bilinear",
                      align_corners=False)
    return ((t - 0.5) / 0.5).to(device)


def _rescale(crop, s):
    """The same view at a different occupancy, padded or centre-cropped.

    s > 1 zooms in. Returns (crop_s, offset) with offset the origin of crop_s
    inside the original crop, so x_orig = x_s + offset.
    """
    n = crop.shape[0]
    m = int(round(n / s))
    off = (n - m) / 2.0
    if m <= n:
        a = int(round(off))
        return crop[a:a + m, a:a + m], float(a)
    out = np.full((m, m, 3), 128, dtype=crop.dtype)
    a = (m - n) // 2
    out[a:a + n, a:a + n] = crop
    return out, float(-a)


def _rotate(crop, deg):
    """Rotate about the crop centre, grey fill -- the pad value the model saw."""
    import cv2
    n = crop.shape[0]
    M = cv2.getRotationMatrix2D((n / 2.0 - 0.5, n / 2.0 - 0.5), deg, 1.0)
    return cv2.warpAffine(crop, M, (n, n), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128)), M


def _zoom(frame, frac):
    """A centred window of the frame, padded when frac > 1. Returns (img, off).

    The detector resizes whatever it is given to its own input size, so taking
    a smaller window IS a zoom; nothing here has to resample.
    """
    h, w = frame.shape[:2]
    n, m = int(round(w * frac)), int(round(h * frac))
    if n == w and m == h:
        return frame, (0.0, 0.0)
    if frac < 1.0:
        a, b = (w - n) // 2, (h - m) // 2
        return frame[b:b + m, a:a + n], (float(a), float(b))
    a, b = (n - w) // 2, (m - h) // 2
    out = np.full((m, n, 3), 128, dtype=frame.dtype)
    out[b:b + h, a:a + w] = frame
    return out, (float(-a), float(-b))


def make_detect(det, views):
    """EarDetector.detect with BlazeEar's view fusion wrapped around it."""
    import torch as _t
    sys.path.insert(0, str(BLAZEEAR_DIR))
    from utils.tta import merge_views                              # noqa: E402
    orig = det.detect

    def detect(frame_rgb):
        B, S = [], []
        for frac, flip in views:
            img, off = _zoom(frame_rgb, frac)
            ox, oy = off
            if flip:
                img = np.ascontiguousarray(img[:, ::-1])
            d = orig(img)
            if not len(d):
                continue
            b, sc = np.asarray(d[:, :4], float).copy(), np.asarray(d[:, 4], float)
            if flip:                      # ymin xmin ymax xmax, mirror x
                W = img.shape[1]
                b[:, [1, 3]] = W - b[:, [3, 1]]
            b += np.array([oy, ox, oy, ox], float)
            B.append(_t.as_tensor(b)); S.append(_t.as_tensor(sc))
        if not B:
            return np.zeros((0, 5))
        fb, fs = merge_views(B, S, len(views), rule="max")
        return np.c_[fb.numpy(), fs.numpy()]
    return detect


def make_predict(pred, arm, scales=(0.9, 1.0, 1.1), rots=(-10.0, 0.0, 10.0)):
    """A drop-in replacement for LandmarkPredictor.predict running one arm."""
    model, dev, tau = pred.model, pred.device, getattr(pred.model, "tau", 1.0)
    has_hm = hasattr(model, "heatmaps")

    @torch.no_grad()
    def predict(crop_rgb, with_confidence=False):
        n = crop_rgb.shape[0]
        sc = float(n) / SIZE                     # model pixels -> crop pixels

        def coords(t):
            """(55,2) in crop pixels, plus per-point confidence."""
            if has_hm:
                hm = model.heatmaps(t)
                out = model.soft_argmax(hm, tau)
                c = model.heatmap_confidence(hm, tau)
                return (out.cpu().numpy().reshape(55, 2) * n,
                        c.cpu().numpy().reshape(-1))
            out = model(t)
            return out.cpu().numpy().reshape(55, 2) * n, None

        def flipped_heatmaps(t):
            """Heatmaps from the mirrored input, flipped back into alignment."""
            return torch.flip(model.heatmaps(torch.flip(t, dims=[3])), dims=[3])

        t = _prep(crop_rgb, dev)

        if arm == "none":
            lm, c = coords(t)

        elif arm in ("flip", "coord_avg"):
            tf = _prep(crop_rgb[:, ::-1], dev)
            lm_f, c_f = coords(tf)
            lm_f[:, 0] = n - lm_f[:, 0]          # pixel-centre decode: W - x
            if arm == "flip":
                lm, c = lm_f, c_f
            else:
                lm_a, c_a = coords(t)
                lm = (lm_a + lm_f) / 2.0
                c = None if c_a is None else (c_a + c_f) / 2.0

        elif arm in ("hm_avg", "hm_avg_shift"):
            if not has_hm:
                raise SystemExit("this checkpoint has no heatmap head")
            hm = model.heatmaps(t)
            hmf = flipped_heatmaps(t)
            if arm == "hm_avg_shift":
                # MMPose's shift_heatmap, included to show it is wrong here.
                hmf = torch.roll(hmf, shifts=1, dims=3)
                hmf[..., 0] = hmf[..., 1]
            hm = (hm + hmf) / 2.0
            lm = model.soft_argmax(hm, tau).cpu().numpy().reshape(55, 2) * n
            c = model.heatmap_confidence(hm, tau).cpu().numpy().reshape(-1)

        elif arm in ("scale", "hm_avg_scale"):
            acc, cs = [], []
            for s in scales:
                cr, off = _rescale(crop_rgb, s)
                m = cr.shape[0]
                ts = _prep(cr, dev)
                if arm == "hm_avg_scale" and has_hm:
                    h = (model.heatmaps(ts) + flipped_heatmaps(ts)) / 2.0
                    u = model.soft_argmax(h, tau).cpu().numpy().reshape(55, 2) * m
                    cc = model.heatmap_confidence(h, tau).cpu().numpy().reshape(-1)
                else:
                    u, cc = coords(ts)
                    u = u / n * m                # coords() scaled by n, not m
                acc.append(u + off)
                cs.append(cc)
            lm = np.mean(acc, axis=0)
            c = None if cs[0] is None else np.mean(cs, axis=0)

        elif arm == "rot":
            acc, cs = [], []
            for d in rots:
                cr, M = _rotate(crop_rgb, d)
                u, cc = coords(_prep(cr, dev))
                Mi = np.vstack([M, [0, 0, 1]])
                Mi = np.linalg.inv(Mi)[:2]
                acc.append(u @ Mi[:, :2].T + Mi[:, 2])
                cs.append(cc)
            lm = np.mean(acc, axis=0)
            c = None if cs[0] is None else np.mean(cs, axis=0)

        else:
            raise SystemExit(f"unknown arm {arm!r}")

        lm = lm.astype(np.float32)
        return (lm, c) if with_confidence else lm

    _ = sc if False else None                   # keep the symbol documented
    return predict


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ears", type=int, default=40)
    p.add_argument("--grid", type=float, default=15.0)
    p.add_argument("--grid-step", type=float, default=15.0)
    p.add_argument("--arms", nargs="*", default=ARMS)
    p.add_argument("--det-tta", choices=sorted(DET_VIEWS), default="none",
                   help="detector-side TTA, fused with BlazeEar's merge_views")
    p.add_argument("--out", default=None, help="directory for per-arm csvs")
    a = p.parse_args()

    cfg = dict(DEFAULTS)
    cfg["interactive"] = False
    pipe, ck = make_pipeline(cfg)
    print(f"landmarker {ck.name}  "
          f"heatmap head: {hasattr(pipe.landmarker.model, 'heatmaps')}")

    ears = sorted(f.stem for f in (WRAP / "corpus/canon2scan").glob("pp*.npz"))[:a.ears]
    g = int(a.grid // a.grid_step)
    angles = [(y * a.grid_step, p_ * a.grid_step)
              for y in range(-g, g + 1) for p_ in range(-g, g + 1)]
    print(f"{len(ears)} ears x {len(angles)} views x 55 points "
          f"= {len(ears) * len(angles) * 55} landmark-views per arm\n")

    if a.det_tta != "none":
        pipe.detector.detect = make_detect(pipe.detector, DET_VIEWS[a.det_tta])
        print(f"detector TTA: {a.det_tta} {DET_VIEWS[a.det_tta]}\n")
    orig = pipe.landmarker.predict
    out = {}
    for arm in a.arms:
        pipe.landmarker.predict = (orig if arm == "none"
                                   else make_predict(pipe.landmarker, arm))
        rows = []
        for i, name in enumerate(ears):
            try:
                rows += audit_ear(name, cfg, pipe, angles, mirror="none")
            except (FileNotFoundError, KeyError, ValueError) as e:
                print(f"  {name}: skipped -- {e}")
        out[arm] = rows
        d = np.array([float(r["d_snap_mm"]) for r in rows
                      if r.get("detected") and r.get("j", -1) >= 0])
        # An off-head miss is an escape at every gate, not a missing row --
        # ear3d.escape has the argument. The arms do not miss equally, so the
        # old `d > t` here flattered whichever arm threw the most landmarks
        # clean off the head.
        print(f"  {arm:14s} n={len(d):6d}  >5mm {100 * escape_rate(d, 5):5.2f}%  "
              f">10mm {100 * escape_rate(d, 10):5.2f}%  "
              f">20mm {100 * escape_rate(d, 20):5.2f}%  "
              f"off-head {100 * off_head(d):4.2f}%", flush=True)
        if a.out:
            import csv
            od = pathlib.Path(a.out); od.mkdir(parents=True, exist_ok=True)
            keys = sorted({k for r in rows for k in r})
            with open(od / f"{arm}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    pipe.landmarker.predict = orig

    print("\n" + "=" * 78)
    for arm in a.arms:
        print(f"\n########## {arm}")
        print(summarise(out[arm]))


if __name__ == "__main__":
    main()
