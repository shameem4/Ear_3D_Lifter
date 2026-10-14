#!/usr/bin/env python3
"""Render ears the detector SHOULD find, with ground-truth boxes, as a test set.

WHY THIS EXISTS. Roughly a third of HUTUBS subjects are invisible to the ear
detector when rendered: not at a bad angle, not at low resolution, not in one
stage mode -- invisible at every view tried, in both two-stage and full-frame,
under both the 128 px and 256 px checkpoints. The renders are clean, lit,
centred ears filling ~70% of frame height. That is a detector-side question, so
this writes a corpus BlazeEar can be measured on without needing any of the 3D
pipeline: a folder of PNGs and a manifest of ground-truth boxes.

THE BOXES ARE REAL GROUND TRUTH, not another model's output. Each ear's 55
landmarks are stored in canonical millimetres by the wrap fit (which sits on the
scan at 0.15-0.62 mm), so projecting them through the same camera that made the
render gives the ear's extent in pixels exactly. Nothing here is seeded from the
detector, which is the trap that would make the measurement circular.

The box is the landmarks' bounding box with a margin, because the 55-point
scheme traces the ear's contours and its hull is slightly inside the silhouette
-- the helix rim sits ON the boundary and the outer edge is a millimetre or so
beyond it. MARGIN_FRAC is applied for that, not to be generous.

    python scripts/make_detector_testset.py --ears 40 --out runs/detector_testset

Output: <out>/images/*.png and <out>/manifest.csv with, per render, the
ground-truth box and what the currently shipped detector did with it.

HUTUBS renders. They stay local and gitignored like everything else derived
from the scans; see CLAUDE.md.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
WRAP = ROOT.parent / "wrap_work"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ear3d import DEFAULTS                                        # noqa: E402
from ear3d.camera import orbit_extrinsic                          # noqa: E402
from ear3d.frames import in_frame                                 # noqa: E402
from ear3d.render import render                                   # noqa: E402

from escape_audit import ear_frame, load_ear                      # noqa: E402

# The 55 points trace contours that sit just inside the silhouette, so their
# hull understates the ear. 6% of the box's own size puts it on the outline.
MARGIN_FRAC = 0.06


def gt_box(lm_frame, cam, w, h):
    """Ground-truth ear box in pixels, from the wrap's own landmarks."""
    uv = cam.project(lm_frame)
    x0, y0 = uv.min(0)
    x1, y1 = uv.max(0)
    mx, my = MARGIN_FRAC * (x1 - x0), MARGIN_FRAC * (y1 - y0)
    return (max(0.0, x0 - mx), max(0.0, y0 - my),
            min(w - 1.0, x1 + mx), min(h - 1.0, y1 + my))


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ears", type=int, default=40, help="how many ears")
    p.add_argument("--out", default="runs/detector_testset")
    p.add_argument("--views", nargs="*", type=float, default=[0, -15, 15],
                   help="yaw offsets; pitch is held at 0 so the ear is upright "
                        "and plainly visible -- this set is about whether the "
                        "detector can see an easy ear, not a hard one")
    a = p.parse_args()

    from PIL import Image
    from ear3d.twod import BLAZEEAR_DIR, DETECTOR_WEIGHTS, EarDetector

    cfg = dict(DEFAULTS)
    cfg["interactive"] = False
    det = EarDetector(BLAZEEAR_DIR / DETECTOR_WEIGHTS, "cuda")

    out = pathlib.Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    names = sorted(f.stem for f in (WRAP / "corpus/canon2scan").glob("pp*.npz"))[:a.ears]

    rows = []
    for i, name in enumerate(names):
        try:
            head, ear, lm = load_ear(name, cfg=cfg)
        except Exception as e:                       # missing head mesh, etc.
            print(f"[{i + 1}/{len(names)}] {name}: skipped -- {e}")
            continue
        R, C, scale = ear_frame(head, lm, cfg)
        g = in_frame(head, R, C, scale)
        lm_f = ((lm - C) @ R.T) / scale
        for yaw in a.views:
            img, _, cam = render(g, cfg, E=orbit_extrinsic(yaw, 0.0, cfg))
            h, w = img.shape[:2]
            box = gt_box(lm_f, cam, w, h)
            b = det.detect(img)
            best, conf = None, 0.0
            if len(b):
                k = int(np.argmax(b[:, 4]))
                best, conf = tuple(float(v) for v in b[k, :4]), float(b[k, 4])
            fn = f"{name}_y{yaw:+03.0f}.png"
            Image.fromarray(img).save(out / "images" / fn)
            rows.append(dict(
                image=fn, ear=name, yaw=yaw,
                gt_x0=round(box[0], 1), gt_y0=round(box[1], 1),
                gt_x1=round(box[2], 1), gt_y1=round(box[3], 1),
                gt_w=round(box[2] - box[0], 1), gt_h=round(box[3] - box[1], 1),
                n_det=int(len(b)), conf=round(conf, 4),
                iou=round(iou(box, best), 4) if best else 0.0,
                hit=int(bool(best) and iou(box, best) >= 0.3)))
        print(f"[{i + 1}/{len(names)}] {name}: "
              f"{sum(r['hit'] for r in rows if r['ear'] == name)}/{len(a.views)} found")

    with open(out / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    hit = np.array([r["hit"] for r in rows])
    print(f"\n{len(rows)} renders, {len({r['ear'] for r in rows})} ears")
    print(f"recall at IoU>=0.3: {100 * hit.mean():.1f}%  ({hit.sum()}/{len(hit)})")
    per = {}
    for r in rows:
        per.setdefault(r["ear"], []).append(r["hit"])
    dead = [e for e, v in per.items() if not any(v)]
    allv = [e for e, v in per.items() if all(v)]
    print(f"ears found at EVERY view: {len(allv)}")
    print(f"ears found at NO view:    {len(dead)}  {sorted(dead)}")
    print(f"\nwrote {out}/manifest.csv")


if __name__ == "__main__":
    main()
