"""The as-is arm minus the mirrored arm, as a DISPLACEMENT FIELD.

Every other measurement in this workstream scores the two test-time arms
against ground truth and compares the scores. This one subtracts the arms from
each other, which drops the target out entirely: whatever the labels are, the
difference between what the model predicts on an image and what it predicts on
that image's mirror is a property of the MODEL.

WHAT IT IS TESTING. Ear_Landmarker traced its right-favoured bias to a half
pixel: preprocessing normalises labels as px/W (corner convention) while the
heatmap head decodes a soft-argmax over pixel centres, (i+0.5)/W, and the flip
augmentation's label transform x -> 1-x is exact only for centres. Under the
corner convention a mirror sends x to 1-x-1/W, so at flip_prob=0.5 the model
splits the difference and carries a permanent ANTI-EQUIVARIANT x offset of
about 0.5/W. Their nine-checkpoint estimate is -0.0023 in normalised crop
units, against 0.5/256 = 0.00195 predicted.

If that is the mechanism, then on renders -- a domain they cannot reach -- the
arm difference must be a near-pure TRANSLATION in x of about 1/W, not an
anatomical pattern. So this reports the field, not a rate:

    dx, dy        the mean displacement, in normalised crop widths
    translation   how much of the field a single constant vector explains
    affine        how much a full affine map explains, their figure to beat
    per strip     because an anatomical story predicts structure here

A near-pure x translation of the predicted size is a third confirmation from a
third domain. Anything else is not.

MEASURED, 12 ears x 25 views, both arms pinned to exactly mirrored crops:

                        anatomy sun     symmetric sun (--sun-x 0)
    mean dx             +0.00518        +0.00532
    consistent sign     42% of views    83% of views
    left ears           -0.00514        +0.00241
    right ears          +0.01550        +0.00824
    constant vector       1.0%            3.1%
    full affine          52.9%           66.1%     (theirs 94.9%)

PIN THE CROP OR MEASURE NOTHING. Free-running, each arm detects its own box and
that scatter swamps the effect -- per-view dx scatter exceeded its own mean and
dy read +0.0048, which collapsed to +0.0002 once the crops were pinned.

PARTIAL CONFIRMATION, AND THE PART THAT FAILS MATTERS. Under symmetric lighting
there is a consistently signed offset of +0.00532 against 0.0039-0.0046
predicted, which is close for a prediction made from a half-pixel in another
repository's preprocessing. But the SIGN IS OPPOSITE to theirs (-0.0046) with
the same un-mirror convention, and the field is not a translation: 3.1% from a
constant vector against their 94.9% affine, with structure that survives the
lighting control (superior_crus +0.0111, concha_border -0.0028). The first
thing to suspect is this pipeline's own crop/resize/uncrop chain carrying a
half-pixel in a fixed direction, which would sit on top of theirs and could
outweigh it -- in which case the magnitude agreement is partly luck.

THE CROP CHAIN IS NOT REMOVED BY PINNING THE DETECTOR, which is what --pin
alone does and why --self-crop exists. Two things in the pipeline sit between
the pinned box and the model:

    refine_roi DEFAULTS TO TRUE, and it re-derives the ROI from the MODEL'S
    OWN LANDMARKS for up to ROI_MAX_REFINE passes. Any asymmetry feeds back,
    so the two arms can settle on different crops entirely. Pinning the
    detector box does nothing about it.

    square_roi_crop places the window at int(round(cx - side/2)). The as-is
    arm rounds cx - s/2 and the mirrored arm rounds W - cx - s/2, and those
    agree only when the fractional parts cooperate. For an odd-sized window
    the centre lands half a pixel off the image centre, and the mirror does
    not cancel it.

--self-crop cuts an integer square window here, mirrors it here, pins the
detector to the whole window and turns refinement off, so the two inputs are
exact mirrors by construction.

    measured, 12 ears x 25 views    pinned box    --self-crop
    mean dx                          +0.00532      +0.00131
    per-view scatter                 +/-0.00750    +/-0.00356
    constant vector                    3.1%         17.4%
    full affine                       66.1%         71.5%

Taking the chain out moves everything toward a translation, which says a good
part of the pinned-box number was this pipeline and not the model. It does
not get there: a constant vector still explains only 17.4% against their
94.9% affine on photographs.

--self-crop IS STILL NOT THE MODEL ALONE. square_roi_crop runs inside the
pipeline regardless, re-cropping at 1.6x with int(round(cx - side/2)); for an
odd window that sits half a pixel off centre, so the mirrored arm's input is
the mirror SHIFTED BY A PIXEL. --raw-predict calls LandmarkPredictor.predict
directly and removes the last of it. 6 ears, 18 views, no crop code at all:

    ear           n      dx px    dy px      dx/n       dy/n
    pp10_left    609    +3.943   -0.157   +0.00647   -0.00026
    pp10_right   839    +4.336   -7.914   +0.00517   -0.00943
    pp11_left    820    +1.171   -7.687   +0.00143   -0.00937
    pp11_right   638    +2.818   -1.826   +0.00442   -0.00286
    pp12_left    642    -0.332   -1.072   -0.00052   -0.00167
    pp12_right   648    +0.903   -2.837   +0.00139   -0.00438
    overall      dx +0.00306 widths    dy -0.00466 widths

dx goes UP when the crop code is removed (+0.00131 through square_roi_crop,
+0.00306 without it), so some of the chain's asymmetry was cancelling some of
the model's. Two wrongs, partly.

THE RESIZE IS NOT THE CULPRIT EITHER. F.interpolate(bilinear,
align_corners=False) maps dst to (dst+0.5)*scale - 0.5, which is
mirror-symmetric at any input size. Max |flip(resize) - resize(flip)| over
random crops at n = 187, 192, 193, 256, 383, 384, 609, 610, 839 is 1e-7 to
6e-5 with no parity structure, so a mirrored odd-width crop resizes to the
mirror of the resize.

THE y COMPONENT IS THE PART NO CONVENTION EXPLAINS. Under --self-crop the two
inputs are exact mirrors and the un-mirror touches x only, so the model's y
output should be identical between the arms. It is not: dy is -0.0078 crop
widths, around 1% of the crop and comparable to dx, and -0.00466 with no
crop code at all. A half-pixel in an x labelling convention cannot produce
it, the crop arithmetic cannot, and neither can the resize. dy is also immune
to the un-mirror convention by construction, since only x is transformed.
This model's response to a mirrored input differs in a direction the mirror
does not act in.

IT IS A DOMAIN EFFECT, not a contradiction of Ear_Landmarker. Their dy_anti
on photographs is -0.000234 and sign-inconsistent across subsets, about 20x
smaller on the SAME checkpoint. Normalisation does not explain it in the
direction needed: this crop is 1.6x the ear box, so a tighter crop would make
their figure larger, not smaller. [[head1-is-render-domain]] already has this
pipeline 5x worse on renders than on photographs, and out of domain is
exactly where a network's incidental asymmetries stop being suppressed.

WHAT THE LIGHTING CONTROL SETTLED. Under the anatomy sun the two sides had
OPPOSITE-signed offsets, which read as chirality; symmetric lighting removes
the reversal and leaves both positive. That reading was the renderer. A smaller
side difference does survive (+0.0024 against +0.0082) with both ears presented
in left form, and is unexplained.

Usage:
    python scripts/flip_displacement.py --ears 10 --grid 30 --grid-step 15
"""

import argparse
import pathlib
import sys

import numpy as np

ROI_EXPAND_HERE = 1.6          # the pipeline's own ROI_EXPAND, for --self-crop

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ears", type=int, default=10)
    p.add_argument("--grid", type=float, default=30.0)
    p.add_argument("--grid-step", type=float, default=15.0)
    p.add_argument("--light", default="anatomy")
    p.add_argument("--same-crop", action=argparse.BooleanOptionalAction, default=True,
                   help="pin the mirrored arm to the mirror of the as-is arm's box")
    p.add_argument("--out", default=None, metavar="CSV",
                   help="per-point displacements, so the field can be re-cut "
                        "without re-rendering")
    p.add_argument("--raw-predict", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="with --self-crop, call LandmarkPredictor.predict "
                        "directly so no crop code runs at all")
    p.add_argument("--self-crop", action=argparse.BooleanOptionalAction, default=False,
                   help="cut and mirror an integer square window here, with the "
                        "detector pinned to it and ROI refinement off, so the "
                        "two inputs are exact mirrors and only the model differs")
    p.add_argument("--int-box", action=argparse.BooleanOptionalAction, default=False,
                   help="round the pinned box to integers so the crop and its "
                        "mirror are exact; isolates the crop chain's own "
                        "sub-pixel from the model's")
    p.add_argument("--sun-x", type=float, default=None,
                   help="override the sun's x. 0.0 makes the lighting symmetric "
                        "about the image's vertical axis, so mirroring the image "
                        "is an exact symmetry of the SHADING too -- the control "
                        "that separates the model from the renderer")
    a = p.parse_args()

    import importlib
    from escape_audit import (WRAP, DEFAULTS, load_ear, ear_frame, pick,
                              _unmirror, light_mirror_for)
    from ear3d.twod import make_pipeline
    from ear3d.scheme import STRIPS
    R3 = importlib.import_module("ear3d.render")
    import open3d as o3d

    cfg = dict(DEFAULTS)
    if a.sun_x is not None:
        d = tuple(cfg["light_dir"])
        cfg["light_dir"] = (a.sun_x, d[1], d[2])
        print(f"sun {cfg['light_dir']}  (x overridden)")
    pipe, _ck = make_pipeline(cfg)
    _orig_detect = pipe.detector.detect
    names = sorted(f.stem for f in (WRAP / "corpus/canon2scan").glob("pp*.npz"))[:a.ears]
    g, st = a.grid, a.grid_step
    angles = [(y, pi) for y in np.arange(-g, g + 1e-9, st)
              for pi in np.arange(-g, g + 1e-9, st)]

    rows = []
    for n, name in enumerate(names):
        try:
            head, ear, lm = load_ear(name, "hutubs", cfg)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"  {name}: skipped -- {e}"); continue
        Rm, C, s = ear_frame(head, lm, cfg)
        h = o3d.geometry.TriangleMesh(head)
        h.vertices = o3d.utility.Vector3dVector(
            (np.asarray(head.vertices) - C) @ Rm.T / s)
        h.compute_vertex_normals()
        c = dict(cfg, light_mirror=light_mirror_for(name, a.light))
        for yaw, pitch in angles:
            img, depth, cam = R3.render(h, dict(c, yaw=float(yaw), pitch=float(pitch)))
            hw = img.shape[:2]
            A0 = pick(pipe(img, timestamp=0.0), c, hw)
            if A0 is None:
                continue
            if a.self_crop:
                # EVERY ROUNDING TAKEN OUT OF THE LOOP. The pipeline crops with
                # x1 = int(round(cx - side/2)), so the as-is arm rounds
                # cx - s/2 and the mirrored arm rounds W - cx - s/2; those are
                # equal only when the fractional parts of the centre and the
                # side happen to cooperate, so the two crops are mirrors to
                # within a pixel rather than exactly. Worse, refine_roi
                # re-derives the ROI from the model's OWN landmarks for up to
                # ROI_MAX_REFINE passes, so an asymmetry feeds back and the two
                # arms can settle on different crops entirely.
                #
                # So: cut an integer square window here, mirror it here, pin
                # the detector to the whole window and turn refinement off.
                # The two inputs are then exact mirrors by construction and
                # nothing but the model can tell them apart.
                y0, x0, y1, x1 = (float(v) for v in A0["bbox"])
                side = max(x1 - x0, y1 - y0) * ROI_EXPAND_HERE
                nn = int(round(side))
                px = int(round((x0 + x1) / 2 - side / 2))
                py = int(round((y0 + y1) / 2 - side / 2))
                if nn < 32:
                    continue
                # Pad rather than skip: at these framings the 1.6x window runs
                # off a 600 px render on most views, and dropping them would
                # keep only the ears that happen to sit small in frame. Grey
                # 128 is what the pipeline's own square_roi_crop pads with.
                pad = max(0, -px, -py, px + nn - img.shape[1], py + nn - img.shape[0])
                P = np.full((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad, 3),
                            128, np.uint8)
                P[pad:pad + img.shape[0], pad:pad + img.shape[1]] = img
                crop = np.ascontiguousarray(P[py + pad:py + pad + nn,
                                              px + pad:px + pad + nn])
                if crop.shape[0] != nn or crop.shape[1] != nn:
                    continue
                if a.raw_predict:
                    # THE LANDMARKER ALONE. square_roi_crop runs even in
                    # --self-crop mode -- the pipeline re-crops at 1.6x with
                    # int(round(cx - side/2)), and for an odd window that
                    # window sits half a pixel off centre, so the mirrored
                    # arm's input is the mirror SHIFTED BY A PIXEL. Calling
                    # the predictor directly removes the last of it: the model
                    # sees exactly this crop and exactly its mirror.
                    la = np.asarray(pipe.landmarker.predict(crop), float)
                    lb = np.asarray(pipe.landmarker.predict(
                        np.ascontiguousarray(crop[:, ::-1])), float).copy()
                    lb[:, 0] = nn - lb[:, 0]
                    rows.append((la, lb, float(nn), name))
                    continue
                box = np.array([[0.0, 0.0, float(nn), float(nn), 1.0]], float)
                keep = pipe.refine_roi
                pipe.refine_roi = False
                pipe.detector.detect = lambda _f, _b=box: _b
                try:
                    A = pick(pipe(crop, timestamp=0.0), c, (nn, nn))
                    B = pick(_unmirror(pipe(np.ascontiguousarray(crop[:, ::-1]),
                                            timestamp=0.0), nn), c, (nn, nn))
                finally:
                    pipe.detector.detect = _orig_detect
                    pipe.refine_roi = keep
                if A is None or B is None:
                    continue
                rows.append((np.asarray(A["landmarks"], float),
                             np.asarray(B["landmarks"], float), float(nn), name))
                continue
            if a.int_box:
                # THE CROP'S OWN HALF-PIXEL. The detector returns a float box.
                # Arm A crops at x0 and arm B at W - x1, and if anything in the
                # crop chain truncates, those two round in opposite directions
                # and leave a sub-pixel asymmetry that no amount of pinning
                # removes -- the same shape of defect Ear_Landmarker found in
                # their translation augmentation. Rounding the box to integers
                # first makes the crop and its mirror exact, so a difference
                # that survives this is the model's and not the chain's.
                W = img.shape[1]
                y0, x0, y1, x1 = (float(np.round(v)) for v in A0["bbox"])
                pipe.detector.detect = lambda _f, _b=np.array(
                    [[y0, x0, y1, x1, 1.0]], float): _b
                try:
                    A = pick(pipe(img, timestamp=0.0), c, hw)
                finally:
                    pipe.detector.detect = _orig_detect
                if A is None:
                    continue
            else:
                A = A0
            if a.same_crop:
                # THE DETECTOR IS THE NUISANCE. Each arm runs its own detection,
                # so the two arms are cropped differently and that difference
                # swamps the half-pixel effect being looked for -- free-running,
                # the per-view dx scatter is larger than its own mean. Pin the
                # mirrored arm to the EXACT mirror of the as-is arm's box, so
                # the only thing left between them is the model.
                W = img.shape[1]
                y0, x0, y1, x1 = A["bbox"]
                if a.int_box:
                    y0, x0, y1, x1 = (float(np.round(v)) for v in (y0, x0, y1, x1))
                box = np.array([[y0, W - x1, y1, W - x0, 1.0]], float)
                pipe.detector.detect = lambda _f, _b=box: _b
            try:
                B = pick(_unmirror(pipe(np.ascontiguousarray(img[:, ::-1]),
                                        timestamp=0.0), img.shape[1]), c, hw)
            finally:
                if a.same_crop:
                    pipe.detector.detect = _orig_detect
            if B is None:
                continue
            ua = np.asarray(A["landmarks"], float)
            ub = np.asarray(B["landmarks"], float)
            wa = abs(A["bbox"][3] - A["bbox"][1]); wb = abs(B["bbox"][3] - B["bbox"][1])
            W = 0.5 * (wa + wb)
            if not np.isfinite(W) or W <= 1:
                continue
            rows.append((ua, ub, W, name))
        print(f"[{n+1}/{len(names)}] {name}: {len(rows)} views so far", flush=True)

    if not rows:
        sys.exit("no paired views")
    D = np.stack([(u - v) / w for u, v, w, _ in rows])          # (N, 55, 2), crop widths
    print(f"\n{len(rows)} paired views, {D.shape[1]} points, "
          f"displacement in NORMALISED CROP WIDTHS\n")
    print(f"  mean dx {D[..., 0].mean():+.5f}   mean dy {D[..., 1].mean():+.5f}")

    # THE DISCRIMINATOR. A property of the MODEL lives in its own input space,
    # so it is constant in NORMALISED units and shrinks in pixels as the crop
    # grows. Arithmetic in the crop/resize chain lives in frame pixels, so it
    # is constant in PIXELS and shrinks in normalised units. Report both and
    # see which one holds still.
    Wv = np.array([r[2] for r in rows])
    pxd = D[..., 0].mean(1) * Wv
    nmd = D[..., 0].mean(1)
    cv = lambda v: float(np.std(v) / abs(np.mean(v))) if np.mean(v) else float("nan")
    print(f"  in crop PIXELS      {pxd.mean():+.3f} px   "
          f"spread/mean {cv(pxd):.3f}")
    print(f"  in CROP WIDTHS      {nmd.mean():+.5f}      "
          f"spread/mean {cv(nmd):.3f}   (crop widths {Wv.min():.0f}-{Wv.max():.0f} px)")
    print(f"  -> {'the crop chain' if cv(pxd) < cv(nmd) else 'the model'} "
          f"is the steadier home for it")
    print(f"  predicted 1/W from a half-pixel label convention: {1/256:+.5f} "
          f"(their dx_anti x 2 = -0.00460)")
    print(f"  per-view mean dx: {D[..., 0].mean(1).mean():+.5f} "
          f"+/- {D[..., 0].mean(1).std():.5f}   "
          f"same sign on {100*np.mean(D[..., 0].mean(1) < 0):.0f}% of views")

    # How much of the field is a single constant vector, and how much is affine?
    v = D.reshape(len(D), -1)
    tot = np.sum((D - 0) ** 2)
    res_t = np.sum((D - D.mean((0, 1))) ** 2)
    print(f"\n  a single constant vector explains "
          f"{100 * (1 - res_t / tot):.1f}% of the field")
    # affine: fit per view  u_b ~ M u_a + t, residual against the raw difference
    ra, rt = 0.0, 0.0
    for ua, ub, w, _ in rows:
        X = np.c_[ua, np.ones(len(ua))]
        M, *_ = np.linalg.lstsq(X, ub, rcond=None)
        ra += np.sum(((ub - X @ M) / w) ** 2); rt += np.sum(((ub - ua) / w) ** 2)
    print(f"  a full affine map explains {100 * (1 - ra / rt):.1f}% "
          f"(Ear_Landmarker measured 94.9% on photographs)")

    side = np.array([r[3].endswith("_left") for r in rows])
    print(f"\n  BY SIDE -- both are presented in LEFT form by ear_frame, so a")
    print(f"  presentation effect must agree and a chirality effect must not:")
    for nm, m in (("left ears", side), ("right ears", ~side)):
        v = D[m][..., 0].mean(1)
        print(f"    {nm:11s} dx {v.mean():+.5f} +/- {v.std():.5f}   "
              f"negative on {100*np.mean(v < 0):3.0f}% of {m.sum()} views")

    if a.out:
        import csv as _csv
        p = pathlib.Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["ear", "crop_px", "j", "dx_px", "dy_px"])
            for (ua, ub, cw, nm) in rows:
                for j in range(len(ua)):
                    w.writerow([nm, f"{cw:.1f}", j,
                                f"{ua[j,0]-ub[j,0]:.4f}", f"{ua[j,1]-ub[j,1]:.4f}"])
        print(f"\n  wrote {p}")

    print(f"\n  {'strip':16s} {'dx':>9s} {'dy':>9s}")
    for nm, idx in zip(("outer_helix", "inner_helix", "concha_border",
                        "superior_crus"), STRIPS):
        sl = list(range(*idx))
        print(f"  {nm:16s} {D[:, sl, 0].mean():+9.5f} {D[:, sl, 1].mean():+9.5f}")


if __name__ == "__main__":
    main()
