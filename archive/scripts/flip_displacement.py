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

MEASURED PROPERLY 2026-10-14, after a bug in THIS script was fixed: it passed
the view angles in the cfg dict, and render() ignores cfg["yaw"]/["pitch"] --
the view is an extrinsic, set by orbit_extrinsic. Every 25-view grid was the
face-on view 25 times, so every per-view figure this file used to print was
one measurement repeated. Caught by asking why 25 different pictures returned
a byte-identical displacement.

20 ears x 25 real views, model only (--self-crop --raw-predict, so no
detector, no square_roi_crop, no refinement), symmetric sun (--sun-x 0, so
mirroring the image is an exact symmetry of the shading):

    mean dx +1.142 px   mean dy -1.906 px   crop widths 89-998 px
    one constant in PIXELS removes  1.0% of the dx field,  6.5% of dy
    one constant in CROP WIDTHS    -0.2%                   8.1%
    a full affine map              79.9%   (theirs: 94.9% on photographs)

THE FIELD IS MOSTLY NOISE HERE, WHICH IS THE HEADLINE. A single constant
absorbs about one percent of dx. On Ear_Landmarker's photographs the same
statistic is essentially a rigid shift. So whatever systematic component
exists on renders sits under scatter that their domain does not have, and the
earlier "35.8% from a constant vector" came from the repeated-view bug.

THE DIFFICULTY CUT, which was pre-registered with its falsifier:

    quartile      NME      dx          dy
    Q1 easiest   0.0274   -0.00097   -0.00351
    Q2           0.0367   +0.00082   -0.00386
    Q3           0.0458   +0.00153   -0.00270
    Q4 hardest   0.1810   +0.01158   +0.00006

THE FALSIFIER FIRED. The stated test was "if dy is flat across quartiles the
domain reading is wrong and I will say so". dy is flat across Q1-Q3 and
vanishes in Q4, so "dy is incidental non-equivariance on inputs the model
handles badly" is REFUTED. dy is largest where the model does best, which is
the signature of a systematic offset rather than difficulty-driven noise.

dx does rise with difficulty, steeply, but Q4's NME is 0.18 of a crop width
-- the model has simply lost the ear there -- so that is disagreement
amplitude, not a systematic effect worth a mechanism.

AND THE WITHIN-EAR CLAIM WAS BACKWARDS. Reported earlier as dx being constant
within an ear and varying between ears, which was the repeated-view bug.
With real views the variation is mostly WITHIN an ear: dx within-ear sd
0.01336 against between-ear 0.00309, dy 0.00674 against 0.00217. Views, not
ears, carry it.

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
    from ear3d.camera import orbit_extrinsic
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
    names = sorted(f.stem for f in (WRAP / __import__("ear3d.corpus", fromlist=["get"]).get("canon2scan")).glob("pp*.npz"))[:a.ears]
    g, st = a.grid, a.grid_step
    angles = [(y, pi) for y in np.arange(-g, g + 1e-9, st)
              for pi in np.arange(-g, g + 1e-9, st)]

    rows = []
    for n, name in enumerate(names):
        try:
            head, ear, lm = load_ear(name, "hutubs", cfg)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"  {name}: skipped -- {e}"); continue
        Rm, C, s_ = ear_frame(head, lm, cfg)
        h = o3d.geometry.TriangleMesh(head)
        h.vertices = o3d.utility.Vector3dVector(
            (np.asarray(head.vertices) - C) @ Rm.T / s_)
        h.compute_vertex_normals()
        c = dict(cfg, light_mirror=light_mirror_for(name, a.light))
        lmf = (lm - C) @ Rm.T / s_
        for yaw, pitch in angles:
            # THE VIEW IS AN EXTRINSIC, NOT A CFG KEY. render() ignores
            # cfg["yaw"]/["pitch"] entirely -- audit_ear sets the view with
            # orbit_extrinsic and passes E. Putting the angles in the cfg
            # rendered the FACE-ON view every time, so a 25-view grid was one
            # view repeated 25 times and every per-view statistic in this
            # script was inflated 25-fold. Found 2026-10-14 by noticing that
            # all 25 views of an ear returned a byte-identical displacement
            # and an identical crop width.
            E = orbit_extrinsic(float(yaw), float(pitch), c)
            img, depth, cam = R3.render(h, c, E=E)
            gt_full = cam.project(lmf)
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
                    gt = gt_full - np.array([px, py], float)
                    e = float(np.linalg.norm(la - gt, axis=1).mean() / nn)
                    rows.append((la, lb, float(nn), name, e))
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
                gt = gt_full - np.array([px, py], float)
                e = float(np.linalg.norm(
                    np.asarray(A["landmarks"], float) - gt, axis=1).mean() / nn)
                rows.append((np.asarray(A["landmarks"], float),
                             np.asarray(B["landmarks"], float), float(nn), name, e))
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
            gt = gt_full
            e = float(np.linalg.norm(ua - gt, axis=1).mean() / W)
            rows.append((ua, ub, W, name, e))
        print(f"[{n+1}/{len(names)}] {name}: {len(rows)} views so far", flush=True)

    if not rows:
        sys.exit("no paired views")
    D = np.stack([(u - v) / w for u, v, w, _, _ in rows])          # (N, 55, 2), crop widths
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
    for ua, ub, w, _, _ in rows:
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
            for (ua, ub, cw, nm, _e) in rows:
                for j in range(len(ua)):
                    w.writerow([nm, f"{cw:.1f}", j,
                                f"{ua[j,0]-ub[j,0]:.4f}", f"{ua[j,1]-ub[j,1]:.4f}"])
        print(f"\n  wrote {p}")

    E = np.array([r[4] for r in rows])
    q = np.quantile(E, [0.25, 0.5, 0.75])
    print(f"\n  BY DIFFICULTY -- Ear_Landmarker's cut. Their bug-driven dx sat")
    print(f"  on the EASIEST samples and the fix cleared it there; what survived")
    print(f"  their fix lived in the hardest quartile. NME is this arm's own")
    print(f"  error against the projected ground truth, in crop widths.")
    print(f"    {'quartile':12s} {'n':>5s} {'NME':>8s} {'dx':>10s} {'dy':>10s}")
    for i, nm in enumerate(("Q1 easiest", "Q2", "Q3", "Q4 hardest")):
        m = ((E >= ([-np.inf] + list(q))[i]) &
             (E < (list(q) + [np.inf])[i]))
        if m.sum() < 3:
            continue
        print(f"    {nm:12s} {m.sum():5d} {E[m].mean():8.4f} "
              f"{D[m][..., 0].mean():+10.5f} {D[m][..., 1].mean():+10.5f}")

    print(f"\n  {'strip':16s} {'dx':>9s} {'dy':>9s}")
    for nm, idx in zip(("outer_helix", "inner_helix", "concha_border",
                        "superior_crus"), STRIPS):
        sl = list(range(*idx))
        print(f"  {nm:16s} {D[:, sl, 0].mean():+9.5f} {D[:, sl, 1].mean():+9.5f}")


if __name__ == "__main__":
    main()
