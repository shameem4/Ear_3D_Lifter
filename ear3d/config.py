"""Shared defaults for the 3D path.

Every knob that affects geometry, appearance or the snaps lives here, so the
viewer and ingest cannot be running at different settings while appearing to
share code. Callers copy it -- dict(DEFAULTS, **overrides) -- and pass the
result down; no module reads it directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import corpus as _corpus
from .camera import cone_angles

ROOT = Path(__file__).resolve().parents[1]

# Third-party research data; not in any repo. Override with EAR3D_DIR.
EAR3D_DIR = Path(os.environ.get("EAR3D_DIR", ROOT.parent / "clean_3d_data"))

DEFAULTS = dict(
    # --- what to load -------------------------------------------------------
    mesh=str(EAR3D_DIR / "3D head meshes" / "pp12_3DheadMesh.ply"),
    ear=0,                  # which detected ear: 0 = highest confidence, 1 = other side

    # --- model --------------------------------------------------------------
    run=None,               # None FOLLOWS Ear_Landmarker's SHIPPED_RUN, which
                            # is the policy: take their defaults, send changes
                            # upstream. Naming a run here pins us to a model
                            # they may have superseded -- theirs moved from
                            # manual_occ_s42 to fixed_conv_s42 on 2026-10-14
                            # and this would not have noticed. Set it to a run
                            # name, or cfg["ckpt"] to a path, or env
                            # EAR3D_CKPT (path or run; wins over both), when a
                            # measurement needs one model held still. See
                            # twod.landmarker_checkpoint; results record it.
                            # This model is for a scan's OWN texture (photo-like).
    render_run="r3p_s42",   # the model for SYNTHETIC-SKINNED renders, i.e. every
                            # mesh without a texture (twod.render_checkpoint).
                            # Render-only, never for photographs. r3p = r2j's recipe
                            # (synthetic_v3r + jewellery, from v4) plus pinna-only
                            # renders (synthetic_v3p). vs r2j_s42 on the user's 32
                            # hand-corrected held-out ears: heads 1.089 -> 1.031 mm;
                            # ear-only fixed (AudioEar3D lm 0-2 5.39 -> 2.16 vs v12,
                            # head-cut ears 1.50 -> 1.10); lm 26-28 +0.1 mm. One
                            # seed. User's choice, 2026-10-25 (r2j was 2026-10-24).
    device="cuda",

    # --- the 2D detector, on RENDERS ----------------------------------------
    # These three defaults are measured on renders and are WRONG for
    # photographs. They live here because this config is the render path.
    #
    # Ground truth for all of it: 168 renders with boxes projected from
    # wrap-fitted 3D landmarks (archive-final:archive/scripts/make_detector_testset.py), plus 112
    # ears x 49 views scored against the wrap (scripts/eval/escape_audit.py).
    #
    #   config                              detect@IoU>=0.3   @0.5
    #   two-stage, shipped threshold              56.5%      54.8%
    #   full-frame, shipped threshold             66.1%      64.9%
    #   two-stage,  conf 0.05 + best-pick         89.3%      83.3%
    #   full-frame, conf 0.05 + best-pick         97.6%      95.8%
    #
    # The detector was never blind to these ears: over all 3584 raw anchors
    # EVERY one of the 168 renders carries a box at IoU>=0.5 (median 0.742).
    # It put a correct box down and scored it under the operating point --
    # 52 of 168 images have a correct box scoring below 0.70. The cause is
    # framing: BlazeEar trained on ears at a median 0.223 of frame height and
    # this path runs them at 0.774, above its 99.6th percentile. Cropping real
    # PHOTOGRAPHS to this framing makes the model worse on photographs (25.0%)
    # than it is on these renders (62.5%), so this is not a render-domain gap
    # and fine-tuning on renders is not indicated.
    detector_conf=0.05,     # NOT a quality gate, and dropping it is not free.
                            # On the views the shipped threshold already
                            # accepted, low and shipped are identical (2.62% vs
                            # 2.66% of landmarks >10 mm off the ear), so this
                            # only ADDS views. The added ones are worse -- 12.25%
                            # against 2.62% -- but there are many more of them:
                            # counting landmarks that pass the on-ear filter and
                            # land within 15 mm of their own ground-truth point,
                            # 93,736 -> 204,810, a 2.18x yield at a 2.57% ->
                            # 4.92% bad rate. Raise it toward 0.70 when a clean
                            # label matters more than a plentiful one.
                            #
                            # CORROBORATED UPSTREAM 2026-10-15. Ear_Landmarker
                            # made the gate an explicit consumer choice rather
                            # than a constant (resolve_confidence, profiles
                            # interactive 0.70 / fewer 0.76 / fewest 0.86) and
                            # sanctions a raw float for a consumer who has
                            # swept its own corpus, which is what this is. They
                            # then rendered 64 sampled "false positives" and
                            # looked: 64 of 64 are REAL EARS their labels do
                            # not mark, so the precision case for a high gate
                            # collapsed and the optimum moves DOWN. An
                            # `automated` profile was withdrawn for that
                            # reason. Their small-ear tilt -- 0.70 to 0.86
                            # discards 45.8% of ears under 32 px against 12.5%
                            # over 80 px -- does not bite here, because these
                            # renders frame the ear at 0.774 of frame height;
                            # it is why their gate advice does not transfer
                            # either way.
    detector_full_frame=True,
                            # Skip BlazeFace and run the ear model on the whole
                            # frame. The face stage exists to make a SMALL ear
                            # bigger; an ear-centred render has no face in it to
                            # crop to. Worth 97.6% against 89.3%.
    select="central",       # Which detection to lift when several come back --
                            # see label.pick(). Only valid because this path
                            # frames the ear centrally on purpose.

    # --- head pose ----------------------------------------------------------
    face_task=os.environ.get(
        "FACE_LANDMARKER_TASK",
        "/mnt/14BE47C2BE479ADE/Code/landmarking_stuff/landmarker/face_landmarker.task"),

    # --- appearance ---------------------------------------------------------
    skin=True,              # colour the mesh with ear3d/skin.py before rendering
    tone=10,                # tone index; None draws a random Fitzpatrick I-VI.
                            # 10 is deeper than real skin and is there because the
                            # landmarker reads the ear most cleanly on it -- see
                            # EXTRA_TONES in ear3d/skin.py. Fine here, where the
                            # point is to inspect landmarks; ingest renders its
                            # training images with a realistic tone instead.
    skin_seed=0,            # also drives the blotching and grain
    skin_noise=True,        # blotching + grain on. The FACE stage needs them:
                            # clean skin turned 2/40 heads upside down.
    ear_tone=5,             # the EAR renders' skin, applied by
    ear_skin_noise=False,   # cache_ear_views. Clean tone 5 vs blotchy tone 10,
                            # 80 ears, GT check: median 3.42 vs 5.17 mm, >20 mm
                            # 0.3% vs 6.0%; clean tone 10 ties (3.56), tone 5
                            # better on 60/80 ears.
    ao_strength=0.75,       # how hard ambient occlusion darkens cavities

    # --- the sun ---
    # Direction the light TRAVELS. Only the x sign was ever in question and it
    # is worth 5.34% -> 3.24% of landmarks landing more than 10 mm off the ear;
    # render.sun_world() carries the table and the mechanism. The old value was
    # the x-negated one, hard-coded, for every ear.
    # THE 2D PIPELINE'S OWN TEST-TIME AUGMENTATION. Both are Ear_Landmarker's
    # to own; None means "take their shipping default" and is the policy.
    # These two reach only the EarLandmarkerPipeline path (label.py). The 3D
    # pipeline's bare LandmarkPredictor takes their FLIP_FUSION_DEFAULT (on)
    # and has no detector, so detector_frame_tta does not apply to it.
    # flip_fusion is left None deliberately -- they default it ON, and at the
    # new sun their in-pipeline fusion plus the flag below scores 1.290%,
    # identical to this project's own two-pass frame-level wrapper (p=0.76)
    # for one extra DETECTOR pass instead of two full pipeline passes. The
    # wrapper is retired on that.
    flip_fusion=None,
    detector_frame_tta=True,  # they ship it OFF; turned on here because the
                            # detector is presentation-sensitive on renders --
                            # without the union the chosen box moves a median
                            # 54 px on the same frame and the same ear
                            # (archive-final:archive/scripts/check_detector_symmetry.py). A union
                            # through NMS is symmetric, so it is in the class
                            # a single-presentation corpus cannot flatter.
                            # CONFIRMED at this sun, which was the open
                            # question: worth +0.224pp, 64 of 112 ears,
                            # p=2.4e-05, against +0.254pp at the old one. The
                            # fusion gains halved when the sun improved and
                            # this barely moved -- consistent with the routes
                            # being different things, since better lighting
                            # fixes a weak occluding edge and not which box
                            # the detector picks. At this sun the flag is
                            # worth MORE than the fusion it sits on top of
                            # (1.710 none, 1.514 fusion, 1.290 both).

    light_dir=(1.2, -0.4, -0.4),   # 2026-10-17 reboot. Side views of the
                            # normalised head, light from the face side via
                            # render.side_light (x flipped for the right ear).
                            # Depth component -0.9 -> -0.4, chosen on the two
                            # left ears BlazeEar missed (pp6, pp80); on all 40
                            # side views of 20 heads its top box lands on the
                            # ear 20/20 + 20/20 (was 19 + 20), median score
                            # 0.83/0.89 (was 0.79/0.87). The earlier value was
                            # swept for landmark accuracy in the old pipeline.
    face_light_dir=(1.2, -0.4, -0.4),  # the sun for head_pose's FACE renders
                            # only; light_dir above is the EAR light. Same
                            # value today: 40 heads x 5 suns, this one gave
                            # the best face frames (median 5.1 deg vs scan,
                            # 40/40 found; a camera-axis sun lost 4).
    target_occupancy=0.75,  # each ear's orbit is re-sized so the face-on
                            # landmarks span this share of the frame (the
                            # model's TRAIN_OCCUPANCY is 0.777; the BlazeEar box
                            # + ear_view.PAD alone gives ~0.5-0.6). Default since
                            # 2026-10-21 with view_table_v13, which was learned
                            # at it: held-out parity overall (-0.004 mm vs v12,
                            # p=0.92), pp16's attached lobe 9.7 -> 5.7 mm.
                            # Without a matching table it is WORSE (v7 +0.14).
                            # None restores box framing.
    core=os.environ.get("EAR3D_CORE", "python"),  # "python", or "rust": the whole pipeline in
                              # ear3d_core (core/), hosted by ear3d.core_host
    ear_detector="blazeear",  # pipeline ear box: "blazeear" (Apache-2.0) or "yolo"
                              # (Ultralytics, AGPL-3.0; local use only). 16 held-out
                              # heads: 1.68 vs 1.61 mm median vs v9, 32/32 ears both.
    use_mesh_texture=False, # pipeline: True renders a scan's own texture and reads
                            # it with the regular model (run); False re-skins every
                            # mesh and uses render_run. False since 2026-10-24: on
                            # three Artec textured heads (BJ030, CD022, CD048) the
                            # synthetic route was 2-9x steadier across 4 orientations
                            # (p95 spread 0.13-0.33 vs 0.35-1.55 mm), routes 1.1-1.9 mm
                            # apart; user's call after looking at the overlays.
    gt_lm3d=_corpus.get("gt_lm3d"),  # the REFERENCE 3D landmarks of the corpus
                            # generation (ear3d/corpus.py; lmw: lm3d_v12, marks
                            # re-derived on the landmark-posed fits). Was, in
                            # wrap_work/templates. v9 (2026-10-17, user's
                            # choice): curves from the shipped landmarker run
                            # multiview on ear_mean_v4, every strip re-spaced to
                            # the spacing it uses on real ears (80 HUTUBS ears);
                            # canal 39 unchanged. Model-derived, so the report
                            # also scores against the alternates below.
    gt_lm3d_also=(() if _corpus.CURRENT == "lmw" else   # v7/v8/v9 address the OLD fits:
                  ("lm3d_v7_canonical_mm.npz",          # human curves (v6_spacing)
                   "lm3d_v8_corpus_canonical_mm.npz")), # median real-ear predictions
    multiview_light_dir=(0.6, -0.4, -1.0),  # the sun for the 25-view orbit
                            # (scripts/caches/3_multiview_collect_rays.py), fixed to the
                            # head. 80 ears vs light_dir: table best-8 + re-seat
                            # 2.29 vs 2.39 mm, better on 51/80 (p=6e-3); >10 mm
                            # 0.2% vs 0.6%. NOT for the single face-on view,
                            # where it is worse (3.31 vs 3.16, p=6e-4).
    light_frame="world",    # "view" rotates the sun with the camera instead.
                            # Measured on the same grid: 3.45% against 3.24%,
                            # so world wins -- but only in the orbited views,
                            # since the two are identical face-on.
    light_mirror=False,     # flips the sun's x. LEAVE IT FALSE FOR EVERY EAR.
                            # c243e8f set it True for geometry that arrived
                            # through a reflection, to keep the sun's relation
                            # to the ANATOMY rather than to the frame, and
                            # measured a gain. That gain was compensation for a
                            # renderer bug: reflected heads were being shaded
                            # with their normals pointing inward (see
                            # escape_audit.load_ear), which is close to negating
                            # the light, so mirroring the sun cancelled it.
                            # With the winding fixed the mirror double-counts.
                            # Measured over 112 ears x 25 views, right ears,
                            # rate past 10 mm: mirrored 6.79%, not mirrored
                            # 3.17%, worse on 55 of 56 subjects. Left ears are
                            # not reflected and are untouched by the choice.
    ao_rays=24,             # rays per vertex. The cost of apply_skin is all here.

    # --- camera / framing (these mirror the training pipeline) --------------
    size=600,               # full render resolution before cropping
    eye_z=3.0,              # camera at (0, 0, eye_z) looking at the origin
    vfov=50.0,              # vertical field of view, degrees

    # NOTE there is no crop setting here, and the two 2D paths differ:
    # - the 3D pipeline (pipeline._multiview) feeds the WHOLE render to a bare
    #   LandmarkPredictor: no detector, no ROI, no occupancy step. Framing is
    #   set once per ear by ear_view (BlazeEar box + ear_view.PAD, then
    #   target_occupancy) and the
    #   orbit keeps it, so occupancy is ours: target_occupancy below.
    # - label.py (single-view viewer) uses EarLandmarkerPipeline, which runs
    #   their detect -> ROI -> refine path and drives occupancy itself.

    # --- back-projection ----------------------------------------------------
    snap=True,              # place landmarks that sit behind a depth cliff onto
                            # the near lip of that cliff. See snap_to_cliff().
    snap_radius=0.08,       # JITTER RANGE: the search window, as a fraction of
                            # ear extent -- ~35 px at the usual framing, a 71x71
                            # window. This is the whole safety bound: the further
                            # it reaches, the more a "snap" becomes a relocation
                            # of a landmark the model put in the wrong place.
                            # 8% is the knee. Over four heads, every one of the 13
                            # landmarks flagged bad is already fixed there, and
                            # going further fixes none and moves steadily more
                            # points that were never flagged:
                            #   radius   fixed   disturbed   moved >0.15
                            #      5%      11        13           10
                            #      8%      13        16           13
                            #     10%      13        20           17
                            #     15%      13        24           21
                            #     20%      13        31           28
                            # Note the flag behind "fixed" only catches
                            # z < median - 0.30, so it is blind to a landmark on
                            # the wrong anatomy at roughly the right depth. It
                            # cannot separate these radii on placement; nothing
                            # here can yet.
    # --- triangulation -----------------------------------------------
    tri_angles=tuple(cone_angles(24, 45.0)),
                            # 24 views, not 12. The view COUNT is what carries
                            # the placement: at a fixed cone, going 12 -> 24 buys
                            # about as much as widening 20 -> 45 does, and at 55
                            # deg with only 12 views the inner helix sinks into
                            # the head again. --tri-views N overrides it.
    tri_thresh=0.03,        # RANSAC inlier distance, as a fraction of ear extent.
                            # It was multiplied by a hardcoded 1.0 rather than the
                            # measured extent until this was fixed, so it did not
                            # mean what it said.
    tri_method="lsq",       # "lsq" over all rays, or "ransac".
                            #
                            # RANSAC STILL DOES NOT WIN, re-measured after the
                            # threshold above was fixed (12 views, 3 heads):
                            #   method          reproj px   helix z span   inliers
                            #   lsq                4.62        0.297         12.0
                            #   ransac t=0.03      4.66        0.344         11.6
                            #   ransac t=0.01      5.38        0.399          7.6
                            #   ransac t=0.06      4.61        0.297         12.0
                            # Every actual rejection hurts, on reprojection and on
                            # depth span alike, and RANSAC matches least squares
                            # exactly at the threshold where it keeps all 12 rays
                            # -- which is the threshold where it is least squares.
                            #
                            # There is nothing here to reject: ray residuals have
                            # a p99/median of 5.7, where a contaminated set would
                            # be over 10 and plain Gaussian noise about 3. The
                            # published method that relies on RANSAC renders ~100
                            # views across the whole head, including ones where a
                            # landmark is invisible and the detector guesses. This
                            # pipeline never builds that failure: the cone is
                            # bounded where everything stays visible, and a view
                            # with no detection is dropped rather than guessed.

    # --- multi-view agreement (--multiview) ---------------------------------
    mv_angles=((0, 0), (-25, 0), (25, 0), (-12, -12), (12, 12), (0, -20), (0, 20)),
                            # yaw/pitch to re-landmark from. Kept inside +/-25
                            # because beyond that the ear starts occluding itself
                            # and spread stops measuring placement and starts
                            # measuring visibility.

    # CHAIN PASS OFF. It works -- it cuts link-length CV by a further 20%, from
    # 0.220 to 0.176 -- but link CV is its own objective, so that only says the
    # optimiser runs. Against multi-view agreement, which it does not optimise,
    # it buys nothing: median 3.96% -> 3.89%, >5% count 68 -> 65 of 220 (inside
    # noise), and p90 gets WORSE, 7.33% -> 8.54%. Per head it is inconsistent --
    # pp10 improves 33 -> 16 while pp12 19 -> 25, pp11 7 -> 13 and pp16 9 -> 11
    # all worsen; one win carries the average.
    #
    # The cliff snap, by contrast, earns its place on that same independent
    # metric: >5% count 90 -> 68 and p90 8.68% -> 7.33%, better on 3 of 4 heads.
    #
    # Left in and reachable with chain=True. n=4 heads, and agreement is
    # precision not accuracy, so this is no evidence of benefit rather than
    # proof of harm -- but nothing measurable says the moves it makes are right.
    chain=False,            # second pass: fix landmarks that break link spacing
    chain_tol=1.0,          # only reached when chain=True. Flag a point whose two
                            # links deviate from the strip median by this much in
                            # total, as a fraction of that median. A clean point
                            # scores ~0.2; displaced ones measured 1.5-1.7.
    chain_radius=0.08,      # search window for the replacement, as a fraction of
                            # ear extent -- same safety bound as the cliff snap.
    chain_passes=4,         # one point re-placed per pass, worst first, so a
                            # corrected neighbour informs the next decision.
    snap_step=0.30,         # how big a depth step counts as a cliff, in mesh
                            # units with the ear spanning ~1. Scalp-behind-pinna
                            # measures 0.85-1.08; concha bowl structure varies by
                            # ~0.3, so this sits between them. Lower (0.15) drags
                            # concha points forward; higher (0.50) fixes nothing.

    # --- display ------------------------------------------------------------
    sphere_frac=0.022,      # landmark sphere radius, in mesh units (ear spans ~1)
    interactive=True,
)
