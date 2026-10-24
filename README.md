# Ear 3D Lifter

Head mesh in, 55 ear landmarks per ear out, in 3D.

The 55-point ear scheme is annotated on photographs, and a 2D model
([Ear_Landmarker](https://github.com/shameem4/Ear_Landmarker)) finds those
points in images. This repository puts them on a head scan: it renders the
ears, reads them with the 2D model from many views, and triangulates.

## Use it

    python scripts/run_pipeline.py head.obj [out.npz]

or `from ear3d.pipeline import Pipeline; Pipeline().run(mesh)`.

- **Input:** a head scan in millimetres, any orientation.
- **Two 2D models, chosen by texture.** A mesh without colour gets synthetic
  skin and is read by the render-only model `r2j_s42` (config `render_run`).
  A scan with its own texture (vertex colours or UV texture) is rendered with
  it and read by Ear_Landmarker's regular `SHIPPED_RUN`. `r2j` was trained
  only on this repo's renders and is 420% worse on photographs: never point it
  at a photo or a real texture.
- **Output:** `L`, `R` (55 x 3 each) in the input mesh's own coordinates,
  NaN where an ear or landmark was not found (`L` is the subject's left),
  and `ckpt`, the 2D checkpoint used.
- **Accuracy (untextured, r2j):** median **1.09 mm** per ear against the
  user's hand-corrected landmarks on 16 HUTUBS heads used nowhere in building
  it (32 ears), each randomly rotated; 0.85% of points beyond 5 mm (v4 was
  1.61 mm and 4.7%). The reference is one annotator's placement, which r2j was
  trained toward. With synthetic earrings added, 1.06 mm. About 15 s per head on one GPU. Validated only on HUTUBS
  and SONICOM scans (no hair); one real scan looked right.

Environment: the `mocap` conda env
(`/home/shameem/.conda/envs/mocap/bin/python`). Also needed: `Ear_Landmarker`
beside this repo (or `$EAR_LANDMARKER`), `BlazeEar` beside it (or
`$BLAZEEAR`) for the ear box, and a MediaPipe `face_landmarker.task`
(`$FACE_LANDMARKER_TASK`).

## How it works

    head scan
      -> orient: FaceMesh face frame, checked (skull above neck, symmetry,
         nose-ear distance), retried turned if a check fails
      -> per ear: BlazeEar box, re-sized so the ear fills the frame as in the
         2D model's training (target_occupancy), 25 views on a yaw x pitch grid
      -> 2D landmarks in every view (Ear_Landmarker)
      -> each landmark triangulated from its best 8 views (the view table),
         RANSAC, then re-seated on the head surface

`ear3d/pipeline.py` lists each stage with the measurement behind its setting.

The view table and the reference landmarks it is scored against both come
from a **corpus**: 476 head ears (HUTUBS, SONICOM) and 112 AudioEar3D ears,
each fitted with one ear template so that every landmark has a place on every
fit. Building it is `scripts/corpus` (12 numbered steps); rendering the views
and learning the table is `scripts/caches` (4 steps). Each folder's README is
the runbook.

## Layout

    ear3d/              the library
      pipeline.py         head mesh -> landmarks
      twod.py             the only coupling to Ear_Landmarker
      frames, render, ear_view, backproject, triangulate, label, camera
      corpus.py           corpus generations and where their files live
      data.py, gt.py      corpus ears and reference landmarks, for scripts
      faceform.py         Faceform Wrap: landmark-posed template fitting
      wrap.py, marks.py   mesh helpers, template marks
      skin.py             procedural skin for renders
    scripts/
      run_pipeline.py
      corpus/   01-12     build a corpus generation
      caches/   1-4       render caches -> view table
      eval/               validate_pipeline (held-out), score_heldout,
                          validate_scans, validate_wrapped_ears,
                          escape_audit, gt_check_ear_views, slid_fit_flag
    tests/
    archive/            finished or failed experiments, frozen; see its README

## Things that change results without changing this code

- **The 2D model.** Untextured meshes use `render_run` (`r2j_s42`); textured
  scans follow Ear_Landmarker's `SHIPPED_RUN`, from its working tree, live. `EAR3D_CKPT=<run or .ckpt>`
  pins the checkpoint (results record it), but not their inference code:
  their 2026-10-20 changes moved one held-out ear by up to 9 mm with the same
  checkpoint. Re-measure after pulling their changes.
- **The corpus generation.** `EAR3D_CORPUS=lmw` (default) or `esc` chooses
  fits, transforms, reference landmarks and view table together. Never mix
  them: the reference marks are tied to the fits.
- **The reference.** Compare models against one reference (`lm3d_v12`, built
  from the current 2D model's contours, so it favours that model; quote `lm3d_v7`, hand marks, beside it);
  switching it moved the score more than most model changes do.

## Findings and prior research

`FINDINGS.md` is the research record, the inherited effort's and this
project's: why each pipeline stage is built the way it is, how the ground
truth is made, the measurement mistakes that produced wrong answers and the
rule each left, the retractions, and what is closed or open.
`RESULTS_inherited.md` is the inherited long form. Learned
depth from images never transferred to photographs across five attempts; a
geometric solve and a general depth model (MoGe-2) did. That code is in
`archive/`.

## Licences and data

No data is committed. Scans, fits and photographs stay local and gitignored,
and their licences travel with them.

- The default ear box is BlazeEar (Apache-2.0). An Ultralytics YOLO box
  (`ear_detector="yolo"`) is AGPL-3.0: local use only.
- The view table is derived from HUTUBS (CC BY 4.0) and credited as such.
  Anything released uses the HUTUBS-only (`ccby`) shape model.
- Anything built on the U-Net or its outputs credits SaifulULL (CC BY 4.0).
- The Faceform Wrap EULA forbids decompiling or reverse engineering it;
  `ear3d/faceform.py` only edits project files and runs its CLI.
