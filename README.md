# Ear 3D Lifter

Head mesh in, 55 ear landmarks per ear out, in 3D.

The 55-point ear scheme is annotated on photographs, and a 2D model
([Ear_Landmarker](https://github.com/shameem4/Ear_Landmarker)) finds those
points in images. This repository puts them on a head scan: it renders the
ears, reads them with the 2D model from many views, and triangulates.

## Use it

    python scripts/run_pipeline.py head.obj [out.npz]

or `from ear3d.pipeline import Pipeline; Pipeline().run(mesh)`.

- **Input:** a head scan in millimetres, any orientation, or an ear-only scan.
  Flow-gate: a mesh under 100 mm on its longest side is treated as one ear
  (`ear3d.ear_only`): no face stage; its pose and side come from fitting the
  mean ear template, then the same 25-view landmarking. Output has only that
  side filled. Held-out ears with the head cut away: 1.10 mm, side right
  32/32; noisy AudioEar3D clouds: orientation within 15 deg on 18/20, one
  wrong side. A pinna-only mesh (AudioEar3D, any scan cut to the ear) gets
  landmarks 0-2, the helix root, to ~2 mm (was 5 mm before r3p).
- **2D model.** Every mesh is re-skinned and read by the render-only model
  `r3p_s42` (config `render_run`); a scan's own texture is ignored by default.
  `use_mesh_texture=True` reads the texture with Ear_Landmarker's regular
  `SHIPPED_RUN` instead; on three real textured heads it was 2-9x less stable.
  It was trained only on this repo's renders (head renders plus pinna-only
  ones) and, like its predecessor r2j, is far worse on photographs: never
  point it at a photo or a real texture.
- **Output:** `L`, `R` (55 x 3 each) in the input mesh's own coordinates,
  NaN where an ear or landmark was not found (`L` is the subject's left),
  and `ckpt`, the 2D checkpoint used.
- **Accuracy (untextured, r3p):** median **1.03 mm** per ear against the
  user's hand-corrected landmarks on 16 HUTUBS heads used nowhere in building
  it (32 ears), each randomly rotated; 0.62% of points beyond 5 mm (r2j was
  1.09 mm and 0.85%, v4 1.61 mm and 4.7%). The reference is one annotator's
  placement, which the model was trained toward. With synthetic earrings
  added, 1.07 mm. The antihelix points 26-28 are ~0.1 mm worse than r2j's. About 15 s per head on one GPU. Validated only on HUTUBS
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
                          escape_audit, gt_check_ear_views
    tests/

## Things that change results without changing this code

- **The 2D model.** `render_run` (`r3p_s42`) for every re-skinned mesh;
  Ear_Landmarker's `SHIPPED_RUN` (live, from its working tree) only with
  `use_mesh_texture=True`. `EAR3D_CKPT=<run or .ckpt>`
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
geometric solve and a general depth model (MoGe-2) did. That code, and every
other finished experiment, was removed from the tree on 2026-10-25 and lives
at the git tag `archive-final`: `git show archive-final:archive/README.md`
indexes it.

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
