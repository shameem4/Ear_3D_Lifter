# Ear 3D Lifter

Take 2D ear landmarks and put them in 3D.

The 55-point ear scheme is annotated on photographs, and a 2D detector
([Ear_Landmarker](https://github.com/shameem4/Ear_Landmarker)) finds those
points reliably. This repository is about the next step: recovering where
those points sit in space — depth, pose, and a surface that agrees with them.

## End-to-end: head mesh in, 3D ear landmarks out

    python scripts/run_pipeline.py head.obj [out.npz]

or `from ear3d.pipeline import Pipeline; Pipeline().run(mesh)`.

- **Input:** a head scan in millimetres (it refuses a mesh whose longest side
  is outside 100-600 units), any orientation; colour is ignored by default.
- **Output:** `L` and `R`, each 55 x 3, in the input mesh's own coordinates;
  NaN where an ear or a landmark could not be found. `L` is the subject's left.
- **No fit, no ground truth** at inference: FaceMesh orients the head, a BlazeEar
  box frames each ear, the 2D landmarker reads 25 rendered views, and
  each landmark is triangulated from its best 8 views (`ear3d/view_table_v9.npz`,
  a 25x55 table of numbers learned on HUTUBS ears) and re-seated on the head.
  `ear3d/pipeline.py` lists the stages and why each setting was chosen.

Measured on 16 HUTUBS heads used nowhere in building it, randomly rotated:
all faces and all 32 ears found; median error 1.7 mm against the reference
landmarks (`lm3d_v9`), 2.6 mm against the human-drawn v7 marks, 3.8% of
points beyond 5 mm. About 15 s per head on one GPU.

Needs, besides this repository: `Ear_Landmarker` beside it (or
`$EAR_LANDMARKER`) for the 2D model; `BlazeEar` beside it (or `$BLAZEEAR`) for
the ear detector (`model_weights/BlazeEar_best_256px.pth`); a MediaPipe `face_landmarker.task`
(`$FACE_LANDMARKER_TASK`).

**Licences.** The default ear box is BlazeEar (Apache-2.0). An Ultralytics
YOLO ear box is available with `ear_detector="yolo"` (1.6 mm, slightly better)
but Ultralytics is AGPL-3.0: use it locally only, not in anything
redistributed or offered as a service. The view table is
derived from HUTUBS (CC BY 4.0) and is credited as such.

Limits: validated only on HUTUBS scans (no hair, one scanner); the view table
must be rebuilt (`scripts/build_view_table.py`) if the 2D model or the view
grid changes; a mesh's own texture can be used (`use_mesh_texture=True`) but
is off, since synthetic skin looked better on the one real scan tried.

## Prior research

A 3D effort preceded this repository. `RESEARCH_BRIEF.md` records what it
measured — the data and templates that exist, the depth and pose results, the
wrap pipeline's behaviour, and the claims that were later withdrawn.
`RESULTS_inherited.md` is the long form, chronological.

It is a research record to consult, not a settled plan. The headline is that
learned depth from an image never transferred to real photographs across five
attempts, while a geometric solve and a general-purpose depth model did; the
brief gives the measurements behind that, and what was never established.

## Layout

    ear3d/      camera, back-projection, triangulation, frames, render, wrap
    model/      the depth, lifter and shape heads (kept for the record)
    scripts/    wrap and template pipeline, depth experiments, evaluation
    data/       depth dataset and the render-to-photo augmentation

`ear3d/scheme.py` is the 55-point convention, held by copy in both
repositories. It is constants only; if the convention changes it changes in
both.

Eight scripts need to run the 2D model to produce landmarks to lift. They
reach it through `ear3d/twod.py`, the single coupling point to
Ear_Landmarker, which resolves that repository as a sibling or from
`$EAR_LANDMARKER`. Everything else stands alone.

## Data

No corpus is committed. Scans, fitted meshes and photographs stay local and
gitignored, and the licence terms they arrived under travel with them.
