# Ear 3D Lifter

A head scan in, landmarks in 3D out: 55 per ear, and MediaPipe's 478 on the
face. It runs in Python or entirely in a browser tab (`demo/`).

## The story

**It started with ears.** A 2D model,
[Ear_Landmarker](https://github.com/shameem4/Ear_Landmarker), finds a 55-point
ear scheme in photographs. The question was whether those points could be put
on a 3D head scan, without a separate 3D model. The plan was simple: render the
ear from many angles, read each picture with the 2D model, and intersect the
rays from each view.

**Lifting alone was not enough.** Intersecting the rays gave points, but too
many were wrong, and almost every fix that worked used what we know about the
ear's structure or the head's anatomy, not better geometry:

- *Most of the error is the 2D model, not the lift.* Given the ground truth's
  own pixels, the lift is 0.36 mm and never misses by 20 mm. With the 2D
  model's predictions, the error is 3.7 mm with gross misses. So the work went
  into what the 2D model sees and which of its answers to trust.
- *A corpus with an ear template.* 476 head ears and 112 AudioEar3D ears were
  each fitted with one template ear, so every landmark has a place on every
  ear. That gave reference landmarks to measure against, and a way to learn
  from them.
- *Not every view is trustworthy for every landmark.* A table learned on the
  corpus picks the best 8 views per landmark before intersecting: 2.70 to
  2.24 mm.
- *The renders have to look like the training images.* Synthetic skin instead
  of a bare mesh, the ear framed to fill the picture as the model was trained,
  and pinna-only renders so an ear without a head still works.
- *Points belong on the skin.* Each one is re-seated on the surface.

The ears now land at a median of **1.03 mm** from hand-corrected landmarks on
32 held-out ears.

**Then the face.** A sister project lifted MediaPipe's 478 face points the same
way: 33 views around the face, FaceMesh in each, and rays intersected per
landmark. It was merged in here, so one run produces both. The face showed the
same lesson more sharply. Lifting each point on its own gave points that were
roughly right but wrong as a face:

- *Neighbours bunched or spread apart*, because each point settled on its own
  set of views. Fitting all 468 points at once, held to the shape of
  MediaPipe's canonical face mesh, keeps the spacing.
- *FaceMesh puts hidden nostrils on the upper lip* in every view where they
  can't be seen (5.4 mm on one head), so voting between views can't fix it.
  Fitting the canonical face to the scan's own nose shape puts the nose base
  back: 1.58 to 1.07 mm.
- *Mouth and eye corners still collapsed*, closed lips most of all. Springs
  that keep each edge near its typical length took collapsed mouth-corner
  edges from 11% to 0.2% without costing accuracy.
- *Each view has a consistent 2D bias.* Learned per-view offsets are
  subtracted before anything else: 4.21 to 2.34 mm.

The general rule: triangulation finds where a landmark is roughly, and
knowledge of what an ear or a face looks like decides the rest.

**This is a stopgap.** Rendering a scan and reading it in 2D is a detour. The
goal is direct 3D landmarking: a model that takes the 3D scan itself and
places the landmarks on it. That model needs training data, many scans with
their landmarks, and there is no such dataset for this scheme. This
repository is the intermediate step that builds it: lifting 2D landmarks onto
scans to produce the 3D labels a direct model can learn from.

## Use it

    python scripts/run_pipeline.py head.obj [out.npz]          # ears
    python scripts/face/lift.py head.ply [out.npz]             # face

or in Python: `Pipeline().run(mesh)` from `ear3d.pipeline` for the ears, and
`FaceLifter().run(mesh)` from `ear3d.face` for the face. The browser demo runs
both on any head: see `demo/README.md`.

- **Input:** a head scan in millimetres, any orientation, or an ear-only scan.
  A mesh under 100 mm on its longest side is treated as one ear: it gets no
  face stage, and its pose and side come from fitting the mean ear.
- **Output:** `L`, `R` (55 x 3 each) and the face's 478 x 3, in the input
  mesh's own coordinates, NaN where nothing was found (`L` is the subject's
  left).
- **Ear accuracy:** median 1.03 mm per ear on 16 held-out HUTUBS heads (32
  ears), each randomly rotated, against the user's hand-corrected landmarks;
  0.62% of points beyond 5 mm. Held-out ears with the head cut away: 1.10 mm.
  Validated only on HUTUBS and SONICOM scans (no hair). About 15 s per head
  on one GPU.
- **Face accuracy:** about 2.05 mm (leave one head out, 50 HUTUBS heads),
  against a pseudo ground truth that is itself a canonical-face fit, so it
  flatters the method. On one real scan's anatomical stickers, the four
  points FaceMesh defines (nose tip, nasal bridge, glabella, chin) are 0.4 to
  2.9 mm off. **The iris and eye region are unverified:** none of the scans we
  can use show the eyes in enough detail.
- **The 2D models.** Every mesh is re-skinned and read by `r3p_s42`, trained
  only on this repository's renders: never point it at a photo or a real
  texture. The face uses MediaPipe's FaceLandmarker.

Environment: the `mocap` conda env
(`/home/shameem/.conda/envs/mocap/bin/python`). Also needed: `Ear_Landmarker`
beside this repo (or `$EAR_LANDMARKER`), `BlazeEar` beside it (or
`$BLAZEEAR`) for the ear box, and a MediaPipe `face_landmarker.task`
(`$FACE_LANDMARKER_TASK`).

## How it works

    head scan
      -> orient: FaceMesh face frame, checked (skull above neck, symmetry,
         nose-ear distance), retried turned if a check fails
      -> ears, each: BlazeEar box, framed as in training, 25 views,
         Ear_Landmarker in each, every landmark triangulated from its best
         8 views (the view table), RANSAC, re-seated on the surface
      -> face: 33 views around the face, FaceMesh in each, per-view offsets
         removed, triangulated, then three corrections: the joint fit to
         the canonical face, the nose-base fit to the scan, the spacing springs

`ear3d/pipeline.py` and `ear3d/face.py` list each step with the measurement
behind its setting. The same pipeline is written once more in Rust (`core/`),
which the browser demo runs as WebAssembly and Python can call. They agree to
0.006 mm on the face.

The corpus behind the view table and the ear references is built by
`scripts/corpus` (12 numbered steps). Its render caches and the table are
built by `scripts/caches` (4 steps), and the face's view offsets by
`scripts/face`. Each folder's README is its runbook.

## Layout

    ear3d/              the library
      pipeline.py         head mesh -> ear landmarks
      face.py             head mesh -> face landmarks
      core_host.py        the Python host for the Rust core
      twod.py             the only coupling to Ear_Landmarker
      frames, render, ear_view, backproject, triangulate, label, camera
      corpus.py           corpus generations and where their files live
      data.py, gt.py      corpus ears and reference landmarks, for scripts
      faceform.py         Faceform Wrap: landmark-posed template fitting
      wrap.py, marks.py   mesh helpers, template marks
      skin.py             procedural skin for renders
    core/               the same pipeline in Rust (PyO3 and WebAssembly)
    demo/               the browser demo
    scripts/
      run_pipeline.py
      corpus/   01-12     build a corpus generation
      caches/   1-4       render caches -> view table
      face/               face lift, view offsets, scoring
      eval/               held-out runs and scoring
    tests/

## Things that change results without changing this code

- **The 2D model.** `render_run` (`r3p_s42`) for every re-skinned mesh.
  `EAR3D_CKPT=<run or .ckpt>` pins the checkpoint (results record it), but
  not Ear_Landmarker's inference code, which has moved a held-out ear by up
  to 9 mm with the same checkpoint. Re-measure after pulling their changes.
- **The corpus generation.** `EAR3D_CORPUS=lmw` (default) or `esc` chooses
  fits, transforms, reference landmarks and view table together. Never mix
  them.
- **The reference.** Compare ear models against one reference (`lm3d_v12`,
  built from the current 2D model, so it favours it; quote `lm3d_v7`, hand
  marks, beside it). Switching references has moved the score more than most
  model changes.

## Findings and prior research

`FINDINGS.md` is the research record: why each step is built the way it is,
how the ground truth is made, the measurement mistakes and the rule each
left, the retractions, and what is closed or open. `RESULTS_inherited.md` is
the inherited long form. Finished experiments live at the git tags
`archive-final` and `face-experiments-2026-10-25`.

## Licences and data

No data is committed. Scans, fits and photographs stay local and gitignored,
and their licences travel with them.

- The default ear box is BlazeEar (Apache-2.0). An Ultralytics YOLO box
  (`ear_detector="yolo"`) is AGPL-3.0: local use only.
- The view table and the face's view offsets are derived from HUTUBS (CC BY
  4.0) and credited as such. Anything released uses the HUTUBS-only (`ccby`)
  shape model.
- MediaPipe's FaceLandmarker and canonical face model are Apache-2.0.
- Anything built on the U-Net or its outputs credits SaifulULL (CC BY 4.0).
- The Faceform Wrap EULA forbids decompiling or reverse engineering it;
  `ear3d/faceform.py` only edits project files and runs its CLI.
