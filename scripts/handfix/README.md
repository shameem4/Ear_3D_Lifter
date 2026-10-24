# Hand-corrected 3D landmarks

The reference (lm3d_v12) is derived from the 2D model, so where the model is
wrong in a consistent way the reference is too. Attached lobes are the case
that started this (FINDINGS: "Attached lobes are a known miss"). These steps
let the user correct the 55 points by hand on the scan, then render the
corrected ears as 2D training data for Ear_Landmarker.

| Step | Script | Out |
|---|---|---|
| 1 | `1_export_ears.py [EAR ...]` | `runs/handfix/<ear>/` surface.ply + points.json |
| 2 | `serve.py`, then open http://127.0.0.1:8765 | `runs/handfix/<ear>/adjusted.json` |
| 3 | `3_render_training.py OUT W N` | crops for Ear_Landmarker's photo model (synthetic_v1/v2) |
| 4 | `4_render_pipeline_views.py OUT W N` | whole pipeline renders for a render-only model (synthetic_v3) |

As run 2026-10-21: step 1 with `pp16_left pp72_left` added; 73 ears, every
corpus ear whose worst lobe point is more than 5 mm off v12 in the
v4_fc05_s42 corpus predictions.

## Editing

- Start points are v12, the reference. Blue is the model's prediction
  (read-only); grey (toggle) is the untouched reference.
- Drag a point and it slides on the scan surface. Or select one and
  shift-click the surface to drop it there.
- Every move saves. **"Mark done" means every one of the 55 points was
  checked**, moved or not. Step 3 treats a done ear's points as trusted
  (anatomical ground truth); points on an ear not marked done are not.
- Held-out ears (the 16 evaluation heads) can be corrected, which improves
  our own evaluation, but they are never rendered for training.
- Frame: each ear's canonical mm, with right ears mirrored about z as stored.

`runs/handfix` holds scan geometry: it stays local and is never committed.
The server binds 127.0.0.1 only.

## For step 3 (agreed with Ear_Landmarker, 2026-10-21)

- Crops: square, side = landmark extent / 0.70, centred on the landmarks'
  box. 384 px, head as background, no added jitter.
- Labels: `<stem>.png` + `<stem>_annotation.txt`, 55 lines `x y` in pixels.
  Occluded points keep their projected position.
- One `manifest.csv`: stem, dataset, subject, side, yaw, pitch, roll,
  occupancy, licence, visible (55 0/1), trusted (55 0/1).
- Naming: `syn_<dataset>_<subject>_<side>_<viewid>.png`.
- ~40 views per ear: the 25 orbit views plus ~15 random poses (yaw ±45,
  pitch -45..+30, roll ±15), ~40% of all views from below. Pitch sign:
  NEGATIVE pitch is the camera below the ear (orbit_extrinsic: camera at
  frame y = sin(pitch), +y toward the helix top). The agreed spec said
  "pitch +15 or more" for below, copying a sign error of ours.
- Right ears rendered as right ears, not mirrored. Vary skin, lighting and
  background.
- Licence: SONICOM is CC BY-SA 3.0 per clean_3d_data/README.md. The user
  takes a permissive view (no SONICOM annotations are used, and the renders
  re-skin the ear), so SONICOM renders are included (decided 2026-10-21).

## Step 4 (render-only model, agreed with Ear_Landmarker 2026-10-22)

The pipeline's own path per head (step 01's rotation, face frame, ear_view,
occupancy_scale), 25 orbit views at 600 px, 4 skins (skin 0 = the pipeline's
exact look; 1-3 random tone/blotch/AO/sun, half with decimation, noise or
holes). Labels: the user's points where done (trusted), else lm3d_v13.
As run 2026-10-23: runs/synthetic_v3, 44,400 images, 444 ears (210 hand,
234 v13), 222 subjects, 15 GB. Long processes start failing the face frame
("no face frame, skipped" and an empty _rows file): delete those and re-run;
fresh processes render them.

2026-10-23 revision: SONICOM STLs share no vertices, so v3's decimation and
noise broke 8,850 images. Damage is now decimation only, on welded vertices
(the user judged noise and holes unrealistic). `--jewellery 0.3
--only-jewellery` renders the jewellery add-on set (ear3d.jewellery: lobe
hoops and studs, helix hoops and studs, geometry only, labels unchanged).
Outputs: runs/synthetic_v3r (replaces v3) and runs/synthetic_v3j.
