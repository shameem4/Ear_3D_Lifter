# Pinna-only renders for the render-only landmarker (synthetic_v3p)

For Ear_Landmarker, 2026-10-25. An add-on to `synthetic_v3r` to train the
next render-only model (successor to `r2j_s42`).

## The problem

The ear-only path (meshes under 100 mm: AudioEar3D, any scan cut down to the
pinna) gets the helix root wrong. Landmarks 0-2 miss by ~5 mm, against ~1.5
on a full head. Measured with r2j_s42 through the pipeline, median vs
lm3d_v12:

| mesh | lm 0-2 |
|---|---|
| full HUTUBS head (16 held-out heads) | ~1.5 mm |
| the same ears cut with a disc of cheek/scalp left round them | 2.2 mm |
| the same ears, pinna only (template wrap, no surrounding skin) | **5.0 mm** |
| AudioEar3D wraps (40 ears) | 5.4 mm |

Ruled out:
- **The lift.** Fed the reference's own pixels, it reproduces the landmarks exactly (0.00 mm).
- **Mesh density.** Subdividing the wraps 16x changes nothing.
- **AudioEar's data.** HUTUBS ears cut the same way fail the same way.

The model has never seen a pinna without its head: every `synthetic_v3r`
image is a head render. In the failing renders it places 0-2 down inside the
helix instead of at the rim-cheek junction. The image model (v4) does
better there (3.2 mm), but worse everywhere else, and a 0-2 hybrid loses on
full heads, so this needs training data, not a model swap.

AudioEar3D's lobe error (3.6 vs 1.4 mm on HUTUBS pinna-only) is a separate,
AudioEar-specific issue and is not addressed here.

## The delivery

`/var/tmp/claude/ear3d/2026-10-25-pinna-renders/synthetic_v3p/`, made by
`Ear_3D_Lifter/scripts/handfix/5_render_pinna_only.py` (as-run copy and
sha256 beside the folder).

- **Ears:** the same ears and labels as `synthetic_v3r`: the user's hand points
  where done (`trusted` 1), else lm3d_v13. The 16 held-out heads are excluded
  (asserted), as are BJ002 and supersecret.
- **Mesh:** the corpus wrap of each ear (`wrap_work/corpus/canonical_lmw/<dataset>/<ear>.obj`).
  That is the template fitted to the scan: pinna only, which is exactly what an
  AudioEar3D mesh is. 1188 vertices; no decimation.
- **Framing:** the ear-only path's own, via `ear3d.ear_only.frame` (the same
  code the pipeline runs, so it cannot drift): template-fit orientation, the
  normalised-head frame, `side_frame("L")`, then `occupancy_scale` at the
  pipeline's target occupancy. Then the 25-view orbit at 600 px with the
  pipeline's light. So an image is exactly what `LandmarkPredictor` gets for an
  ear-only scan.
- **Sides:** right ears are stored mirrored in the corpus, and the ear-only
  path views every ear as a left one. All images are therefore left-ear
  frames; `side` records the ear's real side, and `yaw` is not negated.
- **Skins:** as in step 4. Skin 0 is the pipeline's exact look; skins 1-3 draw
  a Fitzpatrick tone, blotching, AO strength and a sun turned up to 30 deg.
  4 skins x 25 views = 100 images per ear.
- **Format:** identical to `synthetic_v3r`:
  - `syn_<dataset>_<subject>_<side>_<view>.png`, the whole 600 px render.
    **view = 400 + skin*25 + orbit**, so no stem clashes with v3r's 0-99.
  - `<stem>_annotation.txt`: 55 lines `x y` in render pixels. Occluded points
    keep their projected position.
  - `manifest.csv`: v3r's columns plus `mesh` = `pinna_wrap`.
  - `visibility.npz`: the same depth test, with 2 mm slack.
- **Licence:** HUTUBS CC BY 4.0; SONICOM CC BY-SA 3.0 under the user's
  permissive reading (2026-10-21), as in v3r. These are renders of scan
  geometry: local only, never committed or uploaded.

## Delivered (2026-10-25)

444 ears, 44,400 images, 12 GB, no skips. 8,000 HUTUBS and 36,400 SONICOM;
22,200 left and 22,200 right. Labels: 21,200 images hand, 23,200 v13.
Occupancy: median 0.758, range 0.55-1.05.

**Exclude the 35 ears in `labels_off_wrap.csv`.** Some of their labels
lie more than 3 mm off the wrap (mostly landmarks 26-28; P0066_right is a
broken wrap with 20 points off, pp5_right is 15 mm off at 17). On a head
render such a point still lands on scan skin. On a pinna-only render it
floats in the air beside the ear, and that would teach the model to put
points off the ear. Elsewhere labels lie on the wrap (median 0.0 mm).

## Suggested ingest and training

    python scripts/ingest_synthetic_v1.py --name synthetic_v3p --px 600 \
        --carry-split data/synthetic_v3r --addon

`--addon` keeps every subject in its v3r split, so the v3r test set stays
held out.

Train as r2j was trained (init from v4, renders only, lr 1e-3, early
stopping on render val, `--bbox-jitter 0`, `--rotation-deg 5`,
`--degrade-prob 0.3`, flip consistency 0.5, the jewellery add-on). Add v3p at
roughly a quarter of the mix, so head renders, the common case, stay the
majority. The ratio is a guess and worth one sweep.

Please score the new checkpoint on:
1. its own pinna-only val set, at landmarks 0-2;
2. the v3r val set, unchanged, which guards the head case.

## How we will judge it in Ear_3D_Lifter

End to end, against `lm3d_v12` and the hand-marked `lm3d_v7`:
- the 16 held-out heads (full head, guards the head case; r2j: 0.991 mm vs v12);
- the same 16 heads' pinna-only wraps (r2j: lm 0-2 5.0 mm);
- 40 AudioEar3D wraps (r2j: lm 0-2 5.39 mm, per-ear 1.46).

Neither the held-out heads nor AudioEar3D is in the delivery.
