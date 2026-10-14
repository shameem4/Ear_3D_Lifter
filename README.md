# Ear 3D Lifter

Take 2D ear landmarks and put them in 3D.

The 55-point ear scheme is annotated on photographs, and a 2D detector
([Ear_Landmarker](https://github.com/shameem4/Ear_Landmarker)) finds those
points reliably. This repository is about the next step: recovering where
those points sit in space — depth, pose, and a surface that agrees with them.

## What is already settled

These came out of the work that preceded this repository, and they are
recorded so they are not repeated. The long form is in
`RESULTS_inherited.md`.

**Do not regress depth from an image.** Five learned depth heads were trained
and every one of them lost to predicting z = 0 on real photographs. Depth
that works comes from a geometric solve, or from a general-purpose depth model
(MoGe-2 beat every ear-specific head at the shipped 2D model's pixels).

**The wrap is ground truth.** Faceform Wrap fits the template to a scan to
0.002 mm. 589 ears across HUTUBS, SONICOM and AudioEar3D are fitted and in one
frame. That corpus, not a learned prior, is what a lift should be checked
against.

**Solve pose, never regress it.** The shape head transfers on renders and
reverses on photographs; a mean ear plus a solved pose beats the learned
depth head for 0.053 ms of work.

**The template is fixed.** `ear_mean_v4` and the v7 basis, built with a
size-preserving Procrustes. Landmark j is vertex j (`ear_lmv_v3`). Right ears
mirror about Z, not X.

**AudioEar3D cannot be a photo-3D benchmark.** A stranger's ear fits one of
its clouds as well as its own. Costed and closed.

## Layout

    ear3d/      camera, back-projection, triangulation, frames, render, wrap
    model/      the depth, lifter and shape heads (kept for the record)
    scripts/    wrap and template pipeline, depth experiments, evaluation
    data/       depth dataset and the render-to-photo augmentation

`ear3d/scheme.py` is the 55-point convention, shared with Ear_Landmarker by
copy. It is constants only; if the convention ever changes it changes in both.

Five evaluation scripts import the 2D pipeline (`inference.py`) from
Ear_Landmarker to produce landmarks to lift. Everything else stands alone.

## Data

No corpus is committed. Scans, fitted meshes and photographs stay local and
gitignored, and the licence terms they arrived under travel with them.
