# Ear 3D Lifter

Take 2D ear landmarks and put them in 3D.

The 55-point ear scheme is annotated on photographs, and a 2D detector
([Ear_Landmarker](https://github.com/shameem4/Ear_Landmarker)) finds those
points reliably. This repository is about the next step: recovering where
those points sit in space — depth, pose, and a surface that agrees with them.

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
