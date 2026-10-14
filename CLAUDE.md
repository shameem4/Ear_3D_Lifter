# Ear 3D Lifter

Take 2D ear landmarks and put them in 3D: depth, pose, and a surface that
agrees with them. The 2D detection problem is solved elsewhere
(`Ear_Landmarker`); this repository starts from its output.

## Environment

Use the `mocap` conda env — nothing else has the full dependency set:

    /home/shameem/.conda/envs/mocap/bin/python

`torch` lives **inside that env** (2.9.1+cu130), not in `~/.local`. It used
to be in the shared user site and a `pip uninstall torch` run from an
unrelated conda env deleted it, because conda envs do not disable the user
site. When running pip from any env, `PYTHONNOUSERSITE=1` keeps it from
reaching into `~/.local`.

`open3d` is still in `~/.local`, so do **not** set `PYTHONNOUSERSITE=1` when
running this repository's code — `ear3d/__init__` imports open3d.

Run anything heavy under the cgroup guard, which caps memory and wall time:

    scripts/guard_run.sh <mem> <secs> <log> <cmd...>

Prefer GPU, and parallelise on CPU where the GPU is idle.

## The 2D dependency

Eight scripts need to run the 2D model rather than read landmarks from a
file. That coupling lives in exactly one module, `ear3d/twod.py`. Import the
2D pipeline from there, never with `from inference import ...` — that was the
old pattern and it meant eight copies of the same sys.path surgery.

    from ear3d.twod import LandmarkPredictor, CHECKPOINTS

It resolves `Ear_Landmarker` as a sibling directory, or `$EAR_LANDMARKER`.
Checkpoints come from there too, which is where `run="manual_occ_s42"` in
`ear3d/config.py` resolves.

**Do not vendor the 2D model.** It is under active development; a copied
definition would drift and then produce landmarks that silently disagree
with the shipped ones.

## Data

Nothing is committed. Scans, fitted meshes, point clouds and photographs stay
local and gitignored, and the licence they arrived under travels with them.

- `$EAR3D_DIR`, default `../clean_3d_data` — the raw third-party 3D data
- `../wrap_work` — templates, Wrap projects, the fitted corpus

**Before every commit**, check nothing from those has been staged:

    git diff --cached --name-only | grep -iE "secret|BJ002|hsrd|mv_|\.obj$|\.ply$|^data/|^runs/"

`supersecret/` is a real person's head scan. It never leaves this machine,
never gets committed, never gets sent anywhere, and its renders are never
attached to a message.

Non-commercial datasets (Aachen, IIT Indore, York, Mendeley 493s6f753v) were
discarded and must not come back. AudioEar3D is CC BY 4.0 with unverified
provenance. Anything built on the U-Net or its outputs credits **SaifulULL**
(CC BY 4.0). The Faceform Wrap EULA forbids decompiling or reverse
engineering it — that is off the table, not a challenge.

Accepting licence terms, including gated HuggingFace repos, is the user's
action and never Claude's.

## What is settled — do not re-run these

`RESULTS_inherited.md` is the long form. The expensive negatives:

**Learned depth never transfers.** Five learned depth heads were trained and
every one lost to predicting z = 0 on real photographs. Only the geometric
solve works. A general depth model (MoGe-2) at the shipped 2D model's pixels
beat every ear-specific head we trained.

**Solve pose, never regress it.** The shape head works on renders and
reverses on photographs. A mean ear plus a solved pose beats the learned
depth head, for 0.053 ms.

**Back-projection is unreliable** as 3D ground truth — landmarks land on the
scalp. **The wrap is ground truth**: Faceform Wrap fits the template to a
scan to 0.002 mm.

**AudioEar3D cannot be a photo-3D benchmark.** A stranger's ear fits one of
its clouds as well as its own. Costed, closed.

**Head 4 (one network for landmarks + mesh) loses to both.** Ship separate
heads. **Implicit surfaces are not needed** — template topology beats them.

**The source shortcut is real but unused by the depth head** — settled over
six runs, do not re-litigate.

## Current artefacts

- `ear_mean_v4` + the v7 basis — rebuilt with a size-preserving Procrustes.
  Every v5/v6 PCA file was built with a shrinking one and is a few percent
  small.
- `lm3d_v7_canonical_mm.npz` — the re-marked template. 50–54 is superior
  crus, not lobe.
- `ear_lmv_v3` — landmark j IS vertex j.
- `escalated_v1` — the finished 476-ear corpus, 99.6% fold-free.
- The unified 589-ear corpus (HUTUBS + SONICOM + AudioEar3D) in one frame.

Right ears mirror about **Z**, not X. Basis files are unit-height, not mm,
despite the old suffix; `UNITS.md` has the measured table. Cut the canal tube
(54 faces) **after** the fit — it causes most of the folding.

## Working style

**Render and look before reporting.** A number is support for a visual
check, not a substitute. This project has produced several confident wrong
answers that a single rendered image would have caught in seconds.

Do a cynic pass before surfacing a claim: ask what else would produce this
number, and test that, before saying what it means.

Say when you are unsure. A stated gap is worth more than confident wrong.

Commits end with:

    Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
