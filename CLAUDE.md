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

    "$CODE_UTILITIES/guard_run.sh" <mem> <secs> <log> <cmd...>

Prefer GPU, and parallelise on CPU where the GPU is idle.

## The 2D dependency

The pipeline and the ray collector run the 2D model rather than read
landmarks from a file. That coupling lives in exactly one module,
`ear3d/twod.py`. Import the 2D pipeline from there, never with
`from inference import ...` — that was the old pattern and it meant eight
copies of the same sys.path surgery.

    from ear3d.twod import LandmarkPredictor, landmarker_checkpoint

It resolves `Ear_Landmarker` as a sibling directory, or `$EAR_LANDMARKER`.
The checkpoint follows their `SHIPPED_RUN` by default; `EAR3D_CKPT=<run or
.ckpt>` pins it, and results record it as `ckpt`. Pinning the checkpoint does
not pin their inference code, which changes under us: re-run a baseline
before comparing results from different days.

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

## Layout, and where new code goes

- `ear3d/` is the library. Anything two scripts share lives here, imported
  normally: never load a script by path or `exec` part of one.
- `scripts/run_pipeline.py`, then `scripts/corpus/` (01-12, build a corpus
  generation), `scripts/caches/` (1-4, render caches and the view table),
  `scripts/eval/` (held-out runs and scoring). Each stage folder's README is
  its runbook, with the commands as run.
- Experiments start in scratch (`/var/tmp/claude/ear3d/<date-task>/`). One
  that becomes a step others depend on moves into a stage folder; one that is
  finished stays in scratch until its result is in `FINDINGS.md`, then is
  deleted. Don't add experiments to `scripts/` directly. The old `archive/`
  of frozen experiments was removed on 2026-10-25; it is at the git tag
  `archive-final` (`git show archive-final:archive/README.md`).
- `EAR3D_CORPUS` selects a corpus generation (default `lmw`); fits, transforms,
  reference landmarks and view table change together. New fits mean
  re-deriving the reference marks (`scripts/corpus/11-12`).

## Docs

- `README.md`: what it does, how to run it, layout.
- `FINDINGS.md`: the research record, inherited and since 2026-10-13: the
  pipeline's evidence, the ground truth, the measurement mistakes and the
  rules they left, what is closed and open.
- `scripts/{corpus,caches}/README.md`: runbooks.
- `RESULTS_inherited.md`: the inherited effort's long chronological log.
- `docs/`: write-ups for other agents (AudioEar3D cloud wrapping).
- `git show archive-final:archive/README.md`: index of the frozen
  experiments removed on 2026-10-25.

## Prior research

`FINDINGS.md` holds the inherited research record alongside this project's,
items marked *(inherited)*: what the previous effort measured, with its
caveats and retractions, plus where the corpora, templates and bases live.
`RESULTS_inherited.md` is the long form.

Read inherited items as prior work, not as decisions already made for this
project. They are there so results are not rediscovered by accident, and the
record says which findings are strong and which were withdrawn. If a
direction it reports as closed looks worth reopening, it gives the evidence
to argue with.

Two things in it are conventions rather than findings, and breaking them
silently corrupts stored data: right ears mirror about **Z**, not X, and
`lm_basis_*` files are unit-height, not millimetres
(`../wrap_work/templates/UNITS.md` has the measured table).

## Working style

**Render and look before reporting.** A number is support for a visual
check, not a substitute. This project has produced several confident wrong
answers that a single rendered image would have caught in seconds.

Do a cynic pass before surfacing a claim: ask what else would produce this
number, and test that, before saying what it means.

Say when you are unsure. A stated gap is worth more than confident wrong.

Commits end with:

    Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
