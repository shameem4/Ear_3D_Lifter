# Research brief: what the previous effort looked at

**How to read this.** This is a research record, not a set of instructions.
Everything below was measured by the project that ran before this repository
existed (2D landmarker work with a 3D branch, split out 2026-10-13). It is
here so a clean project can find prior results and reuse artefacts instead of
rediscovering them — not to fix the approach. Where a result is strong it says
so; where it was underpowered, confounded, or later withdrawn, it says that
too. Several conclusions in the earlier history were confidently wrong and were
retracted, which is itself the most useful thing in this document.

The long form is `RESULTS_inherited.md` (4,227 lines, chronological, including
the retractions). Section titles there are descriptive, so grep it.

---

## 1. Where things are

| What | Path |
|---|---|
| Raw third-party 3D data | `$EAR3D_DIR`, default `../clean_3d_data` |
| Templates, bases, fitted corpora | `../wrap_work/templates`, `../wrap_work/corpus` |
| Measured unit table | `../wrap_work/templates/UNITS.md` |
| The 2D detector | `../Ear_Landmarker`, reached only via `ear3d/twod.py` |
| 78 scripts | `scripts/` — each has a docstring stating its purpose |

Nothing under those data paths is committed, in either repository.

## 2. The 3D data that exists

**Three scan sources, fitted and unified.** `corpus/canonical_esc` holds 589
ears from HUTUBS, SONICOM and AudioEar3D in one coordinate frame. Built from
`corpus/escalated_v1` (476 ears, the finished HUTUBS+SONICOM fit) plus the
AudioEar3D clouds.

Licence differs by source and matters for anything released: the
HUTUBS-only `ccby` variants of each basis exist for that reason.

**Fit quality, as measured on `escalated_v1`:** 99.6% fold-free on a relative
gate, mean coverage 80.03%, scan-outside-fit 0.1507%.

**Two measurement subtleties that cost real time:**

- The fold test. An absolute 90° dihedral gate flags the template's own sharp
  creases. Measured as *fit angle minus template rest angle*, 19 flagged folds
  over 15 ears became 3 over 2. Both numbers are in the history; they describe
  the same meshes.
- Units. `lm_basis_*` files are **unit-height, not millimetres**, despite an
  `_mm` suffix on older ones. A unit-check failure at ratio 1.114 was traced to
  comparing a bounding box against a caliper measurement, not to a unit error;
  measured landmark-to-landmark, every corpus passes at ~1.01. `UNITS.md` has
  the table.
- `corpus/adaptive_v1` is raw Wrap output in **mixed** units (HUTUBS
  normalised, SONICOM mm). Fits saved at Wrap's work scale 100 need dividing
  by `w` before scoring; `accuracy()` now raises rather than silently
  consuming 21 GB.

## 3. Templates and bases, and their lineage

Current: `ear_mean_v4` (mean mesh), `ear_pca_v7_{all,ccby}_mm` (shape PCA),
`lm_basis_v8_*` (landmark basis), `gt_coeffs_v7`, `ear_lmv_v3` (landmark-vertex
template, where landmark *j* is literally vertex *j*),
`lm3d_v7_canonical_mm.npz` (the re-marked 3D landmarks).

Lineage worth knowing, because older files are still on disk:

- v2 template was **folded**; that folding propagated into cloud fits and was
  misread as "clouds are hard". v3 fixed it.
- Every v5 and v6 PCA file was built with a Procrustes loop that silently
  **shrank** the mean. They are a few percent small. v7 is the first
  size-correct one; v4's GPA is size-normalised.
- The 3D template's landmark marks were revised once ("v6_spacing") to match
  the shipped 2D detector's spacing. Strips 35–49 and 50–54 are concha border
  and **superior crus** — 50–54 is not lobe, which an earlier reading had
  wrong.
- A systematic offset between the 2D scheme's points and the wrap's (about one
  index step along each helix contour, worth ~2.5 mm) was resolved by that
  re-marking. Results dated before Oct 2026 carry the offset.

**Convention:** right ears mirror about **Z**, not X.

## 4. Depth from a photograph — the main negative result

This is the most heavily tested area and the findings are consistent across
many runs, so it is worth reading before designing a lift.

**Learned depth did not transfer.** Five independent learned depth mappings
were trained (heatmap-integral head, a no-image lifter, a frozen-2D variant,
a shape-coefficient route, and a single multi-task network). On real
photographs every one lost to simply predicting z = 0. Head 1, the best in
domain, is ~5.1× worse on photographs than on renders; framing was ruled out
as the cause.

**A geometric solve did transfer.** `archive/scripts/fit_photo_depth.py` recovers
depth to 0.0120 on renders using **scaled orthography, not PnP**, and taking
the sign from the canonical axis rather than from anatomy. Fitting a mean ear
to the 2D landmarks beat the regressed depth head by ~0.4 skill at 0.053 ms.

**A general depth model beat all of the ear-specific ones.** MoGe-2, sampled
at the shipped 2D model's pixel locations, beat every head trained here — with
no ear training and nothing fitted. It was fed the ear crop, not the whole
head.

**Pose: solve it, do not regress it.** The shape-coefficient head (route B)
worked in domain and *reversed sign* on photographs. Solving pose instead was
worth +0.6 to +0.9.

**Caveats on all of the above, stated plainly:**
- There is still **no real-photo depth label** for this domain. The photograph
  benchmark was built from renders and from one scan.
- AudioEar3D cannot supply one: a stranger's ear fits a given photo as well as
  the subject's own ear does. This was costed and closed as a dead end, not
  abandoned for effort.
- Back-projecting 2D landmarks to a scan is not a substitute — landmarks land
  on the scalp depending on viewpoint.
- An 11.9° rotation that looked like pose error turned out to be a **labelling
  convention**, not a pose failure.

## 5. Shape, and what a landmark metric cannot see

- A 40-mode shape basis costs almost nothing against full 3D, and its
  shrinkage curve is concave — so even a weak shape head pays.
- Head 2's mesh is **noise** unless its coefficients are supervised directly.
  Landmark loss alone leaves a gauge freedom, and the landmark metric cannot
  detect it. This is a case where the obvious metric was blind to the actual
  failure.
- Head 4 (one network for landmarks, depth and a mesh) lost to both specialist
  heads.
- Shape coefficients identify SONICOM vs HUTUBS 93.9% of the time — a real
  source shortcut — but head 1's depth demonstrably did not exploit it. Six
  runs went into establishing that.
- Implicit surfaces (m-plicits and similar) were evaluated and not adopted:
  template topology satisfied every consumer, and nothing needed a watertight
  surface.
- Nearest-surface distance **cannot rank views** (+0.012 rank correlation).
  Scoring against the wrap's landmarks point-for-point works.

## 6. The wrap pipeline

Faceform Wrap fits the template to a scan to **0.002 mm**, which is why the
wrap's landmarks are treated as 3D ground truth. Reaching that frame cost two
distinct frame bugs; `tf_mm_*` is broken, and the working chain is `tf_*` plus
the scan centroid.

Fitting behaviour, as found:

- Every fit folds to some degree. No single parameter fixes it; per-ear
  escalation does (`archive/scripts/wrap_refit_adaptive.py`). The ladder clears 77 of
  92 folded ears and takes the corpus to 96.8% fold-free. Its gain is
  concentrated in the >5-fold band — the median escalated ear loses about 1pp
  of coverage.
- The ladder's exit condition must be scored at **base resolution** (the 1,188
  vertex count); scored on Wrap's subdivided output it can anti-correlate.
- **No control points at all** gives 1.89 folds over 476 ears; the generation-1
  bootstrap turned out to be unnecessary.
- Cutting the template's 54-face canal tube **after** the fit takes 1.89 folds
  to 0.62, for free. Scans have no ear canal, so the tube gets crushed flat.
- `rigid_icp` needs the template centred **and pre-scaled** first, or its
  `max_dist` guard does nothing and it lands off the ear.
- On the back of the ear: drop the control points, keep the geometry. The
  back's tension braces the front — deleting it is 3.5× worse.
- Extra control points never helped. The lobe and helix gaps are **missing
  template geometry**, not weak constraints.
- Hand-written `.wrap` projects need `formatVersion` 69 and a full layout
  envelope. WrapCmd does not validate layout, so a bad one fails oddly.

The Faceform Wrap EULA prohibits decompiling and reverse engineering.

## 7. Where the 55-point scheme runs out

Measured, not guessed: the 55 points leave the ear surface undetermined at the
**fossae, the lobe, and the concha floor**. If the scheme is extended, the
prior work's view was to extend in 3D and **append** indices rather than
renumber, because the 2D detector and every stored basis are keyed to the
current numbering.

## 8. Instruments that exist and what they are good for

These were built for 2D annotation QA. That programme was closed on
2026-10-13 (zero-mean label noise is absorbed by L2 training, so cleaning
bought no accuracy). The instruments themselves still work, and some are
relevant to 3D:

- **SAM 2.1 silhouette check** (`archive/scripts/check_helix_silhouette.py`): outer
  points sit 1.2–1.5% of ear height from the mask boundary, interior points
  6–7× deeper. Apache 2.0 for code *and* checkpoints.
- **SAM 3 mask deficit** as a free per-row occlusion score: deep deficit marks
  hair and hands, 57% of hard rows vs 2% of clean ones.
- **Classical CV** (sato dark-ridge) traces the antihelix, crura, antitragus
  and lobe — the internal curves a silhouette cannot reach. Canny is brittle;
  the concha is invisible to all of it.
- **Depth Anything 3**: Base/Small/Metric are Apache 2.0 for code and weights;
  Large/Giant/Nested are CC BY-NC. Outputs relative depth, metric depth,
  intrinsics, extrinsics and point maps.
- **MoGe-2** — see §4; the strongest depth result here.

A hard-won calibration: **every independent instrument has a ~5–6% of ear
height noise floor.** It can flag a 15% disagreement; it cannot correct a row,
and it cannot be ranked against the landmark model's own residual — doing so is
circular, since the model was trained on those labels.

Also relevant: segmenters (SAM 3, Gemini) delineate the **visible** ear and
stop at the hairline, while the annotation convention extrapolates through
hair. That is a category difference, not a tuning problem, and no prompt or
box moved SAM 3's lobe boundary.

## 9. Things that were never resolved

Listed because they are open questions, not because they need doing.

- Whether a per-point training-dynamics signal measures **label error** or
  merely **image difficulty**. Level separated groups strongly and reliably;
  nothing showed which of the two it was reading. This is the literature's own
  stated failure mode, reproduced here.
- Whether the 589-ear wrap corpus can act as an **external consistency
  constraint** during 2D training. Never tried; it was the most interesting
  idea left on the table when the cleanup closed.
- A real-photo depth label (§4). Still absent.

## 10. Retractions, kept deliberately

The history contains claims that were asserted and later withdrawn. They are
preserved in `RESULTS_inherited.md` rather than deleted, and they are the
reason this project's standing habit is to render and look before reporting.

Examples: AudioEar3D was called depth-flat (it was a coordinate-frame
artefact; the real limit is partial support, ~40 of 55 landmarks per ear). A
corpus bad-annotation rate was published at 5.40%, revised to 1.9%, then
**withdrawn entirely** when the instrument behind it proved to be measuring
visible-ear extent. A "shape carries no information" claim was withdrawn as
underpowered (single-epoch split-half reliability 0.417). A claim that a
screen needed ~10 epochs was withdrawn as exactly backwards — reliability
rises monotonically with epochs averaged (0.417 at 1, 0.937 at 25, 0.978 at
78).

Four separate traps in the 2D audit each produced a confident wrong answer
before being caught; `RESULTS_inherited.md:2555` enumerates them.

## 11. Rough edges in this repository as handed over

- `ear3d/config.py` hardcodes `run="manual_occ_s42"`, which resolves through
  `ear3d/twod.py` into the 2D repository's checkpoint directory. It works; it
  means the default config names a specific 2D training run.
- `skin.py` exists in both repositories. Duplication, noted and not resolved.
- `RESULTS_inherited.md` includes a long 2D-annotation-QA section (its "v5"
  chapter) that belongs to the other repository's subject matter. It is kept
  whole because the depth and wrap findings are interleaved with it
  chronologically.
