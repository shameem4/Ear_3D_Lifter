# Building a corpus generation

One generation is one consistent set: ear fits, the canonical corpus, a
transform from each canonical fit to its scan, the mean template and shape
bases, the reference landmarks, and (in `../caches`) the view table. Mixing
generations silently moves the ground truth; `ear3d/corpus.py` names each
generation's files, and `EAR3D_CORPUS` selects one (default `lmw`).

These are the steps that built **lmw** on 2026-10-17/18, in order. Paths
under `corpus/` and `templates/` are relative to `../wrap_work`. Run heavy
steps under `"$CODE_UTILITIES/guard_run.sh"`, and give shards of one script a
shared key (`GUARD_RUN_KEY=<script>`). Python is the `mocap` env.

| Step | Script | In | Out |
|---|---|---|---|
| 1 | `01_predict_heads.py OUT W N` | scans in canon2scan | per-ear 55 landmarks, scan frame |
| 2 | `02_rewrap_heads.py OUT PREDS [--lobe EARS]` | step 1 | head ear fits (Faceform) |
| 3 | `03_rewrap_audioear.py OUT EAR...` | AudioEar3D clouds | cloud fits (`runs/ae3d_rewrap`) |
| 4 | `04_export_fits.py` | steps 2, 3 | `corpus/lmwrap_v1`, `corpus/ae3d_v3` |
| 5 | `05_canonicalise_corpus.py` | step 4 | `corpus/canonical_lmw` |
| 6 | `06_canonical_to_scan.py --direct` | step 5 | `corpus/canon2scan_lmw` |
| 7 | `07_build_mean_template.py` | step 5 | `ear_mean_v5`, `ear_lmv_v4` |
| 8 | `08_build_ear_pca.py` | steps 5, 7 | `ear_pca_v8_{all,ccby}_mm` |
| 9 | `09_build_landmark_basis.py` | step 8 | `lm_basis_v9_*` |
| 10 | `10_build_gt_coeffs.py` | steps 5, 8 | `gt_coeffs_v8_mm` |
| 11 | `11_remark_from_predictions.py` | steps 2, 5, 6 | `mark_mean_v10_55` |
| 12 | `12_build_lm3d.py --marks` | steps 5, 11 | `lm3d_v11_canonical_mm` (the reference) |

Then `../caches` (head frames, ear views, rays, view table), and add the
generation to `ear3d/corpus.py`.

The commands as run (`$W` = wrap_work/corpus, `$T` = wrap_work/templates):

    01_predict_heads.py   $PRED w 4                         # x4 workers, GPU
    02_rewrap_heads.py    $FITS $PRED --lobe P0082_left,P0088_left,P0027_left
    04_export_fits.py     --heads $FITS --audioear runs/ae3d_rewrap \
                          --out-heads corpus/lmwrap_v1 --out-audioear corpus/ae3d_v3
    05_canonicalise_corpus.py --assume-mm --out-root $W/canonical_lmw \
        --sources hutubs=$W/lmwrap_v1/hutubs sonicom=$W/lmwrap_v1/sonicom audioear=$W/ae3d_v3
    06_canonical_to_scan.py --canonical canonical_lmw \
        --direct $W/lmwrap_v1/provenance.json --out $W/canon2scan_lmw
    07_build_mean_template.py --canonical $W/canonical_lmw --sources hutubs sonicom audioear \
        --out $T/ear_mean_v5.obj --recipe $T/ear_lmv_v1.npz --lmv-out $T/ear_lmv_v4.obj
    08_build_ear_pca.py --fits $K/hutubs $K/sonicom $K/audioear --template $T/ear_mean_v5.obj \
        --out $T/ear_pca_v8_all_mm.npz        # and --fits $K/hutubs -> ear_pca_v8_ccby_mm.npz
    09_build_landmark_basis.py --pca $T/ear_pca_v8_${v}_mm.npz --marks $mk \
        --template $T/ear_mean_v5.obj --out $T/lm_basis_v9_${v}_${n}_unitheight.npz
    10_build_gt_coeffs.py --pca $T/ear_pca_v8_all_mm.npz --fits $K/hutubs $K/sonicom $K/audioear \
        --template $T/ear_mean_v5.obj --out $T/gt_coeffs_v8_mm.npz
    11_remark_from_predictions.py --canonical corpus/canonical_lmw --canon2scan corpus/canon2scan_lmw \
        --preds $FITS --template templates/ear_mean_v5.obj --out templates/mark_mean_v10_55.npz
    12_build_lm3d.py --canonical corpus/canonical_lmw --definition templates/lm3d_v9_canonical_mm.npz \
        --marks templates/mark_mean_v10_55.npz --out templates/lm3d_v11_canonical_mm.npz

(`$K` = `$W/canonical_lmw`. lmw's as-run step 2 outputs, with the step 1
predictions in each `P`, are kept in `corpus/lmwrap_v1/asrun/`; step 11 reads
them as `$FITS`.)

## What to know before rebuilding

- **Steps 4, 6, 11, 12 and 2 are verified** to reproduce the stored lmw
  outputs exactly (2026-10-20). Step 1 is not bit-reproducible: lmw was built
  from earlier passes with other rotations and a slightly older pipeline,
  and the 2D model it calls changes under it (`EAR3D_CKPT` pins the
  checkpoint, not Ear_Landmarker's inference code).
- **Marks are coupled to the fits.** A new fit generation needs steps 11 and
  12 again; keeping old marks moved the ground truth 2.3 mm last time.
- **Right ears are stored z-mirrored into left form** (step 4). The bases in
  step 9 are unit-height, not millimetres (`../wrap_work/templates/UNITS.md`).
- The lobe rescue (`--lobe`) is for fits that clip the lobe; check by
  rendering the fit face-on before adding an ear to it.
- The 13 ears that would not register, and the 8 folded fits, were reviewed
  and accepted; see project memory `landmark-posed-head-rewrap`.
