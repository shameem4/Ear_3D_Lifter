# Render caches and the view table

The pipeline triangulates each landmark from its best 8 of 25 rendered views;
which 8 is a learned table (`ear3d/view_table_*.npz`, one per corpus
generation). These four steps rebuild it. Caches go under `runs/cache` and
`runs/mvcache` (gitignored: they hold scan geometry).

| Step | Script | Out |
|---|---|---|
| 1 | `1_cache_head_frames.py OUT ROT_SEED N_HEADS W N [SAMPLE_SEED]` | oriented heads; records its `corpus` generation |
| 2 | `2_cache_ear_views.py HEADS OUT W N` | one framed view per ear |
| 3 | `3_multiview_collect_rays.py EAR_TAG OUT head W N` | 25 views x 55 landmark rays per ear (GPU, 2D model) |
| 4 | `4_build_view_table.py RAY_DIR OUT.npz` | table[view, landmark] = median error |

As run for lmw (2026-10-18/19), 40 heads in two 20-head samples:

    1_cache_head_frames.py runs/cache/v4lmw_s0 7 20 w 4 0     # and v4lmw_s1 ... 1
    2_cache_ear_views.py   runs/cache/v4lmw_s0 runs/cache/ears_v3lmw_s0 w N
    3_multiview_collect_rays.py ears_v3lmw runs/mvcache/sun_0.6_-0.4_-1_head_lmw head w 3
    4_build_view_table.py  runs/mvcache/sun_0.6_-0.4_-1_head_lmw OUT.npz

The table built this way matches the shipped `view_table_v11` (median cell
1.97 vs 1.96 mm, best-8 overlap 7.4/8), which was kept.

- A head the plain face search misses is retried on fresh rotations (pp95
  needs a second draw). Frames are stored with the rotation, so this is safe.
- Caches record which corpus generation and 2D checkpoint made them
  (`corpus`, `ckpt`); `ear3d.gt.gt_in_cache` uses the former, so old and new
  caches both score correctly. Rebuild the table if the 2D model, the view
  grid or the corpus generation changes.
- Held-out heads for evaluation are the 16 HUTUBS heads in neither sample
  (`scripts/eval/validate_pipeline.py OUT 2 16 w N`).
