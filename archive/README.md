# Archive

Frozen code from experiments that are finished, failed or superseded. Kept so
what was tried stays findable; **not maintained and not expected to run**.
Shared helpers they imported moved into `ear3d/` on 2026-10-20 (`data`, `gt`,
`faceform`, `marks`, `skin`), so many of these now fail at import.

The verdicts are recorded elsewhere, with their numbers: `RESEARCH_BRIEF.md`,
`RESULTS_inherited.md`, and the project memory. Each line below is the file's
own docstring, which says what it measured, not what it found.

- `SESSION_STATE.md`, `FIXME_post_run.md`: state notes from 2026-10-15/16, now stale.
- `docs/PROJECT_inherited.log`: the inherited effort's run log.

## Depth from photographs: learned heads (closed, never transferred) and general-model probes (MoGe-2, DA3; MoGe-2 won)

- `compare_depth_heads.py`: Are the depth heads actually different? Paired bootstrap over SUBJECTS.
- `da3_camera_recovery.py`: Recover the AudioEar3D photo camera by 3D-to-3D registration, not 2D-to-3D.
- `da3_depth_baseline.py`: How much of our depth does an off-the-shelf model already give for free?
- `eval_by_source.py`: Score a checkpoint on each SOURCE separately, beside the pooled number.
- `eval_crop_degradation.py`: Was the real-photo result the domain, or was it my own crop step?
- `eval_depth.py`: Evaluate a depth head, and measure the ceiling the shape basis imposes.
- `eval_photo_depth.py`: Score a depth head on real photographs, against labels from their own scans.
- `eval_photo_floor.py`: What is actually achievable on these photographs, and where does head 1 sit?
- `eval_shape_fit.py`: Route B scoring: take a shape head's COEFFICIENTS, solve pose, read depth.
- `final_eval.py`: Score the trained heads on validation AND test, side by side.
- `fit_photo_depth.py`: Turn a photograph's 2D annotation into a DEPTH label, via its own 3D ear.
- `hybrid_depth.py`: Do the heads combine, or is one of them simply best?
- `ingest_depth3d.py`: Render real heads with landmarks whose x, y AND z come from the geometry.
- `moge_infer.py`: Run MoGe-2 on one image and write its point map out as an npz.
- `moge_lift_check.py`: Score a monocular depth map as the depth source, against a real scan.
- `moge_multiview.py`: Multi-view: lift the shipped 2D with a depth map in several views, then RANSAC the 3D coordinates into one consensus set.
- `pinna_depth_mask.py`: Segment the pinna in IMAGE space, by flood-filling to its depth cliff.
- `pnp_camera_recovery.py`: Recover the AudioEar3D photo camera by matching features, then PnP.
- `precompute_xy.py`: Run the trained 2D landmarker over the depth corpus and store what it says.
- `relabel_depth3d.py`: Re-label the rendered depth corpus for new marks, without re-rendering it.
- `render_vs_photo.py`: What, measurably, makes a render not a photograph?
- `run_head2_b2.sh`: Route B: predict SHAPE, solve pose, let depth fall out -- MediaPipe's split.
- `run_photo_aug.sh`: Route C: does making renders look like photographs close the 5.1x gap?
- `run_photo_aug_shading.sh`: Is the shading component of photo_aug buying x,y at the cost of depth?
- `run_source_balance.sh`: Does head 1's depth skill survive when the source cannot be assumed?
- `run_source_balance_seeds.sh`: The noise floor for the A-vs-C comparison.
- `scan_depth_check.py`: Score a depth head against a REAL head scan's own geometry.
- `shape_fit_ceiling.py`: What a 40-mode shape can reach, before training anything to predict one.
- `show_depth_samples.py`: Overlay head 1's predictions on the labels, for samples from all three splits.
- `split_depth3d.py`: Split the rendered depth corpus BY SUBJECT, not by image.
- `train_depth.py`: Train the landmarker with depth on the rendered corpus.
- `model/depth_module.py`: Training for the 55-point landmarker WITH depth.
- `model/lifter.py`: Head 3: a separate network that lifts 55 (x, y) to 55 z.
- `model/shape_feat_head.py`: Head 4: landmarks, depth AND a mesh from one network.
- `model/shape_head.py`: Head 2: predict ear SHAPE and POSE, and let the landmarks follow.
- `data/depth_dataset.py`: Dataset for the rendered depth corpus: K landmarks with x, y AND z.
- `data/photo_aug.py`: Make a render look like a photograph, without re-rendering it.

## Earlier corpus generations: cut, wrap, escalate (superseded by scripts/corpus)

- `audit_landmarks.py`: Check landmarks placed on a 3D ear against how the scheme is actually annotated.
- `check_lmv3.py`: Do ear_lmv_v3's 55 landmark vertices actually lie on ear_mean_v4?
- `cut_ears.py`: Cut every ear out of the 3D corpus, into one canonical frame, unattended.
- `escalate_corpus.py`: Pick, per ear, between the no-control-point fit and the ladder's rescue.
- `gt_landmark_check.py`: Score multi-view landmarks against the wrap's own 3D landmarks, point for point.
- `ingest_trimmed_template.py`: Verify a hand-trimmed template and recover its mapping to the original.
- `landmark_template.py`: Promote the landmarks from barycentric addresses to real template vertices.
- `make_mm_transforms.py`: Restate every cut -> head transform in millimetres, once, with provenance.
- `make_template_edit_project.py`: Build a Faceform Wrap project for marking template polygons to exclude.
- `minimal_ear_mesh.py`: How few vertices does it take to describe an ear? A FaceMesh-style answer.
- `propagate_landmarks.py`: Carry the 55 landmarks from the mean ear to every fitted ear in the corpus.
- `rebuild_corpus.sh`: Rebuild the whole 3D corpus and every artefact derived from it.
- `refit_variants.py`: Compare Wrap graph variants on a few ears, before re-fitting the corpus.
- `regen_mm_artefacts.py`: Rebuild every derived 3D artefact in canonical millimetres, and retire the mixed-unit originals.
- `remark_template.py`: Re-mark the template's 55 landmarks with the shipped landmarker.
- `show_lmv3.py`: ear_lmv_v3 drawn on ITS OWN mesh, with both barycentric recipes checked.
- `show_pca_v7.py`: Look at the new basis before trusting it.
- `snap_landmarks.py`: Snap ray-cast landmarks onto the mesh, then repair any that broke the chain.
- `split_fits_by_dataset.py`: Split a Wrap fit directory into one directory per dataset, at base resolution.
- `suggest_landmarks.py`: Suggest where a misplaced landmark on the 3D mean belongs, from the 2D corpus.
- `validate_marks.py`: Validate a hand-placed scheme extension before any of it is propagated.
- `where_to_add_landmarks.py`: Where on the ear do the 55 points fail to pin the surface down, and which new points would fix it most?
- `wrap_audit.py`: Audit a wrapped corpus: find the bad fits, and ask whether one template is enough.
- `wrap_batch_faceform.py`: Run Faceform Wrap over the corpus, unattended, using our fits as the anchors.
- `wrap_batch_head.py`: Wrap the ear template onto the WHOLE HEAD, not onto a cut-out ear.
- `wrap_cloud.py`: Wrap the template onto a POINT CLOUD, filling the gaps with the template.
- `wrap_cloud_adaptive.py`: Re-fit the AudioEar3D clouds, escalating until the mesh does not fold.
- `wrap_cloud_batch.py`: Wrap the template onto every AudioEar3D point cloud, in parallel.
- `wrap_corpus.py`: Wrap one template onto every cut ear in the corpus, unattended.
- `wrap_correspondence.py`: Does vertex N land on the same anatomy in every wrapped ear?
- `wrap_ear.py`: Wrap an ear template onto a scan, and score it against Faceform Wrap.
- `wrap_fold_debug.py`: Build a self-contained Faceform Wrap project for an ear that folds.
- `wrap_from_landmarks.py`: Wrap the ear template onto a head mesh, posed by its 55 3D landmarks.
- `wrap_marks_to_npz.py`: A Wrap SelectPoints project -> the (triangle, barycentric) file propagation wants.
- `wrap_qa.py`: Score a wrapped ear for defects, with and without the scan it came from.
- `wrap_refit_adaptive.py`: Re-fit an ear with Wrap, escalating parameters until the mesh does not fold.
- `wrap_retry_project.py`: Build a Faceform Wrap project that steps through the fits that failed.

## Rendering, lighting and detector studies (conclusions in memory)

- `bake_texture.py`: Bake an OBJ's texture map into per-vertex colours, and write a PLY.
- `bisect_heads.py`: Split a head into left and right halves for hand-wrapping in Faceform Wrap.
- `check_detector_symmetry.py`: Regression check: is the chosen detector box the same under a mirror?
- `check_helix_silhouette.py`: Check the outer-helix annotations against an ear silhouette SAM finds itself.
- `check_pipeline_pin.py`: Behavioural pin on the whole lift path. Fails if anything moved it.
- `domain_probe.py`: Which part of the render-to-photograph gap costs us accuracy?
- `flip_displacement.py`: The as-is arm minus the mirrored arm, as a DISPLACEMENT FIELD.
- `handedness_vs_illumination.py`: Illumination direction, presented handedness, and ear side, crossed.
- `light_direction.py`: PRE-2026-10-14 RESULTS FROM THIS SCRIPT ARE VOID. It renders through escape_audit.load_ear, which until 856652b applied canon2scan's transform to the …
- `make_detector_testset.py`: Render ears the detector SHOULD find, with ground-truth boxes, as a test set.
- `mixed_handedness.py`: Does a single-arm mirror rule survive when presentation is NOT fixed?
- `normal_error_sensitivity.py`: PRE-2026-10-14 RESULTS FROM THIS SCRIPT ARE VOID. It renders through escape_audit.load_ear, which until 856652b applied canon2scan's transform to the …
- `oracle_2d_lift.py`: THE det_snap ROW BELOW IS STALE, the conclusion is not. It was measured before the NaN gate was fixed and before the winding was, so it both forgave l…
- `pinna_mask.py`: Label which vertices are PINNA and which are the scalp behind it.
- `pinna_raycast.py`: Place a 2D landmark onto the PINNA in 3D, not onto whatever the ray hits first.
- `sun_sweep.py`: The sun's direction against the gross-error rate, on corrected renders.
- `tta_bench.py`: EVERY NUMBER THIS SCRIPT HAS PRODUCED IS STALE, AND IT IS LEFT STALE ON PURPOSE. The table was measured at sun x = -0.3, two sun changes ago, with the…

## View selection and triangulation (shipped as the view table)

- `bootstrap_frame.py`: Find the ear's frame from the mesh alone -- no fit, no ground truth.
- `cache_conf.py`: Cache per-landmark HEATMAP confidence alongside the rays.
- `cache_rays.py`: Collect rays and projected ground truth once, so view selection is offline.
- `escape_report.py`: Read an escape_audit csv and report what it says about view dependence.
- `eval_angle_correlation.py`: Does a landmark's stability depend on the angle it is seen from?
- `eval_angle_split_half.py`: Does a per-landmark best viewing angle generalise, or is it noise?
- `eval_homography_fidelity.py`: Does the perspective AUGMENTATION look like a real head turn?
- `eval_pose_consistency.py`: How far do a model's landmarks drift when the ear really rotates?
- `heuristic_view_table.py`: Predict the per-(landmark, view) table from anatomy, as a check on the learned one.
- `ingest_render3d.py`: Render real 3D ear geometry at varied yaw as an extra training source.
- `multiview_table.py`: Multiview accuracy table from ray caches (multiview_collect_rays.py).
- `render3d_ears.py`: Render real 3D ear geometry at a known pose.
- `review_ear_landmarks.py`: 2D ear landmarks on the framed side view (ear_view), one view per ear.
- `review_ear_view.py`: Ear framing review on cached normalised heads (cache_head_frames.py).
- `review_head_sides.py`: Orientation-stage review, side views.
- `triangulate_audit.py`: Score the EXISTING multi-view triangulator with the escape metric.
- `view_backprojection.py`: Standalone, editable: 3D head -> face-on render -> landmarker -> back-projection.
- `view_pipeline.py`: A window onto every stage of the multi-view lift, not just its answer.
- `view_quality.py`: Which views are worth lifting from, and can the pipeline tell without GT?
- `view_table.py`: A per-(landmark, view) accuracy table, used to choose whose rays to trust.

## Other

- `demo_3d.py`: Show what head 1 actually produces: a 3D landmark set, from three angles.
- `demo_3d_turntable.py`: Rotate head 1's predicted 3D landmarks, next to the truth, as an animation.
