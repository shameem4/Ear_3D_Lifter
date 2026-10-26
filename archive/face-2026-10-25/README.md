# Face experiments, 2026-10-25 (archived)

Scripts from the face-lift work on the `face` branch, frozen with their own scratch paths; results are in
FINDINGS.md (face section). Not maintained: they import ear3d as it was at this commit.

- face-port: parity of ear3d.face.FaceLifter vs Face_3D_Lifter's Lifter (identical).
- face-v1: the 50-head HUTUBS re-lift and GT re-fit (run.sh, gt.sh).
- face-bunch: diagnosis of bunching; Laplacian smoothing, IRLS, joint ray fit (became fit_rays).
- face-vw: per-view weights (became fit_view_weights); pp22 nostril diagnosis.
- nose-snap: geometric subnasale snap (not adopted).
- face-nicp: dense canonical fit to the nose (became fit_dense, nose base only); region sweeps.
- face-wrapkeys: Wrap of the canonical face on lifted key points (sets B, C, D, reliable-keys; not adopted).
- face-pinfill: in-house pins + film, dense pinned film, lip-corner refill (not adopted).
- face: the first draft face pipeline (deleted; repeatability 4.55 mm).
