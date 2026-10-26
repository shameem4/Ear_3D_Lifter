# Face lift (MediaPipe 478) — runbook

Ported from Face_3D_Lifter on 2026-10-25. The library is `ear3d/face.py` (FaceLifter), with
`ear3d/face_gt.py` for the pseudo GT. Its output is bit-identical to Face_3D_Lifter's `Lifter` on pp1.
It is separate from the ear Pipeline for now.

Run with the mocap python from the repo root:

    python scripts/face/lift.py head.obj                     # one mesh (mm; --metres to scale)
    python scripts/face/lift.py --hutubs pp1 pp2 ...         # -> runs/face/v0/<head>.npz with diagnostics
    python scripts/face/proxy_gt.py [heads]                  # Wrap fit -> runs/face/v0/<head>_gtd.npz
    python scripts/face/view_table.py                        # per view x landmark miss -> runs/face/view_table.{npz,csv}
    python scripts/face/score_loho.py                        # leave one head out (v1), by region, every step
    python scripts/face/build_view_offsets.py v1             # new ear3d/face_view_offsets_v1.npz

Data (local, never committed):
- `runs/face/v0/`: 50 HUTUBS heads, copied from Face_3D_Lifter `runs/v0` (older ear3d). `runs/face/v1/`: re-lifted 2026-10-25, the default (`FACE_RUN`). Holds the lifts, with offsets off
  (the offsets were learned from them), and the `_gtd` pseudo GT.
- `../wrap_work/face_points/`: the user's hand-placed named control points, one Wrap project per head
  (`projects/`), plus `canonical.wrap`.

The view offsets (`ear3d/face_view_offsets_v0.npz`) hold mean 2D pixel offsets learned on those heads. They
are valid only for the FACE render settings and the 33-view grid; FaceLifter checks the grid.
