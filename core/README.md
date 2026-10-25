# ear3d_core

The pipeline's geometry in Rust, so the Python pipeline and the browser demo
run one copy of it. Same source, two bindings: PyO3 (`python` feature) and
wasm-bindgen (`wasm` feature). Arrays cross flat, row-major.

Build (toolchain: rustup, wasm32-unknown-unknown target, wasm-pack, maturin):

    cd core
    CONDA_PREFIX=/home/shameem/.conda/envs/mocap PYTHONNOUSERSITE=1 \
        /home/shameem/.conda/envs/mocap/bin/maturin develop --release
    wasm-pack build --release --target web -d pkg . -- --features wasm

Ported so far:

- `triangulate`: robust ray intersection. Every pair instead of 64 random
  RANSAC draws (numpy's RNG cannot be reproduced). 16 held-out heads, 32
  hand-corrected ears: median 1.089 = 1.089 mm, paired p=0.87; WASM and the
  Python extension agree bit for bit. Pipeline: `EAR3D_CORE=rust`.
- `snap`: closest point on the mesh (brute force, Ericson). Matches open3d's
  RaycastingScene to 1.4e-4 mm (its float32) on a 364k-triangle head, 0.2 s.
  The Python path stayed bit-identical after the refactor (`Pipeline._snap`).
- `orient` (+ `kdtree`, `linalg`): the ear-only template fit, with its own
  scaled ICP (open3d's convergence rule kept) and an even-stride subsample
  instead of a random one. Gate test, all-Rust pipeline: 32 head-cut ears
  1.524 vs 1.495 mm, paired +0.005, p=0.19, side 32/32; AudioEar3D within 15
  deg 90% both. 011_left, Python's one wrong side, comes out the right side.
- `pipeline` (+ `camera`, `host`, `skin`, `bvh`, `section`, `rng`,
  `ziggurat`, `mesh`, `exec`): THE WHOLE PIPELINE -- face search and frame
  checks, ear framing, the 25-view landmarking, the ear-only path. It asks
  its host only to hold meshes and colours, render views (a model transform
  plus a camera: exactly in_frame then render) and run a detector on each.
  Python host: `ear3d/core_host.py` (Filament, MediaPipe, BlazeEar, r2j);
  `EAR3D_CORE=rust`. Browser: the same `run` from the WASM build, with a JS
  host. The skin draws numpy's own stream (PCG64 + SeedSequence + numpy's
  ziggurat tables, BSD-3, read from numpy's libnpyrandom), so it is
  skin.py's skin on all but one vertex of a 182k head (a grazing AO ray:
  f64 here, float32 in open3d); WASM matches native to 3e-17.
  16 held-out heads, 32 hand-corrected ears: median 1.089 = 1.089, mean
  1.107 = 1.107, points within a median 0.0005 mm of the Python run (max
  0.24 mm, one point).

Two traps found on the way, both silent:
- The sun is a direction in the RENDERED (in_frame) scene; forgetting the
  view's rotation lights every view from the wrong side.
- Folding in_frame into the camera is exact on paper but Filament shades a
  millimetre scene a few grey levels differently, and a reused renderer
  carries ~0.4 grey levels between frames. FaceMesh flips on both. So the
  host applies the model transform itself and makes a fresh renderer per view.
