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
