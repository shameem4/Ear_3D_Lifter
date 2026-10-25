# Web demo

Pick a HUTUBS head or an AudioEar3D ear (or upload a .ply/.obj/.stl in mm)
and watch the pipeline place 55 landmarks per ear. Everything runs in the
browser tab: `ear3d_core` as WebAssembly makes every decision, `js/host.js`
renders (three.js, `js/renderer.js`) and evaluates the bare networks
(onnxruntime-web, MediaPipe's FaceLandmarker). `app.js` only draws the
pipeline's events.

## Build and run

    python demo/export_models.py      # demo/models: ONNX graphs + pipeline assets
    python demo/export_meshes.py      # demo/meshes: the pick list (local data)
    wasm-pack build --release --target web -d ../demo/pkg core -- --features wasm
    cd demo && python -m http.server 8795 --bind 127.0.0.1

then open http://127.0.0.1:8795/. `models/`, `meshes/` and `pkg/` are
generated and gitignored.

## Accuracy

The same pipeline as `EAR3D_CORE=rust` in Python, on different pictures and
a different FaceMesh build: +0.04 mm paired over the 32 held-out ears
(demo/parity: about half the three.js renderer, half the JS FaceLandmarker).
The AudioEar3D ears are shown as WRAPS (our template fitted to each scan):
smooth, hole-free and sharing the pipeline's template topology, so an easier
input than a raw scan. On the raw cloud surfaces the pipeline is weak (holes):
005 orients poorly there, 012 well.

## Before hosting it publicly

- AudioEar3D's provenance is unverified (CLAUDE.md); decide before serving it.
- The mean-ear template and view table come from the full corpus; anything
  released should use the HUTUBS-only `ccby` variant.
