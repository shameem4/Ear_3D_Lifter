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

## Hosting (GitHub Pages)

https://shameem4.github.io/Ear_3D_Lifter/ is built by
`.github/workflows/pages.yml` on every push to `main` that touches `core/` or
`demo/`: the core is compiled to WebAssembly from that commit and the page
code copied from `demo/`. The data (ONNX models, HUTUBS heads) is NOT in git:
it is the release asset `demo-data-vN.tar.gz`, which holds `models/` and
`meshes/` (heads only, ears list empty; no `manifest.json`, it has local
paths). New data: re-export, pack, publish a new release, bump `DATA` and
`DATA_SHA256` in the workflow:

    tar czf demo-data-v2.tar.gz models meshes     # from the assembled site dir
    gh release create demo-data-v2 demo-data-v2.tar.gz --title "Web demo data v2"

## Accuracy

The face lift (a whole head gets MediaPipe's 478 points as well as the ears)
matches the Python FaceLifter (2.46 mm on average against the canonical-fit
pseudo GT, 16 demo heads). The iris points (468-477) and the eye region are
UNRELIABLE: none of the scans we can use resolve the eyes, so they cannot be
verified, and the iris points only get the per-landmark triangulation.

The same pipeline as `EAR3D_CORE=rust` in Python, on different pictures and
a different FaceMesh build: +0.04 mm paired over the 32 held-out ears
(demo/parity: about half the three.js renderer, half the JS FaceLandmarker).
The AudioEar3D ears are shown as WRAPS (our template fitted to each scan):
smooth, hole-free and sharing the pipeline's template topology, so an easier
input than a raw scan. The back of the helix rim and the canal tube are cut
from each wrap (wrap_work/templates/ear_L_template_back_exclusion.npz): a
one-sided scan holds nothing there, and the template tore through the front
on 62 of 112 wraps; cut, 0. On the raw cloud surfaces the pipeline is weak (holes):
005 orients poorly there, 012 well.

## Before hosting it publicly

- AudioEar3D's provenance is unverified (CLAUDE.md); decide before serving it.
- The mean-ear template and view table come from the full corpus; anything
  released should use the HUTUBS-only `ccby` variant.
