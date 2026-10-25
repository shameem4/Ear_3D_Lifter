# Renderer parity

The browser demo cannot run Open3D's Filament, so `demo/js/renderer.js`
reproduces it: a shader fitted to Filament renders of spheres
(`demo/js/shading.json`: environment SH from Open3D's `default_ibl.ktx`,
Lambert sun with a shadow map, GGX, Filament's tone curve; rmse 2.5 grey
levels), plus the pipeline's own camera and model transform.

`e2e_three.py` runs the held-out validation with every picture from that
renderer in headless Chrome and the detectors still in Python:

| 2026-10-24, 16 heads / 32 hand-corrected ears | faces | median | mean |
|---|---|---|---|
| Filament (`EAR3D_CORE=rust`)                   | 16/16 | 1.089 | 1.107 |
| three.js (`e2e_three.py`)                       | 16/16 | 1.121 | 1.117 |

Paired +0.01 mm, p=0.55. Per view (pp16, pp46): pixel diff 3.2 mean, ear box
IoU 0.93-0.95, r2j points 1.4-1.8 px apart on a ~430 px ear; FaceMesh finds a
face on different views (26 vs 15 on pp16), and the face search still ends in
a good frame. Also ~10x faster than Filament (pp16: 14.5 s).

Not covered here: the browser's own detectors (MediaPipe JS, onnxruntime-web).
