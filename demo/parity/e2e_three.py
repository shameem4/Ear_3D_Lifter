"""scripts/eval/validate_pipeline.py with every picture drawn by the browser
renderer (demo/js/renderer.js in headless Chrome); detection is the crate's
(detect.rs) on onnxruntime, so the renderer is the only difference from the
Filament run.

    EAR3D_CORE=rust python demo/parity/e2e_three.py OUT_DIR 2 16 0 1
"""
import pathlib, runpy, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from browser_host import BrowserHost
import ear3d.core_host as ch
_one = {}
def factory(face_task, verbose=True):
    if "h" not in _one: _one["h"] = BrowserHost(face_task, verbose)
    h = _one["h"]; h.events = []; return h
ch.FilamentHost = factory
sys.argv = ["validate_pipeline.py"] + sys.argv[1:]
try:
    runpy.run_path(str(HERE.parents[1] / "scripts/eval/validate_pipeline.py"), run_name="__main__")
finally:
    if "h" in _one: _one["h"].close()
