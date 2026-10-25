"""Export the demo's networks to demo/models/: the bare graphs only.

Pre- and post-processing live in ear3d_core (core/src/detect.rs), so the
browser and the Python host run the same code around the same graphs:
  r2j_s42.onnx      Ear_Landmarker's own export_onnx.py on the render-only
                    checkpoint (render_checkpoint): (1,3,192,192) in [-1,1]
                    -> landmarks (1,55,2) in [0,1], confidence (1,55)
  blazeear.onnx     BlazeEar's network alone: (1,3,256,256) in [-1,1] ->
                    raw boxes (1,A,16), raw scores (1,A,1); no decode, no NMS
  blazeear_anchors.bin  its anchors (A x 4: cx, cy, w, h, normalised, f64 LE),
                    from BlazeEar's get_anchors -- data, not regenerated
  face_landmarker.task  MediaPipe's model bundle, as the Python host uses it
  table/angles/template/canon.bin, cfg.json  the pipeline's assets and config
Each graph is checked against torch on probe inputs.

    python demo/export_models.py
"""
import json, pathlib, shutil, subprocess, sys
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "models"
sys.path.insert(0, str(ROOT))
from ear3d.config import DEFAULTS
from ear3d.ear_view import BLAZE_EAR, BLAZEEAR
from ear3d.twod import EAR_LANDMARKER, render_checkpoint

OUT.mkdir(parents=True, exist_ok=True)
ck = render_checkpoint(DEFAULTS)
subprocess.run([sys.executable, "export_onnx.py", "--checkpoint", str(ck), "--output", str(OUT / "r2j_s42.onnx")],
               cwd=EAR_LANDMARKER, check=True)

sys.path.insert(0, str(BLAZEEAR))
from blazeear_inference import BlazeEarInference
inf = BlazeEarInference(str(BLAZE_EAR), confidence_threshold=0.05, device="cpu").eval()
net = inf.model.eval()
x = torch.rand(1, 3, inf.input_size, inf.input_size) * 2 - 1
torch.onnx.export(net, x, str(OUT / "blazeear.onnx"), input_names=["image"], output_names=["boxes", "scores"],
                  opset_version=14, do_constant_folding=True, dynamo=False)
side = OUT / "blazeear.onnx.data"
if side.exists():
    import onnx
    onnx.save(onnx.load(str(OUT / "blazeear.onnx"), load_external_data=True), str(OUT / "blazeear.onnx"), save_as_external_data=False)
    side.unlink()
import onnxruntime as ort
sess = ort.InferenceSession(str(OUT / "blazeear.onnx"), providers=["CPUExecutionProvider"])
worst = 0.0
for _ in range(4):
    x = torch.rand(1, 3, inf.input_size, inf.input_size) * 2 - 1
    with torch.no_grad():
        want = [t.numpy() for t in net(x)]
    got = sess.run(None, {"image": x.numpy()})
    worst = max(worst, max(float(np.abs(w - g).max()) for w, g in zip(want, got)))
print(f"blazeear.onnx: max |onnx - torch| = {worst:.2e}")
assert worst < 1e-3, "BlazeEar export does not match torch"
np.ascontiguousarray(inf.anchors.cpu().numpy(), "<f8").tofile(OUT / "blazeear_anchors.bin")
shutil.copy(DEFAULTS["face_task"], OUT / "face_landmarker.task")
# The pipeline's own assets, as ear3d.core_host.run hands them to the crate
# (f64 LE): the learned view table and its angles, the mean ear template, the
# canonical face; and the config subset the crate reads (cfg.json).
from ear3d import core_host, ear_only
from ear3d.pipeline import TABLE
t = np.load(TABLE, allow_pickle=True)
for name, a in [("table", t["table"]), ("angles", t["angles"]), ("template", ear_only._template()),
                ("canon", core_host.canonical_face(DEFAULTS))]:
    np.ascontiguousarray(a, "<f8").tofile(OUT / f"{name}.bin")
json.dump({k: DEFAULTS[k] for k in core_host.CFG_KEYS}, open(OUT / "cfg.json", "w"))
json.dump({"r2j_checkpoint": str(ck), "blazeear_weights": str(BLAZE_EAR), "blazeear_input": inf.input_size,
           "anchors": int(inf.anchors.shape[0]), "face_task": DEFAULTS["face_task"]}, open(OUT / "manifest.json", "w"), indent=1)
print("wrote", sorted(p.name for p in OUT.iterdir()))
