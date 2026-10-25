"""Drift test: ear3d_core's detector pre/post (core/src/detect.rs) on the
exported bare graphs (demo/models) against Ear_Landmarker's live
LandmarkPredictor and BlazeEar's live BlazeEarInference, on rendered views.
Fails if either side changes its inference. Re-export (demo/export_models.py)
after a 2D model change, then run this.

    python demo/parity/detector_drift.py IMG_DIR [IMG_DIR ...]
"""
import pathlib, sys
import numpy as np
from PIL import Image
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import ear3d_core as core
from ear3d.config import DEFAULTS
from ear3d.core_host import FilamentHost, anchors
from ear3d.ear_view import BlazeEarBoxes
from ear3d.twod import LandmarkPredictor, render_checkpoint

LM_TOL_PX, BOX_TOL_PX, SCORE_TOL = 0.5, 2.0, 0.02    # on 600 px renders

host = FilamentHost(DEFAULTS["face_task"], verbose=False)
pred = LandmarkPredictor(str(render_checkpoint(DEFAULTS)), "cpu")
det = BlazeEarBoxes(device="cpu")
A = anchors().astype("<f8").tobytes()
f32 = lambda b: list(np.frombuffer(b, "<f4"))
imgs = sorted(p for d in sys.argv[1:] for p in pathlib.Path(d).glob("fil_*.png"))
lm_err, box_err, sc_err, nobox = [], [], [], 0
for p in imgs:
    img = np.asarray(Image.open(p).convert("RGB")); h, w = img.shape[:2]; b = img.tobytes()
    a, m = core.landmarker_inputs(b, w, h)
    ra = host.infer("r2j", np.asarray(a, "<f4").tobytes()); rm = host.infer("r2j", np.asarray(m, "<f4").tobytes())
    ours = np.array(core.landmarks_fused(f32(ra[0]), f32(rm[0]), w, h))
    lm_err.append(np.abs(ours - np.asarray(pred.predict(img))).max())
    r = host.infer("blazeear", np.asarray(core.blazeear_input(b, w), "<f4").tobytes())
    B = np.array(core.blazeear_boxes(f32(r[0]), f32(r[1]), A, w)).reshape(-1, 5)
    TB, TS = det.boxes(img)
    if len(TB) == 0 or len(B) == 0:
        nobox += (len(TB) == 0) != (len(B) == 0); continue
    box_err.append(np.abs(B[0, :4] - TB[0]).max()); sc_err.append(abs(B[0, 4] - TS[0]))
lm_err, box_err, sc_err = map(np.array, (lm_err, box_err, sc_err))
print(f"{len(imgs)} images | landmarks max {lm_err.max():.3f} px med {np.median(lm_err):.3f} | "
      f"top box max {box_err.max():.2f} px med {np.median(box_err):.2f} | score max {sc_err.max():.4f} | box presence mismatches {nobox}")
ok = lm_err.max() < LM_TOL_PX and box_err.max() < BOX_TOL_PX and sc_err.max() < SCORE_TOL and nobox == 0
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
