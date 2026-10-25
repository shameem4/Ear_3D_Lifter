"""Ear framing on a normalised head: side view -> YOLO-ear box -> pad -> zoom.

Input is the output of frames.normalize_head: centroid at the origin, +x
the subject's left, +y up, +z nose forward. The ear on side "L" is seen
from +x, on side "R" from -x, both with +y up and lit from the face side
(render.side_light).

The box comes from BlazeEar (Apache-2.0) by default, taking its top box;
cfg ear_detector="yolo" uses the YOLO11 ear detector kept locally in BlazeEar
(Ultralytics, AGPL-3.0). End to end on 16 held-out heads: BlazeEar 1.68 mm,
YOLO 1.61 mm median vs v9, 32/32 ears both (2026-10-17). Earlier, before the
current lighting: On side views of 20 randomly rotated heads YOLO's top box
was on the ear 40/40, BlazeEar's 33/40 under the light of the time.

ONE ZOOM, NOT TWO. Measured on 40 heads x 2 sides with the head frame and
ear light of a5a949b/dc484d5, ear height as a fraction of the final frame
(YOLO re-run on the zoom):

    single zoom, smallest box   median 65%  IQR 63-67%  range 57-69%  CV 0.05
    zoom, re-detect, zoom again median 66%  IQR 65-67%  range 53-69%  CV 0.04

Equally consistent; the second pass costs a render and a detection per ear.

SUPERSEDED ZOOMS (archive-final:archive/scripts/bootstrap_frame.py), kept as a note only:
  - committed: ss_z = 0.3 * head extent -- magnification varied with head
    size (170-230 mm), ear at 40-60% of frame.
  - never committed: box-sized, half-width = b_mm / (0.7 * 2 * half_units),
    clamped to 25-80 mm (ear's longer side ~70% of frame). Same idea as
    here but on the old transposed-box detector and the head light; this
    module replaces it, so the edit was discarded 2026-10-17.

YOLO resolution: $BLAZEEAR, else a BlazeEar directory next to this repo.
"""
import os
import pathlib

import numpy as np

from .backproject import backproject
from .frames import frame_from, in_frame
from .render import render, side_light

_DEFAULT = pathlib.Path(__file__).resolve().parents[2] / "BlazeEar"
BLAZEEAR = pathlib.Path(os.environ.get("BLAZEEAR", _DEFAULT))
YOLO_EAR = BLAZEEAR / "model_weights" / "yolov11_ear_detector.pt"
BLAZE_EAR = BLAZEEAR / "model_weights" / "BlazeEar_best_256px.pth"

PAD = 0.25          # each side, as a fraction of the box's longer side
MIN_WIDTH = 0.4     # smallest-box rule ignores boxes narrower than this x top.
                    # 0.6 blocked the correct 78 mm inner box on pp67_L (left a
                    # 102 mm frame, ear at 45%); no false small box seen in 80.


def load_yolo(path=YOLO_EAR):
    from ultralytics import YOLO
    if not pathlib.Path(path).exists():
        raise FileNotFoundError(f"YOLO ear detector not at {path}; set $BLAZEEAR")
    return YOLO(str(path))


class BlazeEarBoxes:
    """BlazeEar (Apache-2.0) behind the same call as the YOLO model: returns
    (boxes [x0, y0, x1, y1] px, scores). BlazeEar itself emits
    [ymin, xmin, ymax, xmax]; the swap happens here and only here."""
    def __init__(self, path=BLAZE_EAR, device="cuda"):
        import sys
        sys.path.insert(0, str(BLAZEEAR))
        from blazeear_inference import BlazeEarInference
        self.net = BlazeEarInference(str(path), confidence_threshold=0.05, device=device).to(device).eval()

    def boxes(self, img):
        import torch
        with torch.no_grad():
            d = self.net(np.ascontiguousarray(img)).detach().cpu().numpy().astype(float).reshape(-1, 5)
        return d[:, [1, 0, 3, 2]], d[:, 4]


def load_detector(kind="yolo"):
    return BlazeEarBoxes() if kind == "blazeear" else load_yolo()


def ear_box(yolo, img):
    """[x0, y0, x1, y1] px, or None.

    Ultralytics keeps the top score after NMS at IoU 0.7. On 5 of 40 side
    views its top box was a LOOSE box around a tighter one (92-106 mm vs
    ~75); NMS keeps both because a box nested in a larger one stays under
    0.7 IoU. So take the SMALLEST box overlapping the top one (intersection
    >= half the smaller box), but not one narrower than MIN_WIDTH of the top
    box, so a small false box on the tragus cannot win.
    """
    if isinstance(yolo, BlazeEarBoxes):
        # Top box only: BlazeEar's NMS (IoU 0.3) already drops nested
        # duplicates, and the rule below swapped a correct 0.79 box on pp30
        # for a 0.21 box on the lobe nested inside it.
        B, S = yolo.boxes(img)
        return B[np.argmax(S)] if len(B) else None
    else:
        r = yolo.predict(np.ascontiguousarray(img[:, :, ::-1]), conf=0.05, verbose=False)[0]
        B, S = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
    if not len(B):
        return None
    t = B[np.argmax(S)]
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    inter = lambda a, b: (max(0, min(a[2], b[2]) - max(a[0], b[0]))
                          * max(0, min(a[3], b[3]) - max(a[1], b[1])))
    ok = [b for b in B if inter(b, t) >= 0.5 * min(area(b), area(t))
          and (b[2] - b[0]) >= MIN_WIDTH * (t[2] - t[0])]
    return min(ok, key=area)


def ear_score(det, img):
    """Top ear-detector confidence on img (0 if nothing)."""
    if isinstance(det, BlazeEarBoxes):
        _, S = det.boxes(img)
    else:
        S = det.predict(np.ascontiguousarray(img[:, :, ::-1]), conf=0.05, verbose=False)[0].boxes.conf.cpu().numpy()
    return float(S.max()) if len(S) else 0.0


def side_frame(side):
    """World -> camera rotation for the side view of a normalised head."""
    sd = np.array([1.0 if side == "L" else -1.0, 0.0, 0.0])
    return frame_from(sd, np.array([0.0, 1.0, 0.0]), front=sd)


def ear_view(mesh_c, side, cfg, yolo):
    """Zoomed, ear-centred view of one ear of a normalised head.

    Returns dict(img, depth, cam, R, centre, scale, box_mm), where the view
    shows mesh_c through in_frame(mesh_c, R, centre, scale), or None if no
    ear is found or the box centre misses the mesh.
    """
    c = side_light(cfg, side)
    R = side_frame(side)
    V = np.asarray(mesh_c.vertices)
    sc = float(np.abs(V @ R.T).max())
    img, depth, cam = render(in_frame(mesh_c, R, np.zeros(3), sc), c)
    b = ear_box(yolo, img)
    if b is None:
        return None
    P0, hit = backproject(depth, np.array([[0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])]]), cam)
    if not hit[0]:
        return None
    half = cfg["eye_z"] * np.tan(np.radians(cfg["vfov"]) / 2)     # frame half-extent, units
    box_mm = max(b[2] - b[0], b[3] - b[1]) / (img.shape[0] / 2) * half * sc
    centre = P0[0] * sc @ R
    scale = 0.5 * box_mm * (1 + 2 * PAD) / half
    img_z, depth_z, cam_z = render(in_frame(mesh_c, R, centre, scale), c)
    return dict(img=img_z, depth=depth_z, cam=cam_z, R=R, centre=centre,
                scale=scale, box_mm=float(box_mm))


def occupancy_scale(mesh, R, centre, scale, cfg, light_cfg, pred, target, iters=2):
    """Re-size an ear frame so the 2D model's landmarks span `target` of the
    frame side in the face-on orbit view (the model's own TRAIN_OCCUPANCY
    definition: landmark extent over crop side). The BlazeEar box is larger
    than the ear and varies with it, so PAD alone leaves occupancy at ~0.5-0.6.
    The orbit keeps the returned scale for every view."""
    from .camera import orbit_extrinsic
    E = orbit_extrinsic(0.0, 0.0, cfg)
    for _ in range(iters):
        img, _, _ = render(in_frame(mesh, R, centre, scale), light_cfg, E=E)
        occ = float(np.ptp(np.asarray(pred.predict(img)), axis=0).max()) / img.shape[1]
        if not np.isfinite(occ) or occ <= 0:
            break
        scale *= occ / target
    return scale
