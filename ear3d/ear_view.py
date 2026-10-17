"""Ear framing on a normalised head: side view -> YOLO-ear box -> pad -> zoom.

Input is the output of frames.normalize_head: centroid at the origin, +x
the subject's left, +y up, +z nose forward. The ear on side "L" is seen
from +x, on side "R" from -x, both with +y up and lit from the face side
(render.side_light).

The box comes from the YOLO11 ear detector shipped in BlazeEar, not from
BlazeEar itself. On side views of 20 randomly rotated heads YOLO's top box
was on the ear 40/40, BlazeEar's 33/40 under the light of the time.

ONE ZOOM, NOT TWO. Measured on 40 heads x 2 sides with the head frame and
ear light of a5a949b/dc484d5, ear height as a fraction of the final frame
(YOLO re-run on the zoom):

    single zoom, smallest box   median 65%  IQR 63-67%  range 57-69%  CV 0.05
    zoom, re-detect, zoom again median 66%  IQR 65-67%  range 53-69%  CV 0.04

Equally consistent; the second pass costs a render and a detection per ear.

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

PAD = 0.25          # each side, as a fraction of the box's longer side
MIN_WIDTH = 0.4     # smallest-box rule ignores boxes narrower than this x top.
                    # 0.6 blocked the correct 78 mm inner box on pp67_L (left a
                    # 102 mm frame, ear at 45%); no false small box seen in 80.


def load_yolo(path=YOLO_EAR):
    from ultralytics import YOLO
    if not pathlib.Path(path).exists():
        raise FileNotFoundError(f"YOLO ear detector not at {path}; set $BLAZEEAR")
    return YOLO(str(path))


def ear_box(yolo, img):
    """[x0, y0, x1, y1] px, or None.

    Ultralytics keeps the top score after NMS at IoU 0.7. On 5 of 40 side
    views its top box was a LOOSE box around a tighter one (92-106 mm vs
    ~75); NMS keeps both because a box nested in a larger one stays under
    0.7 IoU. So take the SMALLEST box overlapping the top one (intersection
    >= half the smaller box), but not one narrower than MIN_WIDTH of the top
    box, so a small false box on the tragus cannot win.
    """
    r = yolo.predict(np.ascontiguousarray(img[:, :, ::-1]), conf=0.05, verbose=False)[0]
    if not len(r.boxes):
        return None
    B, S = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
    t = B[np.argmax(S)]
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    inter = lambda a, b: (max(0, min(a[2], b[2]) - max(a[0], b[0]))
                          * max(0, min(a[3], b[3]) - max(a[1], b[1])))
    ok = [b for b in B if inter(b, t) >= 0.5 * min(area(b), area(t))
          and (b[2] - b[0]) >= MIN_WIDTH * (t[2] - t[0])]
    return min(ok, key=area)


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
