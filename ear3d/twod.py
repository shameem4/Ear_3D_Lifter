"""The one place this repository reaches into the 2D landmarker.

A lifter needs 2D landmarks to lift. Those come from Ear_Landmarker, which
is a separate repository because the two have separate goals -- but eight
scripts here legitimately need to RUN the 2D model rather than read its
output from a file, and before this module they each did their own
sys.path surgery against a hard-coded sibling path.

So the coupling lives here and nowhere else. If the 2D repository moves,
one file changes. If it is absent, the error says so in one sentence
instead of surfacing as ModuleNotFoundError on `inference` three frames
deep.

The 2D model is NOT vendored. It is under active development -- its
checkpoints, its detector and its loss all changed in the week this split
happened -- and a copied model definition would drift silently and then
produce landmarks that disagree with the shipped ones for reasons nobody
could see. One source of truth, located at import time.

    export EAR_LANDMARKER=/path/to/Ear_Landmarker   # if not a sibling

Checkpoints come from that repository too: CHECKPOINTS is its
runs/checkpoints, which is where `run="manual_occ_s42"` in config.py
resolves.
"""
from __future__ import annotations

import os
import pathlib
import sys

_DEFAULT = pathlib.Path(__file__).resolve().parents[2] / "Ear_Landmarker"
EAR_LANDMARKER = pathlib.Path(os.environ.get("EAR_LANDMARKER", _DEFAULT))

if not (EAR_LANDMARKER / "inference.py").exists():
    raise ImportError(
        f"the 2D landmarker is not at {EAR_LANDMARKER}. It is a separate "
        f"repository (github.com/shameem4/Ear_Landmarker); clone it beside "
        f"this one or set EAR_LANDMARKER to where it lives.")

if str(EAR_LANDMARKER) not in sys.path:
    sys.path.insert(0, str(EAR_LANDMARKER))

CHECKPOINTS = EAR_LANDMARKER / "runs" / "checkpoints"

from inference import (                                           # noqa: E402
    BLAZEEAR_DIR, DETECTOR_WEIGHTS, DETECTOR_INPUT_SIZE,
    EarDetector, EarLandmarkerPipeline, LandmarkPredictor,
    find_best_checkpoint,
)

__all__ = ["EAR_LANDMARKER", "CHECKPOINTS", "BLAZEEAR_DIR",
           "DETECTOR_WEIGHTS", "DETECTOR_INPUT_SIZE", "EarDetector",
           "EarLandmarkerPipeline", "LandmarkPredictor",
           "find_best_checkpoint"]
