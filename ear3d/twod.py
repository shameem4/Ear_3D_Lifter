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

from inspect import signature  # noqa: E402


def make_pipeline(cfg, run=None):
    """The shipped 2D pipeline, configured for RENDERS the way cfg says.

    Eight scripts built this by hand and none of them agreed on the detector
    settings, which is how the whole path ran at the photograph defaults on
    renders for as long as it did -- losing 44% of views to a score threshold
    that was never meant for this framing. The settings and their evidence live
    in config.py; this is the one place that applies them.

    cfg["detector_full_frame"] is set AFTER construction because
    EarLandmarkerPipeline does not expose two_stage -- it builds its EarDetector
    with the default. Reaching in is deliberate and is the caller's choice to
    make, not a change to the 2D repository, whose thresholds stay fitted for
    the domain they were fitted on.
    """
    # An explicit checkpoint beats a run name, because find_best_checkpoint
    # picks by the nme in the filename and an A/B probe can tie across epochs
    # -- ab2_fixed_s42 has three files all at 0.0161, so asking for the run
    # would silently pick whichever sorted first rather than the epoch the
    # comparison is about.
    ck = cfg.get("ckpt") or find_best_checkpoint(run or cfg.get("run"))
    # smooth=False: the tracker is for video. On a single still it would smooth
    # a one-frame track against wall-clock time and shift the landmarks.
    # THE 2D REPO'S DEFAULTS ARE INHERITED, NOT RESTATED. cfg["flip_fusion"]
    # is None by default and nothing is passed, so the pipeline uses whatever
    # Ear_Landmarker ships -- currently fusion on, their FLIP_FUSION_DEFAULT.
    # Behaviour changes belong there, not forked here.
    #
    # THE ONE EXCEPTION IS A HARNESS THAT MEASURES FUSION. escape_audit's
    # --tta drives the mirror itself, so inheriting fusion would fuse twice
    # and the `none` arm of a --tta pair would silently BE a fused arm. It
    # passes flip_fusion=False for that reason and says so. A measurement
    # cannot inherit the default of the thing it is measuring.
    #
    # Pinning it matters beyond correctness: their default landed between
    # cells 3 and 4 of a running sun sweep here, and one subprocess per cell
    # meant the later cells silently re-imported it.
    kw = {}
    if cfg.get("flip_fusion") is not None and \
            "flip_fusion" in signature(EarLandmarkerPipeline.__init__).parameters:
        kw["flip_fusion"] = bool(cfg["flip_fusion"])
    pipe = EarLandmarkerPipeline(BLAZEEAR_DIR / DETECTOR_WEIGHTS, ck,
                                 device=cfg.get("device", "cuda"), smooth=False,
                                 detector_confidence=cfg.get("detector_conf"),
                                 **kw)
    if cfg.get("detector_full_frame"):
        pipe.detector.two_stage = False
    return pipe, ck


__all__ = ["make_pipeline", "EAR_LANDMARKER", "CHECKPOINTS", "BLAZEEAR_DIR",
           "DETECTOR_WEIGHTS", "DETECTOR_INPUT_SIZE", "EarDetector",
           "EarLandmarkerPipeline", "LandmarkPredictor",
           "find_best_checkpoint"]
