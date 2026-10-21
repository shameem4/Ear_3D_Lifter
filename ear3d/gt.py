"""Reference landmarks (the ground truth), moved into any frame a script needs.

The reference is config's gt_lm3d -- the current corpus generation's
(ear3d/corpus.py) -- in each ear's CANONICAL millimetres. Two transforms reach
everything else:

    inv_gt(name)          scan -> that ear's canonical, in the REFERENCE's generation
    inv(name, version)    scan -> canonical, in a CACHE's generation

A head cache is built in the canonical frame of its subject's LEFT ear, in the
generation it was rendered in (corpus.cache_version), so gt_in_cache goes
ear canonical -> scan -> left-ear canonical of that generation -> cache frame.

GT_LM3D / GT_CANON2SCAN rebase the REFERENCE on another generation without
touching the caches (e.g. score old caches against lm3d_v11).

Moved from the head of scripts/eval/gt_check_ear_views.py (2026-10-20), which three
scripts used to exec() up to a marker line.
"""
import os

import numpy as np

from . import corpus as _corpus
from .config import DEFAULTS
from .corpus import WRAP


def load_reference(fn):
    """(npz, names) for a templates/ lm3d file."""
    z = np.load(WRAP / "templates" / fn, allow_pickle=True)
    return z, [str(x) for x in z["names"]]


GT_FILE = os.environ.get("GT_LM3D", DEFAULTS["gt_lm3d"])
GT_C2S = os.environ.get("GT_CANON2SCAN", _corpus.get("canon2scan"))
LM, NAMES = load_reference(GT_FILE)
ALSO = {fn: load_reference(fn) for fn in DEFAULTS.get("gt_lm3d_also", ())}


def inv(name, version=_corpus.LEGACY):
    """scan -> canonical for `name` in a CACHE's generation (pass the cache's)."""
    return np.load(WRAP / _corpus.get("canon2scan", version) / f"{name}.npz", allow_pickle=True)["inverse"]


def inv_gt(name):
    """scan -> canonical for `name` in the REFERENCE's generation."""
    return np.load(WRAP / GT_C2S / f"{name}.npz", allow_pickle=True)["inverse"]


def gt_in_cache(hd, side, head, ref=None):
    """Reference landmarks of this ear in the cached normalised head frame (mm)."""
    lmz, names = ref or (LM, NAMES)
    nm = f"{hd}_{'left' if side == 'L' else 'right'}"
    P = lmz["landmarks"][names.index(nm)][:55]
    # ear's canonical -> scan mm -> the _left canonical the head cache was built in
    T = inv(f"{hd}_left", _corpus.cache_version(head)) @ np.linalg.inv(inv_gt(nm))
    P = (np.c_[P, np.ones(len(P))] @ T.T)[:, :3]
    return (P @ head["Rr"].T - head["centre"]) @ head["R_head"].T
