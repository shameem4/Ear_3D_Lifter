"""Corpus ears as the scripts load them: head, fit and reference landmarks
in one canonical frame, plus which ears reached that frame by reflection.

Moved from scripts/eval/escape_audit.py (2026-10-20), where it had become the
de-facto loader for some thirty scripts; the text below is unchanged.
"""
import pathlib

import numpy as np
import open3d as o3d

from . import corpus as _corpus
from .config import DEFAULTS
from .corpus import WRAP

# The raw third-party scans ($EAR3D_DIR in config.py; this is its default).
RAW = WRAP.parent / "clean_3d_data"

# HUTUBS head meshes are in metres; the fit lands on them at 0.19 mm after this
# and at 83 mm without it.
HUTUBS_TO_MM = 1000.0


# Skinned head meshes by subject. Ambient occlusion and the blotching are
# per-vertex and frame-invariant, so the two ears of one head share them.
_SKIN_CACHE = {}


def load_ear(name, dataset="hutubs", cfg=None, version=None):
    """Head, fitted ear and landmarks for one ear, all in canonical millimetres.

    The head is the only thing transformed: corpus/canon2scan's `inverse` takes
    the scan to canonical, and the scan is the raw mesh scaled to millimetres.
    The fitted ear and the landmarks are STORED in canonical mm, so they are
    returned untouched -- which is the whole reason for working in this frame.

    THE HEAD IS SKINNED, and skipping that is not cosmetic. A raw HUTUBS mesh
    renders near-white and almost flat, and the detector -- which is trained on
    photographs -- simply does not fire on it: measured over 12 ears on a 7x7
    grid, 64.5% of ear-views returned NO detection, 50% of them face-on, on
    renders where the ear is large, centred and unoccluded. Skinned, that
    collapses. Every other script that feeds this detector a render calls
    apply_skin first; cfg's tone 10 is documented as the tone the landmarker
    reads most cleanly. An unskinned audit measures the detector's response to
    an out-of-domain image, and worse, it measures escapes only on the subset of
    views where the detector coped with a bad one -- a biased sample.
    """
    cfg = cfg or DEFAULTS
    # One generation for mesh, transform and landmarks (ear3d/corpus.py):
    # `version`, default the current one. load_ear BUILDS the head caches, which
    # record it, so a cache and its GT always pair up.
    version = version or _corpus.CURRENT
    tp = WRAP / _corpus.get("canon2scan", version) / f"{name}.npz"
    if not tp.exists():
        raise FileNotFoundError(f"{tp} not found -- run scripts/corpus/06_canonical_to_scan.py")
    z = np.load(tp, allow_pickle=True)
    if str(z["dataset"]) != dataset:
        raise ValueError(f"{name} is {str(z['dataset'])!r}, not {dataset!r}")

    ear = o3d.io.read_triangle_mesh(
        str(WRAP / _corpus.get("canonical", version) / dataset / f"{name}.obj"))
    if len(ear.vertices) == 0:
        raise FileNotFoundError(f"no fitted ear for {name}")
    ear.compute_vertex_normals()

    # THE SCAN COMES FROM canon2scan, NOT FROM A DATASET-SHAPED PATH. The
    # npz already records which file its transform was built against, so the
    # same code serves HUTUBS, whose scan is a whole head in METRES, and
    # SONICOM, whose scan is a graded head-sized mesh already in mm. The
    # SONICOM entries point at sonicom_graded and not at
    # sonicom_raw_headtorso: the two are in DIFFERENT FRAMES, 347 mm apart
    # and not by a translation, and only the graded one is what the wrap was
    # fitted against. Rendering the head+torso instead would put every
    # landmark on nothing.
    scan = pathlib.Path(str(z["scan"]))
    if not scan.exists():
        raise FileNotFoundError(f"{name}: scan {scan} not found")
    key = str(scan)
    if key not in _SKIN_CACHE:
        h = o3d.io.read_triangle_mesh(key)
        if len(h.vertices) == 0:
            raise FileNotFoundError(f"no scan mesh for {name} at {scan}")
        h.compute_vertex_normals()
        if cfg.get("skin"):
            from .skin import apply_skin
            apply_skin(h, seed=cfg["skin_seed"], tone=cfg["tone"],
                       ao_strength=cfg["ao_strength"], n_rays=cfg["ao_rays"],
                       noise=cfg.get("skin_noise", True))
        _SKIN_CACHE[key] = h
    head = o3d.geometry.TriangleMesh(_SKIN_CACHE[key])
    # Units by extent, the convention gt_landmark_check already uses: a head
    # measured in metres spans under 10, one in millimetres spans hundreds.
    # HUTUBS files are metres, SONICOM's are mm, and canon2scan's `units`
    # field describes the CANONICAL frame rather than the file, so it cannot
    # be used for this.
    V = np.asarray(head.vertices)
    V = V * (HUTUBS_TO_MM if float(np.ptp(V, axis=0).max()) < 10.0 else 1.0)
    V = (np.c_[V, np.ones(len(V))] @ z["inverse"].T)[:, :3]
    head.vertices = o3d.utility.Vector3dVector(V)

    # A REFLECTION NEEDS THE WINDING REVERSED. canon2scan's `inverse` has
    # det < 0 for every right ear -- that is how is_reflected() tells them
    # apart -- so applying it mirrors the vertices and leaves every triangle
    # wound the wrong way round. compute_vertex_normals() then reads that
    # winding and hands the renderer normals pointing INTO the head: measured
    # against the direction away from the mesh centroid, +0.72 on a left ear
    # and -0.72 on the right ear of the same subject, 93% of vertices against
    # 7%. The depth buffer does not care, so the lift was never affected, but
    # the IMAGE is shaded from the inside out and every right-ear render in
    # this project before this line was darker and flatter than the left one
    # beside it. Any left-against-right comparison made on those renders was
    # measuring this.
    if np.linalg.det(z["inverse"][:3, :3]) < 0:
        head.triangles = o3d.utility.Vector3iVector(
            np.asarray(head.triangles)[:, ::-1].copy())
    head.compute_vertex_normals()

    lm = np.load(WRAP / "templates" / _corpus.get("gt_lm3d", version), allow_pickle=True)   # same generation as tp
    names = [str(x) for x in lm["names"]]
    if name not in names:
        raise KeyError(f"{name} has no stored landmarks in the reference lm3d")
    P = lm["landmarks"][names.index(name)][:55]
    return head, ear, P


def is_reflected(name, dataset="hutubs", version=None):
    """Did this ear reach the canonical frame through a REFLECTION?

    Right ears mirror about Z to get there, so their surface is handed the
    opposite way round to a left ear's in the same frame. A world-fixed sun
    then lights the two sides' anatomy from opposite directions, which costs
    a factor of two in gross errors -- see ear3d.render.sun_world. Read from
    the stored matrix rather than the "_right" suffix, because the determinant
    is the invariant and the naming is a convention.
    """
    z = np.load(WRAP / _corpus.get("canon2scan", version) / f"{name}.npz", allow_pickle=True)
    return bool(np.linalg.det(np.asarray(z["inverse"], float)[:3, :3]) < 0)


# Where the sun sits relative to the EAR'S ANATOMY, independently of which way
# round the geometry is handed. is_reflected() says whether this ear reached the
# canonical frame through a reflection, so mirroring the sun for exactly those
# ears is what keeps the sun in the ANATOMY's frame rather than the coordinate
# frame. The three settings are the whole lighting axis:
#
#   anatomy   sun grazes the occluding contour on every ear      (the fix)
#   opposite  sun front-lights the occluding contour on every ear (the worst case)
#   frame     sun fixed in the canonical frame, so it grazes one
#             side's anatomy and front-lights the other          (the old bug)
#
# "frame" is not a third lighting condition so much as the AVERAGE of the other
# two taken one ear-side at a time, which is why it reads as a middling number
# while hiding a factor of two between the sides.
LIGHT_ANCHORS = ("anatomy", "opposite", "frame")


def light_mirror_for(name, anchor, dataset="hutubs", version=None):
    """Should this ear's sun have its x flipped, under this anchor?"""
    if anchor == "frame":
        return False
    refl = is_reflected(name, dataset, version)
    return refl if anchor == "anatomy" else (not refl)


