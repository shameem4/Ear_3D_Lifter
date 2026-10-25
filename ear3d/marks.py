"""Template landmark marks: reading them, and replaying a landmark-vertex recipe.

A mark is a (triangle, barycentric) address on the template connectivity;
see memory marks-are-fit-coupled for why marks are re-derived per fit
generation. Moved here 2026-10-20 from archive-final:archive/scripts/gt_landmark_check.py
(read_marks) and archive-final:archive/scripts/landmark_template.py (place, replay), unchanged.
"""
import json
import pathlib
import sys

import numpy as np

from .corpus import WRAP

# The marks the reference is built from. v6_spacing keeps the hand-marked
# curve and takes only the SPACING along it from the shipped landmarker, which
# is the half of the old disagreement that was a convention rather than an
# error. Scored on hutubs pp10_right, RANSAC consensus: 6.08 mm against the
# hand marks, 3.61 against these. Pass --marks to compare with another file.
DEFAULT_MARKS = WRAP / "templates/mark_mean_v6_spacing.wrap"


def read_marks(path):
    """(triangle, barycentric, names) from a Wrap SelectPoints project or npz.

    The same convention as archive-final:archive/scripts/wrap_marks_to_npz.py: u weights the
    triangle's first vertex, v the second, the remainder the third.
    """
    path = pathlib.Path(path)
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        return (z["triangle"], z["barycentric"],
                [str(n) for n in z["point_names"]])
    d = json.loads(path.read_text())
    sel = [n for n in d["nodes"].values() if n["nodeType"] == "SelectPoints"]
    if not sel:
        sys.exit(f"{path}: no SelectPoints node")
    pts = sel[0]["params"]["points"]["value"]
    n = len(pts)
    tri, bary, names = np.zeros(n, int), np.zeros((n, 3)), [""] * n
    for k, q in pts.items():
        j = int(k[:2])
        names[j] = k
        u, v = float(q["u"]), float(q["v"])
        tri[j], bary[j] = int(q["triangleInd"]), (u, v, 1.0 - u - v)
    return tri, bary, names


def place(V, F, tri, bary):
    return np.einsum("ij,ijk->ik", bary, V[F[tri]])


def replay(V, recipe):
    """Apply a stored insertion to another mesh of the ORIGINAL topology."""
    L = place(V, recipe["F_orig"], recipe["insert_tri"], recipe["insert_bary"])
    return np.vstack([L, V])


