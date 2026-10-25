"""Wrap the ear template onto a head mesh, posed by its 55 3D landmarks.

    from ear3d.faceform import wrap_ear, fit_faces
    V = wrap_ear(head_V, head_F, P55, side, work_dir)   # (1188, 3) in the head's frame
    F = fit_faces(side)                                 # right ears: winding reversed

One frame throughout: the landmarks come from ear3d.pipeline run on this
same head, the ear is cut from this same head around them, and the fit comes
back in the head's coordinates.

THE LANDMARKS ONLY POSE. Faceform's RigidAlignment gets the 55 pairs
(template landmark vertex <-> landmark on the cut) with rotation,
translation AND scale; Wrapping then runs with an EMPTY point list, so the
fit itself is the usual automatic one. Before this, the pose came from a
pre-scale by the cut's extent (which includes head) and a trimmed, scaled
ICP: trimming keeps the matches well inside the ear, the scale keeps
shrinking, and P0057_left / P0075_left / P0022_right came out with the
helix top ~15 mm short of the rim. Posed from landmarks all three cover the
ear; good ears move 0.1-0.3 mm.

Right ears are mirrored in z before fitting (the template is a left ear)
and mirrored back after; any reflection works, since the pose is a full
similarity.

Also the WrapCmd project plumbing every Faceform caller shares (WRAPCMD, and
the node helpers that edit a saved .wrap project). Moved here 2026-10-20 from
archive/scripts/wrap_from_landmarks.py, wrap_batch_faceform.py and refit_variants.py;
the code is unchanged. The Faceform Wrap EULA forbids decompiling or reverse
engineering it: this only edits project files and runs its CLI.
"""
import json
import pathlib
import subprocess

import numpy as np
import trimesh

from .corpus import WRAP

WRAPCMD = pathlib.Path("/home/shameem/snap/Faceform_Wrap/"
                       "Faceform_Wrap_2025.11.14_Linux/WrapCmd.sh")


def corner_barycentric(F):
    """For each vertex, one triangle containing it and which corner it is."""
    out = {}
    for t, f in enumerate(F):
        for c, v in enumerate(f):
            out.setdefault(int(v), (t, c))
    return out


def nodes_by_type(proj, t):
    return [n for n in proj["nodes"].values() if n["nodeType"] == t]


def one(proj, t):
    return nodes_by_type(proj, t)[0]


def connect(node, param, target):
    node["params"][param]["connectedNodeId"] = target["nodeId"]


TEMPLATE = WRAP / "templates/ear_L_template.obj"
BASE = WRAP / "ear_wrap_pp12_left.wrap"
MARKS = WRAP / "templates/lm3d_v9_canonical_mm.npz"
CUT_FACTOR = 0.8      # cut radius = this x the landmarks' largest extent, about their centroid
WORK_SIZE = 100.0     # Wrap's parameters are tuned for a scan ~100 units across
UV = [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]


def _template():
    m = trimesh.load(TEMPLATE, process=False)
    Vt, Ft = np.asarray(m.vertices), np.asarray(m.faces)
    z = np.load(MARKS, allow_pickle=True)
    L = (Vt[Ft[z["triangle"][:55]]] * z["barycentric"][:55][:, :, None]).sum(1)
    lm_vert = [int(np.argmin(np.linalg.norm(Vt - p, axis=1))) for p in L]
    cn = corner_barycentric(Ft)
    left = [[int(cn[v][0]), *UV[cn[v][1]]] for v in lm_vert]
    return Vt, Ft, left


def _wrap_env():
    """The environment WrapCmd needs. Importing cv2 (the pipeline does) points
    QT_QPA_PLATFORM_PLUGIN_PATH, QT_QPA_FONTDIR and LD_LIBRARY_PATH at OpenCV's
    bundled Qt; Wrap then loads the wrong xcb plugin and aborts (signal 6)
    without an ERROR line -- seen as 'no output' on every AudioEar ear."""
    import os
    env = {k: v for k, v in os.environ.items() if not k.startswith("QT_QPA_")}
    ld = [d for d in env.get("LD_LIBRARY_PATH", "").split(":") if d and "cv2" not in d]
    if ld:
        env["LD_LIBRARY_PATH"] = ":".join(ld)
    else:
        env.pop("LD_LIBRARY_PATH", None)
    return env


INLIER_MM = 5.0       # a landmark further than this from the robust similarity is dropped
RANSAC_ITERS = 400


def _similarity(A, B):
    a, b = A.mean(0), B.mean(0); A0, B0 = A - a, B - b
    U, S, Vt = np.linalg.svd(A0.T @ B0)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (A0 ** 2).sum()
    return lambda X: s * (X - a) @ R + b


def robust_inliers(P, seed=0):
    """Landmarks that agree with a rigid + scale placement of the template.

    RANSAC over 4-point similarities of the template's 55 landmark points
    onto P, then a refit on the best consensus. A bad landmark set (005_right,
    037_left: 6-8 mm landmark shifts between passes) drags a least-squares
    pose off the ear; these are the points to drop before RigidAlignment, and
    their count is the confidence of the whole set.
    Returns (mask, median inlier residual mm)."""
    Vt, Ft, _ = _template()
    z = np.load(MARKS, allow_pickle=True)
    T = (Vt[Ft[z["triangle"][:55]]] * z["barycentric"][:55][:, :, None]).sum(1)
    ok = np.isfinite(P).all(1); idx = np.flatnonzero(ok)
    rng = np.random.default_rng(seed); best = np.zeros(55, bool)
    for _ in range(RANSAC_ITERS):
        pick = rng.choice(idx, 4, replace=False)
        f = _similarity(T[pick], P[pick])
        inl = ok & (np.linalg.norm(f(T) - np.where(ok[:, None], P, 0), axis=1) < INLIER_MM)
        if inl.sum() > best.sum():
            best = inl
    for _ in range(2):                                     # refit on the consensus
        f = _similarity(T[best], P[best])
        best = ok & (np.linalg.norm(f(T) - np.where(ok[:, None], P, 0), axis=1) < INLIER_MM)
    r = np.linalg.norm(f(T) - np.where(ok[:, None], P, 0), axis=1)
    return best, float(np.median(r[best]))


def _anchor_pairs(anchors, P, Vs, Fs, mir):
    """Wrapping control points for landmark indices `anchors` (all of them,
    whether or not the robust pose kept them), as (template, scan) pairs."""
    _, _, left_all = _template()
    sm = trimesh.Trimesh(Vs, Fs, process=False)
    js = [j for j in anchors if np.isfinite(P[j]).all()]
    cl, _, tid = trimesh.proximity.closest_point(sm, (P[js] * mir))
    bb = trimesh.triangles.points_to_barycentric(Vs[Fs[tid]], cl)
    keep = [i for i in range(len(js)) if np.isfinite(bb[i]).all()]
    return ([left_all[js[i]] for i in keep],
            [[int(tid[i]), float(bb[i][0]), float(bb[i][1])] for i in keep])


def fit_faces(side):
    """Faces for a fit returned by wrap_ear. A right ear's vertices are a
    mirrored left fit, so its triangles must be wound the other way or every
    normal points into the head (checked: 0% vs 100% agreement with the scan)."""
    Ft = np.asarray(trimesh.load(TEMPLATE, process=False).faces)
    return Ft[:, ::-1] if side == "R" else Ft


def cut_ear(V, F, P):
    """Faces of the head within CUT_FACTOR x the landmark extent of their centroid."""
    c = P.mean(0); r = CUT_FACTOR * float(np.ptp(P, axis=0).max())
    keep = np.linalg.norm(V[F].mean(1) - c, axis=1) < r
    m = trimesh.Trimesh(V, F[keep], process=False)
    m.merge_vertices()                     # STL (SONICOM) stores every triangle's own corners
    m.remove_unreferenced_vertices()
    parts = m.split(only_watertight=False)                # the patch the landmarks sit on
    if len(parts) > 1:
        m = max(parts, key=lambda p: (trimesh.proximity.closest_point(p, P)[1] < 2).sum())
    return m


def wrap_ear(V, F, P, side, work_dir, name="ear", cut=True, use=None, anchors=(), exclude=None):
    """Fitted template vertices (base resolution) in the input's frame, or raise.

    cut=False fits (V, F) as given -- an ear-only surface such as a meshed
    AudioEar3D cloud -- instead of cutting the ear out of a head first.

    exclude: template triangle indices Wrapping does NOT fit (its `polygons`
    input excludes; verified on 006_right: passing the front made the front
    fit worse, 10.8% -> 24.8% of the scan off, passing the back made it
    better, 10.8% -> 4.8%). Tried on AudioEar3D with the back of the rim
    (templates/ear_L_template_back_exclusion.npz), since a one-sided scan
    holds nothing there: over 39 ears 15 better, 14 worse (2026-10-25), not
    adopted; the corpus wraps are unchanged.

    anchors: landmark indices also given to Wrapping as control points. A
    RESCUE, never a default: (15, 17, 19) pulls a lobe the automatic fit left
    short (P0082_left: lobe landmark 8.2 mm off the fit, 1.8 mm off the old
    one). Anchors may be landmarks the robust pose dropped -- that is the case
    they exist for."""
    P = np.asarray(P, float)
    if not np.isfinite(P).all():
        ok = np.isfinite(P).all(1)
        if ok.sum() < 20:
            raise ValueError(f"only {ok.sum()} finite landmarks")
    else:
        ok = np.ones(55, bool)
    if use is not None:                                    # e.g. robust_inliers(P)[0]
        ok = ok & use
        if ok.sum() < 12:
            raise ValueError(f"only {ok.sum()} usable landmarks")
    mir = np.array([1.0, 1.0, -1.0]) if side == "R" else np.ones(3)
    if cut:
        m = cut_ear(np.asarray(V, float), np.asarray(F), P[ok])
        V, F = m.vertices, m.faces
    Vs, Fs = np.asarray(V, float) * mir, np.asarray(F)
    if side == "R":
        Fs = Fs[:, ::-1]
    Pm = P * mir
    Vt, Ft, left = _template()
    sm = trimesh.Trimesh(Vs, Fs, process=False)
    cl, _, tid = trimesh.proximity.closest_point(sm, Pm[ok])
    bb = trimesh.triangles.points_to_barycentric(Vs[Fs[tid]], cl)
    right = [[int(t), float(b[0]), float(b[1])] for t, b in zip(tid, bb)]
    left = [l for l, k in zip(left, ok) if k]
    # a landmark that lands on a near-zero-area triangle (smoothed cloud
    # meshes) gets NaN barycentrics, and Wrap refuses the whole project
    fin = [bool(np.isfinite(r[1:]).all()) for r in right]
    left = [l for l, f in zip(left, fin) if f]; right = [r for r, f in zip(right, fin) if f]
    if len(right) < 12:
        raise ValueError(f"only {len(right)} usable landmark pairs")

    work = WORK_SIZE / float(np.ptp(Vs, axis=0).max())
    wd = pathlib.Path(work_dir); wd.mkdir(parents=True, exist_ok=True)
    sp, dst = wd / f"{name}_scan.obj", wd / f"{name}_fit.obj"
    trimesh.Trimesh(Vs * work, Fs, process=False).export(sp)
    # the template goes in at roughly the right size; RigidAlignment does the rest
    s0 = float(np.ptp(Pm[ok], axis=0).max()) / float(np.ptp(Vt, axis=0).max()) * 1.3
    tp = wd / f"{name}_tpl.obj"
    trimesh.Trimesh(((Vt - Vt.mean(0)) * s0 + Pm[ok].mean(0)) * work, Ft, process=False).export(tp)

    proj = json.load(open(BASE)); by = {n["nodeId"]: n for n in proj["nodes"].values()}
    rig, wrp = one(proj, "RigidAlignment"), one(proj, "Wrapping")
    for k in ("fitRotation", "fitTranslation", "fitScale"):
        rig["params"][k]["value"] = True
    by[rig["params"]["geomFloating"]["connectedNodeId"]]["params"]["fileName"]["value"] = str(tp)
    by[rig["params"]["geomFixed"]["connectedNodeId"]]["params"]["fileName"]["value"] = str(sp)
    for n in (by[rig["params"]["geomFloating"]["connectedNodeId"]], by[rig["params"]["geomFixed"]["connectedNodeId"]]):
        n["params"]["scale"]["value"] = 1
    pr = by[rig["params"]["points"]["connectedNodeId"]]
    pr["params"]["pointsLeft"]["value"], pr["params"]["pointsRight"]["value"] = left, right
    pw = by[wrp["params"]["points"]["connectedNodeId"]]
    pw["params"]["pointsLeft"]["value"], pw["params"]["pointsRight"]["value"] = ([], []) if not anchors else \
        _anchor_pairs(anchors, P, Vs, Fs, mir)
    sav = one(proj, "SaveGeom"); sav["params"]["fileName"]["value"] = str(dst)
    connect(sav, "geom", one(proj, "SubdivideGeom"))          # no ProjectGeom
    if exclude is not None:           # Wrapping's `polygons`: template triangles NOT fitted; they follow by smoothness
        nid = max(n["nodeId"] for n in proj["nodes"].values()) + 1
        proj["nodes"]["SelectPolygons"] = {"isEnabled": True, "nodeId": nid, "nodeType": "SelectPolygons", "params": {
            "displayMode": {"value": 0}, "fileName": {"value": ""}, "invisiblePolygonIndices": {"value": []},
            "selection": {"value": [int(i) for i in exclude]}, "source": {"value": 0},
            "geom": {"connectedNodeId": rig["params"]["geomFloating"]["connectedNodeId"]}}}
        connect(wrp, "polygons", proj["nodes"]["SelectPolygons"])
        proj["layout"]["items"].append({"itemId": max(i["itemId"] for i in proj["layout"]["items"]) + 1,
                                        "itemType": "node", "nodeId": nid, "x": 0, "y": 400})
    proj["timeline"] = {"current": 0, "min": 0, "max": 0}
    pj = wd / f"{name}.wrap"; json.dump(proj, open(pj, "w"), indent=4)
    r = subprocess.run([str(WRAPCMD), "compute", str(pj), "-s", "0", "-e", "0"],
                       capture_output=True, text=True, timeout=1800, env=_wrap_env())
    bad = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("ERROR")]
    if bad or r.returncode != 0 or not dst.exists():
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        raise RuntimeError(bad[0] if bad else f"Wrap exit {r.returncode}: {' | '.join(tail)}")
    Vf = np.asarray(trimesh.load(dst, process=False).vertices)[:len(Vt)] / work
    return Vf * mir


