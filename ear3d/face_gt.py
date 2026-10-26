"""Pseudo ground truth for the face lift: MediaPipe's canonical face fitted to a head with Faceform Wrap.

Canonical vertex i IS landmark i, so the fit's vertices 0-467 are the landmarks (midpoint subdivision for
density leaves them in place; the iris, 468-477, is not in the mesh). The fit is driven by the user's
hand-placed control points, matched BY NAME between canonical.wrap and each head's project
(wrap_work/face_points/projects; scripts/face/proxy_gt.py). Ported from Face_3D_Lifter's face3d/wrap.py,
without its WrapHead two-stage fit (superseded): nothing here derives from WrapHead.
"""
import json, pathlib, subprocess
import numpy as np, open3d as o3d, trimesh
from .faceform import WRAPCMD, BASE, one, connect, _wrap_env, _similarity
from .face import canonical_raw, canonical_dense

N_LM = 468
INLIER_MM = 6.0
WORK_SIZE = 100.0            # Wrap's parameters are tuned for a scan ~100 units across


def canonical(levels=2):
    """canonical_raw, midpoint-subdivided for density (mm)."""
    return canonical_dense(levels=levels)


def wrap_points(pj):
    """Named points of a Wrap project's SelectPoints node: name -> {triangleInd, u, v}."""
    p = json.load(open(pj))
    return [n for n in p["nodes"].values() if n["nodeType"] == "SelectPoints"][0]["params"]["points"]["value"]


def xyz(V, F, q):
    return V[F[q["triangleInd"]]].T @ np.array([q["u"], q["v"], 1 - q["u"] - q["v"]])


def robust_similarity(T, Q, seed=0):
    """Similarity taking template points T onto target points Q, RANSAC over 4-point sets. -> (mask, f)."""
    ok = np.isfinite(Q).all(1) & np.isfinite(T).all(1); idx = np.flatnonzero(ok)
    rng = np.random.default_rng(seed); best = np.zeros(len(T), bool)
    for _ in range(300):
        pick = rng.choice(idx, 4, replace=False); f = _similarity(T[pick], Q[pick])
        inl = ok & (np.linalg.norm(f(T) - np.where(ok[:, None], Q, 0), axis=1) < INLIER_MM)
        if inl.sum() > best.sum(): best = inl
    return best, _similarity(T[best], Q[best])


def bary(V, F, P):
    """Closest triangle and barycentric weights of each point of P on (V, F)."""
    m = trimesh.Trimesh(V, F, process=False)
    cl, _, tid = trimesh.proximity.closest_point(m, P)
    return tid, trimesh.triangles.points_to_barycentric(V[F[tid]], cl)


def wrap_template(Vt, Ft, T_pose, V, F, P_pose, work_dir, name="fit", ctrl=None):
    """Fit template (Vt, Ft) onto target (V, F); both mm, any frame.
    T_pose, P_pose: corresponding pose points (k, 3) on template and target (NaN rows allowed); they drive
    RigidAlignment only. ctrl: optional (left, right) Wrapping control points, lists of [triangle, u, v] on
    (Ft, F) paired by position. Returns (fitted template vertices, info)."""
    use, f = robust_similarity(np.asarray(T_pose, float), np.asarray(P_pose, float))
    if use.sum() < 6:
        raise ValueError(f"only {use.sum()} pose points agree")
    placed = f(Vt)
    V, F = np.asarray(V, float), np.asarray(F)
    ttid, tbb = bary(placed, Ft, f(np.asarray(T_pose, float)[use]))
    stid, sbb = bary(V, F, np.asarray(P_pose, float)[use])
    left = [[int(t), float(b[0]), float(b[1])] for t, b in zip(ttid, tbb)]
    right = [[int(t), float(b[0]), float(b[1])] for t, b in zip(stid, sbb)]
    work = WORK_SIZE / float(np.ptp(V, axis=0).max())
    wd = pathlib.Path(work_dir); wd.mkdir(parents=True, exist_ok=True)
    sp, tp, dst = wd / f"{name}_target.obj", wd / f"{name}_tpl.obj", wd / f"{name}_fit.obj"
    trimesh.Trimesh(V * work, F, process=False).export(sp)
    trimesh.Trimesh(placed * work, Ft, process=False).export(tp)          # already posed; RigidAlignment refines
    proj = json.load(open(BASE)); by = {n["nodeId"]: n for n in proj["nodes"].values()}
    rig, wrp = one(proj, "RigidAlignment"), one(proj, "Wrapping")
    for k in ("fitRotation", "fitTranslation", "fitScale"):
        rig["params"][k]["value"] = True
    for side, path in (("geomFloating", tp), ("geomFixed", sp)):
        n = by[rig["params"][side]["connectedNodeId"]]; n["params"]["fileName"]["value"] = str(path); n["params"]["scale"]["value"] = 1
    pr = by[rig["params"]["points"]["connectedNodeId"]]
    pr["params"]["pointsLeft"]["value"], pr["params"]["pointsRight"]["value"] = left, right
    pw = by[wrp["params"]["points"]["connectedNodeId"]]
    pw["params"]["pointsLeft"]["value"], pw["params"]["pointsRight"]["value"] = ctrl if ctrl is not None else ([], [])
    sav = one(proj, "SaveGeom"); sav["params"]["fileName"]["value"] = str(dst)
    connect(sav, "geom", wrp)                                  # no SubdivideGeom: it would move the landmarks
    proj["timeline"] = {"current": 0, "min": 0, "max": 0}
    pj = wd / f"{name}.wrap"; json.dump(proj, open(pj, "w"), indent=4)
    r = subprocess.run([str(WRAPCMD), "compute", str(pj), "-s", "0", "-e", "0"],
                       capture_output=True, text=True, timeout=1800, env=_wrap_env())
    bad = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("ERROR")]
    if bad or r.returncode != 0 or not dst.exists():
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        raise RuntimeError(bad[0] if bad else f"Wrap exit {r.returncode}: {' | '.join(tail)}")
    Vf = np.asarray(trimesh.load(dst, process=False).vertices) / work
    if len(Vf) != len(Vt):
        raise RuntimeError(f"Wrap returned {len(Vf)} vertices, template has {len(Vt)}")
    return Vf, dict(pose_used=np.flatnonzero(use))
