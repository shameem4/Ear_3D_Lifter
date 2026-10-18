"""Wrap the ear template onto a head mesh, posed by its 55 3D landmarks.

    from wrap_from_landmarks import wrap_ear
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

Usage (test driver): wrap_from_landmarks.py OUT_DIR EAR [EAR...]
  reads the head and pipeline landmarks from runs/corpus_pred/<EAR>.npz.
"""
import json
import pathlib
import subprocess
import sys

import numpy as np
import trimesh

R0 = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R0), str(R0 / "scripts")]
from wrap_batch_faceform import WRAPCMD, corner_barycentric     # noqa: E402
from refit_variants import connect, nodes_by_type, one          # noqa: E402

WRAP = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work")
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


def wrap_ear(V, F, P, side, work_dir, name="ear", cut=True):
    """Fitted template vertices (base resolution) in the input's frame, or raise.

    cut=False fits (V, F) as given -- an ear-only surface such as a meshed
    AudioEar3D cloud -- instead of cutting the ear out of a head first."""
    P = np.asarray(P, float)
    if not np.isfinite(P).all():
        ok = np.isfinite(P).all(1)
        if ok.sum() < 20:
            raise ValueError(f"only {ok.sum()} finite landmarks")
    else:
        ok = np.ones(55, bool)
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
    pw["params"]["pointsLeft"]["value"], pw["params"]["pointsRight"]["value"] = [], []
    sav = one(proj, "SaveGeom"); sav["params"]["fileName"]["value"] = str(dst)
    connect(sav, "geom", one(proj, "SubdivideGeom"))          # no ProjectGeom
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


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    for ear in sys.argv[2:]:
        z = np.load(R0 / f"runs/corpus_pred/{ear}.npz")
        h = trimesh.load(str(z["scan"]), process=False)
        V = np.asarray(h.vertices); V = V * (1000.0 if np.ptp(V, 0).max() < 10 else 1.0)
        P = z["pred"] @ z["Rr"]                                # back to the scan's own frame
        side = "L" if ear.endswith("left") else "R"
        try:
            Vf = wrap_ear(V, np.asarray(h.faces), P, side, out / "work", ear)
        except Exception as e:
            print(ear, "FAILED", e, flush=True); continue
        np.savez(out / f"{ear}.npz", V=Vf, P=P, scan=str(z["scan"]))
        print(ear, f"fit extent {np.ptp(Vf, 0).max():.1f} mm, landmark spread {np.ptp(P, 0).max():.1f} mm", flush=True)
