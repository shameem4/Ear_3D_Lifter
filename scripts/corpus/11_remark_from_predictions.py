#!/usr/bin/env python3
"""Re-derive the 55 template marks for a canonical corpus (the v9 recipe).

WHY. A mark is a (triangle, barycentric) address on the template, and where it
lands on a real ear depends on how the FIT spread the template over that ear.
Landmark-posed fits (lmwrap_v1) sit on the same surface as escalated_v1 but
slide 2-5 mm along the helix and crus curves, so v9's marks evaluated on them
moved the ground truth 2.3 mm (pipeline error 1.49 -> 2.77 mm with no change
to the pipeline). The marks have to be re-derived on each fit generation.

THE RECIPE, as v9 (wrap_work/templates/v9_provenance):
  1. curve SHAPE: RED, the landmarker run multiview on the mean template
     (tpl_marks.npz all_25, on ear_mean_v4), carried to the new mean template
     by its mesh address -- same connectivity, the templates differ <= 2.6 mm;
  2. curve SPACING: every head ear's final pipeline prediction, mapped into
     its canonical fit, read off as a template address, median over ears
     (v8_corpus did this on 80 HUTUBS ears; here all head ears);
  3. RED re-spaced along each strip to that spacing; canal 39 unchanged.

SCORED SPLIT-HALF BY SUBJECT: marks learned on one half, scored on the other,
so no ear is scored against marks its own subject helped make.

    python scripts/corpus/11_remark_from_predictions.py --canonical corpus/canonical_lmw \\
        --canon2scan corpus/canon2scan_lmw --preds DIR_OF_NPZ_WITH_P \\
        --template templates/ear_mean_v5.obj --out templates/mark_mean_v10_55.npz
"""
import argparse
import glob
import pathlib
import sys
import zlib

import numpy as np
import open3d as o3d

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ear3d.wrap import load_mesh                                   # noqa: E402

WRAP = ROOT.parent / "wrap_work"   # == ear3d.corpus.WRAP
STRIPS = ((0, 20), (20, 35), (35, 39), (40, 50), (50, 55))        # canal 39 is its own point
RED_SRC = WRAP / "templates/v9_provenance/tpl_marks.npz"           # all_25: xyz on ear_mean_v4
RED_TPL = WRAP / "templates/ear_mean_v4.obj"


def scene(V, F):
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    return s


def address(sc, X):
    cp = sc.compute_closest_points(o3d.core.Tensor(np.asarray(X, np.float32)))
    uv = cp["primitive_uvs"].numpy()
    return cp["primitive_ids"].numpy().astype(int), np.c_[1 - uv.sum(1), uv]


def at(V, F, tri, bary):
    return np.einsum("ij,ijk->ik", bary, V[F[tri]])


def on_poly(curve, X):
    """Arc-length position of each X along the polyline (ends extrapolate a little)."""
    seg = np.diff(curve, axis=0); L = np.linalg.norm(seg, axis=1); cum = np.r_[0, np.cumsum(L)]
    out = []
    for x in X:
        best = (1e9, 0.0)
        for i in range(len(seg)):
            t = np.clip((x - curve[i]) @ seg[i] / L[i] ** 2, -0.5 if i == 0 else 0, 1.5 if i == len(seg) - 1 else 1)
            best = min(best, (float(np.linalg.norm(x - curve[i] - t * seg[i])), cum[i] + t * L[i]))
        out.append(best[1])
    return np.array(out), cum, seg, L


def at_pos(curve, s):
    _, cum, seg, L = on_poly(curve, curve[:1])
    i = np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1)
    return curve[i] + ((s - cum[i]) / L[i])[:, None] * seg[i]


def respace(red, spacing_xyz):
    X = red.copy()
    for a, b in STRIPS:
        s, *_ = on_poly(red[a:b], spacing_xyz[a:b]); X[a:b] = at_pos(red[a:b], s)
    return X


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--canonical", required=True)
    p.add_argument("--canon2scan", required=True)
    p.add_argument("--preds", required=True, help="dir of <ear>.npz holding P (55+,3) in scan mm")
    p.add_argument("--template", required=True)
    p.add_argument("--compare", nargs="*", default=[],
                   help="lm3d files to score the same predictions against (old corpora use canon2scan)")
    p.add_argument("--out", required=True)
    p.add_argument("--red", default=str(RED_SRC), help="step 11a output (all_25); default: the old model's curves (v9/v11)")
    p.add_argument("--red-template", default=str(RED_TPL), help="the mesh --red was made on")
    a = p.parse_args()
    res = lambda s: pathlib.Path(s) if pathlib.Path(s).is_absolute() else WRAP / s
    can, c2s = res(a.canonical), res(a.canon2scan)
    TV, TF = (np.asarray(x) for x in load_mesh(str(res(a.template))))
    tsc = scene(TV, TF)

    # 1. RED (curve shape, step 11a) carried to the new template by address
    RV, RF = (np.asarray(x) for x in load_mesh(str(res(a.red_template))))
    assert (RF == TF).all(), "RED needs the template connectivity"
    red = at(TV, TF, *address(scene(RV, RF), np.load(res(a.red))["all_25"]))

    # 2. per-ear template positions of the final predictions
    ears = {}
    for f in sorted(glob.glob(str(res(a.preds) / "*.npz"))):
        n = pathlib.Path(f).stem
        tf = c2s / f"{n}.npz"
        if not tf.exists():
            continue
        P = np.load(f)["P"][:55]; ok = np.isfinite(P).all(1)
        Minv = np.load(tf)["inverse"]
        X = P @ Minv[:3, :3].T + Minv[:3, 3]                       # scan mm -> canonical mm
        ds = "hutubs" if n.startswith("pp") else "sonicom"
        FV, FF = (np.asarray(x) for x in load_mesh(str(can / ds / f"{n}.obj")))
        tri, bary = address(scene(FV, FF), np.where(ok[:, None], X, 0))
        tp = at(TV, TF, tri, bary); tp[~ok] = np.nan
        subj = n.rsplit("_", 1)[0]
        ears[n] = dict(P=P, tpos=tp, M=np.load(tf)["matrix"], FV=FV, FF=FF, half=zlib.crc32(subj.encode()) % 2)
    print(f"{len(ears)} head ears with predictions and a canonical fit")

    def marks(keys):
        sp = np.nanmedian([ears[k]["tpos"] for k in keys], axis=0)
        X = respace(red, sp); X[39] = red[39]
        return address(tsc, X), sp

    def place(k, tri, bary):
        e = ears[k]; c = at(e["FV"], e["FF"], tri, bary)
        return c @ e["M"][:3, :3].T + e["M"][:3, 3]

    # 3. split-half score
    D = {"re-derived (held-out half)": [], "corpus median, no RED (held-out half)": []}
    for q in (0, 1):
        (t_, b_), sp = marks([k for k in ears if ears[k]["half"] != q])
        t8, b8 = address(tsc, sp)
        for k in [k for k in ears if ears[k]["half"] == q]:
            P = ears[k]["P"]
            D["re-derived (held-out half)"].append(np.linalg.norm(P - place(k, t_, b_), axis=1))
            D["corpus median, no RED (held-out half)"].append(np.linalg.norm(P - place(k, t8, b8), axis=1))
    for f in a.compare:                                             # e.g. lm3d_v9 on the OLD corpus
        z = np.load(res(f), allow_pickle=True); idx = {str(n): i for i, n in enumerate(z["names"])}
        old = WRAP / ("corpus/canon2scan" if "v9" in f or "v7" in f or "v8" in f else a.canon2scan)
        rows = []
        for k, e in ears.items():
            M = np.load(res(old) / f"{k}.npz")["matrix"] if (res(old) / f"{k}.npz").exists() else e["M"]
            L = z["landmarks"][idx[k], :55]
            rows.append(np.linalg.norm(e["P"] - (L @ M[:3, :3].T + M[:3, 3]), axis=1))
        D[pathlib.Path(f).stem] = rows
    print("pipeline predictions vs each reference, per-ear median over 55 landmarks:")
    for lab, rows in D.items():
        R = np.array(rows); m = np.nanmedian(R, axis=1)
        print(f"  {lab:40s} median {np.nanmedian(m):.3f} mm  p95 {np.nanpercentile(m, 95):.2f}  "
              f"helix {np.nanmedian(R[:, :35]):.2f}  rest {np.nanmedian(R[:, 35:]):.2f}")

    # 4. final marks from every ear
    (tri, bary), sp = marks(list(ears))
    xyz = at(TV, TF, tri, bary)
    np.savez(res(a.out), triangle=tri, barycentric=bary, xyz=xyz, spacing_xyz=sp,
             note=f"v9 recipe on {can.name}: RED (landmarker multiview on the mean template) carried to "
                  f"{pathlib.Path(a.template).name}, every strip re-spaced to the median of {len(ears)} head ears' "
                  f"final pipeline predictions read through their {can.name} fits; canal 39 unchanged.")
    print(f"wrote {res(a.out)}")


if __name__ == "__main__":
    main()
