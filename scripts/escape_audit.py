#!/usr/bin/env python3
"""Does a back-projected landmark land on the EAR, or on the scalp behind it?

THE FAILURE THIS MEASURES. The pinna stands off the skull by 15-20 mm. A 2D
landmark predicted slightly outside the pinna's silhouette -- which happens
constantly on the helix rim, where the annotation sits ON the boundary -- sends
its camera ray past the ear and onto the scalp, so the 3D point is too deep by
the whole protrusion. Face-on it looks right; orbit and it swings far more than
it should. That is a GROSS error, tens of millimetres, and no metric in this
repository reports it:

  surface distance   measures distance to ANY surface, and the scalp IS a
                     surface, so an escaped landmark scores well;
  reprojection       rewards cross-view agreement, and every view is landmarked
                     by the same model, so a shared error reprojects nicely;
  paired error        WOULD see it, but is quoted as a median or a per-strip
                     mean over 55 points, and three escapes do not move a
                     median. It has also only ever been run on one or two ears.

So the headline here is an escape RATE at millimetre thresholds, per point index
and per view angle. Never a median.

WHAT THE REFERENCE IS. The wrap fit for this ear, which IS the ear's surface --
it sits on the raw scan at 0.15-0.62 mm median (measured, see below). The scalp
is not part of it, so "distance to the fitted ear" separates pinna from scalp,
which "distance to the scan" cannot: the scalp belongs to the scan.

Note the record claims this reference sits 0.058 mm from the scan and
corpus/canon2scan stores resid_median_mm = 0.0 for all 476 ears. Neither is the
distance to the raw scan. Measured here against clean_3d_data directly, HUTUBS
pp10/pp12 come out at 0.15-0.62 mm median, p90 0.38-1.56, max 2.0-3.1. That is
still negligible against the ~3.6 mm placement error and the 15-20 mm escapes
this script exists to count, so nothing downstream changes -- but the stored
field is a matrix round-trip, not an accuracy, and should not be quoted as one.

THE DECOMPOSITION, which is the point of the script. Two independent facts are
recorded per landmark:

  d_ear_mm   distance from the back-projected 3D point to the fitted ear
  on_mask    whether the landmark's PIXEL lies on the ear's own silhouette,
             rendered from the fitted ear through the same camera

They separate two failures that need different fixes and that nothing so far
has told apart:

  off-mask           the 2D detector put the point outside the pinna.
                     Back-projection then did its job faithfully on a bad pixel.
                     The fix is 2D, or a clamp onto the silhouette.
  on-mask but far    the pixel is on the ear and the 3D point still is not.
                     That is the depth sample itself -- silhouette bleed, or a
                     cliff pixel resolving to the far surface.

WHY CANONICAL MILLIMETRES. The ear is already axis-aligned there: +y is the long
axis (ascending helix +35 mm, lobe -29 mm on pp10_right) and x is the thin axis
at 12.8 mm extent, so x IS the pinna-plane normal. Working in that frame means
the fitted ear and the stored landmarks need no transform at all, and the only
matrix involved is corpus/canon2scan's own verified `inverse`, applied to the
head. Every other composition in this repository's history cost a frame bug.

FRAMING MATCHES label_two_pass, deliberately: centre on the landmark centroid,
scale by the largest centred coordinate so the ear spans ~2 units, which is what
cfg's eye_z and vfov are set for. The difference is that the frame here comes
from the GROUND TRUTH rather than from a first landmarking pass, which isolates
back-projection from the framing error that pass is subject to. This measures
the lift, not the search for the ear.

HUTUBS only for now: its head meshes sit in clean_3d_data/3D head meshes in
metres, verified by the ×1000 landing the fit on the scan at 0.19 mm instead of
83 mm. SONICOM's raw scans need their own units check before they are added.

    python scripts/escape_audit.py --ears pp10_right --views 9 --render
    python scripts/escape_audit.py --all --out runs/escape_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np
import open3d as o3d

ROOT = pathlib.Path(__file__).resolve().parent.parent
WRAP = ROOT.parent / "wrap_work"
RAW = ROOT.parent / "clean_3d_data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ear3d import DEFAULTS, Camera                                 # noqa: E402
from ear3d.backproject import backproject, backproject_snapped     # noqa: E402
from ear3d.camera import camera_ke, cone_angles, orbit_extrinsic   # noqa: E402
from ear3d.frames import frame_from, in_frame, plane_normal      # noqa: E402
from ear3d.label import pick                                       # noqa: E402
from ear3d.render import MATERIAL, light_scene, render              # noqa: E402
from ear3d.scheme import GROUPS, GROUP_NAMES, group_of             # noqa: E402

# HUTUBS head meshes are in metres; the fit lands on them at 0.19 mm after this
# and at 83 mm without it.
HUTUBS_TO_MM = 1000.0

# An escape is counted at these distances from the fitted ear, in millimetres.
# The pinna's standoff from the skull is 15-20 mm, so a landmark that has gone
# to the scalp shows up beyond about 10; 5 mm is already off the anatomy it
# names, and 2 mm is roughly the placement error the method achieves at best.
THRESHOLDS_MM = (2.0, 5.0, 10.0, 20.0)

# Skinned head meshes by subject. Ambient occlusion and the blotching are
# per-vertex and frame-invariant, so the two ears of one head share them.
_SKIN_CACHE = {}


def load_ear(name, dataset="hutubs", cfg=None):
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
    tp = WRAP / "corpus/canon2scan" / f"{name}.npz"
    if not tp.exists():
        raise FileNotFoundError(f"{tp} not found -- run scripts/canonical_to_scan.py")
    z = np.load(tp, allow_pickle=True)
    if str(z["dataset"]) != dataset:
        raise ValueError(f"{name} is {str(z['dataset'])!r}, not {dataset!r}")

    ear = o3d.io.read_triangle_mesh(
        str(WRAP / "corpus/canonical_esc" / dataset / f"{name}.obj"))
    if len(ear.vertices) == 0:
        raise FileNotFoundError(f"no fitted ear for {name}")
    ear.compute_vertex_normals()

    subj = name.split("_")[0]
    if subj not in _SKIN_CACHE:
        h = o3d.io.read_triangle_mesh(str(RAW / "3D head meshes" / f"{subj}_3DheadMesh.ply"))
        if len(h.vertices) == 0:
            raise FileNotFoundError(f"no head mesh for {subj}")
        h.compute_vertex_normals()
        if cfg.get("skin"):
            from skin import apply_skin
            apply_skin(h, seed=cfg["skin_seed"], tone=cfg["tone"],
                       ao_strength=cfg["ao_strength"], n_rays=cfg["ao_rays"])
        _SKIN_CACHE[subj] = h
    head = o3d.geometry.TriangleMesh(_SKIN_CACHE[subj])
    V = np.asarray(head.vertices) * HUTUBS_TO_MM
    V = (np.c_[V, np.ones(len(V))] @ z["inverse"].T)[:, :3]
    head.vertices = o3d.utility.Vector3dVector(V)
    head.compute_vertex_normals()

    lm = np.load(WRAP / "templates/lm3d_v7_canonical_mm.npz", allow_pickle=True)
    names = [str(x) for x in lm["names"]]
    if name not in names:
        raise KeyError(f"{name} has no stored landmarks in lm3d_v7")
    P = lm["landmarks"][names.index(name)][:55]
    return head, ear, P


def ear_frame(head, lm, cfg):
    """(R, centre, scale) framing the ear the way label_two_pass does.

    Centre on the landmark centroid and scale by the largest centred coordinate,
    so the ear spans ~2 units -- what cfg's eye_z and vfov are set for.

    THE VIEW DIRECTION IS THE PINNA'S OWN PLANE, not the canonical x axis. The
    canonical frame puts the ear's long axis on +y and leaves x as the thin one,
    which makes x look like the obvious face-on direction -- and it is wrong on
    every ear. Measured over all 112 HUTUBS ears, the plane fitted to the 55
    landmarks sits a median 17.7 deg off x (min 12.7, max 22.3, 100% beyond 10
    deg). Framing on x tilts every render by that much, and it shows: on the
    first sweep the detector's hit rate peaked at yaw -15 rather than at 0,
    because the grid was quietly compensating for the tilt.

    This is the same correction label_two_pass makes in its second pass, for the
    same reason -- the pinna's plane is not the head's lateral plane -- except
    that the plane here is fitted to the ground-truth landmarks, so it needs no
    iteration to converge.

    The sign points AWAY from the head rather than into it, read off the head's
    own centroid rather than a left/right convention, because every right ear
    was mirrored into left form before fitting and a convention is one more
    thing to get backwards.
    """
    C = lm.mean(0)
    into_head = np.asarray(head.vertices).mean(0) - C
    front = plane_normal(lm)
    if front @ into_head > 0:                      # pointing into the skull
        front = -front
    R = frame_from(front, np.array([0.0, 1.0, 0.0]), front=front)
    if R is None:
        raise ValueError("degenerate ear frame")
    scale = float(np.abs((lm - C) @ R.T).max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("degenerate ear scale")
    return R, C, scale


def mask_depth(ear_mesh, cfg, E, size):
    """Depth buffer of the FITTED EAR ALONE, through the given camera.

    Finite wherever the ear was drawn. Note this is the silhouette of the whole
    fitted shell, back surface included, which is NOT the visible pinna: seen
    off-axis the back and the skirt project well outside the front, over the
    cheek. Membership of this outline is therefore too weak a test -- see
    on_ear(), which uses this buffer the way it has to be used.
    """
    r = o3d.visualization.rendering.OffscreenRenderer(size, size)
    r.scene.add_geometry("ear", ear_mesh, MATERIAL)
    light_scene(r.scene, (0.5, 0.5, 0.5))
    K, E = camera_ke(size, cfg, E)
    r.setup_camera(K, E, size, size)
    d = np.asarray(r.render_to_depth_image(z_in_view_space=True))
    del r
    return d


def _unmirror(res, w):
    """Put detections made on a mirrored image back into the original's pixels."""
    w1 = w - 1.0
    for d in res:
        d["landmarks"] = [[w1 - x, y] for x, y in d["landmarks"]]
        b = list(d["bbox"])
        d["bbox"] = [b[0], w1 - b[3], b[2], w1 - b[1]]      # ymin xmin ymax xmax
    return res


def landmark_view(img, cfg, pipe, mode, hw):
    """Detect and landmark one render under one test-time-augmentation mode.

    Mirroring only ever touches what the MODEL sees. The depth buffer, the
    camera and the lift never learn about it, so a gain here cannot come from
    the geometry changing underneath.

        none  as rendered
        flip  the mirrored image alone, mirrored back
        avg   the mean of the two arms, per landmark
        best  whichever arm's chosen detection scored higher

    'avg' and 'best' cost a second forward pass and no extra render.
    """
    a = pick(pipe(img, timestamp=0.0), cfg, hw) if mode != "flip" else None
    b = None
    if mode != "none":
        b = pick(_unmirror(pipe(np.ascontiguousarray(img[:, ::-1]), timestamp=0.0),
                           img.shape[1]), cfg, hw)
    if mode == "none":
        return a
    if mode == "flip":
        return b
    if a is None or b is None:
        return a or b
    if mode == "best":
        return a if float(a["confidence"]) >= float(b["confidence"]) else b
    ua, ub = np.asarray(a["landmarks"], float), np.asarray(b["landmarks"], float)
    return dict(a, landmarks=((ua + ub) / 2.0).tolist(),
                confidence=(float(a["confidence"]) + float(b["confidence"])) / 2.0)


def audit_ear(name, cfg, pipe, angles, dataset="hutubs", render_dir=None, radii=None,
              mirror="none"):
    """One row per (view, landmark). Returns (rows, note)."""
    head, ear, lm = load_ear(name, dataset, cfg)
    R, C, scale = ear_frame(head, lm, cfg)
    g_head = in_frame(head, R, C, scale)
    g_ear = in_frame(ear, R, C, scale)
    lm_f = ((lm - C) @ R.T) / scale

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(g_ear))

    rows = []
    for yaw, pitch in angles:
        E = orbit_extrinsic(yaw, pitch, cfg)
        img, depth, cam = render(g_head, cfg, E=E)
        best = landmark_view(img, cfg, pipe, mirror, depth.shape)
        if best is None:
            rows.append(dict(ear=name, yaw=yaw, pitch=pitch, j=-1, detected=0))
            continue
        uv = np.asarray(best["landmarks"], float)
        conf = float(best["confidence"])

        # Both lifts, from the same pixels and the same buffer, so the cliff
        # snap is measured against the thing it is meant to fix.
        P_raw, hit_raw = backproject(depth, uv, cam)
        P_snap, hit_snap, moved = backproject_snapped(depth, uv, cam, cfg)

        # Extra snap radii, if asked. The radius only changes how an EXISTING
        # depth buffer is post-processed, so sweeping it costs no renders --
        # which is the whole reason it is swept here rather than by re-running.
        sweep = {}
        for rr in (radii or ()):
            Pr, _, mr = backproject_snapped(depth, uv, cam, dict(cfg, snap_radius=rr))
            # Paired error too, because surface distance is BLIND to a point
            # sliding along the contour -- a wider snap could cut the distance
            # while making the correspondence worse, and this project has been
            # caught by exactly that blindness before.
            sweep[rr] = (_surface_mm(scene, Pr, scale), mr,
                         np.linalg.norm(Pr - lm_f, axis=1) * scale)

        dm = mask_depth(g_ear, cfg, E, cfg["size"])
        on_mask = on_ear(depth, dm, uv, scale)

        d_raw = _surface_mm(scene, P_raw, scale)
        d_snap = _surface_mm(scene, P_snap, scale)
        paired = np.linalg.norm(P_snap - lm_f, axis=1) * scale

        for j in range(55):
            rows.append(dict(
                ear=name, yaw=yaw, pitch=pitch, j=j, detected=1, conf=conf,
                group=group_of(j),
                on_mask=int(on_mask[j]),
                hit_raw=int(hit_raw[j]), hit_snap=int(hit_snap[j]),
                snapped=int(moved[j]),
                d_raw_mm=d_raw[j], d_snap_mm=d_snap[j], paired_mm=paired[j],
                **{f"d_r{rr:g}_mm": sweep[rr][0][j] for rr in sweep},
                **{f"moved_r{rr:g}": int(sweep[rr][1][j]) for rr in sweep},
                **{f"paired_r{rr:g}_mm": sweep[rr][2][j] for rr in sweep}))

        if render_dir is not None:
            _overlay(render_dir, name, yaw, pitch, img, dm, uv, on_mask, d_snap)
    return rows


def on_ear(d_head, d_ear, uv, scale, tol_mm=5.0):
    """Is the surface the camera SEES at this pixel the ear?

    Not "is the pixel inside the ear's outline". The fitted ear is a shell with
    a back, so off-axis its outline covers cheek the camera never sees as ear,
    and membership of it passes points that sit on the scalp. Rendered through
    the same camera, though, the two buffers answer the question: where the ear
    is what is being looked at the two agree, and where something else is in
    front -- cheek, scalp, the far side of the gap behind the pinna -- the head
    reads nearer and they part. Where the shell does not project at all the ear
    buffer is inf, which fails here too.

    WHY 5 mm AND NOT A TIGHT GATE. Measured on pp10_right over three views, the
    gap (ear minus head, mm) across every co-finite pixel:

        view            p1     p25    p50    p75    p95    p99
        face-on       -1.47  -0.16   0.11   0.49   1.09   2.01
        pitch -45     -2.52  -0.14   0.23   0.72   1.36   2.29
        yaw +40       -3.22  -0.06   0.31   0.63   1.68   5.87

    It is NOT the clean bimodal it would be if the ear were a subset of the head
    mesh. It is a fit, so it rides a few tenths either side of the scan and goes
    NEGATIVE where it pokes in front -- which also matches the 0.15-0.62 mm
    median, 3.1 mm max residual measured in the header. A 1 mm gate would call
    10% of genuine pinna pixels off-ear. The scalp behind the pinna is 15-20 mm
    back, so anything from about 4 mm to 15 mm separates cleanly; 5 mm keeps the
    false off-ear rate near 1% (98.8-99.5% of co-finite pixels pass) while
    leaving a 3x margin to the standoff this is meant to catch.
    """
    H, W = d_head.shape
    x = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
    y = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
    dh, de = d_head[y, x], d_ear[y, x]
    return np.isfinite(dh) & np.isfinite(de) & ((de - dh) * scale < tol_mm)


def _surface_mm(scene, P, scale):
    """Distance from each point to the fitted ear, in millimetres."""
    d = np.full(len(P), np.nan)
    ok = np.isfinite(P).all(axis=1)
    if ok.any():
        q = o3d.core.Tensor(P[ok].astype(np.float32))
        d[ok] = scene.compute_distance(q).numpy() * scale
    return d


def _overlay(out, name, yaw, pitch, img, dm, uv, on_mask, d_mm):
    """Render and LOOK: the image, the exact ear silhouette, and every landmark
    coloured by whether it escaped. Green on the ear, amber off the mask,
    red escaped past 10 mm."""
    from PIL import Image, ImageDraw
    out.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(img.copy()).convert("RGB")
    edge = np.isfinite(dm).astype(np.uint8)
    b = (edge - np.minimum.reduce([np.roll(edge, s, a) for s in (-1, 1) for a in (0, 1)])) > 0
    a = np.array(im)
    a[b] = [60, 130, 255]                       # the ear's own silhouette, blue
    im = Image.fromarray(a)
    dr = ImageDraw.Draw(im)
    for j, (u, v) in enumerate(uv):
        far = np.isfinite(d_mm[j]) and d_mm[j] > 10.0
        col = (235, 60, 60) if far else ((245, 175, 40) if not on_mask[j] else (70, 210, 90))
        dr.ellipse([u - 2.5, v - 2.5, u + 2.5, v + 2.5], fill=col)
    im.save(out / f"{name}_y{yaw:+03.0f}_p{pitch:+03.0f}.png")


def summarise(rows):
    """Escape RATES, per group and per view. Not medians."""
    r = [x for x in rows if x.get("detected") and x["j"] >= 0]
    if not r:
        return "no detections"
    d = np.array([x["d_snap_mm"] for x in r], float)
    dr = np.array([x["d_raw_mm"] for x in r], float)
    om = np.array([x["on_mask"] for x in r])
    ok = np.isfinite(d)
    # A view where the detector found no ear is a view-dependence failure too,
    # and it is invisible in anything computed over landmarks only.
    miss = [x for x in rows if x["j"] < 0]
    nview = len({(x["yaw"], x["pitch"]) for x in rows})
    near_ = len({x["ear"] for x in rows})
    out = [f"{len(r)} landmark-views, {near_} ears, {nview} view angles",
           f"no ear detected: {len(miss)} of {near_ * nview} ear-views "
           f"({100 * len(miss) / max(near_ * nview, 1):.1f}%)",
           f"landmark on a pixel showing something other than the ear: "
           f"{100 * (1 - om.mean()):.1f}%"]
    out.append("")
    out.append("escape rate, distance from the fitted ear (snapped lift):")
    for t in THRESHOLDS_MM:
        out.append(f"  > {t:5.1f} mm   {100 * (d[ok] > t).mean():6.2f}%"
                   f"   (raw lift {100 * (dr[np.isfinite(dr)] > t).mean():6.2f}%)")
    out.append(f"  median {np.median(d[ok]):.2f} mm   p90 {np.percentile(d[ok], 90):.2f}"
               f"   p99 {np.percentile(d[ok], 99):.2f}   max {d[ok].max():.2f}")

    out.append("")
    out.append("whose fault, at the 10 mm gate:")
    far = ok & (d > 10.0)
    if far.any():
        a = (far & (om == 0)).sum()
        b = (far & (om == 1)).sum()
        out.append(f"  pixel OFF the ear mask   {a:5d}  ({100 * a / far.sum():5.1f}% of escapes)"
                   "   -> 2D put it outside the pinna")
        out.append(f"  pixel ON  the ear mask   {b:5d}  ({100 * b / far.sum():5.1f}% of escapes)"
                   "   -> the depth sample itself")
    else:
        out.append("  no escapes past 10 mm")

    out.append("")
    out.append("by scheme group, escape rate past 10 mm:")
    for gname in GROUP_NAMES:
        sel = [i for i, x in enumerate(r) if x["group"] == gname]
        if not sel:
            continue
        dd = d[sel][np.isfinite(d[sel])]
        mm = om[sel]
        if len(dd):
            out.append(f"  {gname:24s} {100 * (dd > 10).mean():6.2f}%   "
                       f"median {np.median(dd):6.2f} mm   off-mask "
                       f"{100 * (1 - mm.mean()):5.1f}%")

    out.append("")
    out.append("by view, escape rate past 10 mm:")
    views = sorted({(x["yaw"], x["pitch"]) for x in r})
    for yaw, pitch in views:
        sel = [i for i, x in enumerate(r) if (x["yaw"], x["pitch"]) == (yaw, pitch)]
        dd = d[sel][np.isfinite(d[sel])]
        if len(dd):
            out.append(f"  yaw {yaw:+6.1f} pitch {pitch:+6.1f}   {100 * (dd > 10).mean():6.2f}%"
                       f"   median {np.median(dd):6.2f} mm")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ears", nargs="*", default=["pp10_right"])
    p.add_argument("--all", action="store_true", help="every hutubs ear in canon2scan")
    p.add_argument("--views", type=int, default=9, help="views in a cone")
    p.add_argument("--cone", type=float, default=45.0, help="cone half-angle, degrees")
    p.add_argument("--grid", type=float, default=None, metavar="LIMIT",
                   help="sweep a yaw x pitch grid to +/-LIMIT instead of a cone. "
                        "A cone samples what production does; a grid is what "
                        "shows WHERE view dependence sets in, which a cone's "
                        "scattered angles cannot.")
    p.add_argument("--grid-step", type=float, default=15.0)
    p.add_argument("--out", default=None, help="csv of every landmark-view")
    p.add_argument("--render", action="store_true", help="write overlay PNGs and look")
    p.add_argument("--render-dir", default="runs/escape_audit")
    p.add_argument("--no-snap", action="store_true", help="cfg['snap'] = False")
    p.add_argument("--detector-conf", type=float, default=None, metavar="C",
                   help="detector score floor. The shipped 0.70+ was fitted for "
                        "PHOTOGRAPHS, where precision matters and several ears "
                        "may be present. These frames contain exactly one ear by "
                        "construction, so a threshold buys nothing and costs "
                        "recall: on the GT test set, 56.5%% -> 97.6%% at IoU>=0.3 "
                        "with this low and --full-frame on.")
    p.add_argument("--select", choices=["score", "central"], default=None,
                   help="which detection to lift when several come back. "
                        "'central' weights the score by distance from frame "
                        "centre, which these frames justify and photographs do "
                        "not -- see pick().")
    p.add_argument("--full-frame", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="run the ear model on the whole frame instead of on "
                        "BlazeFace crops. The face stage exists to make a small "
                        "ear bigger; an ear-centred render has no face in it and "
                        "needs no help, and measured on the GT test set "
                        "full-frame beats two-stage 97.6%% to 89.3%%. "
                        "--no-full-frame measures the two-stage path.")
    p.add_argument("--tta", choices=["none", "flip", "avg", "best"], default="none",
                   help="test-time augmentation by horizontal mirror. 'flip' "
                        "replaces the as-rendered pass, 'avg' averages the two "
                        "per landmark, 'best' keeps whichever scored higher. "
                        "Canonical frame puts left and right ears at opposite "
                        "on-screen handedness and the model is worse on the one "
                        "left ears land in -- detector confidence 0.631 against "
                        "0.837 -- so a flip is not a neutral augmentation here.")
    p.add_argument("--tta-side", choices=["both", "left", "right"], default="both",
                   help="restrict --tta to one side, to separate the handedness "
                        "gain from the flip acting as plain augmentation.")
    p.add_argument("--snap-sweep", nargs="*", type=float, default=None,
                   metavar="FRAC",
                   help="also score these cliff-snap radii, as a fraction of ear "
                        "extent. Costs no extra renders -- the radius only "
                        "post-processes a depth buffer that already exists.")
    a = p.parse_args()

    cfg = dict(DEFAULTS)
    cfg["snap"] = not a.no_snap
    cfg["interactive"] = False

    ears = a.ears
    if a.all:
        ears = sorted(f.stem for f in (WRAP / "corpus/canon2scan").glob("pp*.npz"))

    from ear3d.twod import make_pipeline                                   # noqa
    # The flags OVERRIDE config.py so this harness can still measure the old
    # behaviour -- which is the whole reason the defaults there could be set.
    # None means "leave config.py alone". An --select that defaulted to
    # "score" would have quietly undone the measured default on every run.
    if a.detector_conf is not None:
        cfg["detector_conf"] = a.detector_conf
    if a.full_frame is not None:
        cfg["detector_full_frame"] = a.full_frame
    if a.select is not None:
        cfg["select"] = a.select
    pipe, ck = make_pipeline(cfg)
    print(f"landmarker {ck.name}  (conf {cfg['detector_conf']}, "
          f"full_frame {cfg['detector_full_frame']}, select {cfg['select']})")

    if a.grid is not None:
        ax = np.arange(-a.grid, a.grid + 1e-6, a.grid_step)
        angles = [(float(y), float(p_)) for p_ in ax for y in ax]
    else:
        angles = list(cone_angles(a.views, a.cone))
    print(f"{len(angles)} views per ear, {len(ears)} ears")
    rdir = pathlib.Path(a.render_dir) if a.render else None

    rows = []
    for i, name in enumerate(ears):
        try:
            got = audit_ear(name, cfg, pipe, angles, render_dir=rdir,
                            radii=a.snap_sweep,
                            mirror=a.tta if a.tta_side in
                                   ("both", "left" if name.endswith("_left")
                                            else "right") else "none")
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"[{i + 1}/{len(ears)}] {name}: skipped -- {e}")
            continue
        rows += got
        n_far = sum(1 for x in got if x.get("j", -1) >= 0
                    and np.isfinite(x.get("d_snap_mm", np.nan)) and x["d_snap_mm"] > 10)
        print(f"[{i + 1}/{len(ears)}] {name}: {len(got)} rows, {n_far} past 10 mm")

    print()
    print(summarise(rows))

    if a.out:
        keys = sorted({k for x in rows for k in x})
        op = pathlib.Path(a.out)
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {op}  ({len(rows)} rows)")
    if rdir:
        print(f"overlays in {rdir}")


if __name__ == "__main__":
    main()
