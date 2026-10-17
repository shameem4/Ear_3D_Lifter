"""Find the ear's frame from the mesh alone -- no fit, no ground truth.

This is the FIRST step of the production path and it had no committed
implementation: 2749ddc recorded "chop with the detector box and PCA it,
2.4-3.1 deg on the view axis" but touched only SESSION_STATE.md, and the
session that produced those numbers was lost at a restart. So they were
unreproducible. This is that step, written to be run rather than described.

WHY NOT JUST PCA THE CHOP. The recorded rule is "smallest PCA axis = face-on,
largest = up". On a bald HUTUBS head the chop is a flattish pinna slab, so the
eigenvalues separate cleanly and the rule holds. Add a mesh cap, a hat or hair
and the chop gains a smooth dome: the eigenvalues bunch, the ordering can swap,
and the frame does not drift by 3 deg -- it flips to a different axis. That is
a dataset-shaped assumption, and it is the thing most likely to break on a mesh
we did not render ourselves.

So the axis assignment is DECIDED PER MESH instead of assumed. Every candidate
(front, up) built from the chop's own axes is scored by a criterion that needs
no ground truth -- render face-on, ask the shipped detector what it sees -- and
the winner is used. On a mesh where the convention happens to be right this
agrees with it and costs ~12 cheap renders; on a mesh where it is wrong it
recovers instead of failing silently.

    front sign      convention-free already: point away from the head centroid,
                    as ear_frame does. A left/right rule is one more thing to
                    get backwards, and every right ear is mirrored anyway.
    up sign         genuinely ambiguous -- a 180 deg flip is an upside-down ear
                    that a detector will still happily box. Scored, not assumed.

WHAT THE SCORE IS, AND WHAT IT IS NOT. Detection confidence is worthless for
ranking GOOD views against each other (r=-0.010, measured) and this does not
use it that way. Here it separates a correctly framed ear from one framed on
the scalp or rotated 90 deg, which is a far coarser question, and the margin
below reports whether that separation actually happened rather than assuming it.

Usage:
    python scripts/bootstrap_frame.py --ears 8                   # vs the GT frame
    python scripts/bootstrap_frame.py --mesh /path/to/head.OBJ   # no GT needed
"""

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def chop_to_box(V, cam, box, depth_mm=25.0, scale=1.0):
    """Vertices whose projection lands in `box`, nearest depth shell only.

    THE SHELL MUST BE ABSOLUTE, NOT A FRACTION. The box is a 2D window, so it
    selects a column straight through the head; a fraction of THAT column's
    depth is ~70 mm, which keeps a quarter of the head and makes the PCA
    return the head's axes rather than the pinna's -- measured at 23.6 deg
    against the GT frame, where the pinna plane needs ~3. A pinna's relief is
    10-20 mm, so the shell is given in millimetres and `scale` converts from
    the framed units the vertices arrive in.
    """
    uv = cam.project(V)
    x0, y0, x1, y1 = box
    inb = ((uv[:, 0] >= x0) & (uv[:, 0] <= x1) &
           (uv[:, 1] >= y0) & (uv[:, 1] <= y1))
    if inb.sum() < 50:
        return inb & False
    z = V @ cam.forward() if hasattr(cam, "forward") else uv[:, 0] * 0
    # Depth along the camera axis; Camera may not expose it, so fall back to
    # the frame's own z, which in_frame has already aligned with the view.
    zz = V[:, 2] if not np.any(z) else z
    sel = np.flatnonzero(inb)
    d = zz[sel]
    keep = d >= (d.max() - depth_mm / max(scale, 1e-9))
    out = np.zeros(len(V), bool)
    out[sel[keep]] = True
    return out


def candidates(P, centre, into_head):
    """Every (front, up) the chop's own axes allow, with the front sign fixed.

    Three axes for front x two remaining axes for up x two up signs = 12. The
    recorded convention is one of these twelve; it is not given a head start.
    """
    Q = P - P.mean(0)
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    ev = (Q @ vt.T).var(0)
    out = []
    for fi in range(3):
        f = vt[fi]
        if f @ into_head > 0:                      # point away from the skull
            f = -f
        rest = [k for k in range(3) if k != fi]
        for ui in rest:
            for sgn in (+1.0, -1.0):
                out.append(dict(front=f, up=sgn * vt[ui], fi=fi, ui=ui,
                                sgn=sgn, ev=ev))
    return out, vt, ev


LOBE = [14, 15, 16, 17, 18, 19]
SUP_CRUS = [50, 51, 52, 53, 54]


def upright_by_anatomy(lm_uv):
    """Is this render upright? The lobe sits BELOW the superior crus.

    The up sign is the one thing the chop cannot settle: PCA gives an axis,
    not a direction, and a 180 deg flip is an upside-down ear that a detector
    boxes just as happily. Rather than invent a convention, ask the 2D model,
    which was trained on ear anatomy and labels these groups semantically.
    Image y grows downward, so upright means lobe y > crus y.

    Returns None when either group is missing, so the caller can fall back
    instead of trusting a coin flip.
    """
    ok = np.isfinite(lm_uv).all(1)
    lo = [j for j in LOBE if ok[j]]
    cr = [j for j in SUP_CRUS if ok[j]]
    if len(lo) < 3 or len(cr) < 3:
        return None
    return bool(lm_uv[lo, 1].mean() > lm_uv[cr, 1].mean())


def score_frame(mesh, front, up, cfg, pipe, centre, pts=None):
    """Render face-on in this frame and ask the detector what it sees.

    Returns (score, n_landmarks, conf). Score rewards a confident detection
    that fills a sensible share of the frame: a frame rotated onto the scalp
    either finds nothing or finds something small and uncertain.
    """
    from ear3d.frames import frame_from, in_frame
    from ear3d.render import render
    from ear3d.label import pick
    R = frame_from(front, up, front=front)
    if R is None:
        return -1.0, 0, 0.0
    V = np.asarray(mesh.vertices)
    # SCALE BY THE EAR, NOT THE HEAD. ear_frame scales by the landmark extent;
    # scaling by max|V - centre| over every head vertex renders the whole head
    # with the ear a few pixels across. The angle is unaffected -- rotation is
    # scale-invariant -- but the detector then sees no ear in ANY candidate, so
    # the scorer was blind and picked near-randomly (margin 0.032, 78 deg).
    # A rendered panel showed this in one look; no amount of angle-tuning did.
    ref = V if pts is None else np.asarray(pts)
    s = float(np.abs((ref - centre) @ R.T).max())
    g = in_frame(mesh, R, centre, s)
    img, _, cam = render(g, cfg)
    best = pick(pipe(np.asarray(img), timestamp=0.0), cfg, img.shape[:2])
    if best is None:
        return 0.0, 0, 0.0
    lm = np.asarray(best["landmarks"], float)
    ok = np.isfinite(lm).all(1)
    conf = float(best.get("score", best.get("confidence", 1.0)) or 1.0)
    H, W = img.shape[:2]
    # occupancy: how much of the frame the landmark span covers. A scalp
    # "ear" tends to be small and off-centre; a well-framed one is neither.
    span = (np.ptp(lm[ok, 0]) / W) * (np.ptp(lm[ok, 1]) / H) if ok.sum() > 5 else 0.0
    occ = float(np.clip(span / 0.08, 0, 1))        # 0.08 ~ a well-framed ear
    up_ok = upright_by_anatomy(lm)
    # An upside-down frame is not a near-miss, it is wrong; weight it as such
    # rather than letting a confident inverted detection outscore an upright
    # one. None (groups missing) is treated as neutral, not as a pass.
    mult = 1.0 if up_ok else (0.5 if up_ok is None else 0.1)
    return conf * (0.5 + 0.5 * occ) * mult, int(ok.sum()), conf


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ears", type=int, default=8)
    p.add_argument("--workers", type=int, default=1,
                   help="bootstrap ears in this many processes. Each ear is "
                        "independent and one process left the GPU mostly "
                        "idle; measured peak is ~2.4 GB per process")
    p.add_argument("--shard", default=None, help=argparse.SUPPRESS)
    p.add_argument("--dataset", default="hutubs")
    p.add_argument("--mesh", default=None,
                   help="score a mesh with no ground truth (prints the frame "
                        "and the margin, skips the angle comparison)")
    p.add_argument("--seed-axis", default="x", choices=list("xyz"),
                   help="axis for the SEED render that gets the detector box")
    p.add_argument("--seed-face", action="store_true",
                   help="HYBRID seed: MediaPipe head_pose gives the lateral "
                        "axis and the tragion; the seed render is centred on "
                        "the tragion looking along the lateral axis, i.e. an "
                        "ear-centred frame, the regime the detector and pick() "
                        "were measured in. The frame itself then comes from "
                        "the chop and the template (--rule procrustes).")
    p.add_argument("--seed-sweep", action="store_true",
                   help="PRODUCTION seed: render the six axis views and take "
                        "the one with the most confident ear box, instead of "
                        "seeding toward the GT ear (validation) or a fixed axis")
    p.add_argument("--only", nargs="*", default=None,
                   help="run only these ears (the --random-rot draw stays "
                        "keyed to each ear's index in the full list)")
    p.add_argument("--random-rot", type=int, default=None,
                   help="rotate every head (and its GT) by a random rotation "
                        "drawn from this seed before anything runs. The chop, "
                        "the candidates, the Procrustes pick and the refinement "
                        "are orientation-free BY CONSTRUCTION; this tests it.")
    p.add_argument("--up-from-face", action="store_true", default=True,
                   help="resolve the up SIGN from MediaPipe's head pose "
                        "(default). --no-up-from-face for the old behaviour.")
    p.add_argument("--no-up-from-face", dest="up_from_face",
                   action="store_false")
    p.add_argument("--table", default=None,
                   help="ray cache to learn the per-(landmark, view) accuracy "
                        "table from, e.g. runs/raycache. The refinement lift "
                        "currently feeds EVERY view into RANSAC, which is the "
                        "'all views' arm the table was built to beat (2.70 "
                        "against 2.24 mm).")
    p.add_argument("--table-k", type=int, default=8,
                   help="views per landmark when a table is given")
    p.add_argument("--table-from", type=int, default=2,
                   help="first refinement pass that uses the table. The table "
                        "was learned in the GT frame, so it is misapplied "
                        "exactly while the frame is still wrong -- ~20 deg at "
                        "pass 1. Default 2; set 1 to test that reasoning.")
    p.add_argument("--passes", type=int, default=2,
                   help="frame refinements after the chop. Pass 1 is chop+PCA; "
                        "each further pass lifts the landmarks in the current "
                        "frame and refits the plane to them. 1 = chop only.")
    p.add_argument("--refine-grid", type=float, default=15.0,
                   help="half-width of the view grid used by a refinement "
                        "pass, degrees (3x3 at this half-width)")
    p.add_argument("--pad", type=float, default=0.15,
                   help="box padding as a fraction. Some is needed: a tight "
                        "landmark hull clips the helix rim that defines the "
                        "plane, which is how trimmed PCA made the normal worse.")
    p.add_argument("--depth-mm", type=float, default=25.0,
                   help="depth of the kept shell behind the frontmost vertex, "
                        "in mm. A pinna's relief is 10-20.")
    p.add_argument("--rule", default="score", choices=("score", "smallest", "face", "face-nearest", "procrustes"),
                   help="how to pick the axis assignment. 'smallest' forces "
                        "the recorded convention (front = smallest chop axis, "
                        "up = largest) and so isolates whether chop+PCA is "
                        "sound from whether the scorer can rank candidates.")
    p.add_argument("--save-frames", default=None,
                   help="write the bootstrapped frames to this .npz so the "
                        "triangulation path can render in them. Frames as "
                        "DATA rather than an imported function: cache_rays "
                        "reads the file, nothing imports this script, and "
                        "the frames stay inspectable after the fact.")
    p.add_argument("--render-dir", default=None,
                   help="write the winning frame's render here, to look at")
    a = p.parse_args()

    from escape_audit import DEFAULTS, WRAP, load_ear, ear_frame, light_mirror_for
    from ear3d.frames import frame_from, in_frame, plane_normal, head_pose
    from ear3d.render import render
    from ear3d.label import pick
    from ear3d.twod import make_pipeline
    from ear3d.camera import orbit_extrinsic
    from ear3d.triangulate import ransac_point

    def build_table(cache, exclude):
        """table[v, j] = median 2D error in px, over cached ears.

        Built here rather than imported from view_table.py because that
        script computes it inline inside main(). Subjects in `exclude` are
        dropped: the bootstrap is evaluated on ears whose subjects are in
        the cache, and scoring a frame with a table trained on those same
        subjects would be leakage.
        """
        files = [f for f in sorted(pathlib.Path(cache).glob("*.npz"))
                 if f.stem.rsplit("_", 1)[0] not in exclude]
        if len(files) < 10:
            sys.exit(f"only {len(files)} ears left in {cache} after exclusion")
        err, ang = [], None
        for f in files:
            z = np.load(f, allow_pickle=True)
            err.append(np.linalg.norm(z["uv"] - z["gt_uv"], axis=2))
            ang = z["angles"]
        t = np.nanmedian(np.stack(err), axis=0)          # (V, 55)
        print(f"view table from {len(files)} ears ({len(exclude)} subjects "
              f"held out), {t.shape[0]} views")
        return t, np.argsort(np.nan_to_num(t, nan=1e9), axis=0), ang

    def lift_in_frame(mesh, front, up, centre, cfg, pipe, half, pts=None,
                      order=None, K=8, grid=None):
        """Triangulate the 55 landmarks from a small grid in this frame."""
        R = frame_from(front, up, front=front)
        if R is None:
            return None
        V = np.asarray(mesh.vertices)
        ref = V if pts is None else np.asarray(pts)
        sc = float(np.abs((ref - centre) @ R.T).max())
        g = in_frame(mesh, R, centre, sc)
        # When a table is in play the grid must be the grid it was LEARNED
        # on, or the view indices mean different angles.
        angles = ([(float(y), float(q)) for y, q in grid] if grid is not None
                  else [(y, q) for y in (-half, 0.0, half)
                        for q in (-half, 0.0, half)])
        O = np.full((len(angles), 55, 3), np.nan)
        Dd = np.full((len(angles), 55, 3), np.nan)
        for v, (yy, qq) in enumerate(angles):
            img, _, cam = render(g, cfg, E=orbit_extrinsic(yy, qq, cfg))
            best = pick(pipe(np.asarray(img), timestamp=0.0), cfg, img.shape[:2])
            if best is None:
                continue
            uvv = np.asarray(best["landmarks"], float)
            ok = np.isfinite(uvv).all(1)
            if ok.any():
                oo, dd = cam.rays(uvv[ok])
                O[v, ok] = oo
                Dd[v, ok] = dd
        P = np.full((55, 3), np.nan)
        for j in range(55):
            m = np.isfinite(O[:, j]).all(1) & np.isfinite(Dd[:, j]).all(1)
            if order is not None:
                keep = np.zeros(len(angles), bool)
                keep[order[:K, j]] = True
                if (m & keep).sum() >= 2:        # fall back if too few survive
                    m = m & keep
            if m.sum() < 2:
                continue
            q = ransac_point(O[m, j], Dd[m, j], 0.03)
            q = q[0] if isinstance(q, tuple) else q
            if q is not None:
                P[j] = q
        # back to world mm
        return P * sc @ R + centre

    cfg = dict(DEFAULTS)
    pipe, ck = make_pipeline(cfg)
    TPL = None
    if a.rule == "procrustes":
        # THE GENERIC TEMPLATE, NOT THE CORPUS. Production has ear_lmv_v3 --
        # the landmark-vertex template, landmark j = vertex j -- and no wrap
        # fits, so the 55 reference points are its first 55 vertices (mm,
        # canonical frame, up = +y). They sit 1.25 mm RMS from the corpus
        # mean of lm3d_v7, which was used first and must not be: a per-ear
        # corpus file is exactly the dependency the bootstrap exists to drop.
        # Per-ear RMS of real ears to this template is ~2 mm, so a lift with
        # the RIGHT correspondences aligns to a few mm and one from a
        # wrong-way-up render -- where the orientation-blind 2D model has
        # labelled the lobe as the helix -- does not.
        TPL = np.array([[float(x) for x in l.split()[1:4]]
                        for l in open(WRAP / "templates/ear_lmv_v3.obj")
                        if l.startswith("v ")])[:55]
    print(f"{ck.parent.name}/{ck.name}")

    TBL = None
    if a.table:
        names_eval = sorted(f.stem for f in (WRAP / "corpus/canon2scan").glob("*.npz")
                            if str(np.load(f, allow_pickle=True)["dataset"])
                            == a.dataset)[:a.ears]
        TBL = build_table(a.table, {n.rsplit("_", 1)[0] for n in names_eval})

    if a.render_dir:
        pathlib.Path(a.render_dir).mkdir(parents=True, exist_ok=True)

    GTO = {}        # GT frame override under --random-rot (see call site)

    def one(nm, head, lm_gt):
        """Bootstrap one head. lm_gt may be None (no ground truth)."""
        V = np.asarray(head.vertices)
        hc = V.mean(0)
        hp = head_pose(head, cfg) if (a.up_from_face or a.seed_face) else None
        seed_centre, seed_scale = hc, None
        traj = []                      # how the frame got where it is
        # SEED: a coarse view, only good enough to put AN ear in frame and get
        # a detector box. It is allowed to be a long way off.
        #
        # WHICH EAR, THOUGH. A fixed axis seed views whichever ear faces it,
        # and a head has two. Seeding +x on pp10_left framed the ear at x=+153
        # while that ear's ground truth is at x=-8.4, so the chop was of the
        # OTHER ear and every angle was scored against the wrong target --
        # which is where the constant 161 mm "centre error" came from. With
        # ground truth present we therefore seed TOWARD the ear being scored;
        # that is a validation convenience, exactly like the fit, and it is
        # not available in production. In production the detector picks an ear
        # and that is the ear you lift, so there is nothing to disambiguate.
        if a.seed_face:
            if hp is None:
                return None, "seed-face: no face found on 26 views"
            lat = np.asarray(hp[0], float)
            trR, trL = np.asarray(hp[2], float), np.asarray(hp[3], float)
            if lm_gt is not None:
                # validation: the ear being scored. Production does both.
                gdir = lm_gt.mean(0) - hc
                seed = lat if lat @ gdir > 0 else -lat
            else:
                seed = lat
            # NOT THE TRAGION. FaceMesh's tragion is the landmark the
            # reference implementation marks "most likely not possible", and
            # centring on it put the rotated pp10_left seed on the cheek. In
            # a side view the ear sits close to the head's projected
            # centroid, which is what the validation seed (centre = head
            # centroid, whole-head scale) relied on and what pick()'s
            # centrality prior was measured with. So: centroid and whole
            # head; only the DIRECTION comes from the face.
            seed_centre, seed_scale = hc, None
            tr = trR if (trR - hc) @ seed > (trL - hc) @ seed else trL
            traj.append(("face seed", f"model tragion {np.linalg.norm(tr - hc):.0f} mm from centroid"))
        elif a.seed_sweep:
            # Six axis renders, keep the most confident ear box. With GT
            # present the sweep may land on the OTHER ear; then this ear is
            # skipped rather than scored against the wrong target, and the
            # skip is reported, because "found an ear" and "found THIS ear"
            # are different claims.
            sweep = []
            for ax in range(3):
                for sg in (1.0, -1.0):
                    sd = np.zeros(3); sd[ax] = sg
                    Rq = frame_from(sd, np.array([0.0, 0.0, 1.0]) if ax != 2
                                    else np.array([0.0, 1.0, 0.0]), front=sd)
                    sq = float(np.abs((V - hc) @ Rq.T).max())
                    imq, _, _ = render(in_frame(head, Rq, hc, sq), cfg)
                    # NOT pick(): its centrality prior is for frames built
                    # around one ear. On a whole head it chose a central
                    # scalp false positive (conf 0.82) over the real ear at
                    # the edge of the frame -- seen on pp10_left, pp19_left.
                    dets = pipe(np.asarray(imq), timestamp=0.0)
                    if dets:
                        bq = max(dets, key=lambda d: float(d["confidence"]))
                        sweep.append((float(bq["confidence"]), sd))
            if not sweep:
                return None, "seed sweep found no ear on any of six views"
            sweep.sort(key=lambda t: -t[0])
            seed = sweep[0][1]
            if lm_gt is not None:
                gdir = lm_gt.mean(0) - hc
                if seed @ gdir < 0:
                    return None, (f"seed sweep chose the other ear (conf "
                                  f"{sweep[0][0]:.2f}); not scorable against this GT")
        elif lm_gt is not None:
            seed = lm_gt.mean(0) - hc
            seed = seed / max(np.linalg.norm(seed), 1e-9)
        else:
            ax = {"x": 0, "y": 1, "z": 2}[a.seed_axis]
            seed = np.zeros(3); seed[ax] = 1.0
        # any up that is not the seed itself: a seed along y (which the
        # sweep and a rotated head both produce) made frame_from degenerate
        up_s = np.array([0.0, 1.0, 0.0]) if abs(seed[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
        if a.seed_face and hp is not None:
            # UPRIGHT EAR IN THE SEED RENDER. With a world axis as "up" the
            # side view of a rotated head shows the ear rolled by a random
            # angle, and the detector then boxed the jaw, the cheek, or half
            # the ear (9 of 18 rotated heads failed the residual gate that
            # way). The face vertical is already known; it makes the ear
            # upright, the regime the detector and pick() were measured in.
            up_s = np.asarray(hp[1], float)
        Rs = frame_from(seed, up_s, front=seed)
        ss = seed_scale or float(np.abs((V - hc) @ Rs.T).max())
        gs = in_frame(head, Rs, seed_centre, ss)
        img, _, cam = render(gs, cfg)
        dets = pipe(np.asarray(img), timestamp=0.0)
        if a.seed_sweep:
            # whole-head frame: no centrality prior (see the sweep)
            best = max(dets, key=lambda d: float(d["confidence"])) if dets else None
        else:
            best = pick(dets, cfg, img.shape[:2])
        if best is None:
            return None, "seed render found no ear"
        bx = best.get("box") or best.get("bbox")
        if bx is None:
            lmv = np.asarray(best["landmarks"], float)
            o = np.isfinite(lmv).all(1)
            bx = [lmv[o, 0].min(), lmv[o, 1].min(), lmv[o, 0].max(), lmv[o, 1].max()]
        bx = [float(v) for v in np.asarray(bx).ravel()[:4]]
        # pad, because a tight landmark hull clips the helix rim that defines
        # the plane -- the trimmed-PCA failure was exactly this
        w, h = bx[2] - bx[0], bx[3] - bx[1]
        pd = a.pad
        bx = [bx[0] - pd * w, bx[1] - pd * h, bx[2] + pd * w, bx[3] + pd * h]

        Vs = np.asarray(gs.vertices)
        keep = chop_to_box(Vs, cam, bx, depth_mm=a.depth_mm, scale=ss)
        if keep.sum() < 50:
            return None, f"chop kept only {int(keep.sum())} vertices"
        # back to world coordinates for the PCA
        Pw = Vs[keep] * ss @ Rs + seed_centre
        cc = Pw.mean(0)
        cands, vt, ev = candidates(Pw, cc, hc - cc)

        # UP FROM FACE ANATOMY, NOT FROM THE CHOP. PCA returns an axis, not
        # a direction, and the 2D landmarker cannot supply the sign -- it
        # places its template the same way up whichever way up the ear is
        # (2e6a1c4). MediaPipe can: head_pose gives a vertical built from
        # forehead, chin, nasion and nose tip, which owes nothing to the
        # scan's axis convention and so survives a mesh we did not render.
        # Measured against the GT frame's up: 4.36, 10.48, 6.42 deg.
        up_ref = lat_ref = None
        if a.up_from_face:
            if hp is not None:
                up_ref = np.asarray(hp[1], float)
                lat_ref = np.asarray(hp[0], float)
        if up_ref is not None:
            # keep each candidate's axis, take the SIGN from the face frame
            for q in cands:
                if q["up"] @ up_ref < 0:
                    q["up"] = -q["up"]
        if a.rule == "face" and lat_ref is not None:
            # FRONT FROM THE FACE, NOT THE CHOP. On a protruding ear the
            # chop's smallest PCA axis is anterior-posterior -- the pinna is a
            # flap, thin front-to-back -- and the seed view then looks at the
            # ear from BEHIND (pp70_right 81 deg, pp41_left 41, pp66_left 56,
            # pp33_right 50; all with a correctly found face). Refinement
            # cannot recover from that because the detector is shown the back
            # of the pinna. The face's lateral axis is 8-35 deg from the
            # pinna plane, which refinement does recover from (20 -> 4 deg).
            # The chop still sets centre and scale; the sign points away from
            # the head centroid, as every front here does.
            f0 = lat_ref if lat_ref @ (cc - hc) > 0 else -lat_ref
            u0 = up_ref - f0 * (up_ref @ f0)
            u0 /= np.linalg.norm(u0)
            c = dict(front=f0, up=u0, fi=-1, ui=-1, sgn=0)
            top, margin = (float("nan"), 0, 0.0, c), 0.0
        elif a.rule == "procrustes":
            # THE TEMPLATE PICKS THE FRAME. Lift the 55 landmarks in every
            # chop candidate and align them to the template's 55 by a
            # similarity (reflection allowed); the residual is anatomy's
            # verdict on the candidate, and the alignment hands back the up
            # axis WITH its sign -- which is what the face was for. No
            # FaceMesh, no scalp false positives, no axis convention.
            from suggest_landmarks import similarity
            tsz = float(np.sqrt(((TPL - TPL.mean(0)) ** 2).sum(1).mean()))
            tried = []
            for q in cands:
                try:
                    P3 = lift_in_frame(head, q["front"], q["up"], cc, cfg, pipe,
                                       a.refine_grid, Pw)
                except Exception as e:                  # noqa: BLE001
                    P3 = None
                if P3 is None:
                    tried.append((np.inf, q, None)); continue
                ok = np.isfinite(P3).all(1)
                if ok.sum() < 20:
                    tried.append((np.inf, q, None)); continue
                # ROTATION ONLY. With reflection allowed, a candidate whose up
                # is flipped matched the template through a mirror at the
                # SAME residual as the right one (det -1, tied residuals on
                # the rotated pp10/pp16), so the up sign was undecidable.
                # Handedness is not the template's job: the corpus is in
                # left form, and in production the side comes from the face
                # frame (which side of the head the chop is on), with
                # right-side heads mirrored first, as the corpus was.
                Y, (Rp, kp, _, _) = similarity(P3[ok], TPL[ok], reflect=False)
                resid = float(np.sqrt(((Y - TPL[ok]) ** 2).sum(1).mean())) / tsz
                tried.append((resid, q, (Rp, kp, int(ok.sum()))))
            tried.sort(key=lambda t: t[0])
            if not np.isfinite(tried[0][0]):
                return None, "procrustes: no candidate lifted"
            resid, c, (Rp, kp, nok) = tried[0]
            second = tried[1][0] if len(tried) > 1 else np.inf
            # RESIDUAL GATE. On 18 randomly rotated heads every correct frame
            # had residual <= 0.25 and every wrong one >= 0.67 (0.666 was a
            # 21 deg up error). A frame nothing resembles an ear in is a
            # reported failure, not a result.
            if resid > 0.45:
                return None, f"procrustes: no candidate resembles an ear (resid {resid:.3f})"
            # a world direction d is d @ Rp in template coordinates, so the
            # template's up (0,1,0) is this in the world:
            up_w = np.array([0.0, 1.0, 0.0]) @ Rp.T
            up_w /= np.linalg.norm(up_w)
            up_ref = up_w                 # refinement keeps up from here on
            c = dict(c, up=up_w - c["front"] * (up_w @ c["front"]))
            c["up"] /= np.linalg.norm(c["up"])
            traj.append(("procrustes", f"resid {resid:.3f} next {second:.3f} "
                                       f"det {np.sign(np.linalg.det(Rp)):+.0f} n {nok}"))
            top, margin = (-resid, nok, 0.0, c), float(second - resid)
        elif a.rule == "face-nearest" and lat_ref is not None:
            # THE CHOP AXIS NEAREST THE FACE LATERAL. 'face' alone fixed the
            # posterior-view ears but broke six others (pp48_right 62 deg,
            # pp66_right 64, pp82_left 134, ...) that the chop had right to
            # 1-3 deg: the face lateral is sometimes too far from the pinna
            # plane for refinement to close. The chop's axes are accurate
            # when the right one is chosen; the face is reliable about WHICH
            # one. So the face selects among the chop's candidates instead
            # of replacing them.
            f0 = lat_ref if lat_ref @ (cc - hc) > 0 else -lat_ref
            # tie-break the up axis (two per front) by the face vertical
            c = max(cands, key=lambda q: (round(float(q["front"] @ f0), 6),
                                          float(q["up"] @ up_ref)))
            top, margin = (float(c["front"] @ f0), 0, 0.0, c), 0.0
        elif a.rule == "smallest":
            fi, ui = int(np.argmin(ev)), int(np.argmax(ev))
            pickc = [c for c in cands if c["fi"] == fi and c["ui"] == ui]
            # up sign still has to be resolved somehow; take the better one so
            # this arm is the convention at its BEST, not a straw man.
            sc = [(score_frame(head, c["front"], c["up"], cfg, pipe, cc, Pw)[0], c)
                  for c in pickc]
            sc.sort(key=lambda t: -t[0])
            top, margin = (sc[0][0], 0, 0.0, sc[0][1]), 0.0
            c = sc[0][1]
        else:
            scored = []
            for c in cands:
                sc, n, cf = score_frame(head, c["front"], c["up"], cfg, pipe, cc, Pw)
                scored.append((sc, n, cf, c))
            scored.sort(key=lambda t: -t[0])
            top, second = scored[0], scored[1]
            margin = top[0] - second[0]
            c = top[3]
        # PASS 2+: the chop's smallest axis is close to the HEAD's lateral
        # direction, and the pinna plane is a median 17.7 deg off that (the
        # figure is already in ear_frame's docstring). Measured here, chop+PCA
        # alone lands ~19-20 deg out however the shell is tuned, so refinement
        # is not polish -- it is where the accuracy comes from.
        #
        # The refinement is the lift itself: render a small grid in the current
        # frame, triangulate the 55 landmarks from MULTIPLE views, and refit the
        # plane to those. Doing this from a SINGLE view's back-projection was
        # measured at 5.90 deg because scalp-landed points displace ~20 mm along
        # the normal, the worst direction for a plane fit; a multi-view lift has
        # a 0.065% gross-error rate, which is what makes the same idea work
        # here. Trimmed to 80% for the same reason.
        front_i, up_i = c["front"], c["up"]
        Qall = None            # every lifted landmark, untrimmed; sets the scale
        for it in range(max(0, a.passes - 1)):
            try:
                use_tbl = TBL is not None and (it + 2) >= a.table_from
                P3 = lift_in_frame(head, front_i, up_i, cc, cfg, pipe,
                                   a.refine_grid, Pw,
                                   order=TBL[1] if use_tbl else None,
                                   K=a.table_k,
                                   grid=TBL[2] if use_tbl else None)
            except Exception as e:                      # noqa: BLE001
                traj.append(("lift failed", str(e)[:40])); break
            if P3 is None or np.isfinite(P3).all(1).sum() < 20:
                traj.append(("too few points", "")); break
            Q = P3[np.isfinite(P3).all(1)]
            Qall = Q
            n = plane_normal(Q)
            # trim: drop the 20% furthest from the plane, refit
            dd = np.abs((Q - Q.mean(0)) @ n)
            Q = Q[dd <= np.quantile(dd, 0.8)]
            n = plane_normal(Q)
            if n @ (cc - hc) < 0:
                n = -n
            # UP STAYS WITH THE FACE FRAME. Recomputing it as the lifted
            # points' long axis made it WORSE every pass -- 12.5 -> 10.4 ->
            # 16.0 deg -- because the pinna's major axis is not the head's
            # vertical. The front axis genuinely wants the landmark plane and
            # converges (20.4 -> 9.6 -> 3.8); the up does not, so only the
            # front is refined and up is re-orthogonalised against it.
            if up_ref is not None:
                up_new = up_ref - n * (up_ref @ n)
                if np.linalg.norm(up_new) < 1e-6:
                    up_new = up_i
            else:
                Qc = Q - Q.mean(0)
                Qp = Qc - np.outer(Qc @ n, n)
                _, _, vtp = np.linalg.svd(Qp, full_matrices=False)
                up_new = vtp[0]
                if up_new @ up_i < 0:
                    up_new = -up_new
            ang = float(np.degrees(np.arccos(
                np.clip(abs(n @ front_i), -1, 1))))
            front_i, up_i = n, up_new
            cc = Q.mean(0) * 1.0
            traj.append((f"pass{it+2}", f"moved {ang:.2f} deg"))
        c = dict(c, front=front_i, up=up_i)
        _R = frame_from(c["front"], c["up"], front=c["front"])
        s_chop = float(np.abs((Pw - cc) @ _R.T).max())
        s_lm = (float(np.abs((Qall - cc) @ _R.T).max())
                if Qall is not None else s_chop)
        # CLAMPED TO THE CHOP. When the lift has failed the lifted points are
        # scattered over the head and their extent is meaningless: pp6_right
        # came out at 14x the chop (s=410 mm) and the ear was a dot in every
        # orbit view. Over 112 ears the ratio on good frames is 0.99-1.26
        # (p10-p90); a ratio at the clamp is reported, because it is one of
        # the few failure signs this script gets without ground truth -- but
        # it is NOT a frame detector: only 4 of 16 bad frames hit it.
        # LIFT-SANITY GATE, NOT A CHOP-TIED CLAMP. The clamp (0.8-1.6x the
        # chop) assumed the chop was the better-known size; on a rotated head
        # (pp16_left) the chop was 2x the ear, the lift had the scale right
        # at 0.50x, and the clamp forced a wrong answer. The chop is now only
        # the fallback for a lift that is thin (<40 landmarks) or, under the
        # Procrustes rule, that does not resemble an ear (residual >= 0.45).
        lift_ok = Qall is not None and len(Qall) >= 40
        if lift_ok and a.rule == "procrustes":
            lift_ok = any(k == "procrustes" and float(v.split()[1]) < 0.45
                          for k, v in traj)
        s_fin = s_lm if lift_ok else s_chop
        if not lift_ok:
            traj.append(("scale from chop", f"lift/chop {s_lm / s_chop:.2f}"))
        res = dict(front=c["front"], up=c["up"], centre=cc, ev=ev,
                   traj=traj, R=_R,
                   # SCALE FROM THE LANDMARKS, NOT THE CHOP. ear_frame scales
                   # by the largest centred landmark coordinate; the chop is a
                   # box-and-depth-shell subset of the ear and came out a
                   # median 0.87 of that, 0.57-0.78 on the worst ears. The
                   # orbit views then clipped the helix and the lobe, and that
                   # -- not frame angle -- was the gross-error tail on ears
                   # whose axes were within 20 deg: ratio <0.85 averaged 6.6%
                   # gross, >=0.85 averaged 0.47%. Same definition as the GT
                   # frame, so the two scales are comparable by construction.
                   scale=s_fin, scale_ratio=s_lm / s_chop,
                   fi=c["fi"], ui=c["ui"], sgn=c["sgn"], margin=margin,
                   score=top[0], nlm=top[1], evr=float(ev.min() / max(ev.max(), 1e-12)))
        # did the recorded convention (front = smallest axis) win?
        res["convention_won"] = (c["fi"] == int(np.argmin(ev)))
        if lm_gt is not None:
            # ear_frame's up is the WORLD y axis, i.e. the scan convention;
            # under --random-rot that is no longer anatomical up, so the GT
            # frame is built on the unrotated head and rotated with it.
            if nm in GTO:
                gtR, gtC = GTO[nm]
            else:
                gtR, gtC, _ = ear_frame(head, lm_gt, cfg)
            gt_front = gtR[2]
            # SIGNED, NOT abs(). abs() maps a 180 deg flip onto 0, so an
            # upside-down frame scored as perfect: the real up error here was
            # 171.7 deg and the abs metric printed 8.3. Shameem caught it by
            # looking at the render; no number in this script could have.
            R = frame_from(c["front"], c["up"], front=c["front"])
            res["view_deg"] = float(np.degrees(np.arccos(
                np.clip(R[2] @ gt_front, -1, 1))))
            res["up_deg"] = float(np.degrees(np.arccos(
                np.clip(R[1] @ gtR[1], -1, 1))))
            res["up_flipped"] = bool(R[1] @ gtR[1] < 0)
            # THE THIRD AXIS IS NOT FREE INFORMATION -- R is orthonormal, so
            # x = cross(y, z) and its error is determined by the other two.
            # Reported because a large x error with small front/up errors
            # would mean the frame is not orthonormal, i.e. a bug, not a
            # measurement.
            res["lat_deg"] = float(np.degrees(np.arccos(
                np.clip(R[0] @ gtR[0], -1, 1))))
            res["orthonormal_err"] = float(np.abs(R @ R.T - np.eye(3)).max())
            res["centre_mm"] = float(np.linalg.norm(cc - gtC))
        if a.render_dir:
            from PIL import Image
            R = frame_from(c["front"], c["up"], front=c["front"])
            sv = float(np.abs((Pw - cc) @ R.T).max())
            im, _, _ = render(in_frame(head, R, cc, sv), cfg)
            Image.fromarray(im).save(pathlib.Path(a.render_dir) / f"{nm}_bootstrap.png")
        return res, None

    if a.mesh:
        import open3d as o3d
        mp = pathlib.Path(a.mesh)
        head = o3d.io.read_triangle_mesh(str(mp))
        head.compute_vertex_normals()
        print(f"\n{mp.name}: {len(head.vertices)} verts, "
              f"extent {np.round(np.ptp(np.asarray(head.vertices), 0), 1)}")
        res, err = one(mp.stem, head, None)
        if err:
            sys.exit(f"FAILED: {err}")
        print(f"  eigen ratio min/max {res['evr']:.4f}  "
              f"(low = well separated axes, high = ordering is fragile)")
        print(f"  front = chop axis {res['fi']} "
              f"({'SMALLEST, the recorded convention' if res['convention_won'] else 'NOT the smallest -- convention would have failed here'})")
        print(f"  winning score {res['score']:.3f}, margin over runner-up "
              f"{res['margin']:.3f}, {res['nlm']}/55 landmarks")
        return

    names = sorted(f.stem for f in (WRAP / "corpus/canon2scan").glob("*.npz")
                   if str(np.load(f, allow_pickle=True)["dataset"]) == a.dataset)[:a.ears]
    names_all = list(names)
    if a.only:
        names = [n for n in names if n in set(a.only)]
    if a.workers > 1 and not a.shard:
        rows = run_sharded(a, names)
        summarize(rows)
        return
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        names = names[i::n]
    rows, ok_names = [], []
    for i, nm in enumerate(names):
        c = dict(cfg, light_mirror=light_mirror_for(nm, "frame", a.dataset))
        try:
            head, ear, lm = load_ear(nm, a.dataset, c)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"[{i+1}/{len(names)}] {nm}: skipped -- {e}"); continue
        if a.random_rot is not None:
            # a uniformly random rotation, fixed per (seed, ear) so runs repeat
            rr = np.random.default_rng((a.random_rot, names_all.index(nm)))
            q = rr.normal(size=4); q /= np.linalg.norm(q)
            w_, x_, y_, z_ = q
            Rr = np.array([[1 - 2*(y_*y_ + z_*z_), 2*(x_*y_ - z_*w_), 2*(x_*z_ + y_*w_)],
                           [2*(x_*y_ + z_*w_), 1 - 2*(x_*x_ + z_*z_), 2*(y_*z_ - x_*w_)],
                           [2*(x_*z_ - y_*w_), 2*(y_*z_ + x_*w_), 1 - 2*(x_*x_ + y_*y_)]])
            assert abs(np.linalg.det(Rr) - 1) < 1e-9
            gR0, gC0, _ = ear_frame(head, lm, c)
            GTO[nm] = (gR0 @ Rr.T, Rr @ gC0)      # rows are axes: v -> Rr v
            import open3d as o3d
            head.vertices = o3d.utility.Vector3dVector(np.asarray(head.vertices) @ Rr.T)
            head.compute_vertex_normals()
            lm = lm @ Rr.T
        res, err = one(nm, head, lm)
        if err:
            print(f"[{i+1}/{len(names)}] {nm}: {err}"); continue
        rows.append(res); ok_names.append(nm)
        print(f"[{i+1}/{len(names)}] {nm}: view {res['view_deg']:5.2f} deg, "
              f"up {res['up_deg']:5.2f}, centre {res['centre_mm']:5.2f} mm, "
              f"margin {res['margin']:.3f}, "
              f"lift/chop scale {res['scale_ratio']:.2f}, "
              + "".join(f"[{k}: {v}] " for k, v in res["traj"] if k == "procrustes")
              + f"{'convention' if res['convention_won'] else 'OVERRODE convention'}",
              flush=True)

    if a.save_frames and rows:
        out = {}
        for nm, r in zip(ok_names, rows):
            out[f"{nm}__R"] = r["R"]
            out[f"{nm}__C"] = r["centre"]
            out[f"{nm}__s"] = np.array(r["scale"])
            if "view_deg" in r:        # the GT comparison, so shards can be merged
                out[f"{nm}__m"] = np.array([r[k] for k in METRICS], float)
        pathlib.Path(a.save_frames).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.save_frames, **out)
        print(f"wrote {len(rows)} frames to {a.save_frames}")

    if a.shard:                 # the parent summarises the merged corpus
        return
    summarize(rows)


METRICS = ("view_deg", "up_deg", "centre_mm", "margin", "convention_won",
           "lat_deg", "orthonormal_err")


def run_sharded(a, names):
    """Re-run this script as a.workers shard processes and merge their frames.

    The per-ear work -- renders, detector, lift -- shares nothing between
    ears, so the shards are the same program over names[i::n]. Each writes
    its frames AND its GT metrics to its own npz; the parent merges those
    into --save-frames and prints the same summary a single process would.
    Lines from the shards are streamed through as they come, so the log
    reads the same but out of order.
    """
    import shutil
    import subprocess
    import tempfile
    import threading
    n = min(a.workers, len(names))
    tmp = tempfile.mkdtemp(prefix="bootstrap_shards_", dir=ROOT / "runs")
    argv = list(sys.argv[1:])
    for flag in ("--save-frames", "--workers"):
        while flag in argv:
            k = argv.index(flag); del argv[k:k + 2]
    # ONE THREAD POOL PER CORE, NOT PER SHARD. Left alone, every shard's
    # torch/OpenMP/BLAS opens cpu_count threads; 6 shards on 32 cores ran at
    # load 74, stalled, and were no faster than the serial run.
    import os
    th = str(max(1, (os.cpu_count() or 1) // n))
    env = dict(os.environ, OMP_NUM_THREADS=th, MKL_NUM_THREADS=th,
               OPENBLAS_NUM_THREADS=th, NUMEXPR_NUM_THREADS=th)
    procs = []
    for i in range(n):
        cmd = [sys.executable, sys.argv[0], *argv, "--workers", "1",
               "--shard", f"{i}/{n}", "--save-frames", f"{tmp}/shard{i}.npz"]
        # stderr is inherited, not merged: Filament/MediaPipe chatter on
        # stderr is thousands of lines and would bury the per-ear lines.
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True,
                                      env=env))

    def pump(pr, i):
        for line in pr.stdout:
            if line.startswith("[Open3D"):       # one per render, all identical
                continue
            print(f"[w{i}] {line}", end="", flush=True)
    ts = [threading.Thread(target=pump, args=(pr, i)) for i, pr in enumerate(procs)]
    for t in ts: t.start()
    for t in ts: t.join()
    rcs = [pr.wait() for pr in procs]
    out, rows = {}, []
    for i in range(n):
        f = pathlib.Path(f"{tmp}/shard{i}.npz")
        if not f.exists():
            print(f"shard {i} wrote nothing (rc={rcs[i]})"); continue
        z = np.load(f)
        out.update({k: z[k] for k in z.files})
        for k in z.files:
            if k.endswith("__m"):
                rows.append(dict(zip(METRICS, z[k].tolist())))
        f.unlink()
    shutil.rmtree(tmp, ignore_errors=True)
    if any(rcs):
        print(f"shard exit codes {rcs}")
    if a.save_frames and out:
        pathlib.Path(a.save_frames).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.save_frames, **out)
        nf = sum(k.endswith("__R") for k in out)
        print(f"wrote {nf} frames to {a.save_frames}")
    return rows


def summarize(rows):
    if not rows:
        sys.exit("no ears bootstrapped")
    v = np.array([r["view_deg"] for r in rows])
    u = np.array([r["up_deg"] for r in rows])
    cm = np.array([r["centre_mm"] for r in rows])
    mg = np.array([r["margin"] for r in rows])
    cw = np.array([r["convention_won"] for r in rows])
    print(f"\n{len(rows)} ears, against the GT frame")
    print(f"  view axis  median {np.median(v):5.2f} deg   max {v.max():5.2f}")
    print(f"  up axis    median {np.median(u):5.2f} deg   max {u.max():5.2f}")
    lt = np.array([r["lat_deg"] for r in rows])
    on = np.array([r["orthonormal_err"] for r in rows])
    print(f"  lateral    median {np.median(lt):5.2f} deg   max {lt.max():5.2f}"
          f"   (determined by the other two)")
    print(f"  max |R R^T - I| over all ears {on.max():.2e}  "
          f"(a frame that is not orthonormal is a bug, not an error)")
    print(f"  centre     median {np.median(cm):5.2f} mm")
    print(f"  the recorded convention won on {int(cw.sum())}/{len(rows)} ears")
    print(f"  margin over runner-up: median {np.median(mg):.3f}, min {mg.min():.3f}")
    print("  (a small min margin means the score barely separated the "
          "candidates -- it is the warning sign, not the angle)")


if __name__ == "__main__":
    main()
