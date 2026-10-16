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
    p.add_argument("--dataset", default="hutubs")
    p.add_argument("--mesh", default=None,
                   help="score a mesh with no ground truth (prints the frame "
                        "and the margin, skips the angle comparison)")
    p.add_argument("--seed-axis", default="x", choices=list("xyz"),
                   help="axis for the SEED render that gets the detector box")
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
    p.add_argument("--rule", default="score", choices=("score", "smallest"),
                   help="how to pick the axis assignment. 'smallest' forces "
                        "the recorded convention (front = smallest chop axis, "
                        "up = largest) and so isolates whether chop+PCA is "
                        "sound from whether the scorer can rank candidates.")
    p.add_argument("--render-dir", default=None,
                   help="write the winning frame's render here, to look at")
    a = p.parse_args()

    from escape_audit import DEFAULTS, WRAP, load_ear, ear_frame, light_mirror_for
    from ear3d.frames import frame_from, in_frame, plane_normal
    from ear3d.render import render
    from ear3d.label import pick
    from ear3d.twod import make_pipeline
    from ear3d.camera import orbit_extrinsic
    from ear3d.triangulate import ransac_point

    def lift_in_frame(mesh, front, up, centre, cfg, pipe, half, pts=None):
        """Triangulate the 55 landmarks from a small grid in this frame."""
        R = frame_from(front, up, front=front)
        if R is None:
            return None
        V = np.asarray(mesh.vertices)
        ref = V if pts is None else np.asarray(pts)
        sc = float(np.abs((ref - centre) @ R.T).max())
        g = in_frame(mesh, R, centre, sc)
        angles = [(y, q) for y in (-half, 0.0, half) for q in (-half, 0.0, half)]
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
    print(f"{ck.parent.name}/{ck.name}")

    if a.render_dir:
        pathlib.Path(a.render_dir).mkdir(parents=True, exist_ok=True)

    def one(nm, head, lm_gt):
        """Bootstrap one head. lm_gt may be None (no ground truth)."""
        V = np.asarray(head.vertices)
        hc = V.mean(0)
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
        if lm_gt is not None:
            seed = lm_gt.mean(0) - hc
            seed = seed / max(np.linalg.norm(seed), 1e-9)
        else:
            ax = {"x": 0, "y": 1, "z": 2}[a.seed_axis]
            seed = np.zeros(3); seed[ax] = 1.0
        Rs = frame_from(seed, np.array([0.0, 1.0, 0.0]), front=seed)
        ss = float(np.abs((V - hc) @ Rs.T).max())
        gs = in_frame(head, Rs, hc, ss)
        img, _, cam = render(gs, cfg)
        best = pick(pipe(np.asarray(img), timestamp=0.0), cfg, img.shape[:2])
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
        Pw = Vs[keep] * ss @ Rs + hc
        cc = Pw.mean(0)
        cands, vt, ev = candidates(Pw, cc, hc - cc)

        if a.rule == "smallest":
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
        traj = []
        for it in range(max(0, a.passes - 1)):
            try:
                P3 = lift_in_frame(head, front_i, up_i, cc, cfg, pipe,
                                   a.refine_grid, Pw)
            except Exception as e:                      # noqa: BLE001
                traj.append(("lift failed", str(e)[:40])); break
            if P3 is None or np.isfinite(P3).all(1).sum() < 20:
                traj.append(("too few points", "")); break
            Q = P3[np.isfinite(P3).all(1)]
            n = plane_normal(Q)
            # trim: drop the 20% furthest from the plane, refit
            dd = np.abs((Q - Q.mean(0)) @ n)
            Q = Q[dd <= np.quantile(dd, 0.8)]
            n = plane_normal(Q)
            if n @ (cc - hc) < 0:
                n = -n
            # up: the pinna's long axis within the plane
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
        res = dict(front=c["front"], up=c["up"], centre=cc, ev=ev,
                   traj=traj,
                   fi=c["fi"], ui=c["ui"], sgn=c["sgn"], margin=margin,
                   score=top[0], nlm=top[1], evr=float(ev.min() / max(ev.max(), 1e-12)))
        # did the recorded convention (front = smallest axis) win?
        res["convention_won"] = (c["fi"] == int(np.argmin(ev)))
        if lm_gt is not None:
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
    rows = []
    for i, nm in enumerate(names):
        c = dict(cfg, light_mirror=light_mirror_for(nm, "frame", a.dataset))
        try:
            head, ear, lm = load_ear(nm, a.dataset, c)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"[{i+1}/{len(names)}] {nm}: skipped -- {e}"); continue
        res, err = one(nm, head, lm)
        if err:
            print(f"[{i+1}/{len(names)}] {nm}: {err}"); continue
        rows.append(res)
        print(f"[{i+1}/{len(names)}] {nm}: view {res['view_deg']:5.2f} deg, "
              f"up {res['up_deg']:5.2f}, centre {res['centre_mm']:5.2f} mm, "
              f"margin {res['margin']:.3f}, "
              f"{'convention' if res['convention_won'] else 'OVERRODE convention'}",
              flush=True)

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
    print(f"  centre     median {np.median(cm):5.2f} mm")
    print(f"  the recorded convention won on {int(cw.sum())}/{len(rows)} ears")
    print(f"  margin over runner-up: median {np.median(mg):.3f}, min {mg.min():.3f}")
    print("  (a small min margin means the score barely separated the "
          "candidates -- it is the warning sign, not the angle)")


if __name__ == "__main__":
    main()
