"""Finding the ear, and building the frames to look at it from.

load_head, find_ears, the frame helpers and the MediaPipe head pose. These were
duplicated in ingest_render3d.py and had already diverged -- find_ears there
rendered through a private camera setup of its own."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import open3d as o3d

from .backproject import backproject
from .camera import orbit_extrinsic
from .render import render
from .scheme import CHIN, FOREHEAD, NASION, NOSE_TIP, TRAGION_L, TRAGION_R
CHEEK_R, CHEEK_L = 50, 280          # FaceMesh cheek points, for nose relief

def load_head(path):
    """Head mesh, centred and scaled to unit radius."""
    m = o3d.io.read_triangle_mesh(str(path))
    if len(m.vertices) == 0:
        return None
    V = np.asarray(m.vertices)
    V = (V - V.mean(0)) / np.abs(V - V.mean(0)).max()
    m.vertices = o3d.utility.Vector3dVector(V)
    m.compute_vertex_normals()
    return m


def frame_from(normal, up, front=None):
    """Rotation taking `normal` onto +Z (toward the camera), keeping `up` near +Y."""
    n = np.asarray(normal, float)
    if front is not None and n @ np.asarray(front, float) < 0:
        n = -n
    z = n / max(np.linalg.norm(n), 1e-9)
    y = np.asarray(up, float) - z * (np.asarray(up, float) @ z)
    ny = np.linalg.norm(y)
    if ny < 1e-6:
        return None
    y /= ny
    return np.stack([np.cross(y, z), y, z])            # world -> camera frame


def plane_normal(points):
    """Surface normal of the best-fit plane through the landmarks."""
    P = np.asarray(points, float)
    _, _, vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    return vt[2]


def in_frame(mesh, R, C, scale):
    """A copy of the mesh rotated into frame R about C and scaled by `scale`."""
    g = copy.deepcopy(mesh)
    g.vertices = o3d.utility.Vector3dVector(
        ((np.asarray(mesh.vertices) - C) @ R.T) / scale)
    g.compute_vertex_normals()
    return g


def find_ears(mesh, det, cfg, probe_size=400, dist=2.4):
    """Locate ears by sweeping the detector around the head.

    Geometric rules do not transfer between datasets -- HUTUBS and SONICOM store
    heads on different axes -- so the ear is found by what an ear detector sees.
    """
    out = []
    for axis in (0, 1, 2):
        for sgn in (1, -1):
            front = np.zeros(3); front[axis] = sgn
            up = np.array([0.0, 0.0, 1.0]) if axis != 2 else np.array([0.0, 1.0, 0.0])
            R = frame_from(front, up, front=front)
            if R is None:
                continue
            g = in_frame(mesh, R, np.asarray(mesh.vertices).mean(0), 1.0)
            b = det.detect(render(g, cfg, size=probe_size)[0])
            if not len(b):
                continue
            h, w = b[0, 2] - b[0, 0], b[0, 3] - b[0, 1]
            if (h * w) / (probe_size ** 2) > 0.25:   # a whole-head box is a false positive
                continue
            out.append((float(b[0, 4]), front, up, b[0, :4].copy()))
    out.sort(key=lambda t: -t[0])
    return out[:2]


def head_pose(mesh, cfg):
    """Head frame from MediaPipe FaceMesh, in the mesh's own coordinates.

    Returns (lateral, vertical, tragion_R, tragion_L) or None.

    WHY THIS BEATS SWEEPING THE EAR DETECTOR. The sweep renders six axis-aligned
    views and keeps the two best ear boxes, which on a head gives two nearly equal
    confidences and no way to tell which side is which. MediaPipe finds a face on
    exactly ONE of those six views -- on every head tried -- so the frontal
    direction is unambiguous, and the face landmarks then give a real anatomical
    frame rather than whichever axis the dataset stored the head on. The ear
    direction is the sagittal-plane normal -- 90 degrees from the nose -- built
    from forehead, chin, nasion and nose tip. The tragion landmarks are used only
    to pick which side is which and to centre the view, never for the direction.

    IT DOES NOT REMOVE PASS 2, AND IT IS NOT MEASURABLY MORE ACCURATE. Head to
    head over four heads, pass-2 confidence against the detector sweep: 0.931 vs
    0.955, 0.942 vs 0.927, fails vs 0.881, 0.980 vs 0.981. The remaining offset
    from the pinna plane is the same either way (8-35 deg here, 16-37 for the
    sweep). What it buys is that the SIDE is known -- the sweep returns two ear
    boxes at near-equal confidence with no way to tell left from right, so "ear 0"
    is arbitrary and changes between heads. The pinna's own tilt is per-subject
    anatomy that no face-derived frame can supply, which is why pass 2 stays.

    Measured per side (pass-2 confidence / degrees off the pinna plane), the RIGHT
    ear is consistently better: pp16 fails/0.839@12, pp12 0.931@33/0.990@28,
    pp11 0.942@35/0.986@13, pp10 0.980@19/0.982@8. Left and right ears are mirror
    images, so this is most likely an asymmetry in the landmarker rather than in
    this frame -- it is not explained, and worth a look before trusting left-ear
    labels as much as right.
    """
    task = cfg.get("face_task")
    if not task or not Path(task).exists():
        print(f"mediapipe: no face_landmarker.task at {task} -- using the ear sweep")
        return None
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    fl = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(task)), num_faces=1))
    C0 = np.asarray(mesh.vertices).mean(0)
    cands = []
    axes = [(ax, sg) for ax in (0, 1, 2) for sg in (1, -1)]
    # THE SIX AXIS VIEWS ASSUME A HEAD-ALIGNED SCAN. A head in an arbitrary
    # pose can put the face 55 deg off every axis, beyond what FaceMesh
    # detects reliably, so if the six find nothing the eight cube diagonals
    # are tried as well (max obliquity then ~35 deg).
    diag = [np.array(v, float) / np.sqrt(3.0) for v in
            ((sx, sy, sz) for sx in (1, -1) for sy in (1, -1) for sz in (1, -1))]
    # and the 12 cube-edge midpoints as a last ring (26 directions in all,
    # max obliquity ~22 deg): the rotated pp19_left had no face on 14.
    edge = [np.array(v, float) / np.sqrt(2.0) for v in
            [(sa, sb, 0) for sa in (1, -1) for sb in (1, -1)]
            + [(sa, 0, sb) for sa in (1, -1) for sb in (1, -1)]
            + [(0, sa, sb) for sa in (1, -1) for sb in (1, -1)]]
    # ABSOLUTE GATE, NOT JUST A RANKING. Ranking by relief and stability
    # only helps when a real face is among the hits; on rotated heads
    # (pp16_left, pp19_left) the ONLY hit was a scalp false positive --
    # relief 11.0 and 7.3 mm, stable 2/4 and 1/4 -- and it was accepted,
    # and the ear seed then looked at the top of the head. Real faces
    # measured 15-45 mm and 3-4/4 (facediag, 32 ears), so a hit below
    # 15 mm or under 3/4 does not count, the search goes on to the next
    # ring, and with nothing passing the answer is "no face", not a frame.
    good = lambda: [t for t in cands if t[0] >= 15.0 and t[1] >= 3]
    for rnd, dirs in enumerate(([np.eye(3)[ax] * sg for ax, sg in axes], diag, edge)):
        if rnd >= 1 and good():
            break
        # FOUR ROLLS PER DIRECTION. The face detector is weak beyond ~45 deg
        # of in-plane roll, and with an arbitrary "up" a rotated head shows
        # its face at a random roll: 11 of 18 randomly rotated heads had no
        # face in 26 directions while 32 of 32 upright ones did. The
        # reference implementation's random-rotation loop randomises roll
        # implicitly; this does it explicitly.
        for front in dirs:
            axis = int(np.argmax(np.abs(front))); sgn = int(np.sign(front[axis]))
            up0 = np.array([0.0, 0.0, 1.0]) if abs(front[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
            up0 = up0 - front * (up0 @ front); up0 /= np.linalg.norm(up0)
            side = np.cross(front, up0)
            ups = [up0, side, -up0, -side]
            for up in ups:
              R = frame_from(front, up, front=front)
              if R is None:
                continue
              # SCALE BY THE MESH, NOT 1.0. The camera is set up for a mesh
              # spanning ~2 units; a head in millimetres spans ~300, so a unit
              # scale showed MediaPipe an extreme close-up of skin and it found
              # a face on 0 of 6 views. This worked while heads were
              # HUTUBS-normalised and broke silently when the corpus moved to
              # mm -- the measurements in this docstring predate that. With the
              # mesh framed, faces are found on 2 of 6 views again.
              sc = float(np.abs((np.asarray(mesh.vertices) - C0) @ R.T).max())
              g = in_frame(mesh, R, C0, sc)
              img, depth, cam = render(g, cfg)
              res = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                       data=np.ascontiguousarray(img)))
              if not res.face_landmarks:
                  continue
              # MediaPipe normalises to the IMAGE it was handed, so scale by that
              # image's own shape, never by the size we requested.
              h, w = img.shape[:2]
              uv = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
              P, hit = backproject(depth, uv, cam)
              if not hit[[TRAGION_R, TRAGION_L, FOREHEAD, CHIN,
                          NOSE_TIP, NASION, CHEEK_R, CHEEK_L]].all():
                  continue
              # The landmarks come back in THIS frame's coordinates; everything
              # downstream works in the mesh's own, so convert before using them.
              # Skipping this put the ear view 62-70 deg off instead of 10-24.
              P = (P * sc @ R) + C0
              # THE EAR DIRECTION IS 90 DEG FROM THE NOSE: the normal of the
              # sagittal plane, from the head's own up and forward axes.
              #
              # The obvious alternative is the tragion-to-tragion line, and over 12
              # ears it is no better -- mean 25.1 deg from the pinna plane against
              # 25.7 here, median 26.9 against 24.2. What decides it is that the
              # tragion landmarks are the weak ones: the reference implementation in
              # ../landmarking_stuff/landmarker marks exactly these indices "most
              # likely not possible" and does not use them. Forehead, chin, nasion
              # and nose tip are points FaceMesh is actually good at, so the same
              # answer rests on firmer ground.
              #
              # A third option, the head's local surface normal at the tragion, is
              # better typically (mean 16.4, median 14.4) but has a worse tail
              # (48.2) and failed outright on 1 of 12. Not taken.
              vert = P[FOREHEAD] - P[CHIN]
              vert /= np.linalg.norm(vert)
              fwd = P[NOSE_TIP] - P[NASION]
              fwd -= vert * (fwd @ vert)
              if np.linalg.norm(fwd) < 1e-6:
                  continue
              fwd /= np.linalg.norm(fwd)
              lat = np.cross(vert, fwd)
              lat /= np.linalg.norm(lat)
              # Orient toward the subject's left so "ear 0" keeps its meaning.
              if lat @ (P[TRAGION_L] - P[TRAGION_R]) < 0:
                  lat = -lat
              vert -= lat * (vert @ lat)
              vert /= np.linalg.norm(vert)
              # KEEP THE BEST VIEW, NOT THE FIRST. FaceMesh will return a face
              # from a SIDE view of a head, and taking the first hit meant that
              # answer won whenever it came earlier in the axis order. On
              # pp10_right, axis 0+ gave a vertical 143 deg from the GT frame's
              # up while axis 2- gave 5.3 -- both detected, both with every
              # required landmark, and the wrong one returned.
              #
              # Tragion separation is the discriminator: a frontal view sees
              # both tragions at their true spacing, an oblique one foreshortens
              # them (140.2 mm against 130.4 here). It costs nothing, since the
              # value was already being computed for the log line.
              sep = float(np.linalg.norm(P[TRAGION_L] - P[TRAGION_R]))
              # ...EXCEPT THAT IT IS NOT. Diagnosed over the 16 HUTUBS ears whose
              # bootstrap frame came out >20 deg wrong plus 16 that came out
              # right (facediag, 2026-10-16): FaceMesh returns a "face" from
              # side views of a head and from the cut planes of a cropped
              # scan, and those false faces have WIDER back-projected tragions
              # than the real one (pp6_right: 177 vs 131 mm), so max-separation
              # picked the wrong view on 6 of 16 failing ears and caused every
              # up-flip. Two things separate a real face from a scalp hit on
              # all 32 ears, each on its own:
              #   nose relief   nose tip minus the cheek midpoint, along the
              #                 view axis. Real faces 15-45 mm; false ones
              #                 -11 to 19 mm (one overlap, so not a threshold
              #                 by itself).
              #   stability     the same face is still detected when the view
              #                 is rotated +-15 deg in yaw and pitch. Real
              #                 faces 3-4 of 4; false ones 0-1 (one at 3).
              # The pick is max relief among hits that survive >=2 of the 4
              # perturbed views; if none does, max relief alone. Costs four
              # extra renders per hit.
              relief = float((P[NOSE_TIP] - 0.5 * (P[CHEEK_R] + P[CHEEK_L])) @ front)
              stab = 0
              for yaw, pitch in ((15, 0), (-15, 0), (0, 15), (0, -15)):
                  im2, dp2, cam2 = render(g, cfg, E=orbit_extrinsic(yaw, pitch, cfg))
                  r2 = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                          data=np.ascontiguousarray(im2)))
                  stab += bool(r2.face_landmarks)
              cands.append((relief, stab, sep, axis, sgn, lat, vert, P[TRAGION_R], P[TRAGION_L],
                            P, hit, front))
    if not good():
        weak = ", ".join(f"relief {t[0]:.1f} stab {t[1]}" for t in cands)
        print(f"mediapipe: no face passing the gate on 26 views"
              f"{' (rejected: ' + weak + ')' if weak else ''} -- using the ear sweep")
        return None
    stable = good()
    stable.sort(key=lambda t: -t[0])
    relief, stab, sep, axis, sgn, lat, vert, trR, trL, P, hit, front0 = stable[0]
    others = ", ".join(f"{t[3]}{'+' if t[4] > 0 else '-'} relief {t[0]:.1f} stab {t[1]}"
                       for t in cands if t is not stable[0])
    print(f"mediapipe: face on axis {axis}{'+' if sgn > 0 else '-'}, "
          f"nose relief {relief:.1f} mm, stable {stab}/4, tragion sep {sep:.1f}"
          f" (of {len(cands)}{': ' + others if others else ''})")

    # REFINE FACE-ON, THEN FIT THE CANONICAL FACE. The hit above came from
    # whichever of 14 fixed directions was nearest the face, up to ~35-55
    # deg oblique, and an oblique FaceMesh gives a poor frame: on a rotated
    # pp10_left the "tragion" landed on the cheek and the lateral axis was
    # ~50 deg off, so the ear-seed render framed the face. This is the trick
    # from the reference implementation (landmarking_stuff/landmarker): once
    # a nose is found, re-render looking straight at it with the forehead up
    # and detect again, then take the head frame from a rotation-only
    # similarity fit of MediaPipe's canonical face model (468 points, +x =
    # subject's left, +y up, +z forward) onto the back-projected landmarks,
    # rather than from four of them.
    V = np.asarray(mesh.vertices)
    for _ in range(2):
        fwd = P[NOSE_TIP] - P[NASION]
        fwd -= vert * (fwd @ vert)
        if np.linalg.norm(fwd) < 1e-6:
            break
        fwd /= np.linalg.norm(fwd)
        R = frame_from(fwd, vert, front=fwd)
        if R is None:
            break
        nose = P[NOSE_TIP]
        sc = 0.6 * float(np.abs((V - nose) @ R.T).max())
        g = in_frame(mesh, R, nose, sc)
        img, depth, cam = render(g, cfg)
        res = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=np.ascontiguousarray(img)))
        if not res.face_landmarks:
            break
        h, w = img.shape[:2]
        uv = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
        P2, hit2 = backproject(depth, uv, cam)
        if not hit2[[TRAGION_R, TRAGION_L, FOREHEAD, CHIN, NOSE_TIP, NASION]].all():
            break
        P, hit = (P2 * sc @ R) + nose, hit2
        vert = P[FOREHEAD] - P[CHIN]
        vert /= np.linalg.norm(vert)
    model = Path(cfg.get("face_model") or Path(task).parent / "canonical_face_model.obj")
    if model.exists():
        canon = np.array([[float(x) for x in l.split()[1:4]]
                          for l in open(model) if l.startswith("v ")])
        ok = hit[: len(canon)]
        X, Y = canon[ok], P[: len(canon)][ok]
        cx, cy = X.mean(0), Y.mean(0)
        A, B = X - cx, Y - cy
        U, S, Vt = np.linalg.svd(A.T @ B)
        D = np.eye(3)
        if np.linalg.det(U @ Vt) < 0:
            D[-1, -1] = -1                      # rotation only, no mirror
        Rc = U @ D @ Vt                         # Y ~ k (X - cx) @ Rc + cy
        k = (S * np.diag(D)).sum() / (A ** 2).sum()
        lat, vert, fwd = (np.eye(3) @ Rc)       # canonical axes in the world
        for v_ in (lat, vert, fwd):
            v_ /= np.linalg.norm(v_)
        trR, trL = (k * (canon[[TRAGION_R, TRAGION_L]] - cx) @ Rc + cy)
        rms = float(np.sqrt(((k * A @ Rc + cy - Y) ** 2).sum(1).mean()))
        print(f"mediapipe: canonical face fit on {int(ok.sum())} points, rms {rms:.1f} mm, "
              f"face-on refine moved the view {np.degrees(np.arccos(np.clip(fwd @ front0, -1, 1))):.1f} deg")
    else:
        print(f"mediapipe: no canonical_face_model.obj at {model}; 4-point frame")
    return lat, vert, trR, trL


def normalize_head(mesh, cfg):
    """The head in its own frame: centroid at the origin, +x the subject's
    left, +y scalp up, +z nose forward. Returns (mesh_c, R_head, centre) with
    V_c = (V - centre) @ R_head.T, or None when no face passes head_pose's
    gate.

    WHY A SEPARATE STEP. Every stage after the face search wants a known
    orientation: the seed render wants the ear upright and face-on, the
    chop candidates want a sign convention, the template fit wants left
    form. Threading lateral/vertical through each of them worked, but each
    was one more place to get a transform backwards, and every debug render
    came out at a random roll. Rotating ONCE here puts the whole pipeline
    at the true origin; results go back to the world with the one inverse.
    The per-ear pinna tilt (8-35 deg off the head's lateral axis) is NOT
    part of this -- that is what the ear stage recovers.

    Rows of R_head are the canonical axes expressed in the world, so a
    canonical direction d maps to the world as d @ R_head and a point as
    p @ R_head + centre.
    """
    hp = head_pose(mesh, cfg)
    if hp is None:
        return None
    lat, vert = np.asarray(hp[0], float), np.asarray(hp[1], float)
    vert = vert - lat * (vert @ lat)
    vert /= np.linalg.norm(vert)
    fwd = np.cross(lat, vert)                 # x cross y = z: nose forward
    R_head = np.stack([lat, vert, fwd])
    V = np.asarray(mesh.vertices)
    centre = V.mean(0)
    mesh_c = o3d.geometry.TriangleMesh(mesh)
    mesh_c.vertices = o3d.utility.Vector3dVector((V - centre) @ R_head.T)
    mesh_c.compute_vertex_normals()
    return mesh_c, R_head, centre


def mirror_x(mesh):
    """The mesh reflected about the sagittal plane (x -> -x) with the
    triangle winding reversed, so normals and shading stay outward.

    Every right ear in the corpus is mirrored into left form before the
    detector sees it, and the detector has only been exercised on left-form
    ears here: on a 20-head review sheet it boxed 20/20 left ears and about
    half the right ones (cheek, neck, nose, or a patch inside the concha).
    In the canonical head frame handedness is just the sign of x, so the
    production path mirrors the -x side, runs the left-form pipeline, and
    un-mirrors the answer (x components of R's columns and of the centre).
    """
    V = np.asarray(mesh.vertices) * np.array([-1.0, 1.0, 1.0])
    F = np.asarray(mesh.triangles)[:, ::-1].copy()
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                  o3d.utility.Vector3iVector(F))
    if mesh.has_vertex_colors():          # keep the skin; grey renders detect worse
        m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors).copy())
    m.compute_vertex_normals()
    return m
