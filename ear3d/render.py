"""The renderer: Open3D/Filament, one lighting setup, returning its own camera.

Every render in the 3D path comes through here, including the detector probe
sweep. ingest_render3d.py used to carry its own OffscreenRenderer with a separate
camera setup, which is how the two drifted."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from .camera import Camera, camera_ke

MATERIAL = o3d.visualization.rendering.MaterialRecord()
MATERIAL.shader = "defaultLit"


LINE_MATERIAL = o3d.visualization.rendering.MaterialRecord()
LINE_MATERIAL.shader = "unlitLine"
LINE_MATERIAL.line_width = 2.0


# The sun, as a direction the light TRAVELS, in the render's view frame: x right,
# y down, z into the scene, the same convention camera_ke() builds E in. Stated
# in VIEW space rather than world space so the ear's occluding contour keeps its
# shadow whatever the geometry's handedness -- which is the whole point, below.
SUN_VIEW = (0.3, 0.4, 0.9)
SUN_LUX = 95000


def side_light(cfg, side):
    """cfg for a SIDE VIEW of a normalised head: light from the face side.

    Side views put the camera on +x (side "L", the subject's left ear) or -x
    ("R") of the head frame, with the same up. The configured sun is fixed
    to the camera, so it lights the left ear from in front of the face and
    the right ear from behind the head. Measured on 20 randomly rotated
    HUTUBS heads, BlazeEar's top box on the ear:

        left ear,  sun as configured   18/20  median score 0.79
        left ear,  sun x flipped        6/20               0.40
        right ear, sun as configured    9/20               0.47
        right ear, sun x flipped       20/20               0.87

    No single camera-fixed sun does both (overhead 16/16, x=0 12/16). So
    the right view flips x and both ears are lit from the face side. This
    is NOT the reflected-geometry flip config.light_mirror warns about: the
    mesh is not reflected here, only the camera changes side.
    """
    return dict(cfg, light_mirror=(side == "R"))


def sun_world(cfg, E):
    """The sun as a world direction, for a render whose extrinsic is E.

    THIS IS NOT COSMETIC, AND IT IS THE LARGEST EFFECT IN THE PIPELINE.
    Measured over 40 ears x 25 views x 55 points on CORRECTED renders
    (archive-final:archive/scripts/sun_sweep.py), moving only the sun's x component moves the rate of
    landmarks that land more than 10 mm off the ear:

        sun x    all     left    right
        -0.6    8.98%   9.34%   8.61%
        -0.3    8.32%   8.72%   7.93%
         0.0    6.38%   6.67%   6.09%
        +0.3    3.90%   4.43%   3.37%     the value shipped until 2026-10-15
        +0.6    2.37%   2.60%   2.14%
        +0.9    1.77%   1.91%   1.63%
        +1.2    1.61%   1.73%   1.48%     the default now
        +1.8    1.54%   1.62%   1.45%

    5.84x worst to best, better on 40 of 40 ears. +1.2 against the old +0.3 is
    -2.29pp on 39 of 40, p=2.0e-07.

    WHY +1.2 AND NOT +1.8, which scores marginally lower. The last step is
    -0.069pp, better on 18 of 40 ears at p=0.97 -- the curve has plateaued, so
    +1.8 buys nothing measurable. It is also the edge of the sweep, where an
    optimum is a boundary rather than a turning point, and the render is visibly
    darker: the concha falls into shadow and skin texture washes out, which
    other consumers of these renders depend on. +1.2 is interior, on the
    plateau, and keeps the texture.

    THE SIDES MOVE TOGETHER at every sun, Mann-Whitney p=0.19 to 0.68. An
    earlier version of this docstring reported a large side asymmetry (+4.20
    points on left ears against -0.67 on right, p=3.4e-07) and a per-side sun
    rule built on it. Both were artefacts of reflected heads being rendered
    with inverted normals -- see escape_audit.load_ear. Nothing handed survives
    the winding fix, and `light_mirror` must stay False for every ear.

    What it costs, seen: under the unfavourable sun the helix rim is front-lit
    and runs continuously into the scalp behind it -- no boundary for the 2D
    model to find, and the outer landmark strip drifts onto the scalp. Under the
    favourable one the rim is grazed and throws a dark crease along the
    occluding contour, and the same model puts the same strip on the silhouette.

    So the sun is specified in the VIEW frame and rotated into the world per
    render. `light_mirror` flips its x for geometry that arrived through a
    reflection, which restores the light's relationship to the anatomy rather
    than to the coordinate frame.

    That 5.34 -> 3.24 is lighting and is fixed here. It is NOT the whole of the
    side asymmetry, and an earlier version of this docstring said the 2D model
    was not handedness-biased on the strength of a 0.55% pooled bound measured
    over its own held-out photographs. THAT BOUND WAS WORTHLESS. It pools over a
    side-balanced corpus, where a bias of +d on one presented handedness appears
    as -d on the other and the pooled delta goes to zero however large the bias
    is -- and it scores plain NME, which is mostly tangential annotation noise.
    Stratified by side on the balanced source and scored on the normal
    component, Ear_Landmarker measures the bias at +8.53% (p=1.7e-17),
    antisymmetric across sides and absent from the tangential component. Its
    review is in Ear_Landmarker/runs/flip_equivariance/REVIEW.md. Re-cutting
    this repository's own per-sample file by side reproduces the plain-NME
    figure, 1.38%, which is what the pooled number was hiding.

    So the model does carry a handedness bias, the lighting bug here was real
    and separate, and which of the two owns the residual that survives this fix
    -- 3.24% -> 2.06% under an image mirror, and left still 3.76% against right
    2.72% when each side already has its best sun -- is NOT settled. Right ears
    reach the canonical frame through a reflection, so a chirality effect in the
    lift or the wrap fit would also produce a side-dependent residual that an
    image mirror happens to move. Do not attribute it until that is separated.
    """
    d = np.asarray(cfg.get("light_dir", SUN_VIEW), float)
    if cfg.get("light_mirror"):
        d = d * [-1.0, 1.0, 1.0]
    if cfg.get("light_frame", "view") == "world":
        return d
    return np.asarray(E, float)[:3, :3].T @ d          # view -> world


def light_scene(scene, background, direction=None):
    """The one lighting setup, shared by the snapshot and the 3D window.

    `direction` is a WORLD direction; callers that have a cfg and an extrinsic
    should pass sun_world(cfg, E). The default keeps the pre-2026-10-14
    world-fixed sun, so the viewers and the audit renderers that have no
    extrinsic to hand are unchanged.
    """
    scene.set_background([*background, 1.0])
    d = (-0.3, -0.4, -0.9) if direction is None else tuple(float(x) for x in direction)
    scene.scene.set_sun_light(list(d), [1.0, 1.0, 1.0], SUN_LUX)
    scene.scene.enable_sun_light(True)


def render(mesh, cfg, size=None, background=(0.5, 0.5, 0.5), E=None):
    """Render, and return the depth buffer and camera that produced it.

    Returns (image HxWx3 uint8, depth float32 HxW, Camera). The depth is
    view-space z, with inf where nothing was drawn. Callers back-project through
    THIS depth and THIS camera; see backproject().
    """
    size = size or cfg["size"]
    r = o3d.visualization.rendering.OffscreenRenderer(size, size)
    r.scene.add_geometry("m", mesh, MATERIAL)
    K, E = camera_ke(size, cfg, E)
    light_scene(r.scene, background, sun_world(cfg, E))
    r.setup_camera(K, E, size, size)
    img = np.asarray(r.render_to_image())[:, :, :3]
    depth = np.asarray(r.render_to_depth_image(z_in_view_space=True))
    del r
    return img, depth, Camera(K, E, img.shape[:2], size)


def gui_depth_to_view(depth, scene_camera):
    """The GUI's normalised depth buffer -> view-space distance, background inf.

    The window's own capture has no z_in_view_space flag, and its projection has
    an INFINITE far plane: P[2,2] is -1 and P[2,3] is -2*near, so the buffer
    holds d = 1 + near/z_e with z_e negative in front of the camera, and the
    distance is near / (1 - d). Measured against a sphere whose apex is exactly
    2.5 away: 2.5011, matching what the offscreen renderer reports in view space.

    The projection form is asserted rather than assumed -- if Open3D ever ships a
    finite far plane or reverse-Z, this must fail loudly instead of returning
    quietly wrong depths.
    """
    P = np.asarray(scene_camera.get_projection_matrix(), float)
    if not (abs(P[2, 2] + 1.0) < 1e-3 and P[2, 3] < 0):
        raise RuntimeError(f"unexpected projection; cannot linearise depth:\n{P}")
    near = -P[2, 3] / 2.0
    d = np.asarray(depth, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = near / (1.0 - d)
    return np.where(d >= 1.0, np.inf, z)
