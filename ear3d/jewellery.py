"""Synthetic ear jewellery: geometry added to a head mesh, landmarks unchanged.

The pipeline re-skins every mesh it is given, so a real scan's earring reaches
the 2D model as skin-coloured geometry (pp21_left). Jewellery here is geometry
only and is added BEFORE skinning, so it is rendered the same way.

add_jewellery(mesh, P, head_centre, rng, kind=None) -> (mesh, params)
  P: the ear's 55 landmarks in the mesh's frame and units (mm).
  kind: "hoop" | "stud" (lobe) | "helix" (small hoop or stud on the upper rim);
  None draws hoop 0.5, stud 0.3, helix 0.2.
"""
import numpy as np
import open3d as o3d

KINDS, WEIGHTS = ("hoop", "stud", "helix"), (0.5, 0.3, 0.2)


def _unit(v):
    return v / np.linalg.norm(v)


def _frame(P, head_centre):
    c = P.mean(0)
    n = np.linalg.svd(P - c, full_matrices=False)[2][2]
    if n @ (c - head_centre) < 0:
        n = -n                                                       # out of the head
    up = P[0:9].mean(0) - P[15:20].mean(0); up = _unit(up - (up @ n) * n)
    return c, n, up


def _hoop(pierce, n, d, R, t):
    """A torus through `pierce`, in the plane of n and d, hanging along d."""
    m = o3d.geometry.TriangleMesh.create_torus(torus_radius=R, tube_radius=t, radial_resolution=48, tubular_resolution=12)
    axis = _unit(np.cross(n, d))                                     # create_torus lies in xy, axis z
    Rm = np.stack([d, _unit(np.cross(axis, d)), axis], 1)            # x -> d, z -> axis
    V = np.asarray(m.vertices) @ Rm.T + pierce + R * d
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), m.triangles)


def _front(mesh, pierce, n):
    """Where a ray from outside the ear along -n first meets the surface near `pierce` (else pierce)."""
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    o = pierce + 15.0 * n
    t = sc.cast_rays(o3d.core.Tensor(np.r_[o, -n][None].astype(np.float32)))["t_hit"].numpy()[0]
    return o - t * n if np.isfinite(t) and t < 25.0 else pierce


def _stud(pierce, n, r):
    m = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=12)
    m.translate(pierce + n * 0.5 * r)                                 # sits on the front surface
    return m


def _turn(d, axis, deg):
    a = np.radians(deg)
    return d * np.cos(a) + np.cross(axis, d) * np.sin(a) + axis * (axis @ d) * (1 - np.cos(a))


def add_jewellery(mesh, P, head_centre, rng, kind=None):
    P = np.asarray(P, float)
    kind = kind or str(rng.choice(KINDS, p=WEIGHTS))
    c, n, up = _frame(P, head_centre)
    if kind in ("hoop", "stud"):
        lobe = P[15:20].mean(0)
        pierce = lobe + _unit(c - lobe) * rng.uniform(2.5, 4.5)      # through the lobe, in from its edge
        if kind == "hoop":
            R, t = rng.uniform(3.0, 10.0), rng.uniform(0.4, 1.25)    # 6-20 mm across, 0.8-2.5 mm thick
            d = _turn(-up, n, rng.uniform(-30, 30))                  # hangs down, tilted in the ear plane
            jew = _hoop(pierce, n, d, R, t); prm = dict(R=R, t=t, site="lobe")
        else:
            r = rng.uniform(0.8, 2.0); jew = _stud(_front(mesh, pierce, n), n, r); prm = dict(r=r, site="lobe")
    else:
        j = int(rng.integers(2, 11)); rim = P[j]
        o = rim - c; o = _unit(o - (o @ n) * n)                      # out of the rim, in the ear plane
        pierce = rim - o * rng.uniform(1.0, 2.0)
        if rng.random() < 0.6:
            R, t = rng.uniform(2.5, 4.5), rng.uniform(0.3, 0.8)
            jew = _hoop(pierce, n, o, R, t); prm = dict(R=R, t=t, site=f"helix{j}")
        else:
            r = rng.uniform(0.6, 1.4); jew = _stud(_front(mesh, pierce, n), n, r); prm = dict(r=r, site=f"helix{j}")
    out = o3d.geometry.TriangleMesh(mesh.vertices, mesh.triangles)
    out += o3d.geometry.TriangleMesh(jew.vertices, jew.triangles)
    out.compute_vertex_normals()
    return out, dict(kind=kind, **{k: (round(float(v), 2) if not isinstance(v, str) else v) for k, v in prm.items()})
