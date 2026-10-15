#!/usr/bin/env python3
"""

PRE-2026-10-14 RESULTS FROM THIS SCRIPT ARE VOID. It renders through
escape_audit.load_ear, which until 856652b applied canon2scan's transform
to the head without reversing the triangle winding, so every right-ear
render was shaded with its normals pointing inward. The script is correct
from that commit on; anything measured with it before then mixed correct
left ears with inside-out right ones. Re-run before quoting.
Estimate the IMAGE-PLANE light direction from the shading on the helix rim.

WHY THIS IS POSSIBLE AT ALL. At an occluding contour the surface normal is
perpendicular to the line of sight, so it lies IN the image plane and points
along the contour's outward normal. That removes the usual shape-from-shading
ambiguity: around the rim of the pinna the surface normal sweeps through a full
range of image-plane directions whose values are KNOWN from the annotation
alone, and the brightness just inside the rim samples the lighting at each of
them. A Lambertian read of that is a straight least squares,

    I_j  ~=  a * (L . N_j)  +  b

over the outer-helix points, where N_j is the outward contour normal and L the
image-plane light direction. The fit returns a*L, so the DIRECTION is recovered
and the intensity scale is not -- which is all that is wanted here. b absorbs
ambient and the mean albedo.

WHAT IT CANNOT DO. It recovers the light's direction within the image plane
only. A lamp straight behind the camera and one straight in front both give
L_xy ~= 0 and are indistinguishable; the fit's R^2 is reported so those cases
can be dropped rather than read as "lit from dead centre". Albedo variation,
hair and earrings all enter as noise in I_j, which is why the fit is trimmed.

WHY IT IS CALIBRATED ON RENDERS. A photograph has no ground-truth light
direction, so the estimator would be unfalsifiable on photographs alone. This
project has a renderer whose sun is set by hand, so --calibrate sweeps a known
sun over real ear geometry and asks whether the estimate tracks it. Only after
that does --photos mean anything.

    python scripts/light_direction.py --calibrate --ears pp10_left pp12_right
    python scripts/light_direction.py --photos
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ear3d.scheme import STRIPS                                    # noqa: E402

RIM = slice(0, 20)              # the outer helix: the only closed-ish, mostly
                                # convex, mostly unoccluded contour in the scheme


def rim_normals(uv):
    """Outward unit normals along the outer-helix polyline."""
    s = uv[RIM]
    d = np.empty_like(s)
    d[1:-1] = s[2:] - s[:-2]
    d[0], d[-1] = s[1] - s[0], s[-1] - s[-2]
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    N = np.stack([-d[:, 1], d[:, 0]], axis=1)
    out = s - uv.mean(0)
    N[(N * out).sum(1) < 0] *= -1.0
    return N


def _sample(gray, pts):
    """Bilinear sample, NaN outside the image."""
    H, W = gray.shape
    x, y = pts[:, 0], pts[:, 1]
    ok = (x >= 0) & (x <= W - 1.001) & (y >= 0) & (y <= H - 1.001)
    v = np.full(len(pts), np.nan)
    if not ok.any():
        return v
    x0 = np.floor(x[ok]).astype(int); y0 = np.floor(y[ok]).astype(int)
    fx = (x[ok] - x0)[:, None]; fy = (y[ok] - y0)[:, None]
    g = gray[:, :, None] if gray.ndim == 2 else gray
    a = g[y0, x0] * (1 - fx) * (1 - fy) + g[y0, x0 + 1] * fx * (1 - fy) \
        + g[y0 + 1, x0] * (1 - fx) * fy + g[y0 + 1, x0 + 1] * fx * fy
    v[ok] = a[:, 0]
    return v


def estimate(gray, uv, inset=0.035):
    """(Lx, Ly, r2, n) -- the image-plane light direction from the rim.

    `inset` is how far inside the contour the brightness is read, as a fraction
    of the ear's diagonal. It has to clear the rim's own silhouette blur and
    stay on the helix rather than reaching the concha: a few percent does both.
    y is image-down, so a positive Ly means the light travels DOWNWARD in the
    picture, i.e. the scene is lit from above.
    """
    uv = np.asarray(uv, float)
    N = rim_normals(uv)
    diag = float(np.hypot(*(uv.max(0) - uv.min(0))))
    I = _sample(gray, uv[RIM] - inset * diag * N)
    ok = np.isfinite(I)
    if ok.sum() < 8:
        return np.nan, np.nan, np.nan, int(ok.sum())
    # One trimming pass: drop the worst sixth of the residuals, which is where
    # hair, earrings and specular highlights land.
    idx = np.where(ok)[0]
    for _ in range(2):
        A = np.c_[N[idx], np.ones(len(idx))]
        coef, *_ = np.linalg.lstsq(A, I[idx], rcond=None)
        res = np.abs(I[idx] - A @ coef)
        if len(idx) <= 10:
            break
        idx = idx[np.argsort(res)[:max(10, int(0.84 * len(idx)))]]
    A = np.c_[N[idx], np.ones(len(idx))]
    coef, *_ = np.linalg.lstsq(A, I[idx], rcond=None)
    pred = A @ coef
    ss = float(((I[idx] - I[idx].mean()) ** 2).sum())
    r2 = 1.0 - float(((I[idx] - pred) ** 2).sum()) / ss if ss > 0 else np.nan
    L = coef[:2]
    n = float(np.linalg.norm(L))
    return (float(L[0] / n), float(L[1] / n), r2, len(idx)) if n > 1e-9 \
        else (np.nan, np.nan, r2, len(idx))


def shoelace(uv):
    """Signed area of the ordered outer-rim polyline: the side label."""
    s = np.asarray(uv, float)[RIM]
    x, y = s[:, 0], s[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# --------------------------------------------------------------- calibration
def calibrate(a):
    import open3d as o3d                                            # noqa
    from ear3d import DEFAULTS
    from ear3d.camera import orbit_extrinsic
    from ear3d.frames import in_frame
    from ear3d.render import render
    from escape_audit import load_ear, ear_frame

    cfg = dict(DEFAULTS); cfg["interactive"] = False
    suns = [float(x) for x in a.sun_sweep]
    if a.azimuth:
        # A FAIRER TEST THAN SWEEPING x ALONE. The stock sun is (x, -0.4, -0.9):
        # mostly along the view axis, so its IN-PLANE direction barely moves as
        # x is swept and the estimate is dominated by the fixed y component --
        # the regime where a normalised Lx is a small difference of large
        # numbers. Sweeping the azimuth of a GRAZING light puts the whole
        # signal in the plane the estimator actually measures, which is the
        # only regime in which a failure says something about the estimator
        # rather than about the test.
        th = np.deg2rad(np.arange(0.0, 360.0, 30.0))
        suns = [(float(np.cos(t)), float(np.sin(t)), -0.30) for t in th]
    rows = []
    for name in a.ears:
        head, ear, lm = load_ear(name, "hutubs", cfg)
        R, C, s = ear_frame(head, lm, cfg)
        g = in_frame(head, R, C, s)
        lf = ((lm - C) @ R.T) / s
        for yaw, pitch in [(0.0, 0.0), (-20.0, 0.0), (20.0, 0.0), (0.0, -20.0)]:
            E = orbit_extrinsic(yaw, pitch, cfg)
            for sx in suns:
                d = sx if isinstance(sx, tuple) else (
                    sx, cfg["light_dir"][1], cfg["light_dir"][2])
                c = dict(cfg, light_mirror=False, light_dir=d)
                img, _, cam = render(g, c, E=E)
                gray = img.astype(float).mean(2) / 255.0
                uv = cam.project(lf)
                lx, ly, r2, n = estimate(gray, uv, inset=a.inset)
                rows.append(dict(ear=name, yaw=yaw, pitch=pitch, sun_x=d[0],
                                 sun_y=d[1], sun_z=d[2],
                                 Lx=lx, Ly=ly, r2=r2, n=n,
                                 shoelace=shoelace(uv)))
                print(f"{name:13s} y{yaw:+5.0f} p{pitch:+5.0f} sun {d[0]:+5.2f},{d[1]:+5.2f}"
                      f"   Lx {lx:+.3f}  Ly {ly:+.3f}  r2 {r2:.3f}")
    v = [r for r in rows if np.isfinite(r["Lx"])]
    sx = np.array([r["sun_x"] for r in v]); lx = np.array([r["Lx"] for r in v])
    print()
    print("CALIBRATION: does the estimate track the known sun?")
    print(f"  n = {len(v)}   Pearson r(sun_x, Lx) = {np.corrcoef(sx, lx)[0,1]:+.3f}")
    agree = np.mean(np.sign(lx[sx != 0]) == np.sign(sx[sx != 0]))
    print(f"  sign agreement on non-zero suns: {100*agree:.1f}%")
    for u in sorted(set(sx)):
        m = sx == u
        print(f"    sun_x {u:+5.2f}:  Lx mean {lx[m].mean():+.3f}  "
              f"sd {lx[m].std():.3f}  n {m.sum()}")
    _write(a.out, rows)


# -------------------------------------------------------------------- photos
def photos(a):
    from PIL import Image
    LK = pathlib.Path(a.landmarker)
    lm = np.load(LK / "data/preprocessed/landmarks.npy")
    rows = []
    with open(LK / f"data/preprocessed/{a.split}.csv") as f:
        meta = list(csv.DictReader(f))
    for k, m in enumerate(meta):
        ip = LK / "data/preprocessed" / m["image_file"]
        if not ip.exists():
            continue
        im = Image.open(ip).convert("L")
        gray = np.asarray(im, float) / 255.0
        W, H = im.size
        uv = lm[int(m["idx"])] * [W, H]
        lx, ly, r2, n = estimate(gray, uv, inset=a.inset)
        rows.append(dict(idx=m["idx"], source=m["source"], w=W, h=H,
                         Lx=lx, Ly=ly, r2=r2, n=n, shoelace=shoelace(uv)))
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(meta)}")
    v = [r for r in rows if np.isfinite(r["Lx"]) and r["r2"] >= a.min_r2]
    print(f"\n{len(rows)} images, {len(v)} with a usable rim fit (r2 >= {a.min_r2})")
    lx = np.array([r["Lx"] for r in v])
    print(f"  Lx: mean {lx.mean():+.3f}  sd {lx.std():.3f}  "
          f"median {np.median(lx):+.3f}")
    nl, nr = int((lx < 0).sum()), int((lx > 0).sum())
    print(f"  lit from image-left  (Lx > 0): {nr} ({100*nr/len(v):.1f}%)")
    print(f"  lit from image-right (Lx < 0): {nl} ({100*nl/len(v):.1f}%)")
    # a binomial test on the left/right split
    from math import erf, sqrt
    z = (nr - len(v) / 2) / (sqrt(len(v)) / 2)
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    print(f"  binomial z = {z:+.2f}, p = {p:.2e}")
    print("\nby source:")
    for s in sorted({r["source"] for r in v}):
        q = np.array([r["Lx"] for r in v if r["source"] == s])
        print(f"  {s:12s} n {len(q):4d}  Lx mean {q.mean():+.3f}  "
              f"lit-from-left {100*(q>0).mean():5.1f}%")
    print("\nby recovered side label (shoelace sign):")
    for lab, sel in (("A (shoelace > 0)", lambda r: r["shoelace"] > 0),
                     ("B (shoelace < 0)", lambda r: r["shoelace"] < 0)):
        q = np.array([r["Lx"] for r in v if sel(r)])
        if len(q):
            print(f"  {lab:18s} n {len(q):4d}  Lx mean {q.mean():+.3f}  "
                  f"lit-from-left {100*(q>0).mean():5.1f}%")
    _write(a.out, rows)


def _write(out, rows):
    if not out:
        return
    op = pathlib.Path(out); op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {op}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--photos", action="store_true")
    p.add_argument("--ears", nargs="*", default=["pp10_left", "pp12_right"])
    p.add_argument("--sun-sweep", nargs="*", default=[-0.6, -0.3, 0.0, 0.3, 0.6])
    p.add_argument("--azimuth", action="store_true",
                   help="sweep a grazing light through 360 deg of azimuth "
                        "instead of sweeping only the stock sun's x")
    p.add_argument("--inset", type=float, default=0.035)
    p.add_argument("--min-r2", type=float, default=0.3)
    p.add_argument("--split", default="test")
    p.add_argument("--landmarker", default=str(ROOT.parent / "Ear_Landmarker"))
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.calibrate:
        calibrate(a)
    if a.photos:
        photos(a)
    if not (a.calibrate or a.photos):
        p.error("pick --calibrate or --photos")


if __name__ == "__main__":
    main()
