#!/usr/bin/env python3
"""Step 3: render the hand-corrected ears as 2D training data for Ear_Landmarker.

Usage: 3_render_training.py OUT WORKER N_WORKERS [--dir runs/handfix] [--views 40] [--seed 0]
       3_render_training.py OUT --manifest      (after all workers: join the per-ear rows)

Format agreed with Ear_Landmarker (scripts/handfix/README.md):
- per image syn_<dataset>_<subject>_<side>_<view>.png, 384 px square, cropped
  to landmark extent / 0.70 centred on the landmarks' box, the head behind;
- <stem>_annotation.txt: 55 lines "x y", pixels in that PNG; occluded points
  keep their projected position, points outside the crop just fall outside;
- manifest.csv: stem, dataset, subject, side, yaw, pitch, roll, occupancy,
  licence, visible (55 0/1, depth test), trusted (55 0/1).

Only ears marked done in the editor are rendered; their 55 points are all
trusted (done means every point was checked). Held-out heads are never
rendered. Right ears are rendered as right ears: the corrected points go back
through the corpus transform onto the raw scan, undoing the canonical z-mirror.

Views per ear: the pipeline's 25-view orbit, then random poses (yaw +-45,
pitch -45..30, roll +-15). PITCH SIGN: orbit_extrinsic puts the camera at
frame y = sin(pitch), with +y toward the top of the helix, so NEGATIVE pitch
looks from BELOW (checked 2026-10-22: pitch -30 puts the camera at y -1.5).
About 40% of views come from below (pitch <= -15). Skin tone
(the Fitzpatrick I-VI training pool), skin texture, AO strength, sun direction
and background vary per view.
"""
import argparse
import csv
import json
import pathlib
import sys
import zlib

import numpy as np
import open3d as o3d
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ear3d.backproject import visible                                # noqa: E402
from ear3d.camera import orbit_extrinsic                             # noqa: E402
from ear3d.config import DEFAULTS                                    # noqa: E402
from ear3d.corpus import WRAP                                        # noqa: E402
from ear3d.data import HUTUBS_TO_MM                                  # noqa: E402
from ear3d.frames import frame_from, in_frame, plane_normal          # noqa: E402
from ear3d.gt import GT_C2S                                          # noqa: E402
from ear3d.render import render                                      # noqa: E402
from ear3d.skin import apply_skin                                    # noqa: E402

OUT_PX, OCC_CROP = 384, 0.70        # crop side = landmark extent / OCC_CROP
RENDER_PX, OCC_RENDER = 900, 0.45   # render loose so the crop never leaves the frame
CROP_MM = 110.0                     # scan kept around the ear (speeds AO; the frame is ~90 mm)
SKINS = 4                           # skin variants per ear, cycled over its views
LICENCE = {"hutubs": "CC BY 4.0 (HUTUBS)",
           "sonicom": "CC BY-SA 3.0 (SONICOM); renders used under the user's permissive reading, 2026-10-21"}
FIELDS = ["stem", "dataset", "subject", "side", "yaw", "pitch", "roll", "occupancy", "licence", "visible", "trusted"]


def views(n, rng):
    v = [(float(y), float(p), 0.0) for y in range(-30, 31, 15) for p in range(-30, 31, 15)]
    while len(v) < n:
        p = rng.uniform(-45, 30)                       # NEGATIVE pitch = camera below the ear
        v.append((float(rng.uniform(-45, 45)), float(p), float(rng.uniform(-15, 15))))
    return v[:n]


def ear_scan(nm):
    """Raw scan (mm, native handedness) around the ear, and canonical -> scan."""
    z = np.load(WRAP / GT_C2S / f"{nm}.npz", allow_pickle=True)
    m = o3d.io.read_triangle_mesh(str(z["scan"]))
    V = np.asarray(m.vertices); V = V * (HUTUBS_TO_MM if np.ptp(V, axis=0).max() < 10.0 else 1.0)
    return V, np.asarray(m.triangles), np.linalg.inv(z["inverse"])


def crop(V, F, c, r):
    keep = np.linalg.norm(V - c, axis=1) < r
    F = F[keep[F].all(1)]; used = np.unique(F); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
    return V[used], remap[F]


def roll_E(E, deg):
    a = np.radians(deg); Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    E = E.copy(); E[:3, :3] = Rz @ E[:3, :3]; E[:3, 3] = Rz @ E[:3, 3]
    return E


def render_ear(d, out, n_views, seed):
    pj = json.loads((d / "points.json").read_text()); adj = json.loads((d / "adjusted.json").read_text())
    nm, ds = pj["ear"], pj["dataset"]; subject, side = nm.rsplit("_", 1); side = side[0].upper()
    V, F, C2S = ear_scan(nm)
    P = (np.c_[np.array(adj["points"]), np.ones(55)] @ C2S.T)[:, :3]          # corrected, scan mm
    head_c = V.mean(0)
    c = P.mean(0); n = plane_normal(P)
    if n @ (c - head_c) < 0:
        n = -n                                                                # out of the head
    up = P[0:9].mean(0) - P[15:20].mean(0)
    R = frame_from(n, up)
    Vc, Fc = crop(V, F, c, CROP_MM)
    ext = np.ptp((P - c) @ R.T, axis=0)[:2].max()
    half = DEFAULTS["eye_z"] * np.tan(np.radians(DEFAULTS["vfov"]) / 2)
    s = ext / (OCC_RENDER * 2 * half)
    rng = np.random.default_rng([seed, zlib.crc32(nm.encode())])
    skins = []
    for k in range(SKINS):
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vc), o3d.utility.Vector3iVector(Fc)); m.compute_vertex_normals()
        apply_skin(m, seed=int(rng.integers(1 << 30)), tone=None, ao_strength=float(rng.uniform(0.5, 0.9)),
                   n_rays=DEFAULTS["ao_rays"], noise=bool(rng.random() < 0.75))
        skins.append(in_frame(m, R, c, s))
    Pf = (P - c) @ R.T / s
    rows = []
    for i, (yw, pt, rl) in enumerate(views(n_views, rng)):
        E = roll_E(orbit_extrinsic(yw, pt, DEFAULTS), rl)
        sun = rng.normal(size=3); sun[2] = -abs(sun[2]) - 0.6                 # travels into the scene, from the camera side
        cfg = dict(DEFAULTS, light_dir=tuple(sun / np.linalg.norm(sun)), light_mirror=False, light_frame="view")
        bg = tuple(float(x) for x in rng.uniform(0.15, 0.85) * rng.uniform(0.85, 1.15, 3).clip(0, 1.17))
        img, depth, cam = render(skins[i % SKINS], cfg, size=RENDER_PX, background=tuple(min(x, 1.0) for x in bg), E=E)
        uv = cam.project(Pf); vis = visible(depth, Pf, cam, tol=1.0 / s)
        lo, hi = uv.min(0), uv.max(0); side_px = (hi - lo).max() / OCC_CROP; cc = (lo + hi) / 2
        x0, y0 = cc - side_px / 2
        if x0 < 0 or y0 < 0 or x0 + side_px > img.shape[1] or y0 + side_px > img.shape[0]:
            print(f"{nm} view {i}: crop leaves the render, skipped", flush=True); continue
        im = Image.fromarray(img).crop((int(round(x0)), int(round(y0)), int(round(x0 + side_px)), int(round(y0 + side_px))))
        k = OUT_PX / im.width; im = im.resize((OUT_PX, OUT_PX), Image.LANCZOS)
        q = (uv - [round(x0), round(y0)]) * k
        stem = f"syn_{ds}_{subject}_{side}_{i:03d}"
        im.save(out / f"{stem}.png")
        (out / f"{stem}_annotation.txt").write_text("".join(f"{x:.3f} {y:.3f}\n" for x, y in q))
        rows.append(dict(stem=stem, dataset=ds, subject=subject, side=side, yaw=round(yw, 2), pitch=round(pt, 2),
                         roll=round(rl, 2), occupancy=round(float((q.max(0) - q.min(0)).max() / OUT_PX), 3),
                         licence=LICENCE[ds], visible="".join("1" if v else "0" for v in vis), trusted="1" * 55))
    with open(out / f"_rows_{nm}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)
    print(f"{nm}: {len(rows)} views", flush=True)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("out"); a.add_argument("worker", nargs="?", type=int, default=0); a.add_argument("n_workers", nargs="?", type=int, default=1)
    a.add_argument("--dir", default="runs/handfix"); a.add_argument("--views", type=int, default=40)
    a.add_argument("--seed", type=int, default=0); a.add_argument("--manifest", action="store_true")
    a = a.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.manifest:
        rows = [r for f in sorted(out.glob("_rows_*.csv")) for r in csv.DictReader(open(f))]
        with open(out / "manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)
        print(f"manifest: {len(rows)} images from {len(list(out.glob('_rows_*.csv')))} ears"); return
    ears = []
    for d in sorted(pathlib.Path(a.dir).iterdir()):
        if not (d / "adjusted.json").is_file():
            continue
        pj = json.loads((d / "points.json").read_text())
        if pj["held_out"] or not json.loads((d / "adjusted.json").read_text()).get("done"):
            continue
        ears.append(d)
    for d in ears[a.worker::a.n_workers]:
        if (out / f"_rows_{d.name}.csv").is_file():
            continue
        render_ear(d, out, a.views, a.seed)


if __name__ == "__main__":
    main()
