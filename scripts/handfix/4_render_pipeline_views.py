#!/usr/bin/env python3
"""Step 4: render every training ear exactly as the pipeline sees it, for a render-only landmarker.

Usage: 4_render_pipeline_views.py OUT WORKER N_WORKERS [--skins 4] [--seed 0]
       4_render_pipeline_views.py OUT --manifest      (after all workers: join the per-scan rows)

Each corpus head scan goes through the pipeline's own path (turned by step 01's
seeded rotation, face frame, ear_view, occupancy_scale at target_occupancy),
then the 25-view orbit is rendered at the pipeline's size (600 px) with the
pipeline's light. So an image here is the frame LandmarkPredictor gets at
inference. Format agreed with Ear_Landmarker (2026-10-22):
- syn_<dataset>_<subject>_<side>_<view>.png, the WHOLE render, 600 px, view = skin * 25 + orbit index;
- <stem>_annotation.txt: 55 lines "x y" in render pixels; occluded and
  out-of-frame points keep their projected position;
- manifest.csv: stem, dataset, subject, side, yaw, pitch (anatomical: right
  ears' orbit yaw negated, as the pipeline's view table), occupancy, licence,
  skin_id, light_id, damage, label_source, visible (55 0/1), trusted (55 0/1);
- visibility.npz: stems + (N, 55) bool, the same depth test (2 mm slack).

Skin 0 is the pipeline's exact appearance (ear_tone, clean skin, its sun).
Skins 1.. draw a Fitzpatrick tone, blotching, AO strength and a sun turned up
to 30 deg from the pipeline's, and half of them damage the geometry
(decimation, vertex noise or holes) for scans from the wild; light_id = skin_id.

Labels: the user's hand-corrected points where the ear is done in runs/handfix
(label_source hand, trusted 1), else lm3d_v13 (label_source v13, trusted 0).
Held-out heads and BJ002 are never rendered (asserted). The output holds scan
renders: local only, never committed or uploaded.
"""
import argparse
import collections
import csv
import json
import pathlib
import sys
import zlib

import numpy as np
import open3d as o3d
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ear3d import corpus                                             # noqa: E402
from ear3d.backproject import visible                                # noqa: E402
from ear3d.camera import orbit_extrinsic                             # noqa: E402
from ear3d.corpus import WRAP                                        # noqa: E402
from ear3d.ear_view import ear_view, occupancy_scale                 # noqa: E402
from ear3d.frames import in_frame                                    # noqa: E402
from ear3d.gt import load_reference                                  # noqa: E402
from ear3d.pipeline import Pipeline                                  # noqa: E402
from ear3d.render import render                                      # noqa: E402

HELD_OUT = {f"pp{n}" for n in (16, 19, 21, 22, 33, 40, 46, 47, 49, 57, 69, 70, 71, 72, 77, 78)}
REFERENCE = "lm3d_v13_canonical_mm.npz"
LICENCE = {"hutubs": "CC BY 4.0 (HUTUBS)",
           "sonicom": "CC BY-SA 3.0 (SONICOM); renders used under the user's permissive reading, 2026-10-21"}
VIS_MM = 2.0                        # depth-test slack: face-on, 1 mm hides ~5 rim points that are in plain view
FIELDS = ["stem", "dataset", "subject", "side", "yaw", "pitch", "occupancy", "licence", "skin_id", "light_id",
          "damage", "label_source", "visible", "trusted"]


def rotation(seed):                                                  # step 01's, so the frames match it
    q = np.random.default_rng(seed).normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def labels(handfix):
    """ear -> (55x3 canonical mm, source)."""
    L, N = load_reference(REFERENCE)
    out = {n: (L["landmarks"][i][:55], "v13") for i, n in enumerate(N)}
    for d in pathlib.Path(handfix).iterdir():
        f = d / "adjusted.json"
        if f.is_file() and json.loads(f.read_text()).get("done"):
            out[d.name] = (np.array(json.loads(f.read_text())["points"]), "hand")
    return out


def damage(mesh, near, rng):
    """A copy with scan-like defects; returns (mesh, name). Vertex noise and holes stay near the ear."""
    kind = rng.choice(["decimate", "noise", "holes"])
    V, F = np.asarray(mesh.vertices).copy(), np.asarray(mesh.triangles).copy()
    if kind == "decimate":
        m = mesh.simplify_quadric_decimation(int(len(F) * rng.uniform(0.3, 0.7)))
        return m, kind
    if kind == "noise":
        V += rng.normal(scale=rng.uniform(0.1, 0.3), size=V.shape)
    else:
        for _ in range(int(rng.integers(1, 4))):
            c = near[rng.integers(len(near))] + rng.normal(scale=5.0, size=3)
            gone = np.linalg.norm(V - c, axis=1) < rng.uniform(2.0, 6.0)
            F = F[~gone[F].any(1)]
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)), kind


def turn(d, deg, rng):
    d = np.asarray(d, float) / np.linalg.norm(d); a = rng.normal(size=3); a -= (a @ d) * d; a /= np.linalg.norm(a)
    t = np.radians(deg)
    return tuple(d * np.cos(t) + a * np.sin(t))


def render_scan(pipe, scan, ears, lab, out, n_skins, seed):
    c = pipe.cfg
    m = o3d.io.read_triangle_mesh(scan); V = np.asarray(m.vertices)
    V = V * (1000.0 if np.ptp(V, 0).max() < 10 else 1.0)            # HUTUBS is metres
    Rr = rotation(zlib.crc32(pathlib.Path(scan).name.encode()))
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V @ Rr.T), m.triangles)
    nh = pipe._find_face(pipe._skinned(mesh, c["tone"], c["skin_noise"]))
    if nh is None:
        print(f"{scan}: no face frame, skipped", flush=True); return [], []
    mesh_c, R_head, centre, _ = nh
    ear0 = pipe._skinned(mesh_c, c["ear_tone"], c["ear_skin_noise"])
    rows, vis_all = [], []
    for nm in ears:
        side = "L" if nm.endswith("left") else "R"
        subject = nm.rsplit("_", 1)[0]; ds = "sonicom" if nm.startswith("P0") else "hutubs"
        assert subject not in HELD_OUT and "bj002" not in nm.lower() and "secret" not in scan.lower(), nm
        v = ear_view(ear0, side, c, pipe.yolo)
        if v is None:
            print(f"{nm}: no ear view, skipped", flush=True); continue
        C2S = np.linalg.inv(np.load(WRAP / corpus.get("canon2scan") / f"{nm}.npz", allow_pickle=True)["inverse"])
        G, src = lab[nm]
        Xs = (np.c_[G, np.ones(55)] @ C2S.T)[:, :3]                  # canonical -> scan mm
        Pn = (Xs @ Rr.T - centre) @ R_head.T                          # -> the pipeline's normalised head
        d0 = np.asarray(c["multiview_light_dir"], float) * ([-1.0, 1, 1] if side == "R" else 1.0)
        light0 = dict(c, light_dir=tuple(d0), light_mirror=False, light_frame="world")
        s = occupancy_scale(ear0, v["R"], v["centre"], v["scale"], c, light0, pipe.pred, c["target_occupancy"])
        Pf = (Pn - v["centre"]) @ v["R"].T / s
        rng = np.random.default_rng([seed, zlib.crc32(nm.encode())])
        for k in range(n_skins):
            dmg = "none"
            if k == 0:
                g, light = ear0, light0
            else:
                base = mesh_c
                if rng.random() < 0.5:
                    base, dmg = damage(mesh_c, Pn, rng)
                pipe.cfg = dict(c, skin_seed=int(rng.integers(1 << 30)), ao_strength=float(rng.uniform(0.5, 0.9)))
                try:
                    g = pipe._skinned(base, None, bool(rng.random() < 0.75))
                finally:
                    pipe.cfg = c
                light = dict(light0, light_dir=turn(d0, rng.uniform(0, 30), rng))
            gf = in_frame(g, v["R"], v["centre"], s)
            for i, (yw, pt) in enumerate(pipe.angles):
                img, depth, cam = render(gf, light, E=orbit_extrinsic(float(yw), float(pt), c))
                uv = cam.project(Pf); vis = visible(depth, Pf, cam, tol=VIS_MM / s)
                stem = f"syn_{ds}_{subject}_{side}_{k * len(pipe.angles) + i:03d}"
                Image.fromarray(img).save(out / f"{stem}.png")
                (out / f"{stem}_annotation.txt").write_text("".join(f"{x:.3f} {y:.3f}\n" for x, y in uv))
                rows.append(dict(stem=stem, dataset=ds, subject=subject, side=side,
                                 yaw=float(yw) * (-1 if side == "R" else 1), pitch=float(pt),
                                 occupancy=round(float(np.ptp(uv, axis=0).max() / img.shape[1]), 3),
                                 licence=LICENCE[ds], skin_id=k, light_id=k, damage=dmg, label_source=src,
                                 visible="".join("1" if b else "0" for b in vis),
                                 trusted=("1" if src == "hand" else "0") * 55))
                vis_all.append(vis)
        print(f"{nm}: {n_skins * len(pipe.angles)} views ({src})", flush=True)
    return rows, vis_all


def main():
    a = argparse.ArgumentParser()
    a.add_argument("out"); a.add_argument("worker", nargs="?", type=int, default=0); a.add_argument("n_workers", nargs="?", type=int, default=1)
    a.add_argument("--handfix", default="runs/handfix"); a.add_argument("--skins", type=int, default=4)
    a.add_argument("--seed", type=int, default=0); a.add_argument("--manifest", action="store_true")
    a = a.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.manifest:
        parts = sorted(out.glob("_rows_*.csv"))
        rows = [r for f in parts for r in csv.DictReader(open(f))]
        with open(out / "manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)
        vz = [np.load(f.with_suffix(".npz")) for f in parts]
        np.savez(out / "visibility.npz", stems=np.concatenate([z["stems"] for z in vz]),
                 visible=np.concatenate([z["visible"] for z in vz]))
        print(f"manifest: {len(rows)} images from {len(parts)} scans"); return

    lab = labels(a.handfix)
    by_scan = collections.defaultdict(list)
    for f in sorted((WRAP / corpus.get("canon2scan")).glob("*.npz")):
        if f.stem.rsplit("_", 1)[0] not in HELD_OUT and f.stem in lab:
            by_scan[str(np.load(f, allow_pickle=True)["scan"])].append(f.stem)
    pipe = Pipeline()
    for scan in sorted(by_scan)[a.worker::a.n_workers]:
        key = f"{pathlib.Path(scan).stem}_{zlib.crc32(scan.encode()):08x}"
        if (out / f"_rows_{key}.csv").is_file():
            continue
        rows, vis = render_scan(pipe, scan, by_scan[scan], lab, out, a.skins, a.seed)
        np.savez(out / f"_rows_{key}.npz", stems=np.array([r["stem"] for r in rows]),
                 visible=np.array(vis, bool).reshape(-1, 55))
        with open(out / f"_rows_{key}.csv", "w", newline="") as f:      # last: marks the scan finished
            w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
