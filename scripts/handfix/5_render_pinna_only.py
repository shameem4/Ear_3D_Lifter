#!/usr/bin/env python3
"""Step 5: render every training ear as a PINNA-ONLY mesh, the way the ear-only path sees it.

Usage: 5_render_pinna_only.py OUT WORKER N_WORKERS [--skins 4] [--seed 0]
       5_render_pinna_only.py OUT --manifest      (after all workers: join the per-ear rows)

Why: on an ear-only mesh (AudioEar3D, any scan cut down to the pinna) the
render model r2j_s42 misses the helix root, landmarks 0-2, by ~5 mm against
~1.5 on a head. HUTUBS ears cut the same way fail the same way (5.0 mm); left
with a disc of cheek round them they do not (2.2 mm). The model never saw a
pinna without its head, and step 4 renders only heads (2026-10-25).

Meshes: the corpus wraps (canonical_lmw/<dataset>/<ear>.obj), the template
fitted to each scan -- exactly what an AudioEar3D mesh is, and pinna only.
Each goes through ear_only.frame (template-fit orientation, the normalised
head frame, side_frame("L"), the same box) and occupancy_scale, then the
25-view orbit at the pipeline's size and light. Right ears are stored
mirrored in the corpus and the ear-only path views every ear as a left one,
so all images are left-ear frames; `side` records the ear's real side.

Same format, labels and skins as step 4, so Ear_Landmarker can ingest it as
an add-on to synthetic_v3r (--carry-split data/synthetic_v3r --addon):
- syn_<dataset>_<subject>_<side>_<view>.png, the whole 600 px render;
  view = 400 + skin * 25 + orbit index (no clash with step 4's 0-99);
- <stem>_annotation.txt, manifest.csv (plus mesh="pinna_wrap"), visibility.npz.
No decimation: the wraps are already coarse (1188 vertices). The output holds
renders of scan geometry: local only, never committed or uploaded.
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
from ear3d import corpus, ear_only                                   # noqa: E402
from ear3d.backproject import visible                                # noqa: E402
from ear3d.camera import orbit_extrinsic                             # noqa: E402
from ear3d.corpus import WRAP                                        # noqa: E402
from ear3d.ear_view import occupancy_scale                           # noqa: E402
from ear3d.frames import in_frame                                    # noqa: E402
from ear3d.gt import load_reference                                  # noqa: E402
from ear3d.pipeline import Pipeline                                  # noqa: E402
from ear3d.render import render                                      # noqa: E402

HELD_OUT = {f"pp{n}" for n in (16, 19, 21, 22, 33, 40, 46, 47, 49, 57, 69, 70, 71, 72, 77, 78)}
REFERENCE = "lm3d_v13_canonical_mm.npz"
LICENCE = {"hutubs": "CC BY 4.0 (HUTUBS)",
           "sonicom": "CC BY-SA 3.0 (SONICOM); renders used under the user's permissive reading, 2026-10-21"}
VIS_MM = 2.0
VIEW0 = 400
FIELDS = ["stem", "dataset", "subject", "side", "yaw", "pitch", "occupancy", "licence", "skin_id", "light_id",
          "damage", "label_source", "visible", "trusted", "jewellery", "jewellery_params", "mesh"]


def labels(handfix):
    """ear -> (55x3 canonical mm, source); step 4's rule."""
    L, N = load_reference(REFERENCE)
    out = {n: (L["landmarks"][i][:55], "v13") for i, n in enumerate(N)}
    for d in pathlib.Path(handfix).iterdir():
        f = d / "adjusted.json"
        if f.is_file() and json.loads(f.read_text()).get("done"):
            out[d.name] = (np.array(json.loads(f.read_text())["points"]), "hand")
    return out


def turn(d, deg, rng):
    d = np.asarray(d, float) / np.linalg.norm(d); a = rng.normal(size=3); a -= (a @ d) * d; a /= np.linalg.norm(a)
    t = np.radians(deg)
    return tuple(d * np.cos(t) + a * np.sin(t))


def render_ear(pipe, nm, lab, out, n_skins, seed):
    c = pipe.cfg
    ds = "sonicom" if nm.startswith("P0") else "hutubs"
    subject, side = nm.rsplit("_", 1)[0], "L" if nm.endswith("left") else "R"
    assert subject not in HELD_OUT and "bj002" not in nm.lower(), nm
    m = o3d.io.read_triangle_mesh(str(WRAP / "corpus" / f"canonical_{corpus.CURRENT}" / ds / f"{nm}.obj"))
    f = ear_only.frame(pipe, m)
    if f["side"] != "L":                                              # corpus ears are canonical left ears
        print(f"{nm}: oriented as a right ear, skipped", flush=True); return [], []
    G, src = lab[nm]
    d0 = np.asarray(c["multiview_light_dir"], float)
    light0 = dict(c, light_dir=tuple(d0), light_mirror=False, light_frame="world")
    s = occupancy_scale(f["ear"], f["R"], f["centre"], f["scale"], c, light0, pipe.pred, c["target_occupancy"])
    Pf = (f["to_head"](G) - f["centre"]) @ f["R"].T / s
    rng = np.random.default_rng([seed, zlib.crc32(nm.encode())])
    rows, vis_all = [], []
    for k in range(n_skins):
        if k == 0:
            g, light = f["ear"], light0
        else:
            pipe.cfg = dict(c, skin_seed=int(rng.integers(1 << 30)), ao_strength=float(rng.uniform(0.5, 0.9)))
            try:
                g = pipe._skinned(f["mesh"], None, bool(rng.random() < 0.75))
            finally:
                pipe.cfg = c
            light = dict(light0, light_dir=turn(d0, rng.uniform(0, 30), rng))
        gf = in_frame(g, f["R"], f["centre"], s)
        for i, (yw, pt) in enumerate(pipe.angles):
            img, depth, cam = render(gf, light, E=orbit_extrinsic(float(yw), float(pt), c))
            uv = cam.project(Pf); vis = visible(depth, Pf, cam, tol=VIS_MM / s)
            stem = f"syn_{ds}_{subject}_{side}_{VIEW0 + k * len(pipe.angles) + i:03d}"
            Image.fromarray(img).save(out / f"{stem}.png")
            (out / f"{stem}_annotation.txt").write_text("".join(f"{x:.3f} {y:.3f}\n" for x, y in uv))
            rows.append(dict(stem=stem, dataset=ds, subject=subject, side=side, yaw=float(yw), pitch=float(pt),
                             occupancy=round(float(np.ptp(uv, axis=0).max() / img.shape[1]), 3),
                             licence=LICENCE[ds], skin_id=k, light_id=k, damage="none", label_source=src,
                             visible="".join("1" if b else "0" for b in vis),
                             trusted=("1" if src == "hand" else "0") * 55,
                             jewellery="none", jewellery_params="", mesh="pinna_wrap"))
            vis_all.append(vis)
    print(f"{nm}: {n_skins * len(pipe.angles)} views ({src})", flush=True)
    return rows, vis_all


def main():
    a = argparse.ArgumentParser()
    a.add_argument("out"); a.add_argument("worker", nargs="?", type=int, default=0); a.add_argument("n_workers", nargs="?", type=int, default=1)
    a.add_argument("--handfix", default="runs/handfix"); a.add_argument("--skins", type=int, default=4)
    a.add_argument("--seed", type=int, default=0); a.add_argument("--manifest", action="store_true")
    a.add_argument("--ears", nargs="*", help="limit to these ears (tests)")
    a = a.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.manifest:
        parts = sorted(out.glob("_rows_*.csv"))
        rows = [r for p in parts for r in csv.DictReader(open(p))]
        with open(out / "manifest.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, FIELDS); w.writeheader(); w.writerows(rows)
        vz = [np.load(p.with_suffix(".npz")) for p in parts]
        np.savez(out / "visibility.npz", stems=np.concatenate([z["stems"] for z in vz]),
                 visible=np.concatenate([z["visible"] for z in vz]))
        print(f"manifest: {len(rows)} images from {len(parts)} ears"); return
    lab = labels(a.handfix)
    ears = sorted(f.stem for f in (WRAP / corpus.get("canon2scan")).glob("*.npz")
                  if f.stem.rsplit("_", 1)[0] not in HELD_OUT and f.stem in lab and (not a.ears or f.stem in a.ears))
    pipe = Pipeline(cfg={"core": "python"})
    for nm in ears[a.worker::a.n_workers]:
        if (out / f"_rows_{nm}.csv").is_file():
            continue
        rows, vis = render_ear(pipe, nm, lab, out, a.skins, a.seed)
        np.savez(out / f"_rows_{nm}.npz", stems=np.array([r["stem"] for r in rows]), visible=np.array(vis, bool).reshape(-1, 55))
        with open(out / f"_rows_{nm}.csv", "w", newline="") as fh:     # last: marks the ear finished
            w = csv.DictWriter(fh, FIELDS); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
