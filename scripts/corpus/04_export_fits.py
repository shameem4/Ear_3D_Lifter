#!/usr/bin/env python3
"""Step 4: write the fits from steps 2 and 3 as a corpus: one .obj per ear.

Usage: 04_export_fits.py --heads DIR --audioear DIR --out-heads corpus/lmwrap_vN --out-audioear corpus/ae3d_vN

Head ears (step 2) go to OUT_HEADS/{hutubs,sonicom}/<ear>.obj with
provenance.json (source npz, scan, lobe anchors). RIGHT EARS ARE STORED
Z-MIRRORED INTO LEFT FORM, as every corpus is: scan = mirror_z(stored).
Exporting them unmirrored once folded the mean template 37 times.
AudioEar fits (step 3) go to OUT_AUDIOEAR/<ear>.obj as fitted.

Paths are relative to wrap_work unless absolute. Refuses to overwrite.
Reconstructed from the two inline export passes that built lmwrap_v1 and
ae3d_v3 (2026-10-18).
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ear3d.corpus import WRAP                                        # noqa: E402
from ear3d.faceform import TEMPLATE                                  # noqa: E402

F = np.asarray(trimesh.load(TEMPLATE, process=False).faces)


def save(path, V, header):
    with open(path, "w") as f:
        f.write(f"# {header}\n")
        for v in V: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in F: f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")


def main():
    a = argparse.ArgumentParser()
    for k in ("--heads", "--audioear", "--out-heads", "--out-audioear"): a.add_argument(k, required=True)
    a = a.parse_args(); res = lambda s: pathlib.Path(s) if pathlib.Path(s).is_absolute() else WRAP / s
    oh, oa = res(a.out_heads), res(a.out_audioear)

    prov = {}
    for ds in ("hutubs", "sonicom"): (oh / ds).mkdir(parents=True, exist_ok=False)
    for f in sorted(pathlib.Path(a.heads).glob("*.npz")):
        n = f.stem; z = np.load(f); right = n.endswith("right")
        V = z["V"] * (np.array([1.0, 1.0, -1.0]) if right else 1.0)
        ds = "hutubs" if n.startswith("pp") else "sonicom"
        save(oh / ds / f"{n}.obj", V, f"{oh.name}: landmark-posed Faceform wrap, millimetres; right ears z-mirrored into left form")
        prov[n] = dict(src=str(f), scan=str(z["scan"]), lobe_anchors=bool(len(z["anchors"])) if "anchors" in z else False,
                       stored="z-mirrored into left form; scan = mirror_z(stored)" if right else "scan frame")
    json.dump(prov, open(oh / "provenance.json", "w"), indent=1)

    oa.mkdir(parents=True, exist_ok=False); kept = 0
    for f in sorted(pathlib.Path(a.audioear).glob("*.npz")):
        z = np.load(f, allow_pickle=True); save(oa / f"{f.stem}.obj", z["V"][:1188], f"{oa.name}: AudioEar3D cloud rewrap")
        kept += bool(z["kept_old"])
    print(f"heads {len(prov)} -> {oh} | audioear {len(list(oa.glob('*.obj')))} -> {oa} (kept old fit: {kept})")


if __name__ == "__main__":
    main()
