"""Export the demo's pick list to demo/meshes/ (gitignored, like the data it
comes from): the 16 held-out HUTUBS heads (raw scans, metres -> mm) and the
first 20 AudioEar3D left ears as WRAPPED (the mean-ear template fitted to
each scan, ../wrap_work/corpus/canonical_lmw/audioear, canonical mm). Caveat:
a wrap is smooth, hole-free and shares the pipeline's template topology, so
it is an easier input than a raw scan; the raw cloud surfaces
(runs/ae3d_rewrap) are where the pipeline is weak. The back of the helix rim
and the canal tube are cut from each wrap (BACK_EXCLUSION, chosen by hand):
a one-sided scan holds nothing there, and the template tore through the
front on 62 of 112 ears; cut, 0. Each mesh is one .bin:
u32 nV, u32 nF, f32 V (nV x 3, mm), u32 F (nF x 3); index.json lists them.

    python demo/export_meshes.py
"""
import json, pathlib, sys
import numpy as np, open3d as o3d
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "demo" / "meshes"
HUTUBS = ROOT.parent / "clean_3d_data" / "3D head meshes"
BACK_EXCLUSION = ROOT.parent / "wrap_work" / "templates" / "ear_L_template_back_exclusion.npz"
WRAPS = ROOT.parent / "wrap_work" / "corpus" / "canonical_lmw" / "audioear"
HELD_OUT = ["pp16", "pp19", "pp21", "pp22", "pp33", "pp40", "pp46", "pp47", "pp49", "pp57", "pp69", "pp70", "pp71", "pp72", "pp77", "pp78"]
OUT.mkdir(exist_ok=True)

def write(name, V, F):
    V = np.asarray(V, "<f4"); F = np.asarray(F, "<u4")
    with open(OUT / f"{name}.bin", "wb") as f:
        f.write(np.array([len(V), len(F)], "<u4").tobytes()); f.write(V.tobytes()); f.write(F.tobytes())
    return {"id": name, "file": f"{name}.bin", "vertices": len(V), "span_mm": round(float(np.ptp(V, 0).max()), 1)}

index = {"heads": [], "ears": [], "credits": {
    "heads": "HUTUBS (Brinkmann et al. 2019, JAES 67(9)), CC BY 4.0",
    "ears": "AudioEar3D (Huang et al. 2023, AAAI), CC BY 4.0, shown as our template wrapped onto each scan: smooth and hole-free, so easier than a raw scan"}}
for h in HELD_OUT:
    m = o3d.io.read_triangle_mesh(str(HUTUBS / f"{h}_3DheadMesh.ply"))
    index["heads"].append(write(h, np.asarray(m.vertices) * 1000.0, np.asarray(m.triangles)) | {"label": f"HUTUBS {h}"})
import trimesh
cut = np.load(BACK_EXCLUSION)["triangles"]
for i in range(1, 21):
    t = trimesh.load(WRAPS / f"{i:03d}_left.obj", process=False)       # trimesh: the triangle order the mask indexes
    F = np.delete(np.asarray(t.faces), cut, axis=0)
    used = np.unique(F); remap = np.full(len(t.vertices), -1); remap[used] = np.arange(len(used))
    index["ears"].append(write(f"ae{i:03d}_left", np.asarray(t.vertices)[used], remap[F]) | {"label": f"AudioEar3D {i:03d} left (wrap)"})
json.dump(index, open(OUT / "index.json", "w"), indent=1)
print(len(index["heads"]), "heads,", len(index["ears"]), "ears ->", OUT)
