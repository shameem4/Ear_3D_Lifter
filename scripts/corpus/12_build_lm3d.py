#!/usr/bin/env python3
"""Reference 3D landmarks for a canonical corpus: a fixed landmark DEFINITION
evaluated on every canonical fit.

The definition is (triangle, barycentric) per landmark on the shared template
connectivity, taken unchanged from an existing lm3d file (--definition), so a
rebuild changes the ears and never the meaning of a landmark. Names and
ear_size come from the corpus's own manifest (ear_size = its `ear_mm`, the
largest extent of the canonical fit -- the same quantity lm3d_v7..v9 carry).

Replaces wrap_work/templates/v9_provenance/build_lm3d.py, which hard-coded the
OLD corpus/canonical (588 ears) while canonical_esc, canon2scan and the PCA
used another generation.

    python scripts/corpus/12_build_lm3d.py --canonical corpus/canonical_lmw \\
        --definition templates/lm3d_v9_canonical_mm.npz \\
        --out templates/lm3d_v10_canonical_mm.npz
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ear3d.wrap import load_mesh                                   # noqa: E402

WRAP = ROOT.parent / "wrap_work"   # == ear3d.corpus.WRAP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--canonical", required=True, help="dir with manifest.csv and <dataset>/<name>.obj")
    p.add_argument("--definition", required=True, help="lm3d npz whose triangle/barycentric are reused")
    p.add_argument("--marks", default=None,
                   help="npz with triangle/barycentric for points 0-54 (e.g. a re-derived "
                        "mark_mean_v10_55.npz); 55+ keep the definition's")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    res = lambda s: pathlib.Path(s) if pathlib.Path(s).is_absolute() else WRAP / s
    can, out = res(a.canonical), res(a.definition)
    d = dict(np.load(out, allow_pickle=True))
    tri, bary = d["triangle"].copy(), d["barycentric"].copy()
    if a.marks:
        mk = np.load(res(a.marks))
        tri[:55], bary[:55] = mk["triangle"], mk["barycentric"]
    rows = sorted(csv.DictReader(open(can / "manifest.csv")), key=lambda r: (r["dataset"], r["name"]))
    names, L, size = [], [], []
    for r in rows:
        V, F = load_mesh(str(can / r["dataset"] / f"{r['name']}.obj"))
        names.append(r["name"])
        L.append(np.einsum("ij,ijk->ik", bary, np.asarray(V)[np.asarray(F)[tri]]))
        size.append(float(r["ear_mm"]))
    L, size = np.array(L), np.array(size)
    np.savez(res(a.out), landmarks=L, landmarks_scalefree=L / size[:, None, None], ear_size=size,
             names=np.array(names), triangle=tri, barycentric=bary, strips=d["strips"],
             point_names=d["point_names"], corpus=str(can), definition=str(out),
             marks_0_54=str(a.marks or out))
    print(f"{len(names)} ears x {L.shape[1]} landmarks from {can.name}, definition {out.name} -> {res(a.out)}")


if __name__ == "__main__":
    main()
