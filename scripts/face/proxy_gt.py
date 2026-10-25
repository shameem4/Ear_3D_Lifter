"""Pseudo GT per head: the canonical face Wrap-fitted with the user's named control points.

python scripts/face/proxy_gt.py [heads...]  -> runs/face/v0/<head>_gtd.npz  (G: 468 x 3, same frame as P_norm)

Points are matched BY NAME between face_points/projects/canonical.wrap (on the canonical file as Wrap loads
it: file order, cm) and head_NN_<head>.wrap. Needs runs/face/v0/<head>.npz from lift.py for the mesh.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import sys, json, pathlib, numpy as np
from ear3d.face_gt import canonical, canonical_raw, wrap_template, wrap_points, xyz, bary, N_LM
from ear3d.face import show

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "face" / "v0"
WP = ROOT.parent / "wrap_work" / "face_points"
frames = json.load(open(WP / "frames.json")); only = set(sys.argv[1:])

V0, F0 = canonical_raw()
Vc, Fc = canonical()
T = wrap_points(WP / "projects" / "canonical.wrap")
for f in frames["frames"]:
    h = f["name"]
    if (only and h not in only) or (not only and (RUN / f"{h}_gtd.npz").exists()):
        continue
    z = np.load(RUN / f"{h}.npz"); V, F, P = z["V"], z["F"], z["P_norm"]
    H = wrap_points(WP / "projects" / f"head_{f['frame']:02d}_{h}.wrap")
    names = sorted(set(T) & set(H))
    tp = np.array([xyz(V0, F0, T[n]) for n in names]); hp = np.array([xyz(V, F, H[n]) for n in names])
    ttid, tbb = bary(Vc, Fc, tp)                   # the points re-expressed on the subdivided mesh
    ctrl = ([[int(t), float(b[0]), float(b[1])] for t, b in zip(ttid, tbb)], [[H[n]["triangleInd"], H[n]["u"], H[n]["v"]] for n in names])
    Vfit, info = wrap_template(Vc, Fc, tp, V, F, hp, RUN / "wrapd" / h, name=h, ctrl=ctrl)
    G = Vfit[:N_LM]
    held = np.linalg.norm((Vfit[Fc[ttid]] * tbb[:, :, None]).sum(1) - hp, axis=1)
    d = np.linalg.norm(G - P[:N_LM], axis=1)
    np.savez(RUN / f"{h}_gtd.npz", G=G, fit=Vfit, fit_faces=Fc, names=names, held_mm=held, pose_used=np.array(names)[info["pose_used"]])
    print(h, f"{len(names)} pairs | control held: median {np.median(held):.1f}, max {held.max():.1f} mm ({names[held.argmax()]})"
          f" | lift vs GT: median {np.median(d):.2f}, p90 {np.percentile(d, 90):.2f}", flush=True)
    show(V, F, [(P[:N_LM], (0, 0, 255)), (G, (0, 255, 0))], RUN / f"{h}_gtd.jpg", yaws=(-60, 0, 60), centre=G[1] - np.array([0, 0, 60]))
