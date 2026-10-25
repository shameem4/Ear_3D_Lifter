"""Lift MediaPipe's 478 face landmarks onto head meshes.

python scripts/face/lift.py mesh.ply|obj [out.npz] [--metres]   one mesh -> <mesh>.face478.npz
python scripts/face/lift.py --hutubs pp10 pp12 ...              HUTUBS heads, with per-view diagnostics
                                                                -> runs/face/v0/<head>.npz (+ _views.jpg)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import sys, pathlib, numpy as np, open3d as o3d, cv2
from ear3d.face import FaceLifter
from ear3d.config import EAR3D_DIR

ROOT = pathlib.Path(__file__).resolve().parents[2]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
L = FaceLifter()


def report(name, out):
    ok = np.isfinite(out["P"]).all(1)
    print(name, f"{ok.sum()}/478 lm, {out['n_views']} views, res {np.nanmedian(out['residual_mm']):.2f} mm, "
          f"off {np.nanmedian(out['off_surface_mm']):.2f} mm", flush=True)


if "--hutubs" in sys.argv:
    out_dir = ROOT / "runs" / "face" / "v0"; out_dir.mkdir(parents=True, exist_ok=True)
    for h in args:
        m = o3d.io.read_triangle_mesh(str(EAR3D_DIR / "3D head meshes" / f"{h}_3DheadMesh.ply"))
        m.vertices = o3d.utility.Vector3dVector(np.asarray(m.vertices) * 1000.0)
        r = L.run(m, keep_images=True)
        if r is None:
            print(h, "no face"); continue
        out, dbg = r
        np.savez(out_dir / f"{h}.npz", **out, **{k: dbg[k] for k in ("UV", "inliers", "views", "E", "P_norm", "P_raw_norm", "piv", "V", "F")})
        cv2.imwrite(str(out_dir / f"{h}_views.jpg"), np.vstack([np.hstack([cv2.resize(i[:, :, ::-1], (200, 200))
                    for i in dbg["imgs"][j:j + 11]]) for j in range(0, len(dbg["imgs"]), 11)]))
        report(h, out)
else:
    m = o3d.io.read_triangle_mesh(args[0])
    if "--metres" in sys.argv:
        m.vertices = o3d.utility.Vector3dVector(np.asarray(m.vertices) * 1000.0)
    r = L.run(m)
    if r is None:
        sys.exit("no face found")
    dst = args[1] if len(args) > 1 else pathlib.Path(args[0]).with_suffix(".face478.npz")
    np.savez(dst, **r[0]); report(dst, r[0])
