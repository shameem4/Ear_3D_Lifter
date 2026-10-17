"""GT check: 2D landmarks on cached ear views, back-projected, vs wrap landmarks.

Back-projection is backproject_snapped with config's snap (on, radius 0.08):
landmarks on a rim edge otherwise fall into the cavity behind it. Point 41
sits 2.0 mm off in-plane but 12.5 mm too deep unsnapped; snapped, 4.2 mm.

Usage: gt_check_ear_views.py OUT_PREFIX EAR_TAG [EAR_TAG...]   (tags: ears_v2c10, ears_v2c5, ears_v1)
"""
import sys, glob, pathlib, numpy as np
R0 = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_3D_Lifter")
sys.path.insert(0, str(R0)); sys.path.insert(0, str(R0 / "scripts"))
from PIL import Image, ImageDraw
from escape_audit import WRAP
from ear3d import Camera
from ear3d.backproject import backproject_snapped
from escape_audit import DEFAULTS

LM = np.load(WRAP / "templates/lm3d_v7_canonical_mm.npz", allow_pickle=True)
NAMES = [str(x) for x in LM["names"]]
def inv(name): return np.load(WRAP / "corpus/canon2scan" / f"{name}.npz", allow_pickle=True)["inverse"]

def gt_in_cache(hd, side, head):
    """Wrap landmarks of this ear in the cached normalised head frame (mm)."""
    nm = f"{hd}_{'left' if side == 'L' else 'right'}"
    P = LM["landmarks"][NAMES.index(nm)][:55]
    # ear's canonical -> scan mm -> the _left canonical the head cache was built in
    T = inv(f"{hd}_left") @ np.linalg.inv(inv(nm))
    P = (np.c_[P, np.ones(len(P))] @ T.T)[:, :3]
    return (P @ head["Rr"].T - head["centre"]) @ head["R_head"].T

out = sys.argv[1]
for tag in sys.argv[2:]:
    D, rows, shown = [], [], []
    for f in sorted(glob.glob(str(R0 / f"runs/cache/{tag}_s*/*_[LR].npz"))):
        z = np.load(f); head = np.load(str(z["head"]))
        stem = pathlib.Path(f).stem; hd, side = stem.rsplit("_", 1)
        img = z["img"]; H, W = img.shape[:2]
        cam = Camera(z["K"], z["E"], (H, W), int(z["size"]))
        R, c0, s = z["R"], z["centre"], float(z["scale"])
        gt = gt_in_cache(hd, side, head)
        lm = np.load(f.replace(".npz", "_lm.npy"))
        P, hit, _ = backproject_snapped(z["depth"], lm, cam, DEFAULTS)
        pw = P * s @ R + c0
        d = np.linalg.norm(pw - gt, axis=1); d[~hit] = np.nan      # NaN = escape
        D.append(d); rows.append(stem)
        if len(shown) < 6 and side == ("L" if len(shown) % 2 == 0 else "R"):
            uv = cam.project((gt - c0) @ R.T / s)
            a = Image.fromarray(img); dr = ImageDraw.Draw(a)
            for (x, y), (u, v) in zip(lm, uv):
                dr.line([(x, y), (u, v)], fill=(255, 255, 255))
                dr.ellipse([u - 3, v - 3, u + 3, v + 3], fill=(0, 255, 0))
                dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 0, 0))
            dr.text((4, 4), f"{stem}  green=GT red=pred  median {np.nanmedian(d):.1f} mm", fill=(255, 255, 0))
            shown.append(a.resize((384, 384)))
    D = np.array(D); flat = D.ravel(); esc = ~np.isfinite(flat); fv = np.where(esc, np.inf, flat)
    per_ear = np.nanmedian(D, axis=1); per_lm = np.nanmedian(D, axis=0)
    print(f"{tag}: ears {len(D)} | per-point median {np.nanmedian(flat):.2f} mm mean {np.nanmean(flat):.2f} | "
          + " ".join(f">{t:g}mm {np.mean(fv > t) * 100:.1f}%" for t in (2, 5, 10, 20))
          + f" | no-hit {esc.sum()} | worst ears " + ", ".join(f"{rows[i]} {per_ear[i]:.1f}" for i in np.argsort(per_ear)[::-1][:5]))
    print("   per-landmark median, worst 8:", ", ".join(f"{j}:{per_lm[j]:.1f}" for j in np.argsort(per_lm)[::-1][:8]))
    np.savez(f"{out}_{tag}.npz", D=D, ears=rows)
    S = Image.new("RGB", (384 * len(shown), 384))
    for k, a in enumerate(shown): S.paste(a, (384 * k, 0))
    S.save(f"{out}_{tag}_overlay.png")
