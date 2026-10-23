"""Score held-out runs against the user's hand-corrected held-out ears (runs/handfix).

Usage: score_handfix.py RUN_DIR [RUN_DIR...]     (paired per ear against the first)

The only reference not derived from a 2D model: the user placed all 55 points
anatomically on the scan (scripts/handfix). Rank 2D checkpoints on this, not on
v12 (model-derived) or v7 (sits inside the rim).
"""
import json, pathlib, sys, numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ear3d import corpus
from ear3d.gt import inv, inv_gt
R0 = pathlib.Path(__file__).resolve().parents[2]
ears = []
for d in sorted((R0 / "runs/handfix").iterdir()):
    if (d / "points.json").is_file() and json.loads((d / "points.json").read_text())["held_out"]:
        a = d / "adjusted.json"
        if a.is_file() and json.loads(a.read_text()).get("done"): ears.append(d)
print(f"{len(ears)} hand-corrected held-out ears (done)")
def errs(run):
    E = []
    for d in ears:
        nm = d.name; hd, sd = nm.rsplit("_", 1); z = np.load(R0 / run / f"{hd}.npz")
        A = np.array(json.loads((d / "adjusted.json").read_text())["points"])
        T = inv(f"{hd}_left", corpus.CURRENT) @ np.linalg.inv(inv_gt(nm))
        G = (np.c_[A, np.ones(55)] @ T.T)[:, :3] @ z["Rr"].T
        E.append(np.linalg.norm(z[f"pred_{'L' if sd == 'left' else 'R'}"] - G, axis=1))
    return np.array(E)
REG = {"all": slice(0, 55), "0-8": slice(0, 9), "lobe": slice(15, 20), "26-28": slice(26, 29)}
B = errs(sys.argv[1])
for run in sys.argv[1:]:
    E = errs(run); m = np.nanmedian(E, 1)
    line = f"{run.split('e2e_heldout_')[-1]:28s} median {np.median(m):.3f} mean {m.mean():.3f}  0-8 {np.nanmedian(E[:, :9]):.2f}  lobe {np.nanmedian(E[:, 15:20]):.2f}  >5mm {np.mean(~(E <= 5)):.2%}"
    if run != sys.argv[1]:
        out = []
        for k, sl in REG.items():
            d = np.nanmedian(E[:, sl], 1) - np.nanmedian(B[:, sl], 1)
            out.append(f"{k} {np.median(d):+.2f} ({(d < 0).sum()}/{len(d)}, p={wilcoxon(d).pvalue:.3f})")
        g = (~(E <= 5)).mean(1) - (~(B <= 5)).mean(1)
        out.append(f">5mm {g.mean()*100:+.2f}pp p={wilcoxon(g).pvalue if np.any(g) else 1:.3f}")
        line += "\n    vs first, paired: " + " | ".join(out)
    print(line)
