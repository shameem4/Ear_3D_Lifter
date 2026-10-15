"""Does a single-arm mirror rule survive when presentation is NOT fixed?

Everything either project has measured about presentation comes from a corpus
that cannot answer this. Ear_3D_Lifter's is single-presentation by
construction -- ear_frame puts every ear in left form -- so "always mirror"
is really "always move to the favoured presentation", which is free advice
only because the corpus never varies. Ear_Landmarker's fusion number was
measured on pre-cropped images, where a frame mirror and a crop mirror are
the same operation. Neither licenses a default for a mixed corpus.

THIS NEEDS NO NEW RENDERS. A --tta pair run already contains, for every
(ear, view), a complete pipeline pass on the frame AND a complete pass on its
mirror -- detector, ROI loop and landmarker, not just the landmarker. So a
mixed-presentation corpus is obtained by relabelling: choose half the views
to be "presented mirrored", and for those the `flip` arm IS the as-presented
arm and `none` is the mirror of what was presented. The mirror is an exact
pixel permutation and the lift never sees it, so the distances are identical
either way round; nothing is approximated here.

    as-presented   what you get with no TTA at all, on a corpus whose
                   handedness you do not control
    always-mirror  the single-arm rule, which on a mixed corpus mirrors half
                   the ears INTO the unfavourable presentation
    fusion         the coordinate average, which is symmetric and so is the
                   same arm whichever way a view was presented
    oracle         the better of the two arms per landmark-view. Not
                   achievable -- it needs the answer -- and reported as the
                   ceiling any selection rule is working against.

The assignment is resampled, so the spread is the spread over which views
happened to be mirrored, not over ears.

Usage:
    python scripts/mixed_handedness.py runs/winding_fix/frame_wound.csv
"""

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="a --tta pair run, with none/flip/avg arms")
    p.add_argument("--gate", type=float, default=10.0)
    p.add_argument("--draws", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    from scipy import stats
    from handedness_vs_illumination import load, rate

    D = load(a.csv)
    keys = sorted({k[:4] for k in D if k[4] == "none"})
    keys = [k for k in keys if (*k, "flip") in D and (*k, "avg") in D]
    if not keys:
        sys.exit("no paired landmark-views")
    n = np.array([D[(*k, "none")] for k in keys])
    f = np.array([D[(*k, "flip")] for k in keys])
    g = np.array([D[(*k, "avg")] for k in keys])
    view = np.array([hash(k[:3]) for k in keys])          # ear + yaw + pitch
    ear = np.array([k[0] for k in keys])
    ears = sorted(set(ear))
    print(f"{len(keys)} landmark-views, {len(ears)} ears, "
          f"{len(set(view))} ear-views, gate {a.gate:.0f} mm\n")

    # The ceiling, and the single-presentation baseline for reference.
    best = np.where(np.nan_to_num(n, nan=np.inf) <= np.nan_to_num(f, nan=np.inf), n, f)
    print("single-presentation corpus, as this project actually renders it:")
    for nm, v in (("no TTA", n), ("always-mirror", f), ("fusion", g),
                  ("oracle (unreachable)", best)):
        print(f"  {nm:22s} {100 * rate(v, a.gate):6.3f}%")

    rng = np.random.default_rng(a.seed)
    uv = np.unique(view)
    out = {k: [] for k in ("as-presented", "always-mirror", "fusion", "oracle")}
    for _ in range(a.draws):
        mirrored = set(rng.choice(uv, size=len(uv) // 2, replace=False).tolist())
        m = np.array([v in mirrored for v in view])
        pres = np.where(m, f, n)          # what the pipeline is handed
        other = np.where(m, n, f)         # the mirror of what it is handed
        out["as-presented"].append(100 * rate(pres, a.gate))
        out["always-mirror"].append(100 * rate(other, a.gate))
        out["fusion"].append(100 * rate(g, a.gate))
        out["oracle"].append(100 * rate(best, a.gate))

    print(f"\nMIXED-PRESENTATION corpus, {a.draws} draws of which half is mirrored:")
    print(f"  {'rule':22s} {'mean':>8s} {'sd':>7s} {'2.5%':>8s} {'97.5%':>8s}")
    for k in ("as-presented", "always-mirror", "fusion", "oracle"):
        v = np.array(out[k])
        print(f"  {k:22s} {v.mean():7.3f}% {v.std():6.3f} {np.percentile(v,2.5):7.3f}% "
              f"{np.percentile(v,97.5):7.3f}%")

    ap, am, fu = (np.array(out[k]) for k in
                  ("as-presented", "always-mirror", "fusion"))
    print(f"\n  always-mirror vs as-presented  {am.mean()-ap.mean():+.3f}pp   "
          f"beats it on {100*np.mean(am < ap):.0f}% of draws")
    print(f"  fusion vs as-presented         {fu.mean()-ap.mean():+.3f}pp")
    print(f"  fusion vs always-mirror        {fu.mean()-am.mean():+.3f}pp")

    # Paired over EARS at one assignment, which is the unit that generalises.
    mirrored = set(rng.choice(uv, size=len(uv) // 2, replace=False).tolist())
    m = np.array([v in mirrored for v in view])
    pres, other = np.where(m, f, n), np.where(m, n, f)
    per = lambda v: np.array([100 * rate(v[ear == e], a.gate) for e in ears])
    P, O, G = per(pres), per(other), per(g)
    print(f"\n  paired over {len(ears)} ears at one assignment:")
    for nm, x, y in (("always-mirror vs as-presented", P, O),
                     ("fusion vs as-presented", P, G),
                     ("fusion vs always-mirror", O, G)):
        print(f"    {nm:32s} {y.mean()-x.mean():+.3f}pp  second better on "
              f"{int((y < x).sum()):3d}/{len(ears)}  "
              f"p={stats.wilcoxon(x, y).pvalue:.2e}")


if __name__ == "__main__":
    main()
