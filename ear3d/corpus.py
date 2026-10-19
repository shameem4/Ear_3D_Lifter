"""Which corpus generation the pipeline and the GT scoring use.

One generation is one consistent set: canonical fits, the transform from each
canonical fit to its scan, the reference landmarks evaluated ON those fits,
the view table learned against those landmarks, and the mean template. Mixing
generations silently moves the ground truth -- the template marks address the
fit, and fits of different generations slide 2-5 mm along the helix
(memory: marks-are-fit-coupled).

  esc  escalated_v1 heads + ae3d_v2, canonical_esc, lm3d_v9      (to 2026-10-18)
  lmw  landmark-posed rewrap (lmwrap_v1) + ae3d_v3, canonical_lmw, lm3d_v11

EAR3D_CORPUS selects one; the default is lmw.

CACHE is pinned, not selected. runs/cache/* head caches (and the rays in
runs/mvcache) were rendered in the frame of the OLD canon2scan of each
subject's left ear, and a cache file does not record which transform made it.
Anything that maps into or out of a cache frame must use CACHE's transforms,
whatever the current generation is.
"""
import os

VERSIONS = {
    "esc": dict(canonical="corpus/canonical_esc", canon2scan="corpus/canon2scan",
                gt_lm3d="lm3d_v9_canonical_mm.npz", view_table="view_table_v9.npz",
                mean="ear_mean_v4.obj"),
    "lmw": dict(canonical="corpus/canonical_lmw", canon2scan="corpus/canon2scan_lmw",
                gt_lm3d="lm3d_v11_canonical_mm.npz", view_table="view_table_v11.npz",
                mean="ear_mean_v5.obj"),
}
CACHE = "esc"
CURRENT = os.environ.get("EAR3D_CORPUS", "lmw")
if CURRENT not in VERSIONS:
    raise ValueError(f"EAR3D_CORPUS={CURRENT!r}; known: {sorted(VERSIONS)}")


def get(key, version=None):
    """A path or file name of the current generation (or of `version`)."""
    return VERSIONS[version or CURRENT][key]
