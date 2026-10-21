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

CACHES RECORD THEIR GENERATION. A head cache (runs/cache/<tag>/<head>.npz) is
rendered in the canonical frame of its subject's left ear, which differs per
generation; scripts/caches/1_cache_head_frames.py stores `corpus` in it. Caches made
before that key existed (v4_s0/s1, ears_v3, the mvcache rays) are esc:
cache_version() returns LEGACY for them.
"""
import os
import pathlib

# wrap_work: corpora, templates and the fitted meshes, a sibling of this repo.
WRAP = pathlib.Path(__file__).resolve().parents[2] / "wrap_work"

VERSIONS = {
    "esc": dict(canonical="corpus/canonical_esc", canon2scan="corpus/canon2scan",
                gt_lm3d="lm3d_v9_canonical_mm.npz", view_table="view_table_v9.npz",
                mean="ear_mean_v4.obj"),
    "lmw": dict(canonical="corpus/canonical_lmw", canon2scan="corpus/canon2scan_lmw",
                gt_lm3d="lm3d_v11_canonical_mm.npz", view_table="view_table_v11.npz",
                mean="ear_mean_v5.obj"),
}
LEGACY = "esc"            # generation of any cache without a `corpus` key
CURRENT = os.environ.get("EAR3D_CORPUS", "lmw")
if CURRENT not in VERSIONS:
    raise ValueError(f"EAR3D_CORPUS={CURRENT!r}; known: {sorted(VERSIONS)}")


def cache_version(head):
    """Generation a loaded head-cache npz was built in."""
    return str(head["corpus"]) if "corpus" in getattr(head, "files", head) else LEGACY


def get(key, version=None):
    """A path or file name of the current generation (or of `version`)."""
    return VERSIONS[version or CURRENT][key]
