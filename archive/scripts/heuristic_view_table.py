#!/usr/bin/env python3
"""Predict the per-(landmark, view) table from anatomy, as a check on the learned one.

`scripts/view_table.py` LEARNS which views place which landmark well, from a
corpus. This script PREDICTS the same table from the mean ear's geometry and
nothing else: a landmark is well placed from a view that looks at its surface
face-on, and badly placed from a view where that surface turns edge-on and the
point becomes a silhouette. That is a statement about the scheme and the
viewing angle, which is exactly what the learned table claims to capture.

The point is the comparison, not the prediction. Two tables built from
unrelated information agreeing is evidence both are measuring the geometry; a
disagreement localises to one landmark or one view and says which to look at.
A learned table that matches nothing anatomical is a table that has learned
the corpus.

    python scripts/heuristic_view_table.py
    python scripts/heuristic_view_table.py --compare runs/viewtable/table.npz

WHAT IT PREDICTS, and the reasoning, so a disagreement can be argued with:

  cos(normal, view)   the surface normal at the landmark, from the mean
                      template mesh through the barycentric marks. Face-on is
                      1, edge-on is 0, back-facing is negative. A 2D annotator
                      places a point accurately where the surface faces them
                      and poorly along a silhouette, and a ray from an edge-on
                      view is nearly tangent to the surface it is meant to hit.

  depth rank          how medial the landmark sits. A medial point -- the
                      canal, the inferior crus -- is occluded by the helix rim
                      from oblique views in a way a lateral point never is,
                      and that is not captured by its own normal.

The two are reported separately rather than combined into one score, because a
single number would hide which mechanism is disagreeing with the corpus.

TWO SANITY RESULTS PRINTED FIRST, both cheap and both load-bearing for anyone
reading the table.

  LEFT/RIGHT is already one frame. Over the 294 subjects with both ears,
  centred, the stored left and right differ by 1.03 mm per point as they
  stand, against 8.45 mm if either is mirrored on the depth axis. So the
  chirality transform has been applied upstream, one table covers both ears,
  and the 1.03 mm is a subject's own bilateral asymmetry rather than a frame
  error. Do not build separate left and right tables.

  THE 55-POINT SCHEME ANNOTATES BORDERS, NOT BASINS. This was found by writing
  an anatomy-first depth prediction down and having it fail in three places.
  The concha group is NOT deep (-0.49 mm, mid-pack) because points 43-46 are
  the concha BORDER -- the scheme's own strip name for 35-49 is
  `concha_border` -- and a photograph has no floor to annotate. The two crura
  are 7 mm apart in depth rather than together, correctly: per point, the
  inferior crus runs +3.21 -> +5.68 -> +8.59 as it dives toward the canal
  while the superior crus stays at -2.6 -> +0.8 on the antihelix ridge. And
  the antihelix is marginally MORE lateral than the helix rim (-3.72 against
  -2.21), because the helix is a fold that curls inward, so the traced rim
  point sits medial to the antihelix crest. Each of those is the heuristic
  being wrong and the corpus being right, and together they say what the
  scheme can encode: visible-border depth, never basin depth. A depth prior
  built on group NAMES will be wrong for the concha.

ONE DISAGREEMENT WITH `view_table.py`'S OWN EXAMPLE, worth settling rather
than assuming. Its docstring says "the concha is unoccluded face-on while the
descending helix is a silhouette edge that only resolves from one side". On
surface facing this comes out the other way: face-on, the concha group scores
+0.427 with 35% of views edge-on, and the descending helix +0.727 with 10%.
Both can be true, because they are different failures -- the concha is a basin
whose walls are oblique even when nothing occludes it, while a rim point is
ambiguous ALONG the rim rather than hidden, which a normal-facing score does
not measure. What the facing score does flag as silhouette is a different part
of the same rim: the `helix` group at +0.268 and 47% edge-on, and
`ascending_helix` at +0.246 and 50%. So the prediction and the intuition
disagree about WHICH part of the helix a view table should distrust, and the
learned table can say which is right. That is the comparison this script
exists for.

NOT A VISIBILITY MODEL. Facing is a local surface property; it does not ray
trace, so a point that faces the camera but sits behind the helix rim scores
well here and badly in reality. The depth-rank column is the stand-in for
that, deliberately kept separate rather than folded in. If the learned table
disagrees with the prediction mainly on medial points -- canal, inferior
crus, tragus, the three with depth rank above 0.7 -- occlusion is the missing
term and a real visibility pass would be the fix.
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ear3d.config import cone_angles                                 # noqa: E402
from ear3d.scheme import GROUPS, GROUP_NAMES, group_of               # noqa: E402

TEMPLATES = pathlib.Path('../wrap_work/templates')
LM3D = 'lm3d_v7_canonical_mm.npz'
PCA = 'ear_pca_v7_all_mm.npz'


def load(root):
    root = pathlib.Path(root)
    lm = np.load(root / LM3D, allow_pickle=True)
    pca = np.load(root / PCA, allow_pickle=True)
    return lm, pca


def vertex_normals(verts, faces):
    """Area-weighted vertex normals, oriented OUTWARD.

    The template's winding gives inward normals, which silently turns every
    facing score negative and inverts the whole table. Rather than hard-code
    a flip, the global sign is chosen so that normals point away from the
    mesh centroid on average -- correct for either winding, and it fails
    loudly (a near-zero agreement) rather than quietly if the mesh is ever
    something other than an open ear surface.
    """
    n = np.zeros_like(verts)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    outward = verts - verts.mean(0)
    if float((n * outward).sum()) < 0:
        n = -n
    return n


def landmark_normals(lm, pca):
    """Normal at each landmark, interpolated through its barycentric mark."""
    verts, faces = pca['mean'], pca['faces']
    vn = vertex_normals(verts, faces)
    tri, bary = lm['triangle'], lm['barycentric']
    n = np.einsum('ijk,ij->ik', vn[faces[tri]], bary)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def view_dirs(angles, depth_axis, height_axis, ap_axis):
    """Unit vectors from the ear TOWARD the camera, for (yaw, pitch) degrees.

    Toward the camera rather than along the line of sight, so that a face-on
    surface gives facing = +1 and the table reads the way its column header
    says. The camera sits on the LATERAL side of the ear, which is the
    NEGATIVE depth direction in this corpus: the canal is at +6.2 mm from the
    centroid and the helix rim at -2.2, so +depth is medial. Getting this sign
    wrong turns every facing score negative and inverts the table, which is
    how it was first written.
    """
    out = np.zeros((len(angles), 3))
    for i, (yaw, pitch) in enumerate(np.radians(angles)):
        v = np.zeros(3)
        v[depth_axis] = -np.cos(yaw) * np.cos(pitch)
        v[ap_axis] = np.sin(yaw) * np.cos(pitch)
        v[height_axis] = np.sin(pitch)
        out[i] = v / np.linalg.norm(v)
    return out


def axes(points):
    """Identify depth / height / anterior-posterior by span, not by assumption."""
    span = (points.max(1) - points.min(1)).mean(0)
    depth, height = int(np.argmin(span)), int(np.argmax(span))
    return depth, height, ({0, 1, 2} - {depth, height}).pop(), span


def sanity(lm, P, depth, height, ap):
    """The two checks the docstring asserts, computed rather than quoted.

    A script that states numbers it does not compute is how a stale figure
    outlives the thing it described, which has already happened more than
    once across these repositories.
    """
    import re
    names = [str(x) for x in lm['names']]
    sub = {}
    for i, s in enumerate(names):
        m = re.match(r'(.+)_(left|right)$', s)
        if m:
            sub.setdefault(m.group(1), {})[m.group(2)] = i
    both = [k for k, v in sub.items() if len(v) == 2]
    if both:
        L = np.stack([P[sub[k]['left']] for k in both])
        R = np.stack([P[sub[k]['right']] for k in both])
        Lc, Rc = L - L.mean(1, keepdims=True), R - R.mean(1, keepdims=True)
        plain = np.sqrt(((Lc - Rc) ** 2).sum(-1)).mean()
        flipped = []
        for axis in range(3):
            M = Lc.copy()
            M[:, :, axis] *= -1
            flipped.append(np.sqrt(((M - Rc) ** 2).sum(-1)).mean())
        print(f'\nLEFT/RIGHT, {len(both)} subjects with both ears, centred')
        print(f'  as stored       {plain:6.2f} mm mean per-point difference')
        for axis, v in enumerate(flipped):
            print(f'  mirror axis {axis}   {v:6.2f} mm')
        verdict = ('already one chirality -- one table covers both ears'
                   if plain < min(flipped) else
                   'NOT one chirality; the sides need separate handling')
        print(f'  -> {verdict}')

    J = min(55, P.shape[1])
    c = P[:, :J, :] - P[:, :J, :].mean(1, keepdims=True)
    print(f'\nDEPTH by group, mm from each ear\'s centroid, + is medial')
    print(f'  {"group":<24} {"n":>3} {"depth":>7} {"height":>8} {"ant-post":>9}')
    print('  ' + '-' * 55)
    rows = []
    for (lo, hi), nm in zip(GROUPS, GROUP_NAMES):
        if lo >= J:
            continue
        g = c[:, lo:min(hi, J), :].mean(1)
        rows.append((nm, min(hi, J) - lo, g[:, depth].mean(),
                     g[:, height].mean(), g[:, ap].mean()))
    for nm, k, dp, ht, a in sorted(rows, key=lambda r: -r[2]):
        print(f'  {nm:<24} {k:3d} {dp:+7.2f} {ht:+8.2f} {a:+9.2f}')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--templates', default=str(TEMPLATES))
    p.add_argument('--views', type=int, default=24)
    p.add_argument('--cone', type=float, default=45.0)
    p.add_argument('--compare', help='a learned table .npz with table[j, v]')
    p.add_argument('--out', help='write the predicted table here')
    args = p.parse_args()

    lm, pca = load(args.templates)
    P = lm['landmarks']
    depth, height, ap, span = axes(P)
    print(f'{P.shape[0]} ears, {P.shape[1]} points; per-ear mean span '
          f'{span.round(1)} mm')
    print(f'  axis {depth} = depth, {height} = height, {ap} = anterior-posterior '
          '(by span, not assumed)')
    sanity(lm, P, depth, height, ap)

    n = landmark_normals(lm, pca)
    # THE TWO TABLES MUST BE ON THE SAME ANGLES, and matching SHAPES do not
    # guarantee that: this script's default is a cone of --views plus face-on,
    # while view_table.py learns a 5x5 rectangular grid. Both are 25. Compare
    # them cell-for-cell and you correlate one angle against a different one
    # while everything looks well formed. So when the learned table carries
    # its own angles, they win.
    angles = [(0.0, 0.0)] + list(cone_angles(args.views, args.cone))
    if args.compare:
        _z = np.load(args.compare, allow_pickle=True)
        if 'angles' in _z.files:
            angles = [tuple(map(float, r)) for r in np.asarray(_z['angles'])]
            print(f'\nusing the {len(angles)} view angles from {args.compare} '
                  'rather than this script\'s cone, so the two tables are '
                  'indexed alike')
    V = view_dirs(angles, depth, height, ap)
    facing = n @ V.T                      # (points, views), 1 = face-on

    mean = P.mean(0)
    mean = mean - mean.mean(0)
    dep = mean[:, depth]
    rank = dep.argsort().argsort() / max(len(dep) - 1, 1)   # 0 lateral, 1 medial

    J = min(55, P.shape[1])
    print(f'\nPREDICTED quality by group, over {len(angles)} views '
          '(55-point scheme)')
    print(f'  {"group":<24} {"n":>3} {"face-on":>8} {"best":>7} {"mean":>7} '
          f'{"edge-on":>8} {"depth rank":>11}')
    print('  ' + '-' * 74)
    rows = []
    for (lo, hi), nm in zip(GROUPS, GROUP_NAMES):
        if lo >= J:
            continue
        f = facing[lo:min(hi, J)]
        rows.append((nm, min(hi, J) - lo, f[:, 0].mean(), f.max(1).mean(),
                     f.mean(), float((f < 0.2).mean()),
                     rank[lo:min(hi, J)].mean()))
    for nm, k, fo, best, mn, edge, r in sorted(rows, key=lambda x: -x[2]):
        print(f'  {nm:<24} {k:3d} {fo:+8.3f} {best:+7.3f} {mn:+7.3f} '
              f'{100*edge:7.0f}% {r:11.2f}')
    print('\n  face-on is view 0. The GAP between face-on and best is what a '
          'view table\n  can exploit: a group whose best view is much better '
          'than face-on is one\n  worth choosing views for, and a group '
          'already at its best face-on is not.')
    print('\n  facing: 1 is face-on, 0 edge-on. "edge-on views" is the share of '
          'views\n  below 0.2, where a ray is near-tangent and a 2D annotator '
          'is guessing.\n  Low facing and high edge-on share predict the '
          'landmarks a learned table\n  should also rate badly; depth rank '
          'predicts which are occluded obliquely.')

    if args.out:
        np.savez(args.out, table=facing, views=np.array(angles),
                 depth_rank=rank, axes=np.array([depth, height, ap]))
        print(f'\nwrote {args.out}')

    if args.compare:
        learned = np.load(args.compare, allow_pickle=True)
        key = 'table' if 'table' in learned.files else learned.files[0]
        T = np.asarray(learned[key], float)
        # The template carries more marks than the 55-point scheme; the
        # learned table is the 55. Slice rather than refuse, but only from
        # the front, where the two schemes are the same points by index.
        if facing.shape[0] > T.shape[0] == 55 or facing.shape[0] > T.shape[1] == 55:
            facing = facing[:55]
        if T.shape == facing.shape[::-1]:
            T = T.T          # view_table stores (views, points)
        if T.shape != facing.shape:
            print(f'\n--compare: shape {T.shape} against predicted '
                  f'{facing.shape}; cannot align, so nothing is reported '
                  'rather than something misaligned')
            return
        # Learned tables are usually an ERROR (lower better); predicted is a
        # quality (higher better). Report both signs and say which is which
        # rather than guessing the convention.
        from scipy.stats import spearmanr
        flat_r = spearmanr(facing.ravel(), T.ravel()).statistic
        per_lm = np.array([spearmanr(facing[j], T[j]).statistic
                           for j in range(facing.shape[0])])
        print(f'\nCOMPARISON with {args.compare}')
        print(f'  Spearman over all cells        {flat_r:+.3f}  '
              '(negative is agreement if the learned table is an error)')
        print(f'  per-landmark, median           {np.nanmedian(per_lm):+.3f}')
        worst = np.argsort(np.abs(per_lm))[:8]
        print('  landmarks where the two disagree most -- look at these first:')
        for j in worst:
            print(f'    {j:3d} {group_of(j) if j < 55 else "beyond the 55":<24} '
                  f'rho {per_lm[j]:+.3f}')


if __name__ == '__main__':
    main()
