//! Ear-only pose: fit the mean ear template to a scan with no head (port of
//! ear3d/ear_only.orient). Every principal-axis sign choice x both
//! handednesses, refined by scaled point-to-point ICP; least residual wins.
//!
//! Two deliberate changes from the Python, both because numpy and open3d
//! internals cannot be reproduced: the 6000-point scan subsample is an even
//! stride instead of a random draw, and ICP is written here (open3d's
//! convergence rule and correspondence cap are kept: max_corr 5, 30 iterations,
//! stop when fitness and inlier RMSE both change by < 1e-6).

use crate::kdtree::KdTree;
use crate::linalg::{apply, det, jacobi_eig, matmul, transpose, M3};

type V3 = [f64; 3];
const MAX_CORR: f64 = 5.0;
const MAX_SUB: usize = 6000;

fn mean(p: &[V3]) -> V3 {
    let n = p.len() as f64;
    let mut m = [0.0; 3];
    for q in p {
        for k in 0..3 {
            m[k] += q[k] / n;
        }
    }
    m
}

/// Principal axes as rows, largest first (numpy svd's Vt of the centred cloud).
fn axes(p: &[V3], c: &V3) -> M3 {
    let mut cov = [[0.0; 3]; 3];
    for q in p {
        let d = [q[0] - c[0], q[1] - c[1], q[2] - c[2]];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    jacobi_eig::<3>(cov).1
}

fn extent(p: &[V3], c: &V3, ax: &V3) -> f64 {
    let pr = p.iter().map(|q| (q[0] - c[0]) * ax[0] + (q[1] - c[1]) * ax[1] + (q[2] - c[2]) * ax[2]);
    let (lo, hi) = pr.fold((f64::MAX, f64::MIN), |(l, h), x| (l.min(x), h.max(x)));
    hi - lo
}

/// Similarity (s, R, t) with dst ~ s R src + t, least squares (Horn's
/// quaternion rotation; s = sum dst'.(R src') / sum |src'|^2 = Umeyama's).
pub fn similarity(src: &[V3], dst: &[V3]) -> (f64, M3, V3) {
    let (cs, cd) = (mean(src), mean(dst));
    let mut s = [[0.0; 3]; 3];
    let mut ss = 0.0;
    for (a, b) in src.iter().zip(dst) {
        let (a, b) = ([a[0] - cs[0], a[1] - cs[1], a[2] - cs[2]], [b[0] - cd[0], b[1] - cd[1], b[2] - cd[2]]);
        ss += a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += a[i] * b[j];
            }
        }
    }
    let n = [
        [s[0][0] + s[1][1] + s[2][2], s[1][2] - s[2][1], s[2][0] - s[0][2], s[0][1] - s[1][0]],
        [s[1][2] - s[2][1], s[0][0] - s[1][1] - s[2][2], s[0][1] + s[1][0], s[2][0] + s[0][2]],
        [s[2][0] - s[0][2], s[0][1] + s[1][0], -s[0][0] + s[1][1] - s[2][2], s[1][2] + s[2][1]],
        [s[0][1] - s[1][0], s[2][0] + s[0][2], s[1][2] + s[2][1], -s[0][0] - s[1][1] + s[2][2]],
    ];
    let q = jacobi_eig::<4>(n).1[0];
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let r = [
        [w * w + x * x - y * y - z * z, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), w * w - x * x + y * y - z * z, 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), w * w - x * x - y * y + z * z],
    ];
    let mut num = 0.0;
    for (a, b) in src.iter().zip(dst) {
        let ra = apply(&r, &[a[0] - cs[0], a[1] - cs[1], a[2] - cs[2]]);
        num += ra[0] * (b[0] - cd[0]) + ra[1] * (b[1] - cd[1]) + ra[2] * (b[2] - cd[2]);
    }
    let sc = num / ss;
    let rc = apply(&r, &cs);
    (sc, r, [cd[0] - sc * rc[0], cd[1] - sc * rc[1], cd[2] - sc * rc[2]])
}

/// Scaled ICP from src onto the tree's points. Returns the moved points.
fn icp(mut src: Vec<V3>, tree: &KdTree, tgt: &[V3]) -> Vec<V3> {
    let corr = |src: &[V3]| {
        let mut pairs = Vec::new();
        let mut se = 0.0;
        for (i, p) in src.iter().enumerate() {
            let (j, d2) = tree.nearest(p);
            if d2 < MAX_CORR * MAX_CORR {
                pairs.push((i, j));
                se += d2;
            }
        }
        let fit = pairs.len() as f64 / src.len() as f64;
        let rmse = if pairs.is_empty() { 0.0 } else { (se / pairs.len() as f64).sqrt() };
        (pairs, fit, rmse)
    };
    let (mut pairs, mut fit, mut rmse) = corr(&src);
    for _ in 0..30 {
        if pairs.len() < 3 {
            break;
        }
        let a: Vec<V3> = pairs.iter().map(|&(i, _)| src[i]).collect();
        let b: Vec<V3> = pairs.iter().map(|&(_, j)| tgt[j]).collect();
        let (s, r, t) = similarity(&a, &b);
        for p in src.iter_mut() {
            let q = apply(&r, p);
            *p = [s * q[0] + t[0], s * q[1] + t[1], s * q[2] + t[2]];
        }
        let (p2, f2, r2) = corr(&src);
        let done = (f2 - fit).abs() < 1e-6 && (r2 - rmse).abs() < 1e-6;
        (pairs, fit, rmse) = (p2, f2, r2);
        if done {
            break;
        }
    }
    src
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

pub struct Pose {
    pub r: M3,
    pub t: V3,
    pub s: f64,
    pub mirror: bool,
    pub residual: f64,
}

/// X_canonical ~ s * R W + t, W = V mirrored about z when `mirror` (a right ear).
pub fn orient(v: &[V3], tpl: &[V3]) -> Pose {
    let ct = mean(tpl);
    let et = axes(tpl, &ct);
    let ttree = KdTree::new(tpl.to_vec());
    let mut best: Option<Pose> = None;
    for mirror in [false, true] {
        let w: Vec<V3> = v.iter().map(|p| if mirror { [p[0], p[1], -p[2]] } else { *p }).collect();
        let cm = mean(&w);
        let em = axes(&w, &cm);
        let s0 = extent(tpl, &ct, &et[0]) / extent(&w, &cm, &em[0]);
        let step = (w.len() + MAX_SUB - 1) / MAX_SUB;
        let sub: Vec<V3> = w.iter().step_by(step.max(1)).cloned().collect();
        for sg in 0..8 {
            let sgn = [0, 1, 2].map(|k| if sg >> k & 1 == 1 { -1.0 } else { 1.0 });
            let d = [[sgn[0], 0.0, 0.0], [0.0, sgn[1], 0.0], [0.0, 0.0, sgn[2]]];
            let r = matmul(&matmul(&transpose(&et), &d), &em);
            if det(&r) < 0.0 {
                continue;
            }
            let x: Vec<V3> = sub.iter().map(|p| {
                let q = apply(&r, &[p[0] - cm[0], p[1] - cm[1], p[2] - cm[2]]);
                [s0 * q[0] + ct[0], s0 * q[1] + ct[1], s0 * q[2] + ct[2]]
            }).collect();
            let xf = icp(x.clone(), &ttree, tpl);
            let xtree = KdTree::new(xf.clone());
            let fwd = median(xf.iter().map(|p| ttree.nearest(p).1.sqrt()).collect());
            let back = median(tpl.iter().map(|p| xtree.nearest(p).1.sqrt()).collect());
            let res = fwd + back;
            if best.as_ref().map_or(true, |b| res < b.residual) {
                // the whole chain as one similarity from sub to xf
                let (s, rr, t) = similarity(&sub, &xf);
                best = Some(Pose { r: rr, t, s, mirror, residual: res });
            }
        }
    }
    best.expect("orient: no proper rotation among the sign choices")
}

/// Flat form: [R (9, row-major), t (3), s, mirror (0/1), residual].
pub fn orient_flat(v: &[f64], tpl: &[f64]) -> Vec<f64> {
    let f = |a: &[f64]| a.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect::<Vec<V3>>();
    let p = orient(&f(v), &f(tpl));
    let mut out: Vec<f64> = p.r.iter().flatten().cloned().collect();
    out.extend_from_slice(&p.t);
    out.extend_from_slice(&[p.s, if p.mirror { 1.0 } else { 0.0 }, p.residual]);
    out
}
