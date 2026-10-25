//! Robust ray intersection (port of ear3d/triangulate.py).
//!
//! One deliberate change from the Python: RANSAC tries EVERY pair of rays in
//! order instead of 64 random draws. Python's draws come from numpy's RNG, which
//! nothing else reproduces, and with the pipeline's best-8 views there are only
//! 28 pairs, so exhaustive is both deterministic and no more expensive.

type V3 = [f64; 3];

fn unit(d: &V3) -> V3 {
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    [d[0] / n, d[1] / n, d[2] / n]
}

/// Eigenvalues of a symmetric 3x3 matrix (closed form).
fn sym_eig3(a: &[[f64; 3]; 3]) -> [f64; 3] {
    let p1 = a[0][1].powi(2) + a[0][2].powi(2) + a[1][2].powi(2);
    let q = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
    if p1 == 0.0 {
        return [a[0][0], a[1][1], a[2][2]];
    }
    let p2 = (a[0][0] - q).powi(2) + (a[1][1] - q).powi(2) + (a[2][2] - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    let mut b = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            b[i][j] = (a[i][j] - if i == j { q } else { 0.0 }) / p;
        }
    }
    let det = b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
        - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
        + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0]);
    let r = (det / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;
    let e1 = q + 2.0 * p * phi.cos();
    let e3 = q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
    [e1, 3.0 * q - e1 - e3, e3]
}

fn solve3(a: &[[f64; 3]; 3], b: &V3) -> V3 {
    let det = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(a);
    let mut x = [0.0; 3];
    for k in 0..3 {
        let mut m = *a;
        for i in 0..3 {
            m[i][k] = b[i];
        }
        x[k] = det(&m) / d;
    }
    x
}

/// Point minimising the summed squared distance to the rays; None when the
/// normal equations are ill-conditioned (cond > 1e8), as in the Python.
pub fn lsq_point(o: &[V3], n: &[V3]) -> Option<V3> {
    let mut a = [[0.0; 3]; 3];
    let mut b = [0.0; 3];
    for (oi, ni) in o.iter().zip(n) {
        for i in 0..3 {
            for j in 0..3 {
                let p = if i == j { 1.0 } else { 0.0 } - ni[i] * ni[j];
                a[i][j] += p;
                b[i] += p * oi[j];
            }
        }
    }
    let e = sym_eig3(&a);
    let (hi, lo) = (e.iter().cloned().fold(f64::MIN, f64::max), e.iter().cloned().fold(f64::MAX, f64::min).abs());
    if lo == 0.0 || hi / lo > 1e8 {
        return None;
    }
    Some(solve3(&a, &b))
}

fn ray_dist(p: &V3, o: &V3, n: &V3) -> f64 {
    let v = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
    let t = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
    ((v[0] - t * n[0]).powi(2) + (v[1] - t * n[1]).powi(2) + (v[2] - t * n[2]).powi(2)).sqrt()
}

/// Robust intersection: (point, inlier mask), None when no fit exists.
pub fn ransac_point(o: &[V3], d: &[V3], thresh: f64) -> Option<(V3, Vec<bool>)> {
    let n: Vec<V3> = d.iter().map(unit).collect();
    let m = o.len();
    if m < 2 {
        return None;
    }
    if m == 2 {
        return lsq_point(o, &n).map(|p| (p, vec![true; 2]));
    }
    let mut best: Option<(V3, Vec<bool>, usize)> = None;
    for i in 0..m {
        for j in i + 1..m {
            let Some(p) = lsq_point(&[o[i], o[j]], &[n[i], n[j]]) else { continue };
            let inl: Vec<bool> = (0..m).map(|k| ray_dist(&p, &o[k], &n[k]) < thresh).collect();
            let c = inl.iter().filter(|x| **x).count();
            if best.as_ref().map_or(c > 0, |b| c > b.2) {
                best = Some((p, inl, c));
            }
        }
    }
    match best {
        Some((p, inl, c)) if c >= 2 => {
            let (oi, ni): (Vec<V3>, Vec<V3>) = (0..m).filter(|&k| inl[k]).map(|k| (o[k], n[k])).unzip();
            Some((lsq_point(&oi, &ni).unwrap_or(p), inl))
        }
        _ => lsq_point(o, &n).map(|p| (p, vec![true; m])),
    }
}

/// Flat-array form both bindings call: o, d are m*3; returns [x, y, z] or NaNs.
pub fn ransac_point_flat(o: &[f64], d: &[f64], thresh: f64) -> [f64; 3] {
    let v = |a: &[f64]| a.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect::<Vec<V3>>();
    ransac_point(&v(o), &v(d), thresh).map_or([f64::NAN; 3], |r| r.0)
}
