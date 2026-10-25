//! Closest point on a triangle mesh (replaces open3d's RaycastingScene
//! compute_closest_points). Brute force over every triangle: the pipeline
//! snaps 55 points, so 55 x ~200k tests is tens of milliseconds.

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn dot(a: V3, b: V3) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn mad(a: V3, b: V3, s: f64) -> V3 { [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s] }

/// Closest point to p on triangle abc (Ericson, Real-Time Collision Detection 5.1.5).
pub fn closest_on_triangle(p: V3, a: V3, b: V3, c: V3) -> V3 {
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 { return a; }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 { return b; }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 { return mad(a, ab, d1 / (d1 - d3)); }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 { return c; }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 { return mad(a, ac, d2 / (d2 - d6)); }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return mad(b, sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let den = 1.0 / (va + vb + vc);
    mad(mad(a, ab, vb * den), ac, vc * den)
}

/// Flat form: points n*3, vertices v*3, triangles t*3 (indices). Non-finite
/// points come back unchanged (the pipeline keeps a failed landmark NaN).
pub fn closest_points_flat(pts: &[f64], verts: &[f64], tris: &[u32]) -> Vec<f64> {
    let v = |i: u32| { let k = 3 * i as usize; [verts[k], verts[k + 1], verts[k + 2]] };
    let mut out = pts.to_vec();
    for (i, q) in pts.chunks_exact(3).enumerate() {
        let p = [q[0], q[1], q[2]];
        if !p.iter().all(|x| x.is_finite()) { continue; }
        let mut best = (f64::INFINITY, p);
        for t in tris.chunks_exact(3) {
            let c = closest_on_triangle(p, v(t[0]), v(t[1]), v(t[2]));
            let d = sub(c, p);
            let d2 = dot(d, d);
            if d2 < best.0 { best = (d2, c); }
        }
        out[3 * i..3 * i + 3].copy_from_slice(&best.1);
    }
    out
}
