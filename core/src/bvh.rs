//! Bounding-volume hierarchy over a mesh's triangles, for ray casts (the skin's
//! ambient occlusion) and closest-point queries (the face lift's seating). Median split on the longest axis; Moller-Trumbore hits.

use crate::mesh::{cross, dot, sub, Mesh, V3};
use crate::snap::closest_on_triangle;

struct Node { lo: V3, hi: V3, start: u32, count: u32, left: u32 } // leaf when count > 0

pub struct Bvh { nodes: Vec<Node>, tris: Vec<[V3; 3]>, ids: Vec<u32> }

impl Bvh {
    pub fn new(m: &Mesh) -> Self {
        let tris: Vec<[V3; 3]> = m.f.iter().map(|t| [m.v[t[0] as usize], m.v[t[1] as usize], m.v[t[2] as usize]]).collect();
        let cen: Vec<V3> = tris.iter().map(|t| [0, 1, 2].map(|k| (t[0][k] + t[1][k] + t[2][k]) / 3.0)).collect();
        let mut idx: Vec<u32> = (0..tris.len() as u32).collect();
        let mut nodes = vec![Node { lo: [0.0; 3], hi: [0.0; 3], start: 0, count: 0, left: 0 }];
        build_into(&tris, &cen, &mut idx, 0, &mut nodes, 0);
        let tris = idx.iter().map(|&i| tris[i as usize]).collect();
        Bvh { nodes, tris, ids: idx }
    }

    /// Distance to the first hit along a unit direction, or infinity.
    pub fn cast(&self, o: V3, d: V3) -> f64 {
        let inv = [1.0 / d[0], 1.0 / d[1], 1.0 / d[2]];
        let mut best = f64::INFINITY;
        let mut stack = vec![0u32];
        while let Some(n) = stack.pop() {
            let nd = &self.nodes[n as usize];
            if !slab(o, inv, nd.lo, nd.hi, best) { continue; }
            if nd.count > 0 {
                for t in &self.tris[nd.start as usize..(nd.start + nd.count) as usize] {
                    if let Some(h) = hit(o, d, t) { if h < best { best = h; } }
                }
            } else {
                stack.push(nd.left);
                stack.push(nd.left + 1);
            }
        }
        best
    }
}

impl Bvh {
    /// The closest surface point to p: (point, input triangle index, squared distance).
    pub fn closest(&self, p: V3) -> (V3, u32, f64) {
        let mut best = ([f64::NAN; 3], u32::MAX, f64::INFINITY);
        let mut stack = vec![0u32];
        while let Some(n) = stack.pop() {
            let nd = &self.nodes[n as usize];
            if box_d2(p, nd.lo, nd.hi) >= best.2 { continue; }
            if nd.count > 0 {
                for k in nd.start..nd.start + nd.count {
                    let t = &self.tris[k as usize];
                    let q = closest_on_triangle(p, t[0], t[1], t[2]);
                    let d = sub(q, p);
                    let d2 = dot(d, d);
                    if d2 < best.2 { best = (q, self.ids[k as usize], d2); }
                }
            } else {
                // nearer child last, so it is searched first
                let (a, b) = (nd.left, nd.left + 1);
                let (na, nb) = (&self.nodes[a as usize], &self.nodes[b as usize]);
                if box_d2(p, na.lo, na.hi) < box_d2(p, nb.lo, nb.hi) { stack.push(b); stack.push(a); } else { stack.push(a); stack.push(b); }
            }
        }
        best
    }
}

fn box_d2(p: V3, lo: V3, hi: V3) -> f64 {
    (0..3).map(|k| { let d = (lo[k] - p[k]).max(0.0).max(p[k] - hi[k]); d * d }).sum()
}

fn bounds(tris: &[[V3; 3]], idx: &[u32]) -> (V3, V3) {
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for &i in idx { for p in &tris[i as usize] { for k in 0..3 { lo[k] = lo[k].min(p[k]); hi[k] = hi[k].max(p[k]); } } }
    (lo, hi)
}

/// Fill node `slot` with idx (whose triangles start at `start`), splitting below it.
fn build_into(tris: &[[V3; 3]], cen: &[V3], idx: &mut [u32], start: usize, nodes: &mut Vec<Node>, slot: u32) {
    let (lo, hi) = bounds(tris, idx);
    nodes[slot as usize] = Node { lo, hi, start: start as u32, count: idx.len() as u32, left: 0 };
    if idx.len() <= 4 { return; }
    let ax = (0..3).max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b]))).unwrap();
    let mid = idx.len() / 2;
    idx.select_nth_unstable_by(mid, |a, b| cen[*a as usize][ax].total_cmp(&cen[*b as usize][ax]));
    let l = nodes.len() as u32;
    nodes.push(Node { lo, hi, start: 0, count: 0, left: 0 });
    nodes.push(Node { lo, hi, start: 0, count: 0, left: 0 });
    let (a, b) = idx.split_at_mut(mid);
    build_into(tris, cen, a, start, nodes, l);
    build_into(tris, cen, b, start + mid, nodes, l + 1);
    nodes[slot as usize].count = 0;
    nodes[slot as usize].left = l;
}

fn slab(o: V3, inv: V3, lo: V3, hi: V3, tmax: f64) -> bool {
    let (mut t0, mut t1) = (0.0f64, tmax);
    for k in 0..3 {
        let (mut a, mut b) = ((lo[k] - o[k]) * inv[k], (hi[k] - o[k]) * inv[k]);
        if a > b { std::mem::swap(&mut a, &mut b); }
        if a.is_nan() || b.is_nan() { continue; }
        t0 = t0.max(a);
        t1 = t1.min(b);
        if t0 > t1 { return false; }
    }
    true
}

fn hit(o: V3, d: V3, t: &[V3; 3]) -> Option<f64> {
    let (e1, e2) = (sub(t[1], t[0]), sub(t[2], t[0]));
    let p = cross(d, e2);
    let det = dot(e1, p);
    if det.abs() < 1e-14 { return None; }
    let inv = 1.0 / det;
    let s = sub(o, t[0]);
    let u = dot(s, p) * inv;
    if !(0.0..=1.0).contains(&u) { return None; }
    let q = cross(s, e1);
    let v = dot(d, q) * inv;
    if v < 0.0 || u + v > 1.0 { return None; }
    let h = dot(e2, q) * inv;
    if h > 0.0 { Some(h) } else { None }
}
