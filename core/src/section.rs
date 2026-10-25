//! Plane sections of a mesh: the girth check in Pipeline._find_face
//! (trimesh.section + the longest discrete curve). Crossing points are joined
//! by POSITION, as trimesh's path loader does, so an unwelded scan (SONICOM
//! STLs share no vertices) still gives closed curves.

use crate::mesh::{Mesh, V3};
use std::collections::HashMap;

fn key(p: V3) -> (i64, i64, i64) { let q = |x: f64| (x * 1e6).round() as i64; (q(p[0]), q(p[1]), q(p[2])) }

/// Length of the longest connected curve where the mesh crosses y = y0.
pub fn longest_section_y(m: &Mesh, y0: f64) -> f64 {
    let mut seg: Vec<(V3, V3)> = Vec::new();
    for t in &m.f {
        let mut pts = Vec::new();
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let (mut pa, mut pb) = (m.v[a as usize], m.v[b as usize]);
            if pa > pb { std::mem::swap(&mut pa, &mut pb); }        // same edge, same arithmetic
            let (da, db) = (pa[1] - y0, pb[1] - y0);
            if (da < 0.0) != (db < 0.0) {
                let s = da / (da - db);
                pts.push([0, 1, 2].map(|k| pa[k] + s * (pb[k] - pa[k])));
            }
        }
        if pts.len() == 2 { seg.push((pts[0], pts[1])); }
    }
    let mut id: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut parent: Vec<usize> = Vec::new();
    fn find(p: &mut [usize], x: usize) -> usize { let mut r = x; while p[r] != r { r = p[r]; } let mut y = x; while p[y] != r { let n = p[y]; p[y] = r; y = n; } r }
    let mut ends = Vec::with_capacity(seg.len());
    for s in &seg {
        let mut node = |p: V3| *id.entry(key(p)).or_insert_with(|| { parent.push(parent.len()); parent.len() - 1 });
        let (a, b) = (node(s.0), node(s.1));
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb { parent[ra] = rb; }
        ends.push(a);
    }
    let mut len: HashMap<usize, f64> = HashMap::new();
    for (s, a) in seg.iter().zip(ends) {
        let r = find(&mut parent, a);
        let d = ((s.0[0] - s.1[0]).powi(2) + (s.0[1] - s.1[1]).powi(2) + (s.0[2] - s.1[2]).powi(2)).sqrt();
        *len.entry(r).or_insert(0.0) += d;
    }
    len.values().cloned().fold(0.0, f64::max)
}
