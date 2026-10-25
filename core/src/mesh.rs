//! Triangle mesh: vertices, faces, open3d-style vertex normals, and the
//! similarity transforms the pipeline moves heads and ears through.

pub type V3 = [f64; 3];

#[derive(Clone, Default)]
pub struct Mesh {
    pub v: Vec<V3>,
    pub f: Vec<[u32; 3]>,
}

pub fn sub(a: V3, b: V3) -> V3 { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
pub fn add(a: V3, b: V3) -> V3 { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
pub fn scale(a: V3, s: f64) -> V3 { [a[0] * s, a[1] * s, a[2] * s] }
pub fn dot(a: V3, b: V3) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
pub fn cross(a: V3, b: V3) -> V3 { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }
pub fn norm(a: V3) -> f64 { dot(a, a).sqrt() }
pub fn unit(a: V3) -> V3 { let n = norm(a).max(1e-300); scale(a, 1.0 / n) }

impl Mesh {
    pub fn from_flat(v: &[f64], f: &[u32]) -> Self {
        Mesh {
            v: v.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
            f: f.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
        }
    }
    pub fn flat_v(&self) -> Vec<f64> { self.v.iter().flatten().cloned().collect() }
    pub fn flat_f(&self) -> Vec<u32> { self.f.iter().flatten().cloned().collect() }

    pub fn centroid(&self) -> V3 {
        let n = self.v.len() as f64;
        let mut c = [0.0; 3];
        for p in &self.v { for k in 0..3 { c[k] += p[k]; } }
        scale(c, 1.0 / n)
    }

    /// Area-weighted vertex normals, normalised (open3d compute_vertex_normals).
    pub fn vertex_normals(&self) -> Vec<V3> {
        let mut n = vec![[0.0; 3]; self.v.len()];
        for t in &self.f {
            let (a, b, c) = (self.v[t[0] as usize], self.v[t[1] as usize], self.v[t[2] as usize]);
            let fn_ = cross(sub(b, a), sub(c, a));
            for &i in t { n[i as usize] = add(n[i as usize], fn_); }
        }
        n.iter().map(|x| { let l = norm(*x); if l > 0.0 { scale(*x, 1.0 / l) } else { [0.0, 0.0, 1.0] } }).collect()   // open3d's fallback
    }

    /// Longest side of the axis-aligned bounding box.
    pub fn extent(&self) -> f64 {
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for p in &self.v { for k in 0..3 { lo[k] = lo[k].min(p[k]); hi[k] = hi[k].max(p[k]); } }
        (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max)
    }

    /// V' = (V - c) R^T: the same surface in the frame whose axes are R's rows.
    pub fn rotated(&self, r: &[[f64; 3]; 3], c: V3) -> Mesh {
        Mesh { v: self.v.iter().map(|p| mul_rt(sub(*p, c), r)).collect(), f: self.f.clone() }
    }

    /// Reflected about the plane normal to `axis`, winding reversed so it stays outward.
    pub fn mirrored(&self, axis: usize) -> Mesh {
        Mesh {
            v: self.v.iter().map(|p| { let mut q = *p; q[axis] = -q[axis]; q }).collect(),
            f: self.f.iter().map(|t| [t[2], t[1], t[0]]).collect(),
        }
    }
}

/// p R^T (row vector times R transposed) = R p.
pub fn mul_rt(p: V3, r: &[[f64; 3]; 3]) -> V3 { [dot(r[0], p), dot(r[1], p), dot(r[2], p)] }
/// p R (row vector times R) = R^T p.
pub fn mul_r(p: V3, r: &[[f64; 3]; 3]) -> V3 {
    [r[0][0] * p[0] + r[1][0] * p[1] + r[2][0] * p[2],
     r[0][1] * p[0] + r[1][1] * p[1] + r[2][1] * p[2],
     r[0][2] * p[0] + r[1][2] * p[1] + r[2][2] * p[2]]
}
