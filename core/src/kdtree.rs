//! Static 3-D k-d tree for nearest-neighbour queries (ICP correspondences and
//! the cloud-to-cloud residual). Built once per point set; no dependencies.

pub struct KdTree {
    pts: Vec<[f64; 3]>,
    idx: Vec<usize>,          // permutation; node = middle of a sub-range
}

impl KdTree {
    pub fn new(pts: Vec<[f64; 3]>) -> Self {
        let mut idx: Vec<usize> = (0..pts.len()).collect();
        build(&pts, &mut idx, 0);
        KdTree { pts, idx }
    }

    /// (index, squared distance) of the nearest point.
    pub fn nearest(&self, q: &[f64; 3]) -> (usize, f64) {
        let mut best = (usize::MAX, f64::INFINITY);
        self.search(0, self.idx.len(), 0, q, &mut best);
        best
    }

    fn search(&self, lo: usize, hi: usize, axis: usize, q: &[f64; 3], best: &mut (usize, f64)) {
        if lo >= hi {
            return;
        }
        let mid = (lo + hi) / 2;
        let i = self.idx[mid];
        let p = &self.pts[i];
        let d2 = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2);
        if d2 < best.1 || (d2 == best.1 && i < best.0) {
            *best = (i, d2);
        }
        let diff = q[axis] - p[axis];
        let (near, far) = if diff < 0.0 { ((lo, mid), (mid + 1, hi)) } else { ((mid + 1, hi), (lo, mid)) };
        let next = (axis + 1) % 3;
        self.search(near.0, near.1, next, q, best);
        if diff * diff <= best.1 {
            self.search(far.0, far.1, next, q, best);
        }
    }
}

fn build(pts: &[[f64; 3]], idx: &mut [usize], axis: usize) {
    if idx.len() <= 1 {
        return;
    }
    let mid = idx.len() / 2;
    idx.select_nth_unstable_by(mid, |a, b| pts[*a][axis].total_cmp(&pts[*b][axis]));
    let (l, r) = idx.split_at_mut(mid);
    build(pts, l, (axis + 1) % 3);
    build(pts, &mut r[1..], (axis + 1) % 3);
}
