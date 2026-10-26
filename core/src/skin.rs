//! Synthetic skin for a colourless scan (port of ear3d/skin.py): ambient
//! occlusion, subsurface reddening, a base tone, low-frequency mottling and
//! position-based grain. See skin.py for why each term is there.
//!
//! The random draws are numpy's (rng::Rng reproduces default_rng), in skin.py's
//! order, so a seed gives skin.py's skin. Rays are cast in f64 here and in
//! float32 by open3d, so a grazing ray can rarely land the other way.

use crate::bvh::Bvh;
use crate::linalg::jacobi_eig;
use crate::mesh::{add, cross, dot, scale, sub, unit, Mesh, V3};
use crate::rng::Rng;

pub const ALL_TONES: [[f64; 3]; 13] = [
    [0.96, 0.84, 0.76], [0.93, 0.79, 0.69], [0.88, 0.72, 0.60],
    [0.80, 0.63, 0.50], [0.71, 0.54, 0.42], [0.60, 0.44, 0.34],
    [0.48, 0.34, 0.26], [0.36, 0.25, 0.19], [0.27, 0.18, 0.14],
    [0.20, 0.13, 0.10], [0.14, 0.09, 0.07], [0.09, 0.06, 0.045], [0.05, 0.035, 0.025],
];

/// Per-vertex accessibility in [0, 1]: cosine-weighted hemisphere rays of
/// length radius_frac x extent; the fraction that escape.
pub fn ambient_occlusion(m: &Mesh, normals: &[V3], n_rays: usize, radius_frac: f64, seed: u64) -> Vec<f64> {
    let ext = m.extent();
    let rlen = ext * radius_frac;
    let bvh = Bvh::new(m);
    // numpy's draw order: per ray, u1 for every vertex, then u2 for every vertex
    let mut rng = Rng::new(seed);
    let n = m.v.len();
    let frames: Vec<(V3, V3, V3, V3)> = (0..n).map(|i| {
        let nv = normals[i];
        let helper = if nv[2].abs() > 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 0.0, 1.0] };
        let c = cross(nv, helper);
        let t1 = scale(c, 1.0 / crate::mesh::norm(c).max(1e-9));
        (nv, t1, cross(nv, t1), add(m.v[i], scale(nv, ext * 1e-4)))
    }).collect();
    let mut open = vec![0usize; n];
    for _ in 0..n_rays {
        let u1: Vec<f64> = (0..n).map(|_| rng.random()).collect();
        let u2: Vec<f64> = (0..n).map(|_| rng.random()).collect();
        for i in 0..n {
            let (nv, t1, t2, o) = frames[i];
            let (rr, th) = (u1[i].sqrt(), 2.0 * std::f64::consts::PI * u2[i]);
            let d = add(add(scale(t1, rr * th.cos()), scale(t2, rr * th.sin())), scale(nv, (1.0 - u1[i]).max(0.0).sqrt()));
            let d = scale(d, 1.0 / crate::mesh::norm(d).max(1e-9));
            if !(bvh.cast(o, d) < rlen) { open[i] += 1; }
        }
    }
    open.iter().map(|&c| c as f64 / n_rays as f64).collect()
}

pub struct SkinCfg { pub seed: u64, pub tone: usize, pub ao_strength: f64, pub n_rays: usize, pub noise: bool }

/// Vertex colours (RGB in [0, 1]) for `m` as given (its own axes).
pub fn skin_colors(m: &Mesh, c: &SkinCfg) -> Vec<V3> { skin_colors_ao(m, c, None).0 }

/// skin_colors, with the ambient occlusion given (it depends only on the shape,
/// seed and ray count, not on tone or noise); returns the colours and the AO used.
pub fn skin_colors_ao(m: &Mesh, c: &SkinCfg, ao: Option<&[f64]>) -> (Vec<V3>, Vec<f64>) {
    let mut rng = Rng::new(c.seed);
    let normals = m.vertex_normals();
    let mut base = ALL_TONES[c.tone];
    let expo = rng.uniform(0.94, 1.06);
    for k in 0..3 { base[k] *= expo; }
    let acc = match ao { Some(a) => a.to_vec(), None => ambient_occlusion(m, &normals, c.n_rays, 0.04, c.seed) };
    let ext = m.extent();
    let cen = m.centroid();
    let mut blotch_terms = Vec::new();
    for (f, a) in [(7.0, 0.030), (17.0, 0.016)] {
        let ph = [rng.random(), rng.random(), rng.random()].map(|x| x * 2.0 * std::f64::consts::PI);
        blotch_terms.push((f, a, ph));
    }
    let n_wave = 96;
    let w: Vec<V3> = (0..n_wave).map(|_| unit([rng.normal(), rng.normal(), rng.normal()])).collect();
    let freq: Vec<f64> = (0..n_wave).map(|_| rng.uniform(60.0, 400.0)).collect();
    let ph: Vec<f64> = (0..n_wave).map(|_| rng.random() * 2.0 * std::f64::consts::PI).collect();
    let amp = 0.012 * (2.0 / n_wave as f64).sqrt();
    let col = m.v.iter().enumerate().map(|(i, v)| {
        let p = scale(sub(*v, cen), 1.0 / ext);
        let a = acc[i];
        let shade = (1.0 - c.ao_strength) + c.ao_strength * a;
        let warm = [1.0 + (1.0 - a) * 0.22, 1.0 - (1.0 - a) * 0.06, 1.0 - (1.0 - a) * 0.10];
        let mut tex = 0.0;
        if c.noise {
            for (f, am, ph3) in &blotch_terms {
                tex += am * (p[0] * f + ph3[0]).sin() * (p[1] * f + ph3[1]).sin() * (p[2] * f + ph3[2]).sin();
            }
            tex += amp * (0..n_wave).map(|k| (dot(p, scale(w[k], freq[k])) + ph[k]).sin()).sum::<f64>();
        }
        [0, 1, 2].map(|k| (base[k] * shade * warm[k] + tex).clamp(0.0, 1.0))
    }).collect();
    (col, acc)
}

/// Colours that do not depend on the mesh's orientation: computed in its own
/// PCA frame (axis signs fixed by the third moment), as Pipeline._skinned does.
pub fn skinned(m: &Mesh, c: &SkinCfg) -> Vec<V3> { skinned_ao(m, c, None).0 }

/// skinned, with skin_colors_ao's AO pass-through. The AO is computed in the PCA
/// frame at relative ray length, so it is the same for any rotated, translated or
/// uniformly scaled copy of the same mesh.
pub fn skinned_ao(m: &Mesh, c: &SkinCfg, ao: Option<&[f64]>) -> (Vec<V3>, Vec<f64>) {
    let c0 = m.centroid();
    let mut cov = [[0.0; 3]; 3];
    for p in &m.v { let d = sub(*p, c0); for i in 0..3 { for j in 0..3 { cov[i][j] += d[i] * d[j]; } } }
    let mut vt = jacobi_eig::<3>(cov).1;
    for r in 0..3 {
        let m3: f64 = m.v.iter().map(|p| dot(sub(*p, c0), vt[r]).powi(3)).sum();
        if m3 < 0.0 { vt[r] = scale(vt[r], -1.0); }
    }
    if crate::linalg::det(&vt) < 0.0 { vt[2] = scale(vt[2], -1.0); }
    skin_colors_ao(&m.rotated(&vt, c0), c, ao)
}
