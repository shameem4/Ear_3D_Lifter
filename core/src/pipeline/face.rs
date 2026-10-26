//! The face lift (port of ear3d/face.py FaceLifter): a head mesh in, MediaPipe's
//! 478 face landmarks in 3D out, in the mesh's coordinates. Steps and their
//! rationale are in face.py's docstring and FINDINGS; numbering here follows it.
//!   1. find_face (the ear pipeline's normalize_head, with its retries)
//!   2. orbit render of the normalised head, skinned, from a yaw x pitch grid
//!   3. FaceMesh per view (the host's Detect::Face)
//!   4. per-view offsets subtracted, RANSAC ray intersection, seated
//!   5. fit_rays: joint Cauchy-weighted ray fit, Laplacian prior on the canonical mesh
//!   6. fit_dense: dense canonical face fitted to the scan; the nose base takes it
//!   7. spread: one-sided edge springs against bunching
//! Python solves 5 and 6 with a sparse LU; here they are preconditioned CG.

use super::*;
use crate::bvh::Bvh;
use crate::orient::similarity;

pub const N_LM: usize = 478;
const N_MESH: usize = 468;

/// face.FACE, the keys the lift reads.
#[derive(Deserialize, Clone)]
pub struct FaceCfg {
    pub orbit_tone: usize,
    pub orbit_skin_noise: bool,
    pub orbit_light_dir: V3,     // world frame (orbit_light_frame "world")
    pub yaws: Vec<f64>,
    pub pitches: Vec<f64>,
    pub scale_mm: f64,
    pub pivot_behind_nose_mm: f64,
    pub ransac_mm: f64,
    pub fit_rays: bool,
    pub fit_lambda: f64,
    pub fit_cauchy_mm: f64,
    pub fit_iters: usize,
    pub fit_dense: bool,
    pub dense_lambda: f64,
    pub dense_beta: f64,
    pub dense_region_mm: f64,
    pub dense_apply_mm: f64,
    pub dense_trim_mm: f64,
    pub dense_iters: usize,
    pub fit_spacing: bool,
    pub spacing_rho: f64,
    pub spacing_sweeps: usize,
    pub spacing_anchor: f64,
    pub fit_view_weights: bool,
}

/// What demo/export_models.py writes from face_view_offsets_v1.npz and the canonical face.
pub struct FaceAssets {
    pub off: Vec<f64>,          // view x 478 x 2 px
    pub miss: Vec<f64>,         // view x 468 mm, NaN = never detected
    pub edges: Vec<[u32; 2]>,   // canonical_edges
    pub edge_rest: Vec<f64>,    // per edge, typical length / face_size
    pub canon: Mesh,            // canonical_raw (mm, file order), 468 vertices
    pub dense: Mesh,            // canonical_dense (mm), vertices 0-467 are the landmarks
}

pub struct FaceOut {
    pub p: Vec<V3>,             // 478, NaN where unplaced
    pub n_views: usize,
    pub r_head: M3,
    pub centre: V3,
}

/// Rows of an umbrella Laplacian L = D - A, applied twice for L^T L.
struct Lap { adj: Vec<Vec<u32>> }

impl Lap {
    fn new(n: usize, f: &[[u32; 3]]) -> Lap {
        let mut adj = vec![Vec::new(); n];
        for t in f {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                if (a as usize) < n && (b as usize) < n { adj[a as usize].push(b); adj[b as usize].push(a); }
            }
        }
        for a in adj.iter_mut() { a.sort_unstable(); a.dedup(); }
        Lap { adj }
    }
    fn apply(&self, x: &[V3]) -> Vec<V3> {
        self.adj.iter().enumerate().map(|(i, a)| {
            let mut y = scale(x[i], a.len() as f64);
            for &j in a { y = sub(y, x[j as usize]); }
            y
        }).collect()
    }
    fn ltl(&self, x: &[V3]) -> Vec<V3> { self.apply(&self.apply(x)) }
    /// diag(L^T L) = deg^2 + deg.
    fn ltl_diag(&self, i: usize) -> f64 { let d = self.adj[i].len() as f64; d * d + d }
}

type M33 = [[f64; 3]; 3];

fn mv(a: &M33, x: V3) -> V3 { [0, 1, 2].map(|i| a[i][0] * x[0] + a[i][1] * x[1] + a[i][2] * x[2]) }

fn inv3(m: &M33) -> M33 {
    let d = crate::linalg::det(m);
    let c = |i: usize, j: usize| {
        let (r0, r1) = ((i + 1) % 3, (i + 2) % 3);
        let (c0, c1) = ((j + 1) % 3, (j + 2) % 3);
        m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
    };
    [0, 1, 2].map(|i| [0, 1, 2].map(|j| c(j, i) / d))
}

/// Solve (blockdiag(B) + lam L^T L) x = rhs by block-Jacobi preconditioned CG from x.
fn solve(b: &[M33], lap: &Lap, lam: f64, rhs: &[V3], mut x: Vec<V3>) -> Vec<V3> {
    let n = b.len();
    let a = |x: &[V3]| -> Vec<V3> {
        let l = lap.ltl(x);
        (0..n).map(|i| add(mv(&b[i], x[i]), scale(l[i], lam))).collect()
    };
    let pre: Vec<M33> = (0..n).map(|i| {
        let mut m = b[i];
        for k in 0..3 { m[k][k] += lam * lap.ltl_diag(i); }
        inv3(&m)
    }).collect();
    let dotv = |u: &[V3], v: &[V3]| u.iter().zip(v).map(|(p, q)| dot(*p, *q)).sum::<f64>();
    let ax = a(&x);
    let mut r: Vec<V3> = (0..n).map(|i| sub(rhs[i], ax[i])).collect();
    let mut z: Vec<V3> = (0..n).map(|i| mv(&pre[i], r[i])).collect();
    let mut p = z.clone();
    let mut rz = dotv(&r, &z);
    let stop = 1e-24 * dotv(rhs, rhs).max(1e-300);
    for _ in 0..20 * n {
        if dotv(&r, &r) <= stop { break; }
        let ap = a(&p);
        let alpha = rz / dotv(&p, &ap);
        for i in 0..n { x[i] = add(x[i], scale(p[i], alpha)); r[i] = sub(r[i], scale(ap[i], alpha)); }
        for i in 0..n { z[i] = mv(&pre[i], r[i]); }
        let rz2 = dotv(&r, &z);
        let beta = rz2 / rz;
        rz = rz2;
        for i in 0..n { p[i] = add(z[i], scale(p[i], beta)); }
    }
    x
}

fn finite(p: &V3) -> bool { p.iter().all(|v| v.is_finite()) }

/// _similarity(A, B) applied to X: the least-squares similarity taking A onto B.
fn posed(a: &[V3], b: &[V3], x: &[V3]) -> Vec<V3> {
    let (s, r, t) = similarity(a, b);
    x.iter().map(|p| add(scale(crate::linalg::apply(&r, p), s), t)).collect()
}

fn face_size(x: &[V3]) -> f64 {
    let c = mean(x);
    (x.iter().map(|p| dot(sub(*p, c), sub(*p, c))).sum::<f64>() / x.len() as f64).sqrt()
}

/// The (view, landmark) rays in mm: origin, unit direction, and whether the view saw it.
struct Rays { o: Vec<V3>, d: Vec<V3>, has: Vec<bool>, nv: usize }

impl Rays {
    fn at(&self, k: usize, j: usize) -> usize { k * N_LM + j }
    /// Projector off ray (k, j) and that projector applied to its origin.
    fn proj(&self, i: usize) -> (M33, V3) {
        let d = self.d[i];
        let pm = [0, 1, 2].map(|a| [0, 1, 2].map(|b| (if a == b { 1.0 } else { 0.0 }) - d[a] * d[b]));
        let po = mv(&pm, self.o[i]);
        (pm, po)
    }
    fn miss(&self, i: usize, p: V3) -> f64 {
        let v = sub(p, self.o[i]);
        norm(sub(v, scale(self.d[i], dot(v, self.d[i]))))
    }
    /// Per landmark j < n: sum_k w (P, P o) with w Cauchy on the miss from x[j], times the view weight.
    fn normal_eq(&self, x: &[V3], n: usize, c: f64, vw: Option<&[f64]>, beta: f64, b: &mut [M33], rhs: &mut [V3]) {
        for j in 0..n {
            for k in 0..self.nv {
                let i = self.at(k, j);
                if !self.has[i] { continue; }
                let m = self.miss(i, x[j]) / c;
                let w = beta / (1.0 + m * m) * vw.map_or(1.0, |v| v[k * N_MESH + j]);
                let (pm, po) = self.proj(i);
                for a in 0..3 { for bb in 0..3 { b[j][a][bb] += w * pm[a][bb]; } }
                rhs[j] = add(rhs[j], scale(po, w));
            }
        }
    }
}

impl<'h, H: Host> Pipeline<'h, H> {
    /// FaceLifter.run. None when the mesh is not a head in mm or no face is found.
    pub async fn run_face(&mut self, fc: &FaceCfg, fa: &FaceAssets) -> Option<FaceOut> {
        let ext = self.mesh.extent();
        if !(100.0..=600.0).contains(&ext) {
            log(self.host, format!("face: mesh spans {ext:.3}; need a head in MILLIMETRES"));
            return None;
        }
        self.host.set_mesh(0, &self.mesh).await;
        let c = self.cfg.clone();
        let face = self.skin(&self.mesh, c.tone, c.skin_noise);
        self.host.set_colors(self.face_col, 0, &face).await;
        let ear = self.skin(&self.mesh, c.ear_tone, c.ear_skin_noise);
        self.host.set_colors(self.ear_col, 0, &ear).await;
        self.host.event("stage", r#""face""#);
        let (head, nose) = self.find_face().await?;
        self.host.event("head", &serde_json::json!({"R": head.q, "centre": head.c0}).to_string());
        self.lift_face(fc, fa, head, nose).await
    }

    /// After run() on a head: the face lift on the head it found, without skinning or searching again.
    pub async fn face_after_run(&mut self, fc: &FaceCfg, fa: &FaceAssets) -> Option<FaceOut> {
        let (head, nose) = self.head?;
        self.lift_face(fc, fa, head, nose).await
    }

    /// Steps 2-7 on the normalised head.
    async fn lift_face(&mut self, fc: &FaceCfg, fa: &FaceAssets, head: Frame, nose: V3) -> Option<FaceOut> {
        let c = self.cfg.clone();

        // 2-3. the orbit: the normalised head about a pivot behind the nose, in render units
        let s = fc.scale_mm;
        let piv = sub(nose, [0.0, 0.0, fc.pivot_behind_nose_mm]);
        let hv = self.cur_of(&head);
        let geo = Mesh { v: hv.iter().map(|p| scale(sub(*p, piv), 1.0 / s)).collect(), f: self.mesh.f.clone() };
        self.host.set_mesh(2, &geo).await;
        let col = self.skin(&geo, fc.orbit_tone, fc.orbit_skin_noise);
        self.host.set_colors(2, 2, &col).await;
        self.host.event("stage", r#""face orbit""#);
        let grid: Vec<(f64, f64)> = fc.pitches.iter().flat_map(|&p| fc.yaws.iter().map(move |&y| (y, p))).collect();
        let nv = grid.len();
        if fa.off.len() != nv * N_LM * 2 || fa.miss.len() != nv * N_MESH {
            log(self.host, "face: the view offsets were learned on a different view grid".into());
            return None;
        }
        let views: Vec<View> = grid.iter().map(|&(y, p)| View { r: I3, c: [0.0; 3], s: 1.0, e: orbit(y, p, c.eye_z), sun: fc.orbit_light_dir }).collect();
        let out = self.host.render(&RenderReq { mesh: 2, colors: 2, k: self.k, views: &views, depth: false, image: false, detect: Detect::Face }).await;
        let n_views = out.iter().filter(|o| o.face.is_some()).count();
        log(self.host, format!("face: FaceMesh found the face in {n_views}/{nv} views"));

        // 4. rays (mm, head frame about the pivot), triangulated and seated
        let mut rays = Rays { o: vec![[0.0; 3]; nv * N_LM], d: vec![[0.0; 3]; nv * N_LM], has: vec![false; nv * N_LM], nv };
        for (k, o) in out.iter().enumerate() {
            let Some(uv) = &o.face else { continue };
            if uv.len() < N_LM { continue; }
            for j in 0..N_LM {
                let q = [uv[j][0] - fa.off[(k * N_LM + j) * 2], uv[j][1] - fa.off[(k * N_LM + j) * 2 + 1]];
                let (oo, dd) = camera::ray(&self.k, &views[k].e, q);
                let i = rays.at(k, j);
                rays.o[i] = scale(oo, s);
                rays.d[i] = dd;
                rays.has[i] = true;
            }
        }
        let vg = Mesh { v: geo.v.iter().map(|p| scale(*p, s)).collect(), f: geo.f.clone() };
        let bvh = Bvh::new(&vg);
        let seat = |x: &[V3]| -> Vec<V3> { x.iter().map(|p| if finite(p) { bvh.closest(*p).0 } else { *p }).collect() };
        let mut p: Vec<V3> = (0..N_LM).map(|j| {
            let ks: Vec<usize> = (0..nv).filter(|&k| rays.has[rays.at(k, j)]).collect();
            if ks.len() < 2 { return [f64::NAN; 3]; }
            let oo: Vec<V3> = ks.iter().map(|&k| rays.o[rays.at(k, j)]).collect();
            let dd: Vec<V3> = ks.iter().map(|&k| rays.d[rays.at(k, j)]).collect();
            crate::triangulate::ransac_point(&oo, &dd, fc.ransac_mm).map_or([f64::NAN; 3], |r| r.0)
        }).collect();
        p = seat(&p);
        let vw: Option<Vec<f64>> = fc.fit_view_weights.then(|| fa.miss.iter().map(|&m| {
            let m = if m.is_finite() { m.max(0.5) } else { 50.0 };
            1.0 / (m * m)
        }).collect());
        let vw = vw.as_deref();

        // 5. joint ray fit
        if fc.fit_rays {
            self.host.event("stage", r#""face ray fit""#);
            let vc = &fa.canon.v[..N_MESH];
            let ok: Vec<usize> = (0..N_MESH).filter(|&j| finite(&p[j])).collect();
            if ok.len() >= 4 {
                let lap = Lap::new(N_MESH, &fa.canon.f);
                let a: Vec<V3> = ok.iter().map(|&j| vc[j]).collect();
                let b: Vec<V3> = ok.iter().map(|&j| p[j]).collect();
                let rf = posed(&a, &b, vc);
                let lref = lap.ltl(&rf);
                let mut x: Vec<V3> = (0..N_MESH).map(|j| if finite(&p[j]) { p[j] } else { rf[j] }).collect();
                for _ in 0..fc.fit_iters {
                    let mut bm = vec![[[0.0; 3]; 3]; N_MESH];
                    let mut rhs: Vec<V3> = lref.iter().map(|l| scale(*l, fc.fit_lambda)).collect();
                    rays.normal_eq(&x, N_MESH, fc.fit_cauchy_mm, vw, 1.0, &mut bm, &mut rhs);
                    x = solve(&bm, &lap, fc.fit_lambda, &rhs, x);
                    x = seat(&x);
                }
                p[..N_MESH].copy_from_slice(&x);
            }
        }

        // 6. dense nose fit
        if fc.fit_dense {
            self.host.event("stage", r#""face nose fit""#);
            let vd = &fa.dense.v;
            let nd = vd.len();
            let ok: Vec<usize> = (0..N_MESH).filter(|&j| finite(&p[j])).collect();
            if ok.len() >= 4 {
                let lap = Lap::new(nd, &fa.dense.f);
                let region: Vec<bool> = vd.iter().map(|v| norm(sub(*v, vd[2])) < fc.dense_region_mm).collect();
                let a: Vec<V3> = ok.iter().map(|&j| vd[j]).collect();
                let b: Vec<V3> = ok.iter().map(|&j| p[j]).collect();
                let mut y = posed(&a, &b, vd);
                let tn: Vec<V3> = vg.f.iter().map(|t| unit(cross(sub(vg.v[t[1] as usize], vg.v[t[0] as usize]), sub(vg.v[t[2] as usize], vg.v[t[0] as usize])))).collect();
                for _ in 0..fc.dense_iters {
                    let mut bm = vec![[[0.0; 3]; 3]; nd];
                    let mut rhs = vec![[0.0; 3]; nd];
                    for i in 0..nd {
                        if !region[i] { continue; }
                        let (q, t, d2) = bvh.closest(y[i]);
                        if d2.sqrt() >= fc.dense_trim_mm { continue; }
                        let n = tn[t as usize];
                        bm[i] = [0, 1, 2].map(|a| [0, 1, 2].map(|b| n[a] * n[b]));
                        rhs[i] = mv(&bm[i], q);
                    }
                    rays.normal_eq(&y, N_MESH, fc.fit_cauchy_mm, vw, fc.dense_beta, &mut bm, &mut rhs);
                    let rf = posed(vd, &y, vd);
                    let lref = lap.ltl(&rf);
                    for i in 0..nd { rhs[i] = add(rhs[i], scale(lref[i], fc.dense_lambda)); }
                    y = solve(&bm, &lap, fc.dense_lambda, &rhs, y);
                }
                let ys = seat(&y[..N_MESH]);
                let v2 = fa.canon.v[2];
                for j in 0..N_MESH {
                    if norm(sub(fa.canon.v[j], v2)) < fc.dense_apply_mm { p[j] = ys[j]; }
                }
            }
        }

        // 7. spacing springs
        self.host.event("stage", r#""face spacing""#);
        if fc.fit_spacing && p[..N_MESH].iter().all(finite) {
            let x0: Vec<V3> = p[..N_MESH].to_vec();
            let fs = face_size(&x0);
            let mut y = x0.clone();
            for _ in 0..fc.spacing_sweeps {
                let mut cs = vec![[0.0; 3]; N_MESH];
                let mut cn = vec![0.0f64; N_MESH];
                let mut any = false;
                for (e, &rest) in fa.edges.iter().zip(&fa.edge_rest) {
                    let (a, b) = (e[0] as usize, e[1] as usize);
                    let d = sub(y[b], y[a]);
                    let l = norm(d);
                    let want = fc.spacing_rho * rest * fs;
                    if l >= want { continue; }
                    any = true;
                    let k = scale(d, 0.5 * (want - l) / l.max(1e-9));
                    cs[a] = sub(cs[a], k);
                    cs[b] = add(cs[b], k);
                    cn[a] += 1.0;
                    cn[b] += 1.0;
                }
                if !any { break; }
                let moved: Vec<V3> = (0..N_MESH).map(|i| {
                    let q = add(y[i], scale(cs[i], 1.0 / cn[i].max(1.0)));
                    add(q, scale(sub(x0[i], q), fc.spacing_anchor))
                }).collect();
                y = seat(&moved);
            }
            p[..N_MESH].copy_from_slice(&y);
        }

        self.host.event("stage", r#""face done""#);
        let p = p.iter().map(|x| if finite(x) { head.to_mesh(add(*x, piv)) } else { *x }).collect();
        Some(FaceOut { p, n_views, r_head: head.q, centre: head.c0 })
    }
}
