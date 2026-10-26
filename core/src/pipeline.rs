//! The whole pipeline (port of ear3d/pipeline.py, frames.py, ear_view.py and
//! ear_only.landmarks): a mesh in, 55 3D landmarks per ear out. Every decision
//! is here; the host only renders and runs detectors (host.rs). Rationale for
//! each rule lives with the Python it was ported from -- the comments here say
//! only what the rule is.
//!
//! FRAMES. Python moves vertex copies around (the input turned for a retry,
//! the normalised head, an ear-only scan aligned to the template) and renders
//! in_frame(copy, R, C, s). Here a copy is a Frame over a mesh the host holds,
//! x = sigma (p - c0) Q^T, and a render of in_frame(copy, R, C, s) asks the
//! host for its mesh under the model transform (R Q, c0 + C Q / sigma,
//! s / sigma) through E -- exactly in_frame then render, so Filament draws the
//! same pixels it drew for the Python pipeline. (Folding the transform into the
//! camera is equivalent on paper, but Filament's shading differs by a few grey
//! levels at millimetre scale, and FaceMesh flips on that.)

use crate::camera::{self, backproject, frame_from, intrinsics, orbit, side_frame, Depth, Ext, Intr, M3};
use crate::detect;
use crate::host::{Detect, Host, RenderReq, View, ViewOut};
use crate::kdtree::KdTree;
use crate::mesh::{add, cross, dot, mul_r, mul_rt, norm, scale, sub, unit, Mesh, V3};
use crate::section::longest_section_y;
use crate::skin::{skinned, SkinCfg};
use serde::Deserialize;

pub mod face;

const TRAGION_R: usize = 234;
const TRAGION_L: usize = 454;
const FOREHEAD: usize = 10;
const CHIN: usize = 152;
const NOSE_TIP: usize = 1;
const NASION: usize = 168;
const CHEEK_R: usize = 50;
const CHEEK_L: usize = 280;
const NOSE_CLOSE_MM: f64 = 5.0;
const SYM_P90_MM: f64 = 7.0;
const NOSE_EAR_MIN_MM: f64 = 110.0;
const NOSE_EAR_BALANCE: f64 = 0.7;
const GIRTH_OFFSET_MM: f64 = 60.0;
const GIRTH_RATIO_MIN: f64 = 1.15;
const FLIP: M3 = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]];
const PAD: f64 = 0.25;
const K_BEST: usize = 8;
const RANSAC_MM: f64 = 1.0;
pub const HEAD_MIN_MM: f64 = 100.0;
/// _retry_rotation(1..4): numpy default_rng(1000 + k) quaternions, frozen.
const RETRY: [M3; 4] = [
    [[-0.5007479830629822, -0.5510314089576456, -0.6675446380583903], [0.7175892100881103, 0.16700792973821166, -0.6761464907619343], [0.48406320147629184, -0.8176018210270257, 0.3117853095157216]],
    [[0.20775874624720747, -0.9781756693416, 0.002943342640800005], [-0.9701829542191162, -0.20567529766616888, 0.12822912022080407], [-0.12482523262728945, -0.02949630210798715, -0.9917402026042395]],
    [[-0.34789493971994445, -0.14429254987418388, 0.9263631960349366], [0.3559478164225836, 0.8937734591149415, 0.27289220539565906], [-0.8673351502742472, 0.424674774189047, -0.25957864562831623]],
    [[-0.8800087463953468, -0.4748829189590729, 0.008415435140369007], [-0.43547877880484603, 0.8138078294226081, 0.3848052104391302], [-0.18958596856922488, 0.3349672074268971, -0.9229594414004854]],
];
/// ear_only.CANON_TO_HEAD.
const CANON_TO_HEAD: M3 = [[-0.979385, 0.125842, 0.158014], [0.12899, 0.991598, 0.009783], [-0.155455, 0.029964, -0.987388]];
const I3: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

#[derive(Deserialize, Clone)]
pub struct Cfg {
    pub size: usize,
    pub vfov: f64,
    pub eye_z: f64,
    pub light_dir: V3,
    pub face_light_dir: V3,
    pub multiview_light_dir: V3,
    pub tone: usize,
    pub skin_noise: bool,
    pub ear_tone: usize,
    pub ear_skin_noise: bool,
    pub skin_seed: u64,
    pub ao_strength: f64,
    pub ao_rays: usize,
    pub target_occupancy: Option<f64>,
}

pub struct Assets {
    pub table: Vec<[f64; 55]>,   // V rows, per (view, landmark) error; lower is better
    pub angles: Vec<[f64; 2]>,   // (yaw, pitch) per view
    pub template: Vec<V3>,       // mean ear, canonical left ear
    pub canon: Vec<V3>,          // MediaPipe canonical face model, 468 points
    pub anchors: Vec<[f64; 4]>,  // BlazeEar anchors (cx, cy, w, h), normalised
}

pub struct Output {
    pub left: Option<Vec<V3>>,
    pub right: Option<Vec<V3>>,
    pub ear_only: bool,
    pub fit_residual: f64,
    pub r_head: Option<M3>,
    pub centre: Option<V3>,
}

/// The model transform and camera of one render: y = (p - c) R^T / s, camera e.
#[derive(Clone, Copy)]
pub struct Model { r: M3, c: V3, s: f64, e: Ext }

impl Model {
    fn to_mesh(&self, y: V3) -> V3 { add(mul_r(scale(y, self.s), &self.r), self.c) }
    fn dir_to_mesh(&self, d: V3) -> V3 { mul_r(d, &self.r) }
}

/// A copy of a host mesh: x = sigma (p - c0) Q^T.
#[derive(Clone, Copy)]
struct Frame { mesh: u32, q: M3, c0: V3, sigma: f64 }

fn mm(a: &M3, b: &M3) -> M3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 { for j in 0..3 { c[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum(); } }
    c
}

impl Frame {
    fn to_cur(&self, p: V3) -> V3 { scale(mul_rt(sub(p, self.c0), &self.q), self.sigma) }
    fn to_mesh(&self, x: V3) -> V3 { add(mul_r(scale(x, 1.0 / self.sigma), &self.q), self.c0) }
    fn dir_to_cur(&self, d: V3) -> V3 { mul_rt(d, &self.q) }
    /// The copy (x - c) R^T as a frame of the same host mesh.
    fn sub(&self, r: &M3, c: V3) -> Frame {
        Frame { mesh: self.mesh, q: mm(r, &self.q), c0: add(self.c0, scale(mul_r(c, &self.q), 1.0 / self.sigma)), sigma: self.sigma }
    }
}

pub struct Pipeline<'h, H: Host> {
    host: &'h H,
    cfg: Cfg,
    a: Assets,
    k: Intr,
    mesh: Mesh,          // host mesh 0, the input
    cur: Vec<V3>,        // host mesh 1 (an ear-only scan as a left ear), when there is one
    ear_f: Vec<[u32; 3]>,
    face_col: u32,
    ear_col: u32,
    head: Option<(Frame, V3)>,   // run's normalised head and nose, for the face lift
}

fn log<H: Host>(h: &H, msg: String) { h.event("log", &serde_json::to_string(&msg).unwrap()); }

fn mean(p: &[V3]) -> V3 { let mut c = [0.0; 3]; for q in p { c = add(c, *q); } scale(c, 1.0 / p.len() as f64) }

fn percentile(mut v: Vec<f64>, q: f64) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let pos = q / 100.0 * (v.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
}

fn max_abs_rot(v: &[V3], r: &M3, c: V3) -> f64 {
    v.iter().map(|p| { let x = mul_rt(sub(*p, c), r); x[0].abs().max(x[1].abs()).max(x[2].abs()) }).fold(0.0, f64::max)
}

/// The head pose found by one _head_pose pass, in the frame's coordinates.
#[derive(Clone)]
struct Pose { lat: V3, vert: V3, nose: V3 }

impl<'h, H: Host> Pipeline<'h, H> {
    pub fn new(host: &'h H, mesh: Mesh, cfg: Cfg, a: Assets) -> Self {
        let k = intrinsics(cfg.size, cfg.vfov);
        Pipeline { host, cfg, a, k, cur: Vec::new(), ear_f: Vec::new(), mesh, face_col: 0, ear_col: 1, head: None }
    }

    fn skin(&self, m: &Mesh, tone: usize, noise: bool) -> Vec<V3> {
        let c = &self.cfg;
        skinned(m, &SkinCfg { seed: c.skin_seed, tone, ao_strength: c.ao_strength, n_rays: c.ao_rays, noise })
    }

    fn cur_of(&self, f: &Frame) -> Vec<V3> { self.mesh_of(f.mesh).iter().map(|p| f.to_cur(*p)).collect() }
    fn mesh_of(&self, id: u32) -> &[V3] { if id == 0 { &self.mesh.v } else { &self.cur } }

    async fn render(&self, f: &Frame, colors: u32, views: &[(M3, V3, f64, Ext, V3)], depth: bool, detect: Detect) -> (Vec<ViewOut>, Vec<Model>) {
        // views: (R, C, s, E, sun) for in_frame(copy, R, C, s); as a model
        // transform of the host mesh that is R Q about c0 + C Q / sigma, scale s / sigma.
        let vs: Vec<View> = views.iter().map(|(r, c, s, e, sun)| View { r: mm(r, &f.q), c: f.to_mesh(*c), s: s / f.sigma, e: *e, sun: *sun }).collect();
        let models = vs.iter().map(|v| Model { r: v.r, c: v.c, s: v.s, e: v.e }).collect();
        let ours = matches!(detect, Detect::EarBoxes | Detect::Landmarks);
        let host_detect = if ours { Detect::Nothing } else { detect };
        let mut out = self.host.render(&RenderReq { mesh: f.mesh, colors, k: self.k, views: &vs, depth, image: ours, detect: host_detect }).await;
        if ours {
            let n = self.k.size;
            for o in out.iter_mut() {
                let img = o.image.take().expect("host returned no image");
                match detect {
                    Detect::Landmarks => {
                        let (a, b) = detect::landmarker_inputs(&img, n, n);
                        let ra = self.host.infer("r2j", &a).await;
                        let rb = self.host.infer("r2j", &b).await;
                        o.landmarks = Some(detect::landmarks_fused(&ra[0], &rb[0], n, n));
                    }
                    _ => {
                        let r = self.host.infer("blazeear", &detect::blazeear_input(&img, n, n)).await;
                        o.boxes = detect::blazeear_boxes(&r[0], &r[1], &self.a.anchors, n);
                    }
                }
            }
        }
        (out, models)
    }

    fn bp(&self, f: &Frame, o: &ViewOut, m: &Model, uv: [f64; 2]) -> Option<V3> {
        let d = o.depth.as_ref()?;
        backproject(&Depth { z: d, w: self.k.size, h: self.k.size }, &self.k, &m.e, uv).map(|y| f.to_cur(m.to_mesh(y)))
    }

    pub async fn run(&mut self) -> Output {
        let ext = self.mesh.extent();
        self.host.set_mesh(0, &self.mesh).await;
        if ext < HEAD_MIN_MM {
            return self.ear_only().await;
        }
        let c = self.cfg.clone();
        let face = self.skin(&self.mesh, c.tone, c.skin_noise);
        self.host.set_colors(self.face_col, 0, &face).await;
        let ear = self.skin(&self.mesh, c.ear_tone, c.ear_skin_noise);
        self.host.set_colors(self.ear_col, 0, &ear).await;
        self.host.event("stage", r#""face""#);
        let Some((head, nose)) = self.find_face().await else {
            return Output { left: None, right: None, ear_only: false, fit_residual: f64::NAN, r_head: None, centre: None };
        };
        self.head = Some((head, nose));
        self.host.event("head", &serde_json::json!({"R": head.q, "centre": head.c0}).to_string());
        let mut out = [None, None];
        for (i, left) in [true, false].into_iter().enumerate() {
            self.host.event("stage", &serde_json::to_string(&format!("ear {}", if left { "L" } else { "R" })).unwrap());
            let Some((r, cen, s)) = self.ear_view(&head, left).await else { continue };
            let p = self.multiview(&head, self.ear_col, left, &r, cen, s).await;
            let p: Vec<V3> = p.iter().map(|x| if x[0].is_finite() { head.to_mesh(*x) } else { *x }).collect();
            out[i] = Some(self.snap(0, p));
        }
        let [left, right] = out;
        Output { left, right, ear_only: false, fit_residual: f64::NAN, r_head: Some(head.q), centre: Some(head.c0) }
    }

    fn snap(&self, mesh: u32, p: Vec<V3>) -> Vec<V3> {
        let m = if mesh == 0 { &self.mesh.v } else { &self.cur };
        let flat: Vec<f64> = p.iter().flatten().cloned().collect();
        let mf: Vec<f64> = m.iter().flatten().cloned().collect();
        let tris: Vec<u32> = if mesh == 0 { self.mesh.flat_f() } else { self.ear_f.iter().flatten().cloned().collect() };
        let q = crate::snap::closest_points_flat(&flat, &mf, &tris);
        let out: Vec<V3> = q.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        self.host.event("snapped", &serde_json::to_string(&out.iter().map(|x| x.map(|v| if v.is_finite() { v } else { 0.0 })).collect::<Vec<_>>()).unwrap());
        out
    }

    // ---------------------------------------------------------------- face

    /// _find_face: normalize_head, retried on the mesh turned by fixed rotations,
    /// girth-checked (flipped once if the skull is below), frame-checked; else the
    /// least-bad frame. Returns the normalised head as a frame of mesh 0, and the nose.
    async fn find_face(&mut self) -> Option<(Frame, V3)> {
        let mut best: (f64, Option<(Frame, V3)>) = (f64::INFINITY, None);
        for k in 0..=RETRY.len() {
            let q = if k == 0 { I3 } else { RETRY[k - 1] };
            let turned = Frame { mesh: 0, q, c0: [0.0; 3], sigma: 1.0 };
            let Some((mut head, mut nose)) = self.normalize_head(&turned).await else { continue };
            let hv = self.cur_of(&head);
            let g = girth_ratio(&hv, &self.mesh.f, nose);
            log(self.host, format!("pipeline: girth ratio {g:.2}"));
            if g < GIRTH_RATIO_MIN {
                let flipped = head.sub(&FLIP, [0.0; 3]);
                let nf = mul_rt(nose, &FLIP);
                let gf = girth_ratio(&self.cur_of(&flipped), &self.mesh.f, nf);
                log(self.host, format!("pipeline: face search {k} girth ratio {g:.2}, flipped {gf:.2}"));
                if gf < GIRTH_RATIO_MIN {
                    log(self.host, format!("pipeline: face search {k} rejected (no clear skull above neck); retrying turned"));
                    if GIRTH_RATIO_MIN / g < best.0 { best = (GIRTH_RATIO_MIN / g, Some((head, nose))); }
                    continue;
                }
                head = flipped;
                nose = nf;
            }
            let (ok, why, bad) = self.frame_ok(&head, nose).await;
            if ok { return Some((head, nose)); }
            log(self.host, format!("pipeline: face search {k} rejected ({why}); retrying turned"));
            if bad < best.0 { best = (bad, Some((head, nose))); }
        }
        if best.1.is_some() { log(self.host, format!("pipeline: no face search passed; keeping the least-bad (x{:.2} past its gate)", best.0)); }
        best.1
    }

    /// normalize_head: the head in its own frame (+x subject's left, +y up, +z nose).
    async fn normalize_head(&mut self, f: &Frame) -> Option<(Frame, V3)> {
        let hp = self.head_pose(f).await?;
        let vert = unit(sub(hp.vert, scale(hp.lat, dot(hp.vert, hp.lat))));
        let fwd = cross(hp.lat, vert);
        let rh = [hp.lat, vert, fwd];
        let v = self.cur_of(f);
        let centre = mean(&v);
        let head = f.sub(&rh, centre);
        Some((head, mul_rt(sub(hp.nose, centre), &rh)))
    }

    /// head_pose: one pass, then the nose-profile upside-down check (close
    /// calls judged by ear-detector confidence on both side views).
    async fn head_pose(&mut self, f: &Frame) -> Option<Pose> {
        let r = self.head_pose_pass(f, None).await?;
        let fwd = cross(r.lat, r.vert);
        let v = self.cur_of(f);
        let (flipped, tip, margin) = nose_upside_down(&v, r.vert, fwd, r.nose, self.host);
        if margin.abs() < NOSE_CLOSE_MM {
            let Some(r2) = self.head_pose_pass(f, Some((fwd, r.vert, tip))).await else { return Some(r) };
            let (s1, s2) = (self.ear_judge(f, &r).await, self.ear_judge(f, &r2).await);
            log(self.host, format!("mediapipe: close nose call; upright judge {s1:.2} vs turned over {s2:.2}"));
            return Some(if s2 > s1 { r2 } else { r });
        }
        if !flipped { return Some(r); }
        log(self.host, "mediapipe: face is UPSIDE DOWN; turning it over and fitting again".into());
        match self.head_pose_pass(f, Some((fwd, r.vert, tip))).await {
            None => { log(self.host, "mediapipe: turned-over refit found no face; keeping the first pass".into()); Some(r) }
            some => some,
        }
    }

    /// _ear_judge: summed top ear-box confidence on the two side views of the
    /// head as this pose would normalise it.
    async fn ear_judge(&self, f: &Frame, r: &Pose) -> f64 {
        let v = self.cur_of(f);
        let c0 = mean(&v);
        let vert = unit(sub(r.vert, scale(r.lat, dot(r.vert, r.lat))));
        let rh = [r.lat, vert, cross(r.lat, vert)];
        let g = f.sub(&rh, c0);
        let gv: Vec<V3> = v.iter().map(|p| mul_rt(sub(*p, c0), &rh)).collect();
        let views: Vec<_> = [true, false].iter().map(|&left| {
            let rs = side_frame(left);
            let sc = max_abs_rot(&gv, &rs, [0.0; 3]);
            (rs, [0.0; 3], sc, camera::face_on(self.cfg.eye_z), self.side_light(left))
        }).collect();
        let (out, _) = self.render(&g, self.ear_col, &views, false, Detect::EarBoxes).await;
        out.iter().map(|o| o.boxes.iter().map(|b| b[4]).fold(0.0, f64::max)).sum()
    }

    fn side_light(&self, left: bool) -> V3 {
        let d = self.cfg.light_dir;
        if left { d } else { [-d[0], d[1], d[2]] }
    }

    /// _head_pose: FaceMesh over 26 directions x 4 rolls, gated on nose relief and
    /// stability, refined face-on twice, then the canonical face fit.
    async fn head_pose_pass(&mut self, f: &Frame, turn_over: Option<(V3, V3, V3)>) -> Option<Pose> {
        let v = self.cur_of(f);
        let c0 = mean(&v);
        let sun = self.cfg.face_light_dir;
        let fon = camera::face_on(self.cfg.eye_z);
        let axes: Vec<V3> = (0..3).flat_map(|ax| [1.0, -1.0].map(|sg| { let mut e = [0.0; 3]; e[ax] = sg; e })).collect();
        let r3 = 1.0 / 3f64.sqrt();
        let diag: Vec<V3> = [1.0, -1.0].iter().flat_map(|&sx| [1.0, -1.0].iter().flat_map(move |&sy| [1.0, -1.0].map(move |sz| [sx * r3, sy * r3, sz * r3]))).collect();
        let r2 = 1.0 / 2f64.sqrt();
        let mut edge: Vec<V3> = Vec::new();
        for sa in [1.0, -1.0] { for sb in [1.0, -1.0] { edge.push([sa * r2, sb * r2, 0.0]); } }
        for sa in [1.0, -1.0] { for sb in [1.0, -1.0] { edge.push([sa * r2, 0.0, sb * r2]); } }
        for sa in [1.0, -1.0] { for sb in [1.0, -1.0] { edge.push([0.0, sa * r2, sb * r2]); } }
        struct Cand { relief: f64, stab: usize, sep: f64, axis: usize, sgn: i32, vert: V3, p: Vec<Option<V3>>, front: V3 }
        let mut cands: Vec<Cand> = Vec::new();
        let good = |c: &Vec<Cand>| c.iter().any(|t| t.relief >= 15.0 && t.stab >= 3);
        for (rnd, dirs) in [axes, diag, edge].into_iter().enumerate() {
            if rnd >= 1 && good(&cands) { break; }
            let mut views = Vec::new();
            let mut meta = Vec::new();
            for front in &dirs {
                let amax = front.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                let axis = (0..3).find(|&a| front[a].abs() == amax).unwrap();            // numpy argmax: first max
                let sgn = if front[axis] > 0.0 { 1 } else { -1 };
                let up0 = if front[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [0.0, 1.0, 0.0] };
                let up0 = unit(sub(up0, scale(*front, dot(up0, *front))));
                let side = cross(*front, up0);
                for up in [up0, side, scale(up0, -1.0), scale(side, -1.0)] {
                    let Some(r) = frame_from(*front, up, Some(*front)) else { continue };
                    let sc = max_abs_rot(&v, &r, c0);
                    views.push((r, c0, sc, fon, sun));
                    meta.push((axis, sgn, *front));
                }
            }
            let (out, exts) = self.render(f, self.face_col, &views, true, Detect::Face).await;
            let mut hits = Vec::new();
            for (i, o) in out.iter().enumerate() {
                let Some(uv) = &o.face else { continue };
                let p: Vec<Option<V3>> = uv.iter().map(|q| self.bp(f, o, &exts[i], *q)).collect();
                if ![TRAGION_R, TRAGION_L, FOREHEAD, CHIN, NOSE_TIP, NASION, CHEEK_R, CHEEK_L].iter().all(|&j| p[j].is_some()) { continue; }
                let g = |j: usize| p[j].unwrap();
                let mut vert = unit(sub(g(FOREHEAD), g(CHIN)));
                let fwd0 = sub(g(NOSE_TIP), g(NASION));
                let fwd0 = sub(fwd0, scale(vert, dot(fwd0, vert)));
                if norm(fwd0) < 1e-6 { continue; }
                let fwd = unit(fwd0);
                let mut lat = unit(cross(vert, fwd));
                if dot(lat, sub(g(TRAGION_L), g(TRAGION_R))) < 0.0 { lat = scale(lat, -1.0); }
                vert = unit(sub(vert, scale(lat, dot(vert, lat))));
                let sep = norm(sub(g(TRAGION_L), g(TRAGION_R)));
                let (axis, sgn, front) = meta[i];
                let relief = dot(sub(g(NOSE_TIP), scale(add(g(CHEEK_R), g(CHEEK_L)), 0.5)), front);
                hits.push((i, Cand { relief, stab: 0, sep, axis, sgn, vert, p, front }));
            }
            // stability: the same face from +-15 deg in yaw and pitch
            let mut sviews = Vec::new();
            for (i, _) in &hits {
                let (r, c, s, _, sn) = views[*i];
                for (yw, pt) in [(15.0, 0.0), (-15.0, 0.0), (0.0, 15.0), (0.0, -15.0)] {
                    sviews.push((r, c, s, orbit(yw, pt, self.cfg.eye_z), sn));
                }
            }
            if !sviews.is_empty() {
                let (sout, _) = self.render(f, self.face_col, &sviews, false, Detect::Face).await;
                for (h, (_, cand)) in hits.iter_mut().enumerate() {
                    cand.stab = sout[4 * h..4 * h + 4].iter().filter(|o| o.face.is_some()).count();
                }
            }
            for (_, c) in hits { cands.push(c); }
        }
        self.host.event("face_candidates", &serde_json::to_string(&cands.iter().map(|c| (c.front, c.relief, c.stab)).collect::<Vec<_>>()).unwrap());
        if !good(&cands) {
            let weak: Vec<String> = cands.iter().map(|t| format!("relief {:.1} stab {}", t.relief, t.stab)).collect();
            log(self.host, format!("mediapipe: no face passing the gate on 26 views{} -- using the ear sweep",
                if weak.is_empty() { String::new() } else { format!(" (rejected: {})", weak.join(", ")) }));
            return None;
        }
        let mut stable: Vec<&Cand> = cands.iter().filter(|t| t.relief >= 15.0 && t.stab >= 3).collect();
        stable.sort_by(|a, b| b.relief.total_cmp(&a.relief));
        let top = stable[0];
        log(self.host, format!("mediapipe: face on axis {}{}, nose relief {:.1} mm, stable {}/4, tragion sep {:.1} (of {})",
            top.axis, if top.sgn > 0 { "+" } else { "-" }, top.relief, top.stab, top.sep, cands.len()));
        let front0 = top.front;
        let (mut p, mut vert) = (top.p.clone(), top.vert);
        (p, vert) = self.refine(f, &v, p, vert).await;
        if let Some((f_, v_, n_)) = turn_over {
            let Some(fp) = self.face_view(f, &v, f_, scale(v_, -1.0), n_, &[FOREHEAD, CHIN, NOSE_TIP, NASION]).await else {
                log(self.host, "mediapipe: turned-over view found no face".into());
                return None;
            };
            p = fp;
            vert = unit(sub(p[FOREHEAD].unwrap(), p[CHIN].unwrap()));
            (p, vert) = self.refine(f, &v, p, vert).await;
        }
        let _ = vert;
        // rotation-only similarity of the canonical face onto the landmarks
        let canon = &self.a.canon;
        let ok: Vec<usize> = (0..canon.len()).filter(|&j| p[j].is_some()).collect();
        let x: Vec<V3> = ok.iter().map(|&j| canon[j]).collect();
        let y: Vec<V3> = ok.iter().map(|&j| p[j].unwrap()).collect();
        let (k, rcol, _) = crate::orient::similarity(&x, &y);
        let (cx, cy) = (mean(&x), mean(&y));
        let rc = transpose(&rcol);                                // Y ~ k (X - cx) Rc + cy, row form
        let (er, el, orr, ol) = (33, 263, 230, 450);
        let e_lat = unit(sub(canon[el], canon[er]));
        let e_fwd0 = sub(scale(add(canon[orr], canon[ol]), 0.5), scale(add(canon[TRAGION_R], canon[TRAGION_L]), 0.5));
        let mut e_fwd = unit(sub(e_fwd0, scale(e_lat, dot(e_fwd0, e_lat))));
        let mut e_up = cross(e_lat, e_fwd);
        if e_up[1] < 0.0 { e_up = scale(e_up, -1.0); e_fwd = scale(e_fwd, -1.0); }
        let lat = unit(mul_r(e_lat, &rc));
        let mut vert = unit(mul_r(e_up, &rc));
        let mut fwd = unit(mul_r(e_fwd, &rc));
        if let (Some(a), Some(b)) = (p[164], p[9]) {              // up_rule "164_9"
            let u = sub(b, a);
            vert = unit(sub(u, scale(lat, dot(u, lat))));
            fwd = cross(lat, vert);
        }
        let place = |q: V3| add(scale(mul_r(sub(q, cx), &rc), k), cy);
        let nose = place(canon[NOSE_TIP]);
        let rms = (x.iter().zip(&y).map(|(a, b)| { let d = sub(place(*a), *b); dot(d, d) }).sum::<f64>() / x.len() as f64).sqrt();
        log(self.host, format!("mediapipe: canonical face fit on {} points, rms {rms:.1} mm, face-on refine moved the view {:.1} deg",
            ok.len(), dot(fwd, front0).clamp(-1.0, 1.0).acos().to_degrees()));
        Some(Pose { lat, vert, nose })
    }

    /// Two face-on passes, each re-rendering straight at the nose with the
    /// latest vertical.
    async fn refine(&self, f: &Frame, v: &[V3], mut p: Vec<Option<V3>>, mut vert: V3) -> (Vec<Option<V3>>, V3) {
        for _ in 0..2 {
            let (Some(nt), Some(na)) = (p[NOSE_TIP], p[NASION]) else { break };
            let fwd = sub(sub(nt, na), scale(vert, dot(sub(nt, na), vert)));
            if norm(fwd) < 1e-6 { break; }
            let Some(p2) = self.face_view(f, v, unit(fwd), vert, nt, &[TRAGION_R, TRAGION_L, FOREHEAD, CHIN, NOSE_TIP, NASION]).await else { break };
            p = p2;
            vert = unit(sub(p[FOREHEAD].unwrap(), p[CHIN].unwrap()));
        }
        (p, vert)
    }

    async fn face_view(&self, f: &Frame, v: &[V3], fwd: V3, vert: V3, nose: V3, need: &[usize]) -> Option<Vec<Option<V3>>> {
        let r = frame_from(fwd, vert, Some(fwd))?;
        let sc = 0.6 * max_abs_rot(v, &r, nose);
        let (out, exts) = self.render(f, self.face_col, &[(r, nose, sc, camera::face_on(self.cfg.eye_z), self.cfg.face_light_dir)], true, Detect::Face).await;
        let uv = out[0].face.as_ref()?;
        let p: Vec<Option<V3>> = uv.iter().map(|q| self.bp(f, &out[0], &exts[0], *q)).collect();
        if !need.iter().all(|&j| p[j].is_some()) { return None; }
        Some(p)
    }

    /// _frame_ok: symmetric about the nose, nose far from and between both ears.
    async fn frame_ok(&self, head: &Frame, nose: V3) -> (bool, String, f64) {
        let vn = self.cur_of(head);
        let near: Vec<V3> = vn.iter().map(|p| sub(*p, nose)).filter(|l| l[0].hypot(l[1]) < 30.0).collect();
        if near.is_empty() { return (false, "no surface at the nose".into(), f64::INFINITY); }
        let t = add(*near.iter().max_by(|a, b| a[2].total_cmp(&b[2])).unwrap(), nose);
        let reg: Vec<V3> = vn.iter().filter(|p| norm(sub(**p, t)) < 40.0 && p[2] > t[2] - 40.0).cloned().collect();
        let tree = KdTree::new(vn.clone());
        let sym = percentile(reg.iter().map(|p| tree.nearest(&[2.0 * t[0] - p[0], p[1], p[2]]).1.sqrt()).collect(), 90.0);
        if sym > SYM_P90_MM { return (false, format!("asymmetric about the nose, p90 {sym:.1} mm"), sym / SYM_P90_MM); }
        let mut d: Vec<(bool, f64)> = Vec::new();
        for left in [true, false] {
            if let Some((_, c, _)) = self.ear_view(head, left).await { d.push((left, norm(sub(c, t)))); }
        }
        match d.len() {
            0 => (false, "no ear on either side".into(), f64::INFINITY),
            1 => {
                let (s, x) = d[0];
                if x < NOSE_EAR_MIN_MM { return (false, format!("nose-ear {x:.0} mm (side {})", if s { "L" } else { "R" }), NOSE_EAR_MIN_MM / x); }
                (false, format!("no ear on side {}", if s { "R" } else { "L" }), 1.0)
            }
            _ => {
                let (a, b) = (d[0].1, d[1].1);
                let (lo, hi) = (a.min(b), a.max(b));
                if lo < NOSE_EAR_MIN_MM || lo / hi < NOSE_EAR_BALANCE {
                    return (false, format!("nose-ear {a:.0}/{b:.0} mm"), (NOSE_EAR_MIN_MM / lo).max(NOSE_EAR_BALANCE * hi / lo));
                }
                (true, String::new(), 0.0)
            }
        }
    }

    // ----------------------------------------------------------------- ear

    /// ear_view: side view, top ear box, its centre back-projected, padded box
    /// as the zoom. Returns (R, centre, scale) in the head frame.
    async fn ear_view(&self, head: &Frame, left: bool) -> Option<(M3, V3, f64)> {
        let hv = self.cur_of(head);
        let r = side_frame(left);
        let sc = max_abs_rot(&hv, &r, [0.0; 3]);
        let (out, exts) = self.render(head, self.ear_col, &[(r, [0.0; 3], sc, camera::face_on(self.cfg.eye_z), self.side_light(left))], true, Detect::EarBoxes).await;
        let b = out[0].boxes.iter().cloned().max_by(|a, b| a[4].total_cmp(&b[4]))?;   // BlazeEar: top box
        let p0 = self.bp(head, &out[0], &exts[0], [0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])])?;
        let half = self.cfg.eye_z * (self.cfg.vfov.to_radians() / 2.0).tan();
        let box_mm = (b[2] - b[0]).max(b[3] - b[1]) / (self.k.size as f64 / 2.0) * half * sc;
        let s = 0.5 * box_mm * (1.0 + 2.0 * PAD) / half;
        self.host.event("ear_box", &serde_json::json!({"left": left, "box": b, "centre": head.to_mesh(p0), "box_mm": box_mm}).to_string());
        Some((r, p0, s))
    }

    /// occupancy_scale + the 25-view orbit + per-landmark RANSAC over the best
    /// 8 views (all views as the fallback). Points in the frame's coordinates.
    async fn multiview(&self, f: &Frame, colors: u32, left: bool, r: &M3, c0: V3, mut s: f64) -> Vec<V3> {
        let d0 = self.cfg.multiview_light_dir;
        let sun = if left { d0 } else { [-d0[0], d0[1], d0[2]] };
        if let Some(target) = self.cfg.target_occupancy {
            for _ in 0..2 {
                let (out, _) = self.render(f, colors, &[(*r, c0, s, orbit(0.0, 0.0, self.cfg.eye_z), sun)], false, Detect::Landmarks).await;
                let Some(l) = &out[0].landmarks else { break };
                let ptp = |k: usize| { let (lo, hi) = l.iter().fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p[k]), b.max(p[k]))); hi - lo };
                let occ = ptp(0).max(ptp(1)) / self.k.size as f64;
                if !occ.is_finite() || occ <= 0.0 { break; }
                s *= occ / target;
            }
        }
        let views: Vec<_> = self.a.angles.iter().map(|a| (*r, c0, s, orbit(a[0], a[1], self.cfg.eye_z), sun)).collect();
        let (out, exts) = self.render(f, colors, &views, false, Detect::Landmarks).await;
        let nv = views.len();
        let mut o = vec![[[f64::NAN; 3]; 55]; nv];
        let mut d = o.clone();
        let mut uv_all = vec![vec![[f64::NAN; 2]; 55]; nv];
        for i in 0..nv {
            let Some(l) = &out[i].landmarks else { continue };
            for j in 0..55 {
                let (oo, dd) = camera::ray(&self.k, &exts[i].e, l[j]);
                o[i][j] = f.to_cur(exts[i].to_mesh(oo));
                d[i][j] = unit(f.dir_to_cur(exts[i].dir_to_mesh(dd)));
                uv_all[i][j] = l[j];
            }
        }
        self.host.event("views", &serde_json::json!({"left": left, "E": exts.iter().map(|m| m.e.mat4().to_vec()).collect::<Vec<_>>(), "model": exts.iter().map(|m| (m.r, m.c, m.s)).collect::<Vec<_>>(), "uv": uv_all}).to_string());
        if !left {                                               // anatomical view order
            let key = |a: &[f64; 2]| ((a[0] * 1e6).round() as i64, (a[1] * 1e6).round() as i64);
            let idx: Vec<usize> = self.a.angles.iter().map(|a| self.a.angles.iter().position(|b| key(b) == key(&[-a[0], a[1]])).unwrap()).collect();
            o = idx.iter().map(|&i| o[i]).collect();
            d = idx.iter().map(|&i| d[i]).collect();
        }
        let mut p = vec![[f64::NAN; 3]; 55];
        for j in 0..55 {
            let mut vv: Vec<usize> = (0..nv).collect();
            vv.sort_by(|&a, &b| self.a.table[a][j].total_cmp(&self.a.table[b][j]));
            let pick = |vs: &[usize]| {
                let oo: Vec<V3> = vs.iter().map(|&i| o[i][j]).collect();
                let dd: Vec<V3> = vs.iter().map(|&i| d[i][j]).collect();
                crate::triangulate::ransac_point(&oo, &dd, RANSAC_MM).map(|r| r.0)
            };
            if let Some(q) = pick(&vv[..K_BEST.min(nv)]).or_else(|| pick(&(0..nv).collect::<Vec<_>>())) { p[j] = q; }
        }
        self.host.event("triangulated", &serde_json::to_string(&p.iter().map(|x| { let w = f.to_mesh(*x); w.map(|v| if v.is_finite() { v } else { 0.0 }) }).collect::<Vec<_>>()).unwrap());
        p
    }

    // ------------------------------------------------------------ ear only

    /// ear_only.landmarks: template-fit pose and side, then the multiview step
    /// on the aligned ear seen as a left ear of a normalised head.
    async fn ear_only(&mut self) -> Output {
        self.host.event("stage", r#""ear-only""#);
        let pose = crate::orient::orient(&self.mesh.v, &self.a.template);
        log(self.host, format!("pipeline: ear-only scan ({:.0} mm): {} ear, template fit residual {:.2}",
            self.mesh.extent(), if pose.mirror { "R" } else { "L" }, pose.residual));
        // the scan as a left ear: mirrored about z (winding reversed) for a right one
        let w = if pose.mirror { self.mesh.mirrored(2) } else { self.mesh.clone() };
        self.cur = w.v.clone();
        self.ear_f = w.f.clone();
        self.host.set_mesh(1, &w).await;
        let col = self.skin(&w, self.cfg.ear_tone, self.cfg.ear_skin_noise);
        self.host.set_colors(2, 1, &col).await;
        // Xh = (s W R^T + t - c) H^T, c the centroid of s W R^T + t
        let xc: Vec<V3> = w.v.iter().map(|p| add(scale(mul_rt(*p, &pose.r), pose.s), pose.t)).collect();
        let c = mean(&xc);
        let hr = mm(&CANON_TO_HEAD, &pose.r);
        let w0 = scale(mul_r(sub(c, pose.t), &pose.r), 1.0 / pose.s);
        let fr = Frame { mesh: 1, q: hr, c0: w0, sigma: pose.s };
        self.host.event("head", &serde_json::json!({"R": fr.q, "centre": fr.c0, "ear_only": true, "mirror": pose.mirror}).to_string());
        let xh = self.cur_of(&fr);
        let rv = side_frame(true);
        let wv: Vec<V3> = xh.iter().map(|p| mul_rt(*p, &rv)).collect();
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for p in &wv { for k in 0..2 { lo[k] = lo[k].min(p[k]); hi[k] = hi[k].max(p[k]); } }
        let mid = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
        let zn = wv.iter().filter(|p| (p[0] - mid[0]).hypot(p[1] - mid[1]) < 3.0).map(|p| p[2]).fold(f64::MIN, f64::max);
        let z = if zn > f64::MIN { zn } else { wv.iter().map(|p| p[2]).fold(f64::MIN, f64::max) };
        let p0 = mul_r([mid[0], mid[1], z], &rv);
        let half = self.cfg.eye_z * (self.cfg.vfov.to_radians() / 2.0).tan();
        let s = 0.5 * (hi[0] - lo[0]).max(hi[1] - lo[1]) * (1.0 + 2.0 * PAD) / half;
        self.host.event("stage", r#""ear L""#);
        let p = self.multiview(&fr, 2, true, &rv, p0, s).await;
        let p: Vec<V3> = p.iter().map(|x| if x[0].is_finite() { fr.to_mesh(*x) } else { *x }).collect();
        let p = self.snap(1, p);
        let p: Vec<V3> = p.iter().map(|x| if pose.mirror { [x[0], x[1], -x[2]] } else { *x }).collect();
        let (l, r) = if pose.mirror { (None, Some(p)) } else { (Some(p), None) };
        Output { left: l, right: r, ear_only: true, fit_residual: pose.residual, r_head: None, centre: None }
    }
}

fn transpose(a: &M3) -> M3 { [[a[0][0], a[1][0], a[2][0]], [a[0][1], a[1][1], a[2][1]], [a[0][2], a[1][2], a[2][2]]] }

/// nose_upside_down: (upside down, geometric tip, margin = fall below - fall above).
fn nose_upside_down<H: Host>(v: &[V3], vert: V3, fwd: V3, nose: V3, host: &H) -> (bool, V3, f64) {
    let lat = cross(vert, fwd);
    let l: Vec<V3> = v.iter().map(|p| { let d = sub(*p, nose); [dot(d, lat), dot(d, vert), dot(d, fwd)] }).collect();
    let near: Vec<&V3> = l.iter().filter(|q| q[0].hypot(q[1]) < 30.0).collect();
    let Some(t) = near.iter().max_by(|a, b| a[2].total_cmp(&b[2])) else { return (false, nose, f64::NAN) };
    let t = **t;
    let mid: Vec<&V3> = l.iter().filter(|q| (q[0] - t[0]).abs() < 4.0).collect();
    let fall = |dy: f64| {
        let m = mid.iter().filter(|q| (q[1] - (t[1] + dy)).abs() < 2.5).map(|q| q[2]).fold(f64::MIN, f64::max);
        if m > f64::MIN { t[2] - m } else { f64::NAN }
    };
    let (up, dn) = (fall(15.0), fall(-15.0));
    log(host, format!("mediapipe: nose falls back {up:.1} mm above the tip, {dn:.1} below"));
    let tip = add(nose, add(add(scale(lat, t[0]), scale(vert, t[1])), scale(fwd, t[2])));
    (up > dn, tip, dn - up)
}

fn girth_ratio(v: &[V3], f: &[[u32; 3]], nose: V3) -> f64 {
    let m = Mesh { v: v.to_vec(), f: f.to_vec() };
    longest_section_y(&m, nose[1] + GIRTH_OFFSET_MM) / longest_section_y(&m, nose[1] - GIRTH_OFFSET_MM).max(1.0)
}
