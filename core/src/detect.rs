//! Pre- and post-processing around the two bare 2D networks, so the Python
//! host and the browser run identical code and differ only in the runtime that
//! evaluates the graph (onnxruntime vs onnxruntime-web).
//!
//! Ported from, and checked against (demo/parity/detector_drift.py, at tag docs-parity-2026-10-26):
//!  - Ear_Landmarker inference.LandmarkPredictor.predict: bilinear resize
//!    (torch, align_corners=False, no antialias) to 192, [-1, 1], flip fusion
//!    with the mirror arm un-mirrored as 1 - x and NO index permutation.
//!  - BlazeEar blazeear_inference.BlazeEarInference.forward at threshold 0.05:
//!    cv2.resize INTER_LINEAR to 256, bit-exact (cv2_resize: its fixed-point
//!    SIMD path; +-1 grey level flips near-tied top boxes), /127.5 - 1, scores clipped to +-100 and sigmoid, anchor
//!    decode, corners sorted, usable boxes only. NMS is not ported: it never
//!    removes the top box, and the pipeline reads only the top box and score.
//! If Ear_Landmarker or BlazeEar change their inference, the drift test fails.

/// Bilinear resize of an RGB u8 image to ow x oh, as torch's interpolate with
/// align_corners=False and no antialias.
/// Returns HWC f32 in 0..255.
pub fn resize(img: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<f32> {
    let idx = |n_in: usize, n_out: usize, i: usize| {
        let s = ((i as f32 + 0.5) * (n_in as f32 / n_out as f32) - 0.5).max(0.0);
        let i0 = (s.floor() as usize).min(n_in - 1);
        let i1 = if i0 < n_in - 1 { i0 + 1 } else { i0 };
        (i0, i1, s - i0 as f32)
    };
    let xs: Vec<_> = (0..ow).map(|x| idx(w, ow, x)).collect();
    let mut out = vec![0f32; ow * oh * 3];
    for y in 0..oh {
        let (y0, y1, ly) = idx(h, oh, y);
        for (x, &(x0, x1, lx)) in xs.iter().enumerate() {
            for c in 0..3 {
                let p = |yy: usize, xx: usize| img[(yy * w + xx) * 3 + c] as f32;
                let top = (1.0 - lx) * p(y0, x0) + lx * p(y0, x1);
                let bot = (1.0 - lx) * p(y1, x0) + lx * p(y1, x1);
                out[(y * ow + x) * 3 + c] = (1.0 - ly) * top + ly * bot;
            }
        }
    }
    out
}

/// cv2.resize(img, (ow, oh)) with INTER_LINEAR on u8 RGB, bit-exact with
/// OpenCV's vectorised path: 11-bit coefficients, horizontal sums kept as
/// int, then (((b0 (r0 >> 4)) >> 16) + ((b1 (r1 >> 4)) >> 16) + 2) >> 2.
/// Checked against cv2 on renders (detector_drift.py).
pub fn cv2_resize(img: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    let coef = |n_in: usize, n_out: usize, i: usize| {
        let mut fx = ((i as f64 + 0.5) * (n_in as f64 / n_out as f64) - 0.5) as f32;
        let mut sx = fx.floor() as i64;
        fx -= sx as f32;
        if sx < 0 { fx = 0.0; sx = 0; }
        if sx as usize >= n_in - 1 { fx = 0.0; sx = n_in as i64 - 1; }
        let a0 = ((1.0 - fx) * 2048.0).round_ties_even() as i64;
        let s = sx as usize;
        (s, (s + 1).min(n_in - 1), a0, 2048 - a0)
    };
    let cx: Vec<_> = (0..ow).map(|i| coef(w, ow, i)).collect();
    let hrow = |y: usize| -> Vec<i64> {
        let mut r = vec![0i64; ow * 3];
        for (x, &(s0, s1, a0, a1)) in cx.iter().enumerate() {
            for c in 0..3 { r[x * 3 + c] = img[(y * w + s0) * 3 + c] as i64 * a0 + img[(y * w + s1) * 3 + c] as i64 * a1; }
        }
        r
    };
    let mut out = vec![0u8; ow * oh * 3];
    for y in 0..oh {
        let (s0, s1, b0, b1) = coef(h, oh, y);
        let (r0, r1) = (hrow(s0), hrow(s1));
        for i in 0..ow * 3 {
            let v = (((b0 * (r0[i] >> 4)) >> 16) + ((b1 * (r1[i] >> 4)) >> 16) + 2) >> 2;
            out[y * ow * 3 + i] = v.clamp(0, 255) as u8;
        }
    }
    out
}

fn chw(hwc: &[f32], n: usize, f: impl Fn(f32) -> f32) -> Vec<f32> {
    let mut out = vec![0f32; 3 * n * n];
    for i in 0..n * n { for c in 0..3 { out[c * n * n + i] = f(hwc[i * 3 + c]); } }
    out
}

pub const LANDMARKER_INPUT: usize = 192;

/// The two arms of LandmarkPredictor.predict: the [-1, 1] tensor and its mirror.
pub fn landmarker_inputs(img: &[u8], w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let n = LANDMARKER_INPUT;
    let a = chw(&resize(img, w, h, n, n), n, |v| (v / 255.0 - 0.5) / 0.5);
    let mut b = a.clone();
    for c in 0..3 { for y in 0..n { for x in 0..n { b[c * n * n + y * n + x] = a[c * n * n + y * n + (n - 1 - x)]; } } }
    (a, b)
}

/// Fused landmarks in pixels from the two arms' (55 x 2, [0, 1]) outputs.
pub fn landmarks_fused(a: &[f32], b: &[f32], w: usize, h: usize) -> Vec<[f64; 2]> {
    (0..55).map(|i| {
        let x = 0.5 * (a[2 * i] as f64 + (1.0 - b[2 * i] as f64));
        let y = 0.5 * (a[2 * i + 1] as f64 + b[2 * i + 1] as f64);
        [x * w as f64, y * h as f64]
    }).collect()
}

pub const BLAZEEAR_INPUT: usize = 256;
const CONF: f32 = 0.05;
const SCORE_CLIP: f32 = 100.0;
const MIN_SIDE: f32 = 1e-5;

/// BlazeEar's input for a SQUARE image (the pipeline's renders): no padding.
pub fn blazeear_input(img: &[u8], w: usize, h: usize) -> Vec<f32> {
    assert_eq!(w, h, "blazeear_input: the pipeline renders square images");
    let n = BLAZEEAR_INPUT;
    let r: Vec<f32> = cv2_resize(img, w, h, n, n).into_iter().map(|v| v as f32).collect();
    chw(&r, n, |v| v / 127.5 - 1.0)
}

/// Boxes [x0, y0, x1, y1, score] in pixels of a w x w image, score-descending.
pub fn blazeear_boxes(raw_boxes: &[f32], raw_scores: &[f32], anchors: &[[f64; 4]], w: usize) -> Vec<[f64; 5]> {
    let n = BLAZEEAR_INPUT as f32;
    let mut out = Vec::new();
    for (k, a) in anchors.iter().enumerate() {
        let s = 1.0 / (1.0 + (-raw_scores[k].clamp(-SCORE_CLIP, SCORE_CLIP)).exp());
        if !(s >= CONF) { continue; }
        let r = &raw_boxes[k * 16..k * 16 + 4];
        let (ax, ay, aw, ah) = (a[0] as f32, a[1] as f32, a[2] as f32, a[3] as f32);
        let xc = r[0] / n * aw + ax;
        let yc = r[1] / n * ah + ay;
        let (bw, bh) = (r[2] / n * aw, r[3] / n * ah);
        let (y0, x0, y1, x1) = (yc - bh / 2.0, xc - bw / 2.0, yc + bh / 2.0, xc + bw / 2.0);
        let (ymin, ymax, xmin, xmax) = (y0.min(y1), y0.max(y1), x0.min(x1), x0.max(x1));
        if !(xmax - xmin > MIN_SIDE && ymax - ymin > MIN_SIDE) { continue; }
        let px = w as f64;
        out.push([xmin as f64 * px, ymin as f64 * px, xmax as f64 * px, ymax as f64 * px, s as f64]);
    }
    out.sort_by(|a, b| b[4].total_cmp(&a[4]));
    out
}
