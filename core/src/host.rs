//! What the pipeline asks of its host: render views of a mesh it was given,
//! return the pictures (and run FaceMesh on them), evaluate bare graphs, and
//! listen to events. Everything
//! else -- every decision -- is in this crate. Python hosts with Filament and
//! torch; the browser with three.js and onnxruntime-web.

use crate::camera::{Ext, Intr, M3};
use crate::mesh::{Mesh, V3};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Detect { Nothing, Face, EarBoxes, Landmarks }

/// One render: the host's mesh moved by the model transform x = (p - c) R^T / s
/// (Python's in_frame), seen through camera e, lit by a sun travelling along
/// `sun` in those moved coordinates. Depth comes back in moved units.
pub struct View { pub r: M3, pub c: V3, pub s: f64, pub e: Ext, pub sun: V3 }

pub struct RenderReq<'a> {
    pub mesh: u32,
    pub colors: u32,
    pub k: Intr,
    pub views: &'a [View],
    pub depth: bool,
    pub image: bool,       // return the picture (RGB) for the crate's own detectors
    pub detect: Detect,    // host-side detection: only Face (MediaPipe) is asked for
}

#[derive(Default, Clone)]
pub struct ViewOut {
    pub depth: Option<Vec<f32>>,          // size x size, view-space z, inf = background
    pub face: Option<Vec<[f64; 2]>>,      // FaceMesh's 478 points in pixels, or None
    pub boxes: Vec<[f64; 5]>,             // ear boxes [x0, y0, x1, y1, score], px
    pub landmarks: Option<Vec<[f64; 2]>>, // filled in by the crate (detect.rs), not the host
    pub image: Option<Vec<u8>>,           // size x size x 3 RGB, when asked for
}

#[allow(async_fn_in_trait)]
pub trait Host {
    async fn set_mesh(&self, id: u32, m: &Mesh);
    async fn set_colors(&self, id: u32, mesh: u32, c: &[V3]);
    async fn render(&self, r: &RenderReq<'_>) -> Vec<ViewOut>;
    /// Evaluate a bare graph ("r2j": 1x3x192x192 -> [landmarks 55x2, confidence];
    /// "blazeear": 1x3x256x256 -> [boxes Ax16, scores A]) on one input.
    async fn infer(&self, model: &str, input: &[f32]) -> Vec<Vec<f32>>;
    /// Progress for logs and the demo's animation: a kind and a JSON payload.
    fn event(&self, kind: &str, json: &str);
}
