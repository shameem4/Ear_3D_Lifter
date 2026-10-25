//! The camera (port of ear3d/camera.py and the depth back-projection of
//! ear3d/backproject.py). Coordinates here are the RENDERED scene's: the host
//! mesh after the view's model transform (pipeline::Model maps back).

use crate::mesh::{cross, dot, norm, scale, sub, unit, V3};

pub type M3 = [[f64; 3]; 3];

#[derive(Clone, Copy, Debug)]
pub struct Ext { pub r: M3, pub t: V3 }   // x_cam = R p + t (OpenCV: x right, y down, z forward)

/// The face-on extrinsic of camera_ke: y and z flipped, camera at +eye_z.
pub fn face_on(eye_z: f64) -> Ext {
    Ext { r: [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], t: [0.0, 0.0, eye_z] }
}

/// orbit_extrinsic: camera orbited to (yaw, pitch) degrees about the origin.
pub fn orbit(yaw: f64, pitch: f64, eye_z: f64) -> Ext {
    let (y, p) = (yaw.to_radians(), pitch.to_radians());
    let c = [eye_z * y.sin() * p.cos(), eye_z * p.sin(), eye_z * y.cos() * p.cos()];
    let z = scale(c, -1.0 / norm(c));
    let x0 = cross(z, [0.0, 1.0, 0.0]);
    let x = if norm(x0) < 1e-9 { [1.0, 0.0, 0.0] } else { unit(x0) };
    let yv = cross(z, x);
    let r = [x, yv, z];
    let rc = [dot(r[0], c), dot(r[1], c), dot(r[2], c)];
    Ext { r, t: scale(rc, -1.0) }
}

/// A view-frame direction (x right, y down, z in) as a mesh-coordinate direction.
pub fn view_to_world(e: &Ext, d: V3) -> V3 {
    [0, 1, 2].map(|k| e.r[0][k] * d[0] + e.r[1][k] * d[1] + e.r[2][k] * d[2])
}

/// Pinhole intrinsics for a size x size render (camera_ke).
#[derive(Clone, Copy, Debug)]
pub struct Intr { pub f: f64, pub c: f64, pub size: usize }

pub fn intrinsics(size: usize, vfov: f64) -> Intr {
    Intr { f: (size as f64 / 2.0) / (vfov.to_radians() / 2.0).tan(), c: size as f64 / 2.0 - 0.5, size }
}

impl Ext {
    pub fn centre(&self) -> V3 {
        // -R^T t
        [0, 1, 2].map(|k| -(self.r[0][k] * self.t[0] + self.r[1][k] * self.t[1] + self.r[2][k] * self.t[2]))
    }
    pub fn to_cam(&self, p: V3) -> V3 { [0, 1, 2].map(|k| dot(self.r[k], p) + self.t[k]) }
    pub fn to_world(&self, x: V3) -> V3 { view_to_world(self, sub(x, self.t)) }
    /// 4x4 row-major [R t; 0 0 0 1], what the host's renderer is given.
    pub fn mat4(&self) -> [f64; 16] {
        let r = &self.r;
        [r[0][0], r[0][1], r[0][2], self.t[0], r[1][0], r[1][1], r[1][2], self.t[1],
         r[2][0], r[2][1], r[2][2], self.t[2], 0.0, 0.0, 0.0, 1.0]
    }
}

/// Pixel -> (origin, unit direction) in world coordinates.
pub fn ray(k: &Intr, e: &Ext, uv: [f64; 2]) -> (V3, V3) {
    let d = unit(view_to_world(e, [(uv[0] - k.c) / k.f, (uv[1] - k.c) / k.f, 1.0]));
    (e.centre(), d)
}

pub fn project(k: &Intr, e: &Ext, p: V3) -> [f64; 2] {
    let x = e.to_cam(p);
    let z = x[2].max(1e-9);
    [k.f * x[0] / z + k.c, k.f * x[1] / z + k.c]
}

/// Depth buffer (view-space z, inf = background), row-major size x size.
pub struct Depth<'a> { pub z: &'a [f32], pub w: usize, pub h: usize }

/// backproject(): bilinear depth unless the four neighbours straddle a
/// silhouette (spread > max_jump x near), then the nearest finite sample.
pub fn backproject(d: &Depth, k: &Intr, e: &Ext, uv: [f64; 2]) -> Option<V3> {
    const MAX_JUMP: f64 = 0.02;
    if !(uv[0].is_finite() && uv[1].is_finite()) { return None; }
    let x0 = (uv[0].floor() as i64).clamp(0, d.w as i64 - 2) as usize;
    let y0 = (uv[1].floor() as i64).clamp(0, d.h as i64 - 2) as usize;
    let (fx, fy) = (uv[0] - x0 as f64, uv[1] - y0 as f64);
    let q = [d.z[y0 * d.w + x0], d.z[y0 * d.w + x0 + 1], d.z[(y0 + 1) * d.w + x0], d.z[(y0 + 1) * d.w + x0 + 1]].map(|v| v as f64);
    let w = [(1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy];
    let good: Vec<usize> = (0..4).filter(|&i| q[i].is_finite()).collect();
    if good.is_empty() { return None; }
    let near = good.iter().map(|&i| q[i]).fold(f64::INFINITY, f64::min);
    let far = good.iter().map(|&i| q[i]).fold(f64::NEG_INFINITY, f64::max);
    let z = if good.len() == 4 && far - near <= MAX_JUMP * near.max(1e-9) {
        (0..4).map(|i| q[i] * w[i]).sum()
    } else {
        // first of the finite samples closest to the nearest one (numpy nanargmin)
        q[*good.iter().min_by(|&&a, &&b| (q[a] - near).abs().total_cmp(&(q[b] - near).abs())).unwrap()]
    };
    Some(e.to_world([(uv[0] - k.c) / k.f * z, (uv[1] - k.c) / k.f * z, z]))
}

/// Rotation taking `normal` onto +Z, keeping `up` near +Y (rows = new axes).
pub fn frame_from(normal: V3, up: V3, front: Option<V3>) -> Option<M3> {
    let mut n = normal;
    if let Some(f) = front { if dot(n, f) < 0.0 { n = scale(n, -1.0); } }
    let z = scale(n, 1.0 / norm(n).max(1e-9));
    let y0 = sub(up, scale(z, dot(up, z)));
    let ny = norm(y0);
    if ny < 1e-6 { return None; }
    let y = scale(y0, 1.0 / ny);
    Some([cross(y, z), y, z])
}

pub fn side_frame(left: bool) -> M3 {
    let sd = [if left { 1.0 } else { -1.0 }, 0.0, 0.0];
    frame_from(sd, [0.0, 1.0, 0.0], Some(sd)).unwrap()
}
