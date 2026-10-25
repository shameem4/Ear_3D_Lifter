//! Geometry shared by the Python pipeline and the browser demo: one copy,
//! two bindings (`python` feature: PyO3; `wasm` feature: wasm-bindgen).
//! Arrays cross the boundary flat, row-major: n x 3 is a slice of 3n f64.

pub mod bvh;
pub mod camera;
pub mod host;
pub mod kdtree;
pub mod linalg;
pub mod mesh;
pub mod orient;
pub mod pipeline;
pub mod rng;
pub mod section;
pub mod skin;
pub mod snap;
pub mod triangulate;
pub mod ziggurat;

pub mod exec;

#[cfg(feature = "python")]
mod py;
#[cfg(feature = "wasm")]
mod wasm;
