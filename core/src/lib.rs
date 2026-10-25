//! Geometry shared by the Python pipeline and the browser demo: one copy,
//! two bindings (`python` feature: PyO3; `wasm` feature: wasm-bindgen).
//! Arrays cross the boundary flat, row-major: n x 3 is a slice of 3n f64.

pub mod triangulate;

#[cfg(feature = "python")]
mod py;
#[cfg(feature = "wasm")]
mod wasm;
