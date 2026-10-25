use wasm_bindgen::prelude::*;

/// ransacPoint(originsFlat, dirsFlat, thresh) -> Float64Array [x, y, z], NaN when no fit.
#[wasm_bindgen(js_name = ransacPoint)]
pub fn ransac_point(o: &[f64], d: &[f64], thresh: f64) -> Vec<f64> {
    crate::triangulate::ransac_point_flat(o, d, thresh).to_vec()
}
