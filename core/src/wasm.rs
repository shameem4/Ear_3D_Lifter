use wasm_bindgen::prelude::*;

/// ransacPoint(originsFlat, dirsFlat, thresh) -> Float64Array [x, y, z], NaN when no fit.
#[wasm_bindgen(js_name = ransacPoint)]
pub fn ransac_point(o: &[f64], d: &[f64], thresh: f64) -> Vec<f64> {
    crate::triangulate::ransac_point_flat(o, d, thresh).to_vec()
}

/// closestPoints(pointsFlat, verticesFlat, trianglesFlat) -> Float64Array on the mesh.
#[wasm_bindgen(js_name = closestPoints)]
pub fn closest_points(p: &[f64], v: &[f64], t: &[u32]) -> Vec<f64> {
    crate::snap::closest_points_flat(p, v, t)
}

/// orient(verticesFlat, templateFlat) -> Float64Array [R (9), t (3), s, mirror, residual].
#[wasm_bindgen]
pub fn orient(v: &[f64], tpl: &[f64]) -> Vec<f64> {
    crate::orient::orient_flat(v, tpl)
}
