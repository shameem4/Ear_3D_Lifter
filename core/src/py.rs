use pyo3::prelude::*;

/// ransac_point(origins_flat, dirs_flat, thresh) -> (x, y, z), NaN when no fit.
#[pyfunction]
fn ransac_point(o: Vec<f64>, d: Vec<f64>, thresh: f64) -> (f64, f64, f64) {
    let p = crate::triangulate::ransac_point_flat(&o, &d, thresh);
    (p[0], p[1], p[2])
}

#[pymodule]
fn ear3d_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ransac_point, m)?)
}
