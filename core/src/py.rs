use pyo3::prelude::*;

/// ransac_point(origins_flat, dirs_flat, thresh) -> (x, y, z), NaN when no fit.
#[pyfunction]
fn ransac_point(o: Vec<f64>, d: Vec<f64>, thresh: f64) -> (f64, f64, f64) {
    let p = crate::triangulate::ransac_point_flat(&o, &d, thresh);
    (p[0], p[1], p[2])
}

/// closest_points(points_flat, vertices_flat, triangles_flat) -> points_flat on the mesh.
#[pyfunction]
fn closest_points(p: Vec<f64>, v: Vec<f64>, t: Vec<u32>) -> Vec<f64> {
    crate::snap::closest_points_flat(&p, &v, &t)
}

#[pymodule]
fn ear3d_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ransac_point, m)?)?;
    m.add_function(wrap_pyfunction!(closest_points, m)?)
}
