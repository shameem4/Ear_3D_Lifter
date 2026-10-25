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

/// orient(vertices_flat, template_flat) -> [R (9), t (3), s, mirror, residual].
#[pyfunction]
fn orient(v: Vec<f64>, tpl: Vec<f64>) -> Vec<f64> {
    crate::orient::orient_flat(&v, &tpl)
}

#[pymodule]
fn ear3d_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ransac_point, m)?)?;
    m.add_function(wrap_pyfunction!(closest_points, m)?)?;
    m.add_function(wrap_pyfunction!(orient, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(skinned, m)?)?;
    m.add_function(wrap_pyfunction!(rng_draws, m)?)
}

// ------------------------------------------------------------- the pipeline

use crate::exec::{assets, block_on, bytes_f64, f32s, f64s, u32s};
use crate::host::{Detect, Host, RenderReq, ViewOut};
use crate::mesh::{Mesh, V3};
use pyo3::types::{PyBytes, PyDict, PyList};

/// The Python host object: set_mesh(id, V bytes, F bytes), set_colors(id, mesh,
/// C bytes), render(dict) -> list of dicts, event(kind, json).
struct PyHost<'py> { obj: Bound<'py, PyAny> }

impl<'py> Host for PyHost<'py> {
    async fn set_mesh(&self, id: u32, m: &Mesh) {
        let py = self.obj.py();
        let v = PyBytes::new(py, &bytes_f64(&m.flat_v()));
        let f = PyBytes::new(py, &m.flat_f().iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        self.obj.call_method1("set_mesh", (id, v, f)).expect("host.set_mesh");
    }
    async fn set_colors(&self, id: u32, mesh: u32, c: &[V3]) {
        let py = self.obj.py();
        let flat: Vec<f64> = c.iter().flatten().cloned().collect();
        self.obj.call_method1("set_colors", (id, mesh, PyBytes::new(py, &bytes_f64(&flat)))).expect("host.set_colors");
    }
    async fn render(&self, r: &RenderReq<'_>) -> Vec<ViewOut> {
        let py = self.obj.py();
        let d = PyDict::new(py);
        d.set_item("mesh", r.mesh).unwrap();
        d.set_item("colors", r.colors).unwrap();
        d.set_item("size", r.k.size).unwrap();
        d.set_item("f", r.k.f).unwrap();
        d.set_item("c", r.k.c).unwrap();
        d.set_item("depth", r.depth).unwrap();
        d.set_item("detect", match r.detect { Detect::Nothing => "none", Detect::Face => "face", Detect::EarBoxes => "ear_boxes", Detect::Landmarks => "landmarks" }).unwrap();
        let views = PyList::empty(py);
        for v in r.views {
            let vd = PyDict::new(py);
            vd.set_item("E", v.e.mat4().to_vec()).unwrap();
            vd.set_item("R", v.r.iter().flatten().cloned().collect::<Vec<f64>>()).unwrap();
            vd.set_item("C", v.c.to_vec()).unwrap();
            vd.set_item("s", v.s).unwrap();
            vd.set_item("sun", v.sun.to_vec()).unwrap();
            views.append(vd).unwrap();
        }
        d.set_item("views", views).unwrap();
        let out = self.obj.call_method1("render", (d,)).expect("host.render");
        let mut res = Vec::new();
        for item in out.try_iter().expect("render returns a list") {
            let item = item.unwrap();
            let get = |k: &str| item.get_item(k).ok().filter(|x| !x.is_none());
            let mut o = ViewOut::default();
            if let Some(b) = get("depth") { o.depth = Some(f32s(b.extract::<&[u8]>().expect("depth bytes"))); }
            if let Some(b) = get("face") { o.face = Some(f64s(b.extract::<&[u8]>().unwrap()).chunks_exact(2).map(|c| [c[0], c[1]]).collect()); }
            if let Some(b) = get("landmarks") { o.landmarks = Some(f64s(b.extract::<&[u8]>().unwrap()).chunks_exact(2).map(|c| [c[0], c[1]]).collect()); }
            if let Some(b) = get("boxes") { o.boxes = f64s(b.extract::<&[u8]>().unwrap()).chunks_exact(5).map(|c| [c[0], c[1], c[2], c[3], c[4]]).collect(); }
            res.push(o);
        }
        res
    }
    fn event(&self, kind: &str, json: &str) {
        self.obj.call_method1("event", (kind, json)).expect("host.event");
    }
}

/// skinned(V bytes, F bytes, seed, tone, ao_strength, n_rays, noise) -> colours (f64 bytes).
#[pyfunction]
fn skinned<'py>(py: Python<'py>, v: &[u8], f: &[u8], seed: u64, tone: usize, ao_strength: f64, n_rays: usize, noise: bool) -> Bound<'py, PyBytes> {
    let m = Mesh::from_flat(&f64s(v), &u32s(f));
    let c = crate::skin::skinned(&m, &crate::skin::SkinCfg { seed, tone, ao_strength, n_rays, noise });
    PyBytes::new(py, &bytes_f64(&c.iter().flatten().cloned().collect::<Vec<f64>>()))
}

/// rng_draws(seed, kind, n): numpy-compatible draws, for the parity test.
#[pyfunction]
fn rng_draws(seed: u64, kind: &str, n: usize) -> Vec<f64> {
    let mut r = crate::rng::Rng::new(seed);
    (0..n).map(|_| match kind { "normal" => r.normal(), "raw" => r.u64() as f64, _ => r.random() }).collect()
}

/// run(host, V bytes (f64), F bytes (u32), cfg JSON, table, angles, template,
/// canon (f64 bytes)) -> dict(L, R as f64 bytes or None, ear_only, fit_residual, R_head, centre)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run<'py>(host: Bound<'py, PyAny>, v: &[u8], f: &[u8], cfg: &str, table: &[u8], angles: &[u8], template: &[u8], canon: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    let py = host.py();
    let mesh = Mesh::from_flat(&f64s(v), &u32s(f));
    let cfg: crate::pipeline::Cfg = serde_json::from_str(cfg).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let a = assets(&f64s(table), &f64s(angles), &f64s(template), &f64s(canon));
    let h = PyHost { obj: host };
    let mut p = crate::pipeline::Pipeline::new(&h, mesh, cfg, a);
    let out = block_on(p.run());
    let d = PyDict::new(py);
    let pts = |x: &Option<Vec<V3>>| x.as_ref().map(|p| PyBytes::new(py, &bytes_f64(&p.iter().flatten().cloned().collect::<Vec<f64>>())));
    d.set_item("L", pts(&out.left))?;
    d.set_item("R", pts(&out.right))?;
    d.set_item("ear_only", out.ear_only)?;
    d.set_item("fit_residual", out.fit_residual)?;
    d.set_item("R_head", out.r_head.map(|r| r.iter().flatten().cloned().collect::<Vec<f64>>()))?;
    d.set_item("centre", out.centre.map(|c| c.to_vec()))?;
    Ok(d)
}
