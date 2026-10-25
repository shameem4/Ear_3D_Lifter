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

// ------------------------------------------------------------- the pipeline

use crate::exec::assets;
use crate::host::{Detect, Host, RenderReq, ViewOut};
use crate::mesh::{Mesh, V3};
use js_sys::{Array, Float32Array, Float64Array, Function, Object, Promise, Reflect, Uint32Array};
use wasm_bindgen_futures::JsFuture;

/// The browser host: an object with async setMesh(id, Float64Array V,
/// Uint32Array F), async setColors(id, mesh, Float64Array C), async
/// render(request) -> [{depth, face, boxes, landmarks}], and event(kind, json).
struct JsHost { obj: JsValue }

fn call(obj: &JsValue, name: &str, args: &[JsValue]) -> JsValue {
    let f: Function = Reflect::get(obj, &name.into()).expect(name).into();
    let a = Array::new();
    for x in args { a.push(x); }
    f.apply(obj, &a).unwrap_or_else(|e| panic!("host.{name} threw: {e:?}"))
}

async fn wait(v: JsValue) -> JsValue {
    if v.is_instance_of::<Promise>() { JsFuture::from(Promise::from(v)).await.unwrap_or_else(|e| panic!("host promise rejected: {e:?}")) } else { v }
}

fn set(o: &Object, k: &str, v: JsValue) { Reflect::set(o, &k.into(), &v).unwrap(); }
fn f64a(v: &[f64]) -> JsValue { Float64Array::from(v).into() }
fn get_f64(o: &JsValue, k: &str) -> Option<Vec<f64>> {
    let v = Reflect::get(o, &k.into()).ok()?;
    if v.is_null() || v.is_undefined() { return None; }
    Some(Float64Array::new(&v).to_vec())
}

impl Host for JsHost {
    async fn set_mesh(&self, id: u32, m: &Mesh) {
        wait(call(&self.obj, "setMesh", &[id.into(), f64a(&m.flat_v()), Uint32Array::from(&m.flat_f()[..]).into()])).await;
    }
    async fn set_colors(&self, id: u32, mesh: u32, c: &[V3]) {
        let flat: Vec<f64> = c.iter().flatten().cloned().collect();
        wait(call(&self.obj, "setColors", &[id.into(), mesh.into(), f64a(&flat)])).await;
    }
    async fn render(&self, r: &RenderReq<'_>) -> Vec<ViewOut> {
        let q = Object::new();
        set(&q, "mesh", r.mesh.into());
        set(&q, "colors", r.colors.into());
        set(&q, "size", (r.k.size as u32).into());
        set(&q, "f", r.k.f.into());
        set(&q, "c", r.k.c.into());
        set(&q, "depth", r.depth.into());
        set(&q, "detect", match r.detect { Detect::Nothing => "none", Detect::Face => "face", Detect::EarBoxes => "ear_boxes", Detect::Landmarks => "landmarks" }.into());
        let views = Array::new();
        for v in r.views {
            let o = Object::new();
            set(&o, "E", f64a(&v.e.mat4()));
            set(&o, "R", f64a(&v.r.iter().flatten().cloned().collect::<Vec<f64>>()));
            set(&o, "C", f64a(&v.c));
            set(&o, "s", v.s.into());
            set(&o, "sun", f64a(&v.sun));
            views.push(&o);
        }
        set(&q, "views", views.into());
        let out = Array::from(&wait(call(&self.obj, "render", &[q.into()])).await);
        out.iter().map(|o| {
            let mut v = ViewOut::default();
            if let Ok(d) = Reflect::get(&o, &"depth".into()) { if !d.is_null() && !d.is_undefined() { v.depth = Some(Float32Array::new(&d).to_vec()); } }
            v.face = get_f64(&o, "face").map(|a| a.chunks_exact(2).map(|c| [c[0], c[1]]).collect());
            v.landmarks = get_f64(&o, "landmarks").map(|a| a.chunks_exact(2).map(|c| [c[0], c[1]]).collect());
            v.boxes = get_f64(&o, "boxes").map(|a| a.chunks_exact(5).map(|c| [c[0], c[1], c[2], c[3], c[4]]).collect()).unwrap_or_default();
            v
        }).collect()
    }
    fn event(&self, kind: &str, json: &str) {
        call(&self.obj, "event", &[kind.into(), json.into()]);
    }
}

/// run(host, V, F, cfgJson, table, angles, template, canon) -> Promise of
/// {L, R (Float64Array or null), earOnly, fitResidual, R_head, centre}.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub async fn run(host: JsValue, v: Vec<f64>, f: Vec<u32>, cfg: String, table: Vec<f64>, angles: Vec<f64>, template: Vec<f64>, canon: Vec<f64>) -> Result<JsValue, JsValue> {
    let cfg: crate::pipeline::Cfg = serde_json::from_str(&cfg).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let h = JsHost { obj: host };
    let mut p = crate::pipeline::Pipeline::new(&h, Mesh::from_flat(&v, &f), cfg, assets(&table, &angles, &template, &canon));
    let out = p.run().await;
    let o = Object::new();
    let pts = |x: &Option<Vec<V3>>| x.as_ref().map_or(JsValue::NULL, |p| f64a(&p.iter().flatten().cloned().collect::<Vec<f64>>()));
    set(&o, "L", pts(&out.left));
    set(&o, "R", pts(&out.right));
    set(&o, "earOnly", out.ear_only.into());
    set(&o, "fitResidual", out.fit_residual.into());
    set(&o, "R_head", out.r_head.map_or(JsValue::NULL, |r| f64a(&r.iter().flatten().cloned().collect::<Vec<f64>>())));
    set(&o, "centre", out.centre.map_or(JsValue::NULL, |c| f64a(&c)));
    Ok(o.into())
}

/// skinned(V, F, seed, tone, aoStrength, nRays, noise) -> Float64Array colours.
#[wasm_bindgen]
pub fn skinned(v: Vec<f64>, f: Vec<u32>, seed: u64, tone: usize, ao_strength: f64, n_rays: usize, noise: bool) -> Vec<f64> {
    let c = crate::skin::skinned(&Mesh::from_flat(&v, &f), &crate::skin::SkinCfg { seed, tone, ao_strength, n_rays, noise });
    c.iter().flatten().cloned().collect()
}
