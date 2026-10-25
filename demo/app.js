// The demo page: pick or upload a mesh, run ear3d_core (WASM) against
// js/host.js, and draw what the pipeline does from its events. Everything the
// pipeline decides happens in the crate; this file is browser logistics.
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { mergeVertices } from "three/addons/utils/BufferGeometryUtils.js";
import init, { run } from "./pkg/ear3d_core.js";
import { BrowserHost, rgb } from "./js/host.js";

const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const tick = () => new Promise((r) => { const t = setTimeout(r, 30); requestAnimationFrame(() => { clearTimeout(t); setTimeout(r, 0); }); });

// ------------------------------------------------------------------ viewer
const canvas = $("viewer");
const gl = new THREE.WebGLRenderer({ canvas, antialias: true });
gl.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
const cam = new THREE.PerspectiveCamera(35, 1, 1, 5000);
const controls = new OrbitControls(cam, canvas);
controls.enableDamping = true; controls.autoRotate = true; controls.autoRotateSpeed = 1.2;
scene.add(new THREE.HemisphereLight(0xffffff, 0x666660, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(300, 400, 500); scene.add(sun);
cam.add(new THREE.DirectionalLight(0xffffff, 0.6)); scene.add(cam);
const world = new THREE.Group(); scene.add(world);      // the head frame, once the pipeline has one
const body = new THREE.Group(); world.add(body);        // the mesh (mirrored for a right ear-only scan)
const marks = new THREE.Group(); world.add(marks);      // everything in the pipeline's mesh coordinates
let meshObj = null, earFocus = null, this_head = null;
const anims = {};                                        // name -> step(now), returns true when finished
window.__demo = { cam, controls, world, get mesh() { return meshObj; }, get ear() { return earFocus; }, frameAll, THREE };   // for the parity driver

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  gl.setSize(r.width, r.height, false); cam.aspect = r.width / r.height; cam.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas.parentElement);
gl.setAnimationLoop(() => {
  for (const [k, f] of Object.entries(anims)) if (f(performance.now())) delete anims[k];
  controls.update(); gl.setClearColor(new THREE.Color(css("--view-bg")), 1); gl.render(scene, cam);
});

function frame(center, radius, along) {
  const dir = along ? along.clone().normalize() : cam.position.clone().sub(controls.target).normalize();
  if (!isFinite(dir.x) || dir.length() < 0.5) dir.set(0, 0, 1);
  controls.target.copy(center); cam.position.copy(center).add(dir.multiplyScalar(radius * 3.2));
  cam.near = radius / 50; cam.far = radius * 50; cam.updateProjectionMatrix();
}
function frameAll() {
  const b = new THREE.Box3().setFromObject(body), s = b.getBoundingSphere(new THREE.Sphere());
  frame(s.center, s.radius);
}

function showMesh(V, F) {
  if (meshObj) { body.remove(meshObj); meshObj.geometry.dispose(); }
  marks.clear(); world.matrix.identity(); this_head = null; world.matrixAutoUpdate = false; body.scale.set(1, 1, 1);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(V), 3));
  g.setIndex(new THREE.BufferAttribute(F, 1)); g.computeVertexNormals();
  meshObj = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: 0xd8d4cc, roughness: 0.85, side: THREE.DoubleSide }));
  body.add(meshObj);
  const b = new THREE.Box3().setFromBufferAttribute(g.attributes.position), c = b.getCenter(new THREE.Vector3());
  world.matrix.makeTranslation(-c.x, -c.y, -c.z); world.updateMatrixWorld(true);
  earFocus = null; $("frameEar").disabled = true;
  cam.position.set(0, 0, 1); controls.target.set(0, 0, 0); frameAll();
}

function colourMesh(C) {
  if (!meshObj) return;
  // the skin is dark linear albedo that the pipeline's renderer lifts with its
  // tone curve; for the viewer, scale it so the mean lands at a mid skin tone
  let mean = 0; for (let i = 0; i < C.length; i++) mean += C[i]; mean /= C.length;
  const k = 0.45 / Math.max(mean, 1e-3), c = new Float32Array(C.length);
  for (let i = 0; i < C.length; i++) c[i] = Math.min(1, C[i] * k);
  meshObj.geometry.setAttribute("color", new THREE.BufferAttribute(c, 3));
  meshObj.material.vertexColors = true; meshObj.material.color.set(0xffffff); meshObj.material.needsUpdate = true;
}

// head event: show the mesh in the pipeline's head frame (y up, face +z)
function toHeadFrame(R, c0, mirror) {
  const Q = R, m = new THREE.Matrix4().set(Q[0][0], Q[0][1], Q[0][2], 0, Q[1][0], Q[1][1], Q[1][2], 0, Q[2][0], Q[2][1], Q[2][2], 0, 0, 0, 0, 1);
  m.multiply(new THREE.Matrix4().makeTranslation(-c0[0], -c0[1], -c0[2]));
  const from = world.matrix.clone(), to = m, t0 = performance.now();
  body.scale.set(1, 1, mirror ? -1 : 1);              // ear-only right ear: mesh 1 is the scan mirrored about z
  const pf = new THREE.Vector3(), qf = new THREE.Quaternion(), sf = new THREE.Vector3(), pt = new THREE.Vector3(), qt = new THREE.Quaternion(), st = new THREE.Vector3();
  from.decompose(pf, qf, sf); to.decompose(pt, qt, st);
  this_head = { centre: new THREE.Vector3(...c0).applyMatrix4(m) };
  const step = (now) => {
    const a = m.determinant() < 0 ? 1 : Math.min(1, (now - t0) / 900), e = a * a * (3 - 2 * a);   // a reflection cannot be slerped
    if (a >= 1) world.matrix.copy(m);
    else world.matrix.compose(pf.clone().lerp(pt, e), qf.clone().slerp(qt, e), new THREE.Vector3(1, 1, 1));
    world.updateMatrixWorld(true);
    if (a >= 1) frameAll();
    return a >= 1;
  };
  anims.head = step; step.finish = () => { step(Infinity); delete anims.head; };
}

const sphereGeo = new THREE.SphereGeometry(1, 12, 8);
function dots(P, color, r) {
  const m = new THREE.InstancedMesh(sphereGeo, new THREE.MeshStandardMaterial({ color, roughness: 0.4 }), P.length);
  P.forEach((p, i) => m.setMatrixAt(i, new THREE.Matrix4().compose(new THREE.Vector3(...p), new THREE.Quaternion(), new THREE.Vector3(r, r, r))));
  marks.add(m); return m;
}
const valid = (p) => p && (p[0] !== 0 || p[1] !== 0 || p[2] !== 0);

// ------------------------------------------------------------------ stages
const STAGES_HEAD = [["load", "Mesh loaded"], ["skin", "Synthetic skin"], ["face", "Find the face (FaceMesh)"], ["pose", "Head pose"],
  ["earL", "Left ear"], ["earR", "Right ear"]];
const STAGES_EAR = [["load", "Mesh loaded"], ["orient", "Template fit: pose and side"], ["skin", "Synthetic skin"], ["ear", "Landmark the ear"]];
let stageT = {}, stageList = [];
function setStages(list) {
  stageList = list; stageT = {};
  $("stages").innerHTML = list.map(([k, t]) => `<li id="st_${k}">${t}<small></small></li>`).join("");
}
function stage(k, state, note) {
  const li = $(`st_${k}`); if (!li) return;
  if (state === "run") { stageT[k] = performance.now(); li.className = "run"; }
  if (state === "done" && li.className !== "done") {
    li.className = "done";
    if (stageT[k]) li.querySelector("small").textContent = `${((performance.now() - stageT[k]) / 1000).toFixed(1)} s`;
  }
  if (note) li.querySelector("small").textContent = note;
}
const status = (t) => { $("status").textContent = t; };

// ------------------------------------------------------------------ camera panel
const camCtx = $("cam").getContext("2d");
let lastBatch = [];
function showImage(rgba, S, caption) {
  const c = $("cam"); c.width = S; c.height = S;
  camCtx.putImageData(new ImageData(new Uint8ClampedArray(rgba), S, S), 0, 0);
  $("camcap").textContent = caption || "";
}
function drawPoints(ctx, uv, scale, color, r) {
  ctx.fillStyle = color;
  for (const p of uv) if (p && p[0] != null && isFinite(p[0])) { ctx.beginPath(); ctx.arc(p[0] * scale, p[1] * scale, r, 0, 7); ctx.fill(); }
}
function fillStrip(images, uv, S) {
  const strip = $("strip"); strip.innerHTML = "";
  images.forEach((rgba, i) => {
    const c = document.createElement("canvas"); c.width = 120; c.height = 120;
    const t = document.createElement("canvas"); t.width = S; t.height = S;
    t.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(rgba), S, S), 0, 0);
    const x = c.getContext("2d"); x.drawImage(t, 0, 0, 120, 120);
    if (uv[i]) drawPoints(x, uv[i], 120 / S, css("--accent"), 1.6);
    c.title = `view ${i + 1}`;
    c.onclick = () => { showImage(rgba, S, `view ${i + 1} of ${images.length}, with the 2D landmarks`); drawPoints(camCtx, uv[i] || [], 1, css("--accent"), 4); };
    strip.appendChild(c);
  });
}

// the multiview pass renders assets.angles in order; the (0, 0) one faces the ear
function faceOnView() {
  const a = assets.angles; let best = 0;
  for (let i = 0; i < a.length / 2; i++) if (Math.hypot(a[2 * i], a[2 * i + 1]) < Math.hypot(a[2 * best], a[2 * best + 1])) best = i;
  return best;
}

// ------------------------------------------------------------------ host
let host, assets, current = null, running = false;
class DemoHost {
  constructor(h, earOnly) { this.h = h; this.mirror = false; this.left = true; this.earOnly = earOnly; this.skins = 0; }
  async setMesh(id, V, F) {
    this.h.setMesh(id, V, F);
    if (id === (this.earOnly ? 1 : 0)) stage("skin", "run");
    await tick();
  }
  async setColors(id, mesh, C) {
    this.h.setColors(id, mesh, C);
    if (this.skins++ === 0) colourMesh(C);              // the face skin (head) or the ear's (ear only)
    if (this.earOnly || this.skins === 2) { stage("skin", "done"); if (this.earOnly) stage("ear", "run"); }
    await tick();
  }
  async render(q) {
    const S = q.size, out = [], keep = [];
    for (const v of q.views) {
      const { rgba, depth } = this.h.renderer.render(q.mesh, q.colors, v, q.f, q.c, q.depth);
      const o = {};
      if (q.depth) o.depth = depth;
      if (q.image) { o.image = rgb(rgba, S); keep.push(rgba); }
      if (q.detect === "face") o.face = this.h.facePoints(rgba, S);
      out.push(o);
      const what = q.detect === "face" ? (o.face ? "FaceMesh found a face" : "looking for a face") : q.image ? "finding the ear and its landmarks" : "depth";
      showImage(rgba, S, what);
      if (o.face) drawPoints(camCtx, Array.from({ length: o.face.length / 2 }, (_, i) => [o.face[2 * i], o.face[2 * i + 1]]), 1, css("--ok"), 1.5);
      await tick();
    }
    if (keep.length > 1) lastBatch = keep;
    return out;
  }
  infer(model, x) { return this.h.infer(model, x); }
  event(kind, payload) {
    const p = JSON.parse(payload);
    if (kind === "log") { console.log(p); return; }
    if (kind === "stage") {
      if (p === "face") stage("face", "run");
      if (p === "ear-only") stage("orient", "run");
      if (p === "ear L" || p === "ear R") {
        this.left = p === "ear L";
        if (!this.earOnly) { stage("pose", "done"); stage(this.left ? "earL" : "earR", "run"); }
        status(this.earOnly ? "Rendering the ear from 25 views" : `${this.left ? "Left" : "Right"} ear: locating it`);
      }
    } else if (kind === "face_candidates") {
      stage("face", "done", `${p.length} candidate${p.length === 1 ? "" : "s"}`); stage("pose", "run");
      status("Face found; fixing the head pose");
    } else if (kind === "head") {
      window.__demo.head = p;
      this.mirror = !!p.mirror;
      if (this.earOnly) stage("orient", "done", `${p.mirror ? "right" : "left"} ear`);
      toHeadFrame(p.R, p.centre, this.mirror);
    } else if (kind === "ear_box") {
      const c = p.centre, r = Math.max(8, (p.box_mm || 60) / 2);
      if (anims.head) anims.head.finish();               // the ear is placed in the final head frame
      earFocus = { c: new THREE.Vector3(...c), r, left: this.earOnly || p.left, earOnly: this.earOnly }; $("frameEar").disabled = false;
      controls.autoRotate = false; $("spin").className = ""; focusEar();
      status(`${p.left ? "Left" : "Right"} ear found; rendering 25 views around it`);
    } else if (kind === "views") {
      const pts = [], lines = [];
      p.model.forEach(([R, C, s], i) => {
        const E = p.E[i], t = [E[3], E[7], E[11]];
        const y = [0, 1, 2].map((k) => -(E[k] * t[0] + E[4 + k] * t[1] + E[8 + k] * t[2]));   // -R^T t
        const w = [0, 1, 2].map((k) => s * (y[0] * R[0][k] + y[1] * R[1][k] + y[2] * R[2][k]) + C[k]);
        pts.push(w);
      });
      dots(pts, css("--ray"), 1.6);
      if (earFocus) {
        const g = new THREE.BufferGeometry().setFromPoints(pts.flatMap((w) => [new THREE.Vector3(...w), earFocus.c]));
        const l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: css("--ray"), transparent: true, opacity: 0.18 }));
        marks.add(l);
      }
      const S = lastBatch.length ? Math.round(Math.sqrt(lastBatch[0].length / 4)) : 600;
      fillStrip(lastBatch, p.uv, S);
      this.faceOn = { rgba: lastBatch[faceOnView()], uv: p.uv[faceOnView()], S, left: p.left };
      status("2D landmarks on every view; triangulating the rays");
    } else if (kind === "triangulated") {
      this.tri = p.filter(valid);
      this.triMesh = dots(this.tri, css("--accent"), 0.55);
      status("Triangulated; snapping onto the surface");
    } else if (kind === "snapped") {
      const to = p, from = this.tri || [], idx = p.map((q, i) => i).filter((i) => valid(p[i]));
      if (this.earOnly && idx.length) {                  // no ear-box event: centre on the landmarks
        const c = new THREE.Vector3(); idx.forEach((i) => c.add(new THREE.Vector3(...p[i]))); c.divideScalar(idx.length);
        earFocus = { c, r: 32, left: true, earOnly: true }; $("frameEar").disabled = false;
        controls.autoRotate = false; $("spin").className = "";
      }
      const m = dots(idx.map((i) => to[i]), css("--ok"), 0.6), t0 = performance.now();
      const fromAll = idx.map((i, k) => from[k] || to[i]);
      anims.points = (now) => {
        const a = Math.min(1, (now - t0) / 900);
        idx.forEach((i, k) => {
          const v = new THREE.Vector3(...fromAll[k]).lerp(new THREE.Vector3(...to[i]), a);
          m.setMatrixAt(k, new THREE.Matrix4().compose(v, new THREE.Quaternion(), new THREE.Vector3(0.6, 0.6, 0.6)));
        });
        m.instanceMatrix.needsUpdate = true;
        return a >= 1;
      };
      if (this.faceOn) { const f = this.faceOn; showImage(f.rgba, f.S, `${this.earOnly ? "The ear" : f.left ? "Left ear" : "Right ear"}, face-on view, with its 2D landmarks`); drawPoints(camCtx, f.uv, 1, css("--accent"), 4); }
      if (this.earOnly) stage("ear", "done"); else stage(this.left ? "earL" : "earR", "done", `${idx.length}/55 points`);
      focusEar();
    }
  }
}

// ------------------------------------------------------------------ meshes
let index, tab = "heads";
const HEAD_MIN_MM = 100;      // ear_only.HEAD_MIN_MM: the crate's flow gate, mirrored only to pick the stage list
function spanOf(V) { let m = 0; for (let k = 0; k < 3; k++) { let lo = Infinity, hi = -Infinity; for (let i = k; i < V.length; i += 3) { lo = Math.min(lo, V[i]); hi = Math.max(hi, V[i]); } m = Math.max(m, hi - lo); } return m; }
async function loadBin(file) {
  const b = await (await fetch(`meshes/${file}`)).arrayBuffer();
  const [nv, nf] = new Uint32Array(b, 0, 2);
  return { V: Float64Array.from(new Float32Array(b, 8, nv * 3)), F: new Uint32Array(b, 8 + nv * 12, nf * 3).slice() };
}
function renderList() {
  $("tabHeads").className = tab === "heads" ? "on" : ""; $("tabEars").className = tab === "ears" ? "on" : "";
  $("list").innerHTML = index[tab].map((m) => `<li data-id="${m.id}" class="${current && current.id === m.id ? "sel" : ""}">${m.label}<span>${Math.round(m.span_mm)} mm</span></li>`).join("");
  for (const li of $("list").children) li.onclick = () => pick(index[tab].find((m) => m.id === li.dataset.id));
}
async function pick(m) {
  if (running) return;
  status(`Loading ${m.label}…`);
  const { V, F } = await loadBin(m.file);
  setCurrent({ id: m.id, label: m.label, V, F });
}
function setCurrent(c) {
  current = c; renderList(); showMesh(c.V, c.F);
  setStages(spanOf(c.V) < HEAD_MIN_MM ? STAGES_EAR : STAGES_HEAD); stage("load", "done", `${(c.V.length / 3).toLocaleString()} vertices`);
  $("result").innerHTML = `<span style="color:var(--mute)">Not run yet.</span>`; $("strip").innerHTML = "";
  status(`${c.label}: ready. Press Run.`); $("run").disabled = !host; if (host) $("run").textContent = "Run";
}

$("file").onchange = async (e) => {
  const f = e.target.files[0]; if (!f || running) return;
  try {
    const ext = f.name.split(".").pop().toLowerCase(), buf = await f.arrayBuffer();
    let g;
    if (ext === "ply") g = new PLYLoader().parse(buf);
    else if (ext === "stl") g = new STLLoader().parse(buf);
    else if (ext === "obj") {
      const o = new OBJLoader().parse(new TextDecoder().decode(buf)); o.traverse((x) => { if (x.isMesh && !g) g = x.geometry; });
    } else throw new Error("use .ply, .obj or .stl");
    if (!g) throw new Error("no mesh in the file");
    for (const k of Object.keys(g.attributes)) if (k !== "position") g.deleteAttribute(k);
    g = mergeVertices(g, 1e-6);
    if (!g.index || g.index.count === 0) throw new Error("the file has no faces (a point cloud?): the pipeline needs a surface");
    let V = Float64Array.from(g.attributes.position.array);
    const span = () => { let m = 0; for (let k = 0; k < 3; k++) { let lo = Infinity, hi = -Infinity; for (let i = k; i < V.length; i += 3) { lo = Math.min(lo, V[i]); hi = Math.max(hi, V[i]); } m = Math.max(m, hi - lo); } return m; };
    let note = "";
    if (span() < 1) { V = V.map((x) => x * 1000); note = " (metres → mm)"; }
    const s = span();
    if (!(s >= 25 && s <= 600)) throw new Error(`it spans ${s.toFixed(1)} units; the pipeline needs millimetres: a head (~150-350) or an ear (~40-90)`);
    setCurrent({ id: "upload", label: f.name + note, V, F: new Uint32Array(g.index.array) });
  } catch (err) { status(`Could not use ${f.name}: ${err.message}`); $("status").className = "err"; setTimeout(() => ($("status").className = ""), 4000); }
  e.target.value = "";
};

$("run").onclick = async () => {
  if (!current || running) return;
  running = true; $("run").disabled = true; $("run").textContent = "Running…";
  const earOnly = spanOf(current.V) < HEAD_MIN_MM;
  showMesh(current.V, current.F); setStages(earOnly ? STAGES_EAR : STAGES_HEAD); stage("load", "done");
  $("strip").innerHTML = ""; $("result").innerHTML = `<span style="color:var(--mute)">Running…</span>`;
  status(earOnly ? "An ear on its own: fitting the mean ear for its pose and side…" : "Skinning the mesh (synthetic skin and ambient occlusion)…");
  const dh = new DemoHost(host, earOnly); const t0 = performance.now();
  try {
    await tick();
    const o = await run(dh, current.V, current.F, assets.cfg, assets.table, assets.angles, assets.template, assets.canon, assets.anchors);
    showResult(o, (performance.now() - t0) / 1000);
  } catch (err) {
    console.error(err); status(`The pipeline failed: ${err.message || err}`);
    $("result").innerHTML = `<span class="err">Failed: ${String(err.message || err)}</span>`;
  }
  running = false; $("run").disabled = false; $("run").textContent = "Run again";
};

function showResult(o, secs) {
  const n = (P) => (P ? P.length / 3 : 0);
  if (!o.L && !o.R) {
    status("No face found: the pipeline gives up rather than guess."); stageList.forEach(([k]) => { if ($(`st_${k}`).className === "run") stage(k, "done", "no face"); });
    $("result").innerHTML = `<dl><dt>Result</dt><dd>no face found</dd><dt>Time</dt><dd>${secs.toFixed(1)} s</dd></dl>`; return;
  }
  status(`Done in ${secs.toFixed(0)} s.`);
  const rows = [["Scan", o.earOnly ? `ear only (${o.L ? "left" : "right"})` : "head"],
    ["Left ear", o.L ? `${n(o.L)} landmarks` : "—"], ["Right ear", o.R ? `${n(o.R)} landmarks` : "—"],
    ...(o.earOnly ? [["Template fit", `residual ${o.fitResidual.toFixed(2)}`]] : []), ["Time", `${secs.toFixed(1)} s, all in this tab`]];
  $("result").innerHTML = `<dl>${rows.map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join("")}</dl><button id="dl">Download landmarks (JSON)</button>`;
  $("dl").onclick = () => {
    const pts = (P) => (P ? Array.from({ length: P.length / 3 }, (_, i) => [P[3 * i], P[3 * i + 1], P[3 * i + 2]]) : null);
    const blob = new Blob([JSON.stringify({ mesh: current.label, units: "mm, the input mesh's own frame", left: pts(o.L), right: pts(o.R) }, null, 1)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `${current.id}_ear_landmarks.json`; a.click();
  };
}

$("tabHeads").onclick = () => { tab = "heads"; renderList(); };
$("tabEars").onclick = () => { tab = "ears"; renderList(); };
$("spin").onclick = () => { controls.autoRotate = !controls.autoRotate; $("spin").className = controls.autoRotate ? "on" : ""; };
$("frameAll").onclick = frameAll;
$("frameEar").onclick = focusEar;
// the head frame has y up, the face along +z and the left ear at +x: look at an
// ear from its own side, a little from the front
// look at an ear from outside: along the head centre -> ear direction (a head),
// or the side axis of the frame (an ear on its own, where the centre IS the ear)
function focusEar() {
  if (!earFocus) return;
  const c = earFocus.c.clone().applyMatrix4(world.matrix);
  let out = c.clone().sub(this_head ? this_head.centre : new THREE.Vector3());
  out.y = 0;                                           // level: the head frame has y up
  if (earFocus.earOnly || out.length() < 20) out = new THREE.Vector3(1, 0, 0);
  out.normalize().add(new THREE.Vector3(0, 0.1, 0.3));
  frame(c, earFocus.r * 1.5, out);
}

// ------------------------------------------------------------------ start
(async () => {
  index = await (await fetch("meshes/index.json")).json();
  $("credits").textContent = `Heads: ${index.credits.heads}. Ears: ${index.credits.ears}. 2D landmarker builds on SaifulULL's U-Net (CC BY 4.0). Ear boxes: BlazeEar. Face: MediaPipe FaceLandmarker.`;
  renderList();
  const f64 = async (n) => new Float64Array(await (await fetch(`models/${n}.bin`)).arrayBuffer());
  const [, h, cfg, table, angles, template, canon, anchors] = await Promise.all([init(), BrowserHost.create("models/", 600),
    fetch("models/cfg.json").then((r) => r.text()), f64("table"), f64("angles"), f64("template"), f64("canon"), f64("blazeear_anchors")]);
  host = h; assets = { cfg, table, angles, template, canon, anchors };
  $("run").textContent = current ? "Run" : "Pick a mesh"; $("run").disabled = !current;
  await pick(index.heads[0]);
})().catch((e) => { console.error(e); status(`Could not start: ${e.message || e}`); });
