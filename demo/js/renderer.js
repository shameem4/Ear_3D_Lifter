// The browser renderer: draws what Open3D/Filament draws for the pipeline,
// closely enough that FaceMesh, BlazeEar and r2j see the same picture.
//
// Filament is not available here in the version Open3D uses, so its output is
// reproduced by a fitted model (shading.json, fitted to 530k Filament pixels of
// spheres under six suns and eight albedos, rmse 2.5 grey levels):
//   L = albedo * (ks * max(n.l, 0) * shadow + k * SH(n)) + GGX(n, v, l) + kib * SH(r)
//   pixel = LUT(L)
// SH is the irradiance of Open3D's default_ibl.ktx (its own coefficients);
// albedo is the raw vertex colour (Filament takes it as linear: rmse 2.5 vs 10
// read as sRGB); the LUT is Filament's tone curve and output encoding together.
// The sun casts shadows, as it does in Filament. Background 0.5 -> 127.
//
// A view is the pipeline's: the host mesh moved by x = (p - C) R^T / s, seen
// through the OpenCV extrinsic E with intrinsics (f, c) at size x size.

import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

const SH = [
  [0.711666, 0.747167, 0.767062], [0.455257, 0.48366, 0.504973], [0.243303, 0.26168, 0.279423],
  [-0.157977, -0.169309, -0.18347], [-0.138656, -0.146425, -0.158374], [0.283602, 0.302589, 0.320814],
  [-0.000149071, -0.000875601, -0.00169063], [0.000923434, 0.00044748, -0.00646363], [-0.00313725, -0.00498679, -0.00790974],
];

const COMMON = /* glsl */ `
uniform mat3 uR; uniform vec3 uC; uniform float uS;
vec3 moved(vec3 p) { return uR * (p - uC) / uS; }
`;

const VERT = COMMON + /* glsl */ `
attribute vec3 color;
uniform mat4 uView; uniform mat4 uProj; uniform mat4 uLight;
varying vec3 vN; varying vec3 vP; varying vec3 vA; varying vec4 vL;
void main() {
  vec3 x = moved(position);
  vN = uR * normal; vP = x; vA = color;
  vL = uLight * vec4(x, 1.0);
  gl_Position = uProj * uView * vec4(x, 1.0);
}`;

const FRAG = /* glsl */ `
precision highp float;
uniform vec3 uSun; uniform vec3 uEye; uniform vec3 uSH[9];
uniform float ks, k, sp, rough, kib, lutMax;
uniform sampler2D uLut; uniform sampler2D uShadow; uniform float uShadowTexel;
varying vec3 vN; varying vec3 vP; varying vec3 vA; varying vec4 vL;
vec3 sh(vec3 n) {
  float x = n.x, y = n.y, z = n.z;
  vec3 r = uSH[0] + uSH[1]*y + uSH[2]*z + uSH[3]*x + uSH[4]*(y*x) + uSH[5]*(y*z) + uSH[6]*(3.0*z*z-1.0) + uSH[7]*(z*x) + uSH[8]*(x*x-y*y);
  return max(r, 0.0);
}
float shadow() {
  vec3 q = vL.xyz / vL.w * 0.5 + 0.5;
  if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) return 1.0;
  float lit = 0.0;
  for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
    float d = texture2D(uShadow, q.xy + vec2(float(i), float(j)) * uShadowTexel).r;
    lit += (q.z - 0.0015 <= d) ? 1.0 : 0.0;
  }
  return lit / 9.0;
}
void main() {
  vec3 n = normalize(vN);
  vec3 v = normalize(uEye - vP);
  if (dot(n, v) < 0.0) n = -n;                 // two-sided, as Filament draws back faces
  vec3 l = -normalize(uSun);
  float ndl = max(dot(n, l), 0.0), ndv = clamp(dot(n, v), 1e-4, 1.0);
  vec3 h = normalize(l + v);
  float ndh = clamp(dot(n, h), 0.0, 1.0);
  float al = max(rough, 0.05); al *= al;
  float D = al*al / (3.14159265 * pow(ndh*ndh*(al*al-1.0)+1.0, 2.0));
  float V = 0.5 / (ndl*sqrt(ndv*ndv*(1.0-al*al)+al*al) + ndv*sqrt(ndl*ndl*(1.0-al*al)+al*al) + 1e-9);
  float F = 0.04 + 0.96 * pow(1.0 - clamp(dot(l, h), 0.0, 1.0), 5.0);
  float s = shadow();
  vec3 r = reflect(-v, n);
  vec3 L = vA * (ks * ndl * s + k * sh(n)) + vec3(sp * D * V * F * ndl * s) + kib * sh(r);
  vec3 t = L / lutMax;
  gl_FragColor = vec4(texture2D(uLut, vec2(t.r, 0.5)).r, texture2D(uLut, vec2(t.g, 0.5)).r, texture2D(uLut, vec2(t.b, 0.5)).r, 1.0);
}`;

const DEPTH_FRAG = /* glsl */ `
precision highp float;
varying float vZ;
void main() { gl_FragColor = vec4(vZ, 0.0, 0.0, 1.0); }`;
const DEPTH_VERT = COMMON + /* glsl */ `
uniform mat4 uView; uniform mat4 uProj; uniform mat4 uE;
varying float vZ;
void main() {
  vec3 x = moved(position);
  vZ = (uE * vec4(x, 1.0)).z;                    // OpenCV view z: distance along the axis
  gl_Position = uProj * uView * vec4(x, 1.0);
}`;
const SHADOW_VERT = COMMON + /* glsl */ `
uniform mat4 uLight;
void main() { gl_Position = uLight * vec4(moved(position), 1.0); }`;
const SHADOW_FRAG = /* glsl */ `
precision highp float;
void main() { gl_FragColor = vec4(gl_FragCoord.z, 0.0, 0.0, 1.0); }`;

function mat4RowMajor(a) { const m = new THREE.Matrix4(); m.set(...a); return m; }

export class PipelineRenderer {
  constructor(shading, size = 600) {
    this.size = size;
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    this.gl = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
    this.gl.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.gl.toneMapping = THREE.NoToneMapping;
    const lut = new Float32Array(shading.lut.map((v) => v / 255));
    const lutTex = new THREE.DataTexture(lut, lut.length, 1, THREE.RedFormat, THREE.FloatType);
    lutTex.magFilter = lutTex.minFilter = THREE.LinearFilter; lutTex.needsUpdate = true;
    this.shadowSize = 2048;
    this.shadowRT = new THREE.WebGLRenderTarget(this.shadowSize, this.shadowSize, { type: THREE.FloatType, format: THREE.RGBAFormat });
    this.colorRT = new THREE.WebGLRenderTarget(size, size, { type: THREE.UnsignedByteType });
    this.depthRT = new THREE.WebGLRenderTarget(size, size, { type: THREE.FloatType });
    const p = shading.params;
    this.uni = {
      uR: { value: new THREE.Matrix3() }, uC: { value: new THREE.Vector3() }, uS: { value: 1 },
      uView: { value: new THREE.Matrix4() }, uProj: { value: new THREE.Matrix4() }, uLight: { value: new THREE.Matrix4() },
      uE: { value: new THREE.Matrix4() },
      uSun: { value: new THREE.Vector3() }, uEye: { value: new THREE.Vector3() },
      uSH: { value: SH.map((c) => new THREE.Vector3(...c)) },
      ks: { value: p.ks }, k: { value: p.k }, sp: { value: p.sp }, rough: { value: p.rough }, kib: { value: p.kib },
      lutMax: { value: shading.lut_max }, uLut: { value: lutTex },
      uShadow: { value: this.shadowRT.texture }, uShadowTexel: { value: 1 / this.shadowSize },
    };
    const mk = (v, f, side = THREE.DoubleSide) => new THREE.ShaderMaterial({ vertexShader: v, fragmentShader: f, uniforms: this.uni, side });
    this.matColor = mk(VERT, FRAG);
    this.matDepth = mk(DEPTH_VERT, DEPTH_FRAG);
    this.matShadow = mk(SHADOW_VERT, SHADOW_FRAG);
    this.scene = new THREE.Scene();
    this.cam = new THREE.Camera();                   // matrices come from uniforms
    this.meshes = {};
  }

  /** Host mesh: positions (Float64Array n*3), faces (Uint32Array). Normals as open3d computes them. */
  setMesh(id, V, F) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(V), 3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(F), 1));
    const n = new Float32Array(V.length);
    for (let t = 0; t < F.length; t += 3) {
      const a = 3 * F[t], b = 3 * F[t + 1], c = 3 * F[t + 2];
      const e1 = [V[b] - V[a], V[b + 1] - V[a + 1], V[b + 2] - V[a + 2]], e2 = [V[c] - V[a], V[c + 1] - V[a + 1], V[c + 2] - V[a + 2]];
      const x = e1[1] * e2[2] - e1[2] * e2[1], y = e1[2] * e2[0] - e1[0] * e2[2], z = e1[0] * e2[1] - e1[1] * e2[0];
      for (const i of [a, b, c]) { n[i] += x; n[i + 1] += y; n[i + 2] += z; }
    }
    for (let i = 0; i < n.length; i += 3) {
      const l = Math.hypot(n[i], n[i + 1], n[i + 2]);
      if (l > 0) { n[i] /= l; n[i + 1] /= l; n[i + 2] /= l; } else { n[i + 2] = 1; }
    }
    g.setAttribute("normal", new THREE.BufferAttribute(n, 3));
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < V.length; i += 3) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], V[i + k]); hi[k] = Math.max(hi[k], V[i + k]); }
    this.meshes[id] = { g, lo, hi, colors: {} };
  }

  setColors(id, mesh, C) { this.meshes[mesh].colors[id] = new THREE.BufferAttribute(new Float32Array(C), 3); }

  /** One view -> { rgba: Uint8Array (top row first), depth: Float32Array (inf = background) }. */
  render(meshId, colorId, view, f, c, wantDepth) {
    const M = this.meshes[meshId], S = this.size;
    M.g.setAttribute("color", M.colors[colorId]);
    const R = view.R, Cc = view.C;
    this.uni.uR.value.set(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8]);
    this.uni.uC.value.set(Cc[0], Cc[1], Cc[2]); this.uni.uS.value = view.s;
    const E = mat4RowMajor(view.E);
    this.uni.uE.value.copy(E);
    const flip = new THREE.Matrix4().makeScale(1, -1, -1);
    this.uni.uView.value.copy(flip).multiply(E);        // OpenCV -> OpenGL camera
    // the moved mesh's bounding sphere sets near/far and the shadow frustum
    const ctr = [0, 1, 2].map((k) => 0.5 * (M.lo[k] + M.hi[k])), rad = 0.5 * Math.hypot(M.hi[0] - M.lo[0], M.hi[1] - M.lo[1], M.hi[2] - M.lo[2]) / view.s;
    const cm = new THREE.Vector3(...[0, 1, 2].map((k) => ctr[k] - Cc[k])).applyMatrix3(this.uni.uR.value).multiplyScalar(1 / view.s);
    const camZ = cm.clone().applyMatrix4(E).z;
    const near = Math.max(1e-3, camZ - 1.5 * rad), far = camZ + 1.5 * rad;
    const P = new THREE.Matrix4();
    P.set(2 * f / S, 0, 1 - 2 * (c + 0.5) / S, 0,
          0, 2 * f / S, 2 * (c + 0.5) / S - 1, 0,
          0, 0, -(far + near) / (far - near), -2 * far * near / (far - near),
          0, 0, -1, 0);
    this.uni.uProj.value.copy(P);
    const Ei = E.clone().invert();
    this.uni.uEye.value.set(0, 0, 0).applyMatrix4(Ei);
    const sun = new THREE.Vector3(...view.sun).normalize();
    this.uni.uSun.value.copy(sun);
    // sun shadow map: orthographic, looking along the sun at the moved mesh
    const up = Math.abs(sun.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
    const lv = new THREE.Matrix4().lookAt(cm.clone().sub(sun.clone().multiplyScalar(2 * rad)), cm, up);
    lv.setPosition(cm.clone().sub(sun.clone().multiplyScalar(2 * rad)));
    const lview = lv.clone().invert();
    const lproj = new THREE.Matrix4().makeOrthographic(-rad, rad, rad, -rad, 0.01, 4 * rad);
    this.uni.uLight.value.copy(lproj).multiply(lview);
    const mesh = new THREE.Mesh(M.g, this.matShadow);
    mesh.frustumCulled = false;
    this.scene.clear(); this.scene.add(mesh);
    this.gl.setRenderTarget(this.shadowRT); this.gl.setClearColor(0xffffff, 1); this.gl.clear(); this.gl.render(this.scene, this.cam);
    mesh.material = this.matColor;
    this.gl.setRenderTarget(this.colorRT); this.gl.setClearColor(new THREE.Color(0.5, 0.5, 0.5), 1); this.gl.clear(); this.gl.render(this.scene, this.cam);
    const raw = new Uint8Array(S * S * 4); this.gl.readRenderTargetPixels(this.colorRT, 0, 0, S, S, raw);
    const rgba = new Uint8Array(S * S * 4);
    for (let y = 0; y < S; y++) rgba.set(raw.subarray((S - 1 - y) * S * 4, (S - y) * S * 4), y * S * 4);
    let depth = null;
    if (wantDepth) {
      mesh.material = this.matDepth;
      this.gl.setRenderTarget(this.depthRT); this.gl.setClearColor(0x000000, 0); this.gl.clear(); this.gl.render(this.scene, this.cam);
      const d = new Float32Array(S * S * 4); this.gl.readRenderTargetPixels(this.depthRT, 0, 0, S, S, d);
      depth = new Float32Array(S * S);
      for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) { const z = d[((S - 1 - y) * S + x) * 4]; depth[y * S + x] = z > 0 ? z : Infinity; }
    }
    this.gl.setRenderTarget(null);
    return { rgba, depth };
  }
}
