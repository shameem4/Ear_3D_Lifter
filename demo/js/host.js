// The browser host for ear3d_core's pipeline (core/src/pipeline.rs), the twin
// of ear3d/core_host.py: hold meshes and colours, render views (three.js,
// renderer.js), run FaceMesh on them when asked (MediaPipe tasks-vision, the
// same face_landmarker.task the Python host loads), and evaluate the bare 2D
// graphs (onnxruntime-web; pre/post is the crate's, detect.rs).
import { PipelineRenderer } from "./renderer.js";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.wasm.min.mjs";
import { FilesetResolver, FaceLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";
const GRAPHS = { r2j: ["r2j_s42.onnx", 192], blazeear: ["blazeear.onnx", 256] };

export class BrowserHost {
  // models: URL of demo/models/; onEvent(kind, payload) gets the pipeline's events.
  static async create(models, size, onEvent = () => {}) {
    const h = new BrowserHost();
    const base = new URL(models, location.href);
    const shading = await (await fetch(new URL("shading.json", import.meta.url))).json();
    h.renderer = new PipelineRenderer(shading, size);
    h.graphs = {};
    for (const [k, [f]] of Object.entries(GRAPHS))
      h.graphs[k] = await ort.InferenceSession.create(new URL(f, base).href, { executionProviders: ["wasm"] });
    const files = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm");
    h.face = await FaceLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath: new URL("face_landmarker.task", base).href, delegate: "CPU" }, numFaces: 1, runningMode: "IMAGE" });
    h.anchors = new Float64Array(await (await fetch(new URL("blazeear_anchors.bin", base))).arrayBuffer());
    h.onEvent = onEvent;
    h.meshes = {};
    return h;
  }

  setMesh(id, V, F) { this.meshes[id] = true; this.renderer.setMesh(id, V, F); }
  setColors(id, mesh, C) { this.renderer.setColors(id, mesh, C); }

  render(q) {
    const S = q.size, out = [];
    for (const v of q.views) {
      const { rgba, depth } = this.renderer.render(q.mesh, q.colors, v, q.f, q.c, q.depth);
      const o = {};
      if (q.depth) o.depth = depth;
      if (q.image) o.image = rgb(rgba, S);
      if (q.detect === "face") o.face = this.facePoints(rgba, S);
      out.push(o);
    }
    return out;
  }

  facePoints(rgba, S) {
    const res = this.face.detect(new ImageData(new Uint8ClampedArray(rgba.buffer, rgba.byteOffset, S * S * 4), S, S));
    if (!res.faceLandmarks.length) return null;
    const p = res.faceLandmarks[0], uv = new Float64Array(p.length * 2);
    p.forEach((q, i) => { uv[2 * i] = q.x * S; uv[2 * i + 1] = q.y * S; });
    return uv;
  }

  async infer(model, x) {
    const g = this.graphs[model], n = GRAPHS[model][1];
    const r = await g.run({ [g.inputNames[0]]: new ort.Tensor("float32", x, [1, 3, n, n]) });
    return g.outputNames.map((k) => r[k].data);
  }

  event(kind, payload) { this.onEvent(kind, payload); }
}

function rgb(rgba, S) {
  const o = new Uint8Array(S * S * 3);
  for (let i = 0; i < S * S; i++) { o[3 * i] = rgba[4 * i]; o[3 * i + 1] = rgba[4 * i + 1]; o[3 * i + 2] = rgba[4 * i + 2]; }
  return o;
}
