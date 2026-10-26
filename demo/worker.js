// ear3d_core in a Web Worker, so its long steps (skinning a big mesh takes
// seconds) never block the page and the browser never offers to kill it. The
// host it talks to lives on the page (it needs WebGL and MediaPipe): every
// host call is a message, answered by the page with a promise reply.
import init, { run, runFace } from "./pkg/ear3d_core.js";

let next = 0;
const pending = {};
const call = (method, args) => new Promise((resolve) => {
  const id = ++next; pending[id] = resolve;
  postMessage({ type: "call", id, method, args });
});
const host = {
  setMesh: (...a) => call("setMesh", a),
  setColors: (...a) => call("setColors", a),
  render: (q) => call("render", [q]),
  infer: (...a) => call("infer", a),
  event: (kind, payload) => postMessage({ type: "event", kind, payload }),
};

onmessage = async (e) => {
  const m = e.data;
  if (m.type === "reply") { pending[m.id](m.result); delete pending[m.id]; return; }
  if (m.type === "run") {
    try {
      await init();
      if (m.face) {
        const o = await runFace(host, ...m.args);
        postMessage({ type: "done", result: o && { P: o.P, nViews: o.nViews } });
      } else {
        const o = await run(host, ...m.args);
        postMessage({ type: "done", result: { L: o.L, R: o.R, earOnly: o.earOnly, fitResidual: o.fitResidual } });
      }
    } catch (err) { postMessage({ type: "error", message: String(err && err.message || err) }); }
  }
};
