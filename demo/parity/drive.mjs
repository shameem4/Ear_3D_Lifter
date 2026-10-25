// Drive the demo in headless Chrome over CDP: pick a mesh (or upload one),
// press Run, screenshot at each status change, save the console.
//   node demo/parity/drive.mjs http://127.0.0.1:8795/index.html OUT pp16
//   node demo/parity/drive.mjs URL OUT ae012_left | file:/path/to/mesh.ply
import { spawn } from "node:child_process"; import fs from "node:fs";
const [url, out, meshId, W = 1440, H = 900] = process.argv.slice(2);
fs.mkdirSync(out, { recursive: true });
const prof = fs.mkdtempSync((process.env.TMPDIR || "/var/tmp") + "/ear3d_drive_");
const ch = spawn("google-chrome", ["--headless=new", `--user-data-dir=${prof}`, "--use-gl=angle", "--use-angle=gl-egl", "--ignore-gpu-blocklist",
  "--remote-debugging-port=9333", `--window-size=${W},${H}`, "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let tabs; for (let i = 0; i < 50; i++) { try { tabs = await (await fetch("http://127.0.0.1:9333/json")).json(); break; } catch { await sleep(200); } }
const ws = new WebSocket(tabs.find((t) => t.type === "page").webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0; const pend = {}, logs = [];
ws.onmessage = (m) => { const j = JSON.parse(m.data); if (j.id && pend[j.id]) { pend[j.id](j); delete pend[j.id]; }
  if (j.method === "Runtime.consoleAPICalled") logs.push(`[${j.params.type}] ` + j.params.args.map((a) => a.value ?? a.description).join(" "));
  if (j.method === "Runtime.exceptionThrown") logs.push("[exception] " + JSON.stringify(j.params.exceptionDetails).slice(0, 600)); };
const cdp = (method, params = {}) => new Promise((r) => { const i = ++id; pend[i] = r; ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (e) => (await cdp("Runtime.evaluate", { expression: e, awaitPromise: true, returnByValue: true })).result?.result?.value;
const flush = () => fs.writeFileSync(`${out}/console.txt`, logs.join("\n"));
const shot = async (n) => { flush(); const r = await cdp("Page.captureScreenshot", { format: "png" }); fs.writeFileSync(`${out}/${n}.png`, Buffer.from(r.result.data, "base64")); };
await cdp("Runtime.enable"); await cdp("Page.enable");
await cdp("Emulation.setDeviceMetricsOverride", { width: +W, height: +H, deviceScaleFactor: 1, mobile: false });
await cdp("Page.navigate", { url });
for (let i = 0; i < 120; i++) { await sleep(500); if (await ev(`!document.getElementById("run").disabled`)) break; }
if (meshId.startsWith("file:")) {
  await cdp("DOM.enable"); const doc = await cdp("DOM.getDocument");
  const q = await cdp("DOM.querySelector", { nodeId: doc.result.root.nodeId, selector: "#file" });
  await cdp("DOM.setFileInputFiles", { nodeId: q.result.nodeId, files: [meshId.slice(5)] });
  await sleep(6000);
} else if (meshId !== "-") {
  await ev(`(async()=>{ const want=${JSON.stringify(meshId)}; if(!document.querySelector('#list li[data-id="'+want+'"]')) document.getElementById("tabEars").click(); document.querySelector('#list li[data-id="'+want+'"]').click(); })()`);
  await sleep(2500);
}
await shot("0_ready");
await ev(`document.getElementById("run").click()`);
const t0 = Date.now(); let k = 1, last = "";
while (Date.now() - t0 < 600000) {
  await sleep(1500);
  const st = await ev(`document.getElementById("status").textContent`);
  if (st !== last) { last = st; await shot(`${k++}_${st.replace(/[^a-z0-9]+/gi, "_").slice(0, 40)}`); }
  if (/^Done|failed|No face/.test(st)) break;
}
await sleep(2500); await shot("z_final");
console.log(await ev(`(()=>{const d=window.__demo,T=d.THREE; const m=d.mesh; const g=m.geometry; const ray=new T.Raycaster(); const c=d.controls.target.clone(); ray.set(d.cam.position, c.clone().sub(d.cam.position).normalize()); m.material.side=T.DoubleSide; const hits=ray.intersectObject(m); return JSON.stringify({cam:d.cam.position.toArray().map(x=>+x.toFixed(1)), target:c.toArray().map(x=>+x.toFixed(1)), det:d.world.matrix.determinant().toFixed(3), firstHit: hits[0] ? {dist:+hits[0].distance.toFixed(1), facing: +hits[0].face.normal.clone().transformDirection(m.matrixWorld).dot(ray.ray.direction).toFixed(2)} : null, colorMean: [0,1,2].map(k=>{let s=0,n=0;const a=g.attributes.color.array;for(let i=k;i<a.length;i+=3){s+=a[i];n++;}return +(s/n).toFixed(3);})});})()`));
console.log(await ev(`JSON.stringify({head: window.__demo.head, world: window.__demo.world.matrix.elements.map(x=>+x.toFixed(3))})`));
await ev(`window.__demo.frameAll(); window.__demo.controls.autoRotate=false;`); await sleep(1500); await shot("z_whole");
fs.writeFileSync(`${out}/console.txt`, logs.join("\n"));
console.log("status:", last, "|", ((Date.now() - t0) / 1000).toFixed(0), "s |", logs.filter((l) => /error|exception/i.test(l)).length, "errors");
ws.close(); ch.kill();
