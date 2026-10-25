"""scripts/eval/validate_pipeline.py with the WHOLE pipeline in headless
Chrome: ear3d_core as WASM, demo/js/host.js rendering with three.js and
detecting with onnxruntime-web and MediaPipe's JS FaceLandmarker. Python only
loads the heads and scores them, so this is the browser demo's accuracy.

    python demo/parity/full_browser.py OUT_DIR 2 16 0 1
(build core/pkg first: wasm-pack build --release --target web -d pkg core -- --features wasm)
"""
import http.server, json, pathlib, queue, runpy, subprocess, sys, tempfile, threading, urllib.parse
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import ear3d.core_host as ch
from ear3d.config import DEFAULTS
DEFAULTS["core"] = "rust"
PORT = 8793
blobs, jobs, results = {}, queue.Queue(), queue.Queue()

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(s, *a, **k): super().__init__(*a, directory=str(ROOT / "demo"), **k)
    def log_message(s, *a): pass
    def translate_path(s, path):
        if path.startswith("/pkg/"): return str(ROOT / "core" / "pkg" / path[5:].split("?")[0])
        return super().translate_path(path)
    def end_headers(s):
        s.send_header("Cache-Control", "no-store"); super().end_headers()
    def do_GET(s):
        if s.path.startswith("/next"): body = json.dumps(jobs.get()).encode()
        elif s.path.startswith("/blob/"): body = blobs[s.path[6:]]
        else: return super().do_GET()
        s.send_response(200); s.send_header("Content-Length", str(len(body))); s.end_headers(); s.wfile.write(body)
    def do_POST(s):
        data = s.rfile.read(int(s.headers["Content-Length"]))
        results.put((int(urllib.parse.parse_qs(urllib.parse.urlparse(s.path).query)["seq"][0]), json.loads(data)))
        s.send_response(200); s.end_headers(); s.wfile.write(b"ok")

srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
prof = tempfile.mkdtemp(prefix="ear3d_chrome_")
chrome = subprocess.Popen(["google-chrome", "--headless=new", f"--user-data-dir={prof}", "--use-gl=angle", "--use-angle=gl-egl",
                           "--ignore-gpu-blocklist", "--enable-logging=stderr", f"http://localhost:{PORT}/parity/full.html"],
                          stdout=subprocess.DEVNULL, stderr=open(pathlib.Path(prof) / "chrome.log", "w"))
print("chrome log:", pathlib.Path(prof) / "chrome.log", flush=True)
seq = [0]

def run(mesh, cfg, table, angles, template, verbose=True, host=None):
    b = lambda a: np.ascontiguousarray(a, "<f8").tobytes()
    blobs.update(V=b(np.asarray(mesh.vertices)), F=np.asarray(mesh.triangles, "<u4").tobytes(), table=b(table), angles=b(angles),
                 template=b(template), canon=b(ch.canonical_face(cfg)), anchors=b(ch.anchors()))
    seq[0] += 1
    jobs.put({"seq": seq[0], "cfg": json.dumps({k: cfg[k] for k in ch.CFG_KEYS})})
    while True:
        s, o = results.get(timeout=1800)
        if s == seq[0]: break
    if "error" in o: raise RuntimeError("browser: " + o["error"])
    print(f"browser: {o['seconds']:.1f} s", flush=True)
    pts = lambda x: None if x is None else np.reshape(x, (-1, 3))
    return {"L": pts(o["L"]), "R": pts(o["R"])}, {
        "ear_only": o["earOnly"], "fit_residual": o["fitResidual"], "events": o["events"], "seconds": o["seconds"],
        "R_head": None if o["R_head"] is None else np.reshape(o["R_head"], (3, 3)), "centre": o["centre"]}

ch.run = run
sys.argv = ["validate_pipeline.py"] + sys.argv[1:]
try:
    runpy.run_path(str(ROOT / "scripts/eval/validate_pipeline.py"), run_name="__main__")
finally:
    jobs.put({"stop": True}); chrome.terminate(); chrome.wait(); srv.shutdown()
