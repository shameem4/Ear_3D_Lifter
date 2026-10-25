"""The browser's detectors against the Python host's on the same pictures:
MediaPipe JS FaceLandmarker vs Python, onnxruntime-web vs onnxruntime on
the crate's own input tensors. Rendering is not involved.

    python demo/parity/detector_js.py IMG_DIR [IMG_DIR ...]
"""
import http.server, json, pathlib, queue, subprocess, sys, tempfile, threading, urllib.parse
import numpy as np
from PIL import Image
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import ear3d_core as core
from ear3d.config import DEFAULTS
from ear3d.core_host import FilamentHost
PORT = 8794
blobs, jobs, results = {}, queue.Queue(), queue.Queue()

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(s, *a, **k): super().__init__(*a, directory=str(ROOT / "demo"), **k)
    def log_message(s, *a): pass
    def do_GET(s):
        if s.path.startswith("/next"): body = json.dumps(jobs.get()).encode()
        elif s.path.startswith("/blob/"): body = blobs[s.path[6:]]
        else: return super().do_GET()
        s.send_response(200); s.send_header("Content-Length", str(len(body))); s.end_headers(); s.wfile.write(body)
    def do_POST(s):
        d = s.rfile.read(int(s.headers["Content-Length"]))
        results.put((int(urllib.parse.parse_qs(urllib.parse.urlparse(s.path).query)["seq"][0]), json.loads(d)))
        s.send_response(200); s.end_headers(); s.wfile.write(b"ok")

srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
prof = tempfile.mkdtemp(prefix="ear3d_chrome_")
chrome = subprocess.Popen(["google-chrome", "--headless=new", f"--user-data-dir={prof}", "--use-gl=angle", "--use-angle=gl-egl",
                           "--ignore-gpu-blocklist", f"http://localhost:{PORT}/parity/detector_js.html"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
seq = [0]
def js(kind, data, size=None):
    seq[0] += 1; blobs["x"] = data
    jobs.put({"seq": seq[0], "kind": kind, "key": "x", "size": size})
    while True:
        s, o = results.get(timeout=600)
        if s == seq[0]: return [np.asarray(a, np.float64) for a in o]

host = FilamentHost(DEFAULTS["face_task"], verbose=False)
f4 = lambda a: np.asarray(a, "<f4").tobytes()
face, miss, r2j, bz = [], [0, 0], [], []
try:
    for p in sorted(p for d in sys.argv[1:] for p in pathlib.Path(d).glob("fil_*.png")):
        img = np.ascontiguousarray(np.asarray(Image.open(p).convert("RGB"))); S = img.shape[0]
        py = host._face_points(img); jv = js("face", img.tobytes(), S)
        if (py is None) != (not jv): miss[py is None] += 1
        elif py is not None: face.append(np.linalg.norm(py[:468] - jv[0].reshape(-1, 2)[:468], axis=1).mean())
        a, _ = core.landmarker_inputs(img.tobytes(), S, S)
        r2j.append(np.abs(np.frombuffer(host.infer("r2j", f4(a))[0], "<f4") - js("r2j", f4(a))[0]).max())
        x = core.blazeear_input(img.tobytes(), S)
        bz.append(np.abs(np.frombuffer(host.infer("blazeear", f4(x))[1], "<f4") - js("blazeear", f4(x))[1]).max())
finally:
    jobs.put({"stop": True}); chrome.terminate(); chrome.wait(); srv.shutdown()
face, r2j, bz = map(np.array, (face, r2j, bz))
print(f"{len(r2j)} images | FaceMesh: both found {len(face)}, only JS {miss[0]}, only Python {miss[1]}; "
      f"mean point gap median {np.median(face):.3f} px, p90 {np.percentile(face, 90):.3f}, max {face.max():.3f}")
print(f"r2j |web - ort| max {r2j.max():.2e} (x600 = {r2j.max() * 600:.3f} px) | blazeear raw score max {bz.max():.2e}")
