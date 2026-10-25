"""A pipeline host whose pictures come from the browser renderer
(demo/js/renderer.js) in headless Chrome, on the GPU through ANGLE/GL-EGL (the
Vulkan path draws black). Detection stays in Python, so the renderer is the
only difference from FilamentHost. Chrome loads demo/parity/bridge.html, which
long-polls /next for render batches and posts each picture and depth back."""
import http.server, json, pathlib, queue, subprocess, tempfile, threading, urllib.parse, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ear3d.core_host import FilamentHost
WWW = str(pathlib.Path(__file__).resolve().parents[1])   # demo/: js/ and parity/bridge.html

class BrowserHost(FilamentHost):
    def __init__(self, *a, port=8792, **k):
        super().__init__(*a, **k)
        self.blobs, self.req_q, self.res_q, self.seq = {}, queue.Queue(), queue.Queue(), 0
        host = self
        class H(http.server.SimpleHTTPRequestHandler):
            def __init__(s, *a_, **k_): super().__init__(*a_, directory=WWW, **k_)
            def log_message(s, *a_): pass
            def do_GET(s):
                if s.path.startswith("/next"):
                    body = json.dumps(host.req_q.get()).encode()
                elif s.path.startswith("/blob/"):
                    body = host.blobs[s.path[6:]]
                else:
                    return super().do_GET()
                s.send_response(200); s.send_header("Content-Length", str(len(body))); s.end_headers(); s.wfile.write(body)
            def do_POST(s):
                n = int(s.headers["Content-Length"]); data = s.rfile.read(n)
                q = urllib.parse.parse_qs(urllib.parse.urlparse(s.path).query)
                host.res_q.put((int(q["seq"][0]), int(q["i"][0]), data))
                s.send_response(200); s.end_headers(); s.wfile.write(b"ok")
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.prof = tempfile.mkdtemp(prefix="ear3d_chrome_")
        self.chrome = subprocess.Popen(["google-chrome", "--headless=new", f"--user-data-dir={self.prof}", "--use-gl=angle", "--use-angle=gl-egl",
                                        "--ignore-gpu-blocklist", f"http://localhost:{port}/parity/bridge.html"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.versions = {}

    def close(self):
        self.req_q.put({"stop": True}); self.chrome.terminate(); self.chrome.wait(); self.srv.shutdown()

    def set_mesh(self, id, v, f):
        super().set_mesh(id, v, f)
        self.versions[f"m{id}"] = self.versions.get(f"m{id}", 0) + 1
        self.blobs[f"m{id}_V_{self.versions[f'm{id}']}"] = bytes(v); self.blobs[f"m{id}_F_{self.versions[f'm{id}']}"] = bytes(f)

    def set_colors(self, id, mesh, c):
        super().set_colors(id, mesh, c)
        self.versions[f"c{id}"] = self.versions.get(f"c{id}", 0) + 1
        self.blobs[f"c{id}_{self.versions[f'c{id}']}"] = bytes(c)

    def _draw(self, req):
        self.seq += 1
        m, c = req["mesh"], req["colors"]
        job = dict(seq=self.seq, mesh=m, colors=c, meshKey=f"m{m}_{self.versions[f'm{m}']}", colorKey=f"c{c}_{self.versions[f'c{c}']}",
                   colorMesh=self.colors[c][0], size=req["size"], f=req["f"], c=req["c"], depth=req["depth"],
                   views=[{k: (list(map(float, np.ravel(v[k]))) if k != "s" else float(v[k])) for k in ("E", "R", "C", "s", "sun")} for v in req["views"]])
        self.req_q.put(job)
        S = int(req["size"]); got = {}
        while len(got) < len(req["views"]):
            seq, i, data = self.res_q.get(timeout=300)
            if seq == self.seq: got[i] = data
        out = []
        for i in range(len(req["views"])):
            d = got[i]; rgba = np.frombuffer(d[:S * S * 4], np.uint8).reshape(S, S, 4)[:, :, :3].copy()
            depth = np.frombuffer(d[S * S * 4:], "<f4").reshape(S, S).copy() if req["depth"] else None
            out.append((rgba, depth))
        return out
