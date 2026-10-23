#!/usr/bin/env python3
"""Step 2: the hand-correction editor, served on localhost only.

Usage: serve.py [--dir runs/handfix] [--port 8765]   then open http://127.0.0.1:PORT

Serves editor.html and the ears exported by 1_export_ears.py, and writes
<ear>/adjusted.json on save (atomic; the previous version is kept as
adjusted.prev.json). Binds 127.0.0.1: the scans never leave the machine.
"""
import argparse
import http.server
import json
import os
import pathlib
import re
import time

HERE = pathlib.Path(__file__).resolve().parent
NAME = re.compile(r"^[A-Za-z0-9_]+$")


def make_handler(root):
    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def _file(self, p, ctype):
            if not p.is_file():
                return self.send_error(404)
            b = p.read_bytes()
            self.send_response(200); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            parts = self.path.split("?")[0].strip("/").split("/")
            if parts == [""]:
                return self._file(HERE / "editor.html", "text/html; charset=utf-8")
            if parts == ["api", "ears"]:
                rows = []
                for d in sorted(p for p in root.iterdir() if (p / "points.json").is_file()):
                    pj = json.loads((d / "points.json").read_text())
                    adj = d / "adjusted.json"
                    done = json.loads(adj.read_text()).get("done", False) if adj.is_file() else False
                    rows.append(dict(ear=d.name, dataset=pj["dataset"], held_out=pj["held_out"],
                                     lobe_max=pj["lobe_max"], edited=adj.is_file(), done=done))
                return self._json(sorted(rows, key=lambda r: (r["done"], not r["edited"], -r["lobe_max"])))  # started, untouched, done
            if len(parts) == 3 and parts[0] == "data" and NAME.match(parts[1]) and \
                    parts[2] in ("surface.ply", "points.json", "adjusted.json"):
                ctype = "application/octet-stream" if parts[2].endswith(".ply") else "application/json"
                return self._file(root / parts[1] / parts[2], ctype)
            self.send_error(404)

        def do_POST(self):
            parts = self.path.strip("/").split("/")
            if len(parts) != 3 or parts[:2] != ["api", "save"] or not NAME.match(parts[2]) \
                    or not (root / parts[2] / "points.json").is_file():
                return self.send_error(404)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            pts = body.get("points")
            if not (isinstance(pts, list) and len(pts) == 55 and all(len(p) == 3 for p in pts)):
                return self._json({"error": "need 55 points"}, 400)
            d = root / parts[2]; adj = d / "adjusted.json"
            rec = dict(ear=parts[2], points=pts, moved=sorted(set(body.get("moved", []))),
                       done=bool(body.get("done", False)), note=str(body.get("note", ""))[:2000],
                       frame="canonical mm (right ears mirrored about z)", saved=time.strftime("%Y-%m-%dT%H:%M:%S"))
            if adj.is_file():
                os.replace(adj, d / "adjusted.prev.json")
            tmp = d / "adjusted.tmp"; tmp.write_text(json.dumps(rec)); os.replace(tmp, adj)
            return self._json({"ok": True, "saved": rec["saved"]})
    return H


def main():
    a = argparse.ArgumentParser(); a.add_argument("--dir", default="runs/handfix"); a.add_argument("--port", type=int, default=8765)
    a = a.parse_args()
    root = pathlib.Path(a.dir).resolve()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(root))
    print(f"editor: http://127.0.0.1:{a.port}   ({root})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
