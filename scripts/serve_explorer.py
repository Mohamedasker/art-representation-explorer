#!/usr/bin/env python3
"""
Serve the explorer (pages, thumbnails, feature-map viewer) with caching switched off, so a browser refresh always
shows the newest files.  Use it instead of ``python -m http.server``.

    python scripts/serve_explorer.py                 # http://127.0.0.1:8877/  (directory visualizations/explorer)
    python scripts/serve_explorer.py --port 8890

Reach it from your laptop through an SSH tunnel to the SAME login node, e.g. in one session:
    ssh -L 8877:127.0.0.1:8877 <jureca>      and then start this script inside that session.
"""

import argparse
import functools
import http.server
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def log_message(self, *a):          # quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(REPO_ROOT / "visualizations/explorer"))
    ap.add_argument("--port", type=int, default=8877)
    ap.add_argument("--bind", default="127.0.0.1")
    a = ap.parse_args()
    handler = functools.partial(NoCacheHandler, directory=a.dir)
    with http.server.ThreadingHTTPServer((a.bind, a.port), handler) as srv:
        print(f"serving {a.dir} at http://{a.bind}:{a.port}/  (Ctrl+C to stop)", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()
