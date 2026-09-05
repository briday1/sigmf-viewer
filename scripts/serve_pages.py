"""Preview the built Pages artifact under its project subpath."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


class PagesHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, prefix, **kwargs):
        self.prefix = prefix
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if not urlsplit(self.path).path.startswith(self.prefix):
            self.send_error(404, f"Open {self.prefix}")
            return
        super().do_GET()

    def do_HEAD(self):
        if not urlsplit(self.path).path.startswith(self.prefix):
            self.send_error(404, f"Open {self.prefix}")
            return
        super().do_HEAD()

    def translate_path(self, path):
        return super().translate_path("/" + path[len(self.prefix):])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1] / "site")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefix", default="/sigmf-viewer/")
    args = parser.parse_args()
    if not args.prefix.startswith("/") or not args.prefix.endswith("/"):
        parser.error("--prefix must start and end with /")
    handler = partial(PagesHandler, directory=str(args.directory.resolve()), prefix=args.prefix)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Preview: http://127.0.0.1:{args.port}{args.prefix}", flush=True)
    server.serve_forever()
