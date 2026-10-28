"""Serve an exported web map folder: the viewer, the style and the tiles.

The export is a folder of plain files (index.html, ml/, ol/, style/,
tiles.pmtiles) that works from any web server or static host, such as
GitHub Pages. It has to be served over HTTP rather than opened as file://,
because browsers refuse to load ES modules and to fetch() from file://
pages; this small server does that locally.

The viewers read tiles.pmtiles in pieces with HTTP range requests, which
this server answers (Python's own http.server does not).

Directory layout assumed:
    <root>/
        index.html                    # the viewer page
        ml/, ol/                      # the viewers' scripts
        style/                        # style.json, sprite/, glyphs/, search.json
        tiles.pmtiles
        server/
            tiles_server.py           # ← this file
"""

import argparse
import gzip
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
HERE     = Path(__file__).resolve().parent
ROOT_DIR = HERE.parent

# Files requested with a version (?v=..., unique per export) never change
# under that URL, so browsers may keep them; others are revalidated.
VERSIONED_CACHE = "public, max-age=31536000, immutable"

MIME_TYPES = {
    ".json":    "application/json",
    ".html":    "text/html; charset=utf-8",
    ".js":      "application/javascript",
    ".mjs":     "application/javascript",
    ".css":     "text/css",
    ".png":     "image/png",
    ".jpg":     "image/jpeg",
    ".jpeg":    "image/jpeg",
    ".webp":    "image/webp",
    ".svg":     "image/svg+xml",
    ".ico":     "image/x-icon",
    ".woff":    "font/woff",
    ".woff2":   "font/woff2",
    ".map":     "application/json",
    ".md":      "text/markdown; charset=utf-8",
    ".pbf":     "application/x-protobuf",
    ".pmtiles": "application/octet-stream",
}

_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")

# Files read at once; bigger ones (the tiles) are streamed in pieces.
_CHUNK = 1 << 20

# Text files sent gzip-compressed to browsers accepting it (the style, the
# search index, the viewers' scripts: several times smaller), compressed
# once per version of the file.
_COMPRESSIBLE = frozenset({".json", ".html", ".js", ".mjs", ".css", ".svg", ".map", ".md"})
_MIN_COMPRESSED_SIZE = 1024
_compressed = {}
_compressed_lock = threading.Lock()


# ----------------------------------------------------------------------
# HTTP handler
# ----------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    """Serves the files under ROOT_DIR (GET and HEAD, with byte ranges)."""

    # HTTP/1.1 keep-alive (tile bursts reuse a single TCP connection)
    # and TCP_NODELAY for low-latency response delivery.
    protocol_version        = "HTTP/1.1"
    disable_nagle_algorithm = True

    def do_GET(self):
        self._serve(send_body=True)

    def do_HEAD(self):
        self._serve(send_body=False)

    def _serve(self, send_body: bool):
        path, _, query = unquote(self.path).partition("?")
        path = path.split("#", 1)[0].lstrip("/")
        target = self._resolve(path or "index.html")
        if target is not None and target.is_dir():
            target = self._resolve(f"{path.rstrip('/')}/index.html")
        if target is None or not target.is_file():
            return self._send_status(404)

        stat = target.stat()
        size = stat.st_size
        # Strong ETag from the file's size and time: a PMTiles reader checks
        # that every piece comes from the same version of the file.
        etag = f'"{stat.st_mtime_ns:x}-{size:x}"'
        cache_control = VERSIONED_CACHE if "v=" in query else "no-cache"
        mime = MIME_TYPES.get(target.suffix.lower(), "application/octet-stream")

        start, end = 0, size - 1
        partial = False
        requested = self.headers.get("Range")
        if requested and self.headers.get("If-Range", etag) == etag:
            match = _RANGE.match(requested.strip())
            if match is None or (not match.group(1) and not match.group(2)):
                return self._send_range_error(size)
            if match.group(1):
                start = int(match.group(1))
                if match.group(2):
                    end = min(int(match.group(2)), size - 1)
            else:  # The last N bytes.
                start = max(0, size - int(match.group(2)))
            if start >= size or start > end:
                return self._send_range_error(size)
            partial = True
        elif self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", cache_control)
            self.send_header("Content-Length", "0")
            self._common_headers()
            self.end_headers()
            return

        body = None
        if (not partial and size >= _MIN_COMPRESSED_SIZE
                and target.suffix.lower() in _COMPRESSIBLE
                and "gzip" in self.headers.get("Accept-Encoding", "")):
            body = self._gzipped(target, etag)
        length = len(body) if body is not None else (end - start + 1 if size else 0)
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", mime)
        # Glyphs may be stored gzip-compressed; tiles in the PMTiles file
        # are decompressed by the viewer itself.
        if mime == "application/x-protobuf" and not partial and self._is_gzip(target):
            self.send_header("Content-Encoding", "gzip")
        if body is not None:
            self.send_header("Content-Encoding", "gzip")
        if target.suffix.lower() in _COMPRESSIBLE:
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", cache_control)
        self._common_headers()
        self.end_headers()
        if not send_body or not length:
            return
        if body is not None:
            self.wfile.write(body)
            return
        with open(target, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(_CHUNK, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    # ---- helpers ----
    @staticmethod
    def _resolve(path: str):
        """The file for a URL path, or None outside the served folder."""
        target = (ROOT_DIR / path).resolve()
        try:
            target.relative_to(ROOT_DIR)
        except ValueError:
            return None  # path-traversal guard
        return target

    @staticmethod
    def _gzipped(target: Path, etag: str) -> bytes:
        """The file gzip-compressed, from memory after its first request."""
        key = str(target)
        with _compressed_lock:
            cached = _compressed.get(key)
        if cached is not None and cached[0] == etag:
            return cached[1]
        body = gzip.compress(target.read_bytes(), compresslevel=6)
        with _compressed_lock:
            _compressed[key] = (etag, body)
        return body

    @staticmethod
    def _is_gzip(target: Path) -> bool:
        with open(target, "rb") as f:
            return f.read(2) == b"\x1f\x8b"

    def _common_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Expose-Headers", "ETag, Content-Range, Content-Length")

    def _send_range_error(self, size: int):
        self.send_response(416)
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", "0")
        self._common_headers()
        self.end_headers()

    def _send_status(self, code: int):
        self.send_response(code)
        self.send_header("Content-Length", "0")
        self._common_headers()
        self.end_headers()

    def log_message(self, fmt, *args):   # noqa: A002
        # Silence per-request stderr noise
        pass


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
class _IPv6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def run(port: int = 18111991):
    server                = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    # The web map opens at localhost, which browsers try as IPv6 (::1)
    # first: answered there too, a new connection doesn't wait for that
    # attempt to fail (up to a second or more on Windows).
    server6 = None
    try:
        server6 = _IPv6Server(("::1", port), Handler)
        server6.daemon_threads = True
        threading.Thread(target=server6.serve_forever, daemon=True).start()
    except OSError:
        server6 = None  # No IPv6, or the port is taken there.

    print(f"\nServing {ROOT_DIR}")
    print(f"  Web map: http://localhost:{port}/\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()
        if server6 is not None:
            server6.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve an exported web map folder")
    parser.add_argument("--port", type=int, default=18111991,
                        help="TCP port to listen on (1–65535, default %(default)s)")
    args = parser.parse_args()
    run(args.port)
