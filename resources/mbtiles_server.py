"""Serve tiles from .mbtiles files with a simple MapLibre viewer."""

import argparse
import json
import math
import mimetypes
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote
from os.path import dirname

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("application/json", ".json")

ROOT_DIR = Path(dirname(__file__)).parent.resolve()
UTILITIES_DIR = Path(__file__).parent.resolve()
_Q2VTPORT = 9000

def xyz_y_to_tms(zoom, y):
    """Convert XYZ tile Y coordinate to TMS format."""
    return (1 << zoom) - 1 - y


def _mercator_bounds_to_wgs84(bounds_str: str) -> str:
    try:
        parts = [float(v) for v in bounds_str.split(",")]
        xmin, ymin, xmax, ymax = parts

        if abs(xmin) <= 180 and abs(xmax) <= 180 and abs(ymin) <= 90 and abs(ymax) <= 90:
            return bounds_str

        def x_to_lon(x):
            return max(-180.0, min(180.0, x / 20037508.34 * 180))

        def y_to_lat(y):
            y_c = max(-20037508.34, min(20037508.34, y))
            return math.degrees(math.atan(math.sinh(math.pi * y_c / 20037508.34)))

        west = x_to_lon(xmin)
        south = y_to_lat(ymin)
        east = x_to_lon(xmax)
        north = y_to_lat(ymax)
        return f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}"

    except (ValueError, TypeError):
        return "-180,-85.05,180,85.05"


def load_mbtiles(directory: Path):
    """Load .mbtiles files from the specified directory and return a dict of layer info."""
    layers = {}
    for f in directory.glob("*.mbtiles"):
        name = f.stem
        try:
            conn = sqlite3.connect(f"file:{f}?mode=ro", uri=True, check_same_thread=False)
            cursor = conn.cursor()

            cursor.execute("SELECT name, value FROM metadata")
            meta = dict(cursor.fetchall())

            raw_bounds = meta.get("bounds", "-180,-85.05,180,85.05")
            wgs84_bounds = _mercator_bounds_to_wgs84(raw_bounds)

            layers[name] = {
                "conn": conn,
                "cursor": cursor,
                "lock": threading.Lock(),
                "minzoom": meta.get("minzoom", 0),
                "maxzoom": meta.get("maxzoom", 14),
                "bounds": wgs84_bounds,
            }

            print(f"Loaded {f.name} (zoom {layers[name]['minzoom']}-{layers[name]['maxzoom']})")

        except (sqlite3.Error, OSError) as e:
            print(f"Failed to load {f.name}: {e}")

    return layers


class Handler(BaseHTTPRequestHandler):
    """HTTP request handler for serving tiles and metadata."""
    def do_GET(self):
        """ Handle GET requests for tiles, metadata, and static files. """
        path = unquote(self.path).lstrip("/")

        # Root → viewer
        if 'maplibre_viewer' in path:
            return self._serve_file(UTILITIES_DIR / "maplibre_viewer.html")

        # Vector tiles
        elif path.startswith('tiles/'):
            return self._serve_tile(path)

        # Layers metadata
        elif path == 'layers':
            return self._serve_layers()

        # Style + sprites (served from root/style)
        elif path.startswith('style'):
            return self._serve_file(ROOT_DIR / path)

        # Static files (maplibre js/css in utils)
        elif 'maplibre-gl' in path:
            return self._serve_file(UTILITIES_DIR / path)

        else:
            self.send_error(404)

    def _serve_file(self, file_path: Path):
        try:
            data = file_path.read_bytes()
            self.send_response(200)
            mime, _ = mimetypes.guess_type(str(file_path))
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        except (FileNotFoundError, PermissionError, OSError) as e:
            print(f"File not found: {file_path} — {e}")
            self.send_error(404)

    def _serve_tile(self, path: str):
        # path format: tiles/{layer}/{z}/{x}/{y}.pbf
        parts = path.split("/")
        if len(parts) < 5:
            return self.send_error(400)

        layer_name = parts[1]

        try:
            z = int(parts[2])
            x = int(parts[3])
            y = int(parts[4].split(".")[0])
        except ValueError:
            return self.send_error(400)

        layer = self.server.layers.get(layer_name)
        if not layer:
            return self.send_error(404)

        tms_y = xyz_y_to_tms(z, y)

        with layer["lock"]:
            layer["cursor"].execute(
                "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                (z, x, tms_y),
            )
            row = layer["cursor"].fetchone()

        if row:
            data = row[0]
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            if data[:2] == b"\x1f\x8b":
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

    def _serve_layers(self):
        data = [
            {
                "layer": name,
                "minzoom": info["minzoom"],
                "maxzoom": info["maxzoom"],
                "bounds": info["bounds"],
            }
            for name, info in self.server.layers.items()
        ]

        body = json.dumps(data).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args): # pylint: disable=W0622, W0621
        # Suppress per-request console noise; remove this to re-enable access logs
        pass


def run(port: int = _Q2VTPORT):
    """Start the MBTiles server on the specified port."""
    layers = load_mbtiles(ROOT_DIR)

    if not layers:
        print("No .mbtiles found in root directory:", ROOT_DIR)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.layers = layers

    print(f"\nServer running at http://localhost:{port}")
    print(f"Viewer:  http://localhost:{port}/maplibre_viewer.html")
    print(f"Layers:  http://localhost:{port}/layers\n")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MBTiles tile server")
    parser.add_argument("--port", type=int, default=_Q2VTPORT)
    args = parser.parse_args()
    run(args.port)
