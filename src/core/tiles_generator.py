"""
tiles_generator.py

GDALTilesGenerator — builds a multi-layer OGR VRT from the exported
datasets and calls ogr2ogr to produce MVT (Mapbox Vector Tiles) in MBTiles
format, then copies them into the export's single PMTiles file
(pmtiles_writer).

The MVT driver works a run's zoom levels one after another and gains
nothing from GDAL_NUM_THREADS (measured slower), so the zoom levels are
split into parallel ogr2ogr runs, one process each, whose MBTiles are then
merged into one (zoom levels never share a tile). Workers left over when
there are fewer zoom levels than workers split zoom levels by area into
strips of tile columns: a strip's run reads only the features near it and
keeps only its own tiles.

Each run's temporary database, which the driver writes every feature of
every tile to, is kept in memory while the runs' estimated sizes fit in
half the free memory (several times faster than on disk), else on disk.

Depends on: config (for constants only — no custom class dependencies)
"""

import ctypes
import json
import math
import os
import shutil
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass
from os import cpu_count
from os.path import join, basename
from typing import Dict, List, Optional, Tuple
from uuid import uuid4
from xml.sax.saxutils import escape, quoteattr
from osgeo import ogr
from qgis.core import QgsVectorLayer, QgsProcessingFeedback, QgsProcessingUtils

from .pmtiles_writer import mbtiles_to_pmtiles
from ..utils.config import (
    _EPSG_CRS, _FIELD_PREFIX, _SIMPLIFICATION, _SIMPLIFICATION_MAX_ZOOM, _TILES_MAX_ZOOM,
    _TILE_MAX_FEATURES, _TILE_MAX_SIZE, _TILE_MAX_SIZE_FEW, _TILE_VIEW_SIZE,
)

_CREATE_NO_WINDOW = 0x08000000

# Half the width of the EPSG:3857 world, in metres.
_WORLD_HALF = 20037508.342789244

# A run's temporary database in memory takes about this many times the size
# of the datasets it reads, per zoom level (measured: 3.3).
_MEMORY_PER_DATASET_BYTE = 4

# Share of the free memory the runs' in-memory databases may take together.
_MEMORY_SHARE = 0.5


@dataclass
class _Run:
    """One ogr2ogr run: zoom levels lo..hi, of all tile columns or of
    columns[0]..columns[1] only, written to ``output``."""
    lo: int
    hi: int
    columns: Optional[Tuple[int, int]] = None
    output: str = ""
    in_memory: bool = False


class GDALTilesGenerator:
    """Generate MBTiles vector tiles using GDAL CLI with an OGR VRT intermediary."""

    def __init__(
        self,
        layers: List[QgsVectorLayer],
        style: dict,
        output_dir: str,
        extent,
        cpu_percent: int,
        feedback: QgsProcessingFeedback,
    ):
        self.layers = layers
        self.style = style
        self.output_dir = output_dir
        self.extent = extent
        self.cpu_percent = cpu_percent
        self.feedback = feedback
        # Set by cancel(), from another thread.
        self._cancelled = threading.Event()

    def cancel(self):
        """Stop generating (from another thread): the runs are stopped and
        generate() raises."""
        self._cancelled.set()

    def generate(self) -> Tuple[str, int]:
        """Build the VRTs, run ogr2ogr, return (PMTiles path, min_zoom).

        The runs write MBTiles (which merge cheaply), copied at the end into
        the export's one PMTiles file.
        """
        pmtiles = join(self.output_dir, "tiles.pmtiles")
        output = join(QgsProcessingUtils.tempFolder(), f"tiles_{uuid4().hex}.mbtiles")
        try:
            min_zoom = self._generate_mbtiles(output)
            if self._cancelled.is_set() or (self.feedback is not None and self.feedback.isCanceled()):
                raise RuntimeError("Tile generation cancelled.")
            mbtiles_to_pmtiles(output, pmtiles)
        finally:
            self._remove(output)
        return pmtiles, min_zoom

    def _generate_mbtiles(self, output: str) -> int:
        """Make the tiles into the MBTiles ``output``; returns min_zoom."""
        min_zoom = self._get_global_min_zoom()
        max_zoom = self._get_global_max_zoom()
        first = min(min_zoom, _TILES_MAX_ZOOM)
        last = min(max_zoom, _TILES_MAX_ZOOM)
        self._last_zoom = last
        workers = self._worker_count()
        zoom_runs = [
            run for run in self._zoom_runs(first, last, workers)
            if self._run_layers(*run)
        ] or [(first, last)]
        runs = self._split_by_area(zoom_runs, workers)
        self._place_temporary_dbs(runs)
        if len(runs) == 1:
            runs[0].output = output
            self._run_ogr2ogr_runs(runs)
        else:
            for run in runs:
                run.output = join(
                    QgsProcessingUtils.tempFolder(), f"tiles_{run.lo}_{run.hi}_{uuid4().hex}.mbtiles"
                )
            try:
                self._run_ogr2ogr_runs(runs)
                self._merge(output, runs, first, last)
            finally:
                for run in runs:
                    self._remove(run.output)
        return min_zoom

    def _worker_count(self) -> int:
        return max(1, int((cpu_count() or 1) * self.cpu_percent / 100))

    # --- Tile grid over the extent ---

    @staticmethod
    def _tile_index(coordinate: float, zoom: int) -> int:
        """The tile column (or row from the south) holding an EPSG:3857
        coordinate."""
        tiles = 2 ** zoom
        index = int(math.floor((coordinate + _WORLD_HALF) / (2 * _WORLD_HALF) * tiles))
        return min(max(index, 0), tiles - 1)

    def _tile_range(self, zoom: int) -> Tuple[int, int, int, int]:
        """(first column, last column, first row, last row) of the tiles
        covering the extent; the whole world without one."""
        e = self.extent
        if e is None or e.isEmpty():
            return 0, 2 ** zoom - 1, 0, 2 ** zoom - 1
        return (self._tile_index(e.xMinimum(), zoom), self._tile_index(e.xMaximum(), zoom),
                self._tile_index(e.yMinimum(), zoom), self._tile_index(e.yMaximum(), zoom))

    def _tile_count(self, zoom: int) -> int:
        c0, c1, r0, r1 = self._tile_range(zoom)
        return (c1 - c0 + 1) * (r1 - r0 + 1)

    def _max_tile_size(self, zoom: int) -> int:
        """MAX_SIZE of a run whose last zoom level is ``zoom``."""
        return max(_TILE_MAX_SIZE, min(_TILE_MAX_SIZE_FEW, _TILE_VIEW_SIZE // self._tile_count(zoom)))

    # --- Splitting zoom levels by area ---

    def _split_by_area(self, zoom_runs: List[Tuple[int, int]], workers: int) -> List[_Run]:
        """The runs: one per zoom run, single-zoom ones split into strips of
        tile columns while workers are left. Every zoom level costs about
        the same (each writes every feature), so the zoom with the fewest
        strips is split further, the one with most columns first."""
        strips = {run: 1 for run in zoom_runs}
        columns = {}
        for lo, hi in zoom_runs:
            c0, c1, _, _ = self._tile_range(hi)
            columns[(lo, hi)] = c1 - c0 + 1
        spare = workers - len(zoom_runs)
        while spare > 0:
            candidates = [
                run for run in zoom_runs
                if run[0] == run[1] and strips[run] < columns[run]
            ]
            if not candidates:
                break
            run = min(candidates, key=lambda r: (strips[r], -columns[r], -r[0]))
            strips[run] += 1
            spare -= 1
        runs = []
        for lo, hi in zoom_runs:
            count = strips[(lo, hi)]
            if count == 1:
                runs.append(_Run(lo, hi))
                continue
            c0, c1, _, _ = self._tile_range(hi)
            total = c1 - c0 + 1
            start = c0
            for i in range(count):
                width = total // count + (1 if i < total % count else 0)
                runs.append(_Run(lo, hi, (start, start + width - 1)))
                start += width
        return runs

    def _strip_filter(self, run: _Run) -> List[str]:
        """ogr2ogr's spatial filter for a strip: its columns, widened by a
        tile on each side so that every feature reaching into its tiles
        (or their buffer) is read; nothing for a whole run."""
        if run.columns is None:
            return []
        tile = 2 * _WORLD_HALF / 2 ** run.hi
        west = -_WORLD_HALF + run.columns[0] * tile - tile
        east = -_WORLD_HALF + (run.columns[1] + 1) * tile + tile
        return ["-spat", repr(west), repr(-_WORLD_HALF), repr(east), repr(_WORLD_HALF)]

    # --- Temporary databases ---

    def _place_temporary_dbs(self, runs: List[_Run]):
        """Keep runs' temporary databases in memory while their estimated
        sizes fit in _MEMORY_SHARE of the free memory."""
        budget = self._free_memory() * _MEMORY_SHARE
        sizes = {}
        for run in runs:
            layers = self._run_layers(run.lo, run.hi)
            data = sum(self._dataset_size(layer) for layer, _, _ in layers.values())
            strips = 1
            if run.columns is not None:
                c0, c1, _, _ = self._tile_range(run.hi)
                strips = max(1, (c1 - c0 + 1) / (run.columns[1] - run.columns[0] + 1))
            sizes[id(run)] = _MEMORY_PER_DATASET_BYTE * data * (run.hi - run.lo + 1) / strips
        for run in sorted(runs, key=lambda r: sizes[id(r)]):
            if sizes[id(run)] <= budget:
                run.in_memory = True
                budget -= sizes[id(run)]

    @staticmethod
    def _dataset_size(layer: QgsVectorLayer) -> int:
        try:
            return os.path.getsize(layer.source().split("|")[0])
        except OSError:
            return 0

    @staticmethod
    def _free_memory() -> int:
        """Free physical memory in bytes; 0 if unknown."""
        try:
            if os.name == "nt":
                class _MemoryStatus(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                status = _MemoryStatus()
                status.dwLength = ctypes.sizeof(status)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                    return int(status.ullAvailPhys)
                return 0
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (AttributeError, OSError, ValueError):
            return 0

    @staticmethod
    def _zoom_runs(first: int, last: int, workers: int) -> List[Tuple[int, int]]:
        """Zoom ranges of the parallel runs: one zoom level each, as many as
        there are workers; the lowest levels, which have the fewest tiles,
        share one run when there are more levels than workers."""
        zooms = list(range(first, last + 1))
        if workers <= 1 or len(zooms) <= 1:
            return [(first, last)]
        if len(zooms) <= workers:
            return [(z, z) for z in zooms]
        single = zooms[len(zooms) - (workers - 1):]
        return [(first, single[0] - 1)] + [(z, z) for z in single]

    # --- Per-layer zoom configuration ---

    def _layer_zooms(self, layer: QgsVectorLayer) -> Tuple[int, int]:
        """The layer's tile zoom range. Tiles stop at _TILES_MAX_ZOOM, where
        viewers enlarge them, so a layer shown only beyond it is written to
        that zoom's tiles."""
        return (min(self._parse_layer_zoom(layer, "o"), _TILES_MAX_ZOOM),
                min(self._parse_layer_zoom(layer, "i"), _TILES_MAX_ZOOM))

    def _run_layers(self, lo: int, hi: int) -> Dict[str, Tuple[QgsVectorLayer, int, int]]:
        """The layers a run of zooms lo..hi writes, with their zoom range
        within it. A layer not configured for a run would be written at all
        its zooms, so each run gets only its own layers."""
        result = {}
        for layer in self.layers:
            minzoom, maxzoom = self._layer_zooms(layer)
            if minzoom <= hi and maxzoom >= lo:
                result[self._layer_name(layer)] = (layer, max(minzoom, lo), min(maxzoom, hi))
        return result

    def _build_layer_conf(self, conf_path: str, run_layers: dict):
        """Write the MVT driver's CONF file: each layer's own zoom range.

        Without it every layer is written to every zoom level of the
        dataset, although each style only shows it within its own range.
        """
        conf = {
            name: {"minzoom": minzoom, "maxzoom": maxzoom}
            for name, (_, minzoom, maxzoom) in run_layers.items()
        }
        with open(conf_path, "w", encoding="utf-8") as f:
            json.dump(conf, f)

    # --- VRT construction ---

    def _build_vrt(self, vrt_path: str, layers: List[QgsVectorLayer] = None):
        """Write an OGR VRT containing one entry per layer."""
        style_text = str(self.style)
        with open(vrt_path, "w", encoding="utf-8") as f:
            f.write("<OGRVRTDataSource>\n")
            for layer in (self.layers if layers is None else layers):
                f.write(self._vrt_layer_block(layer, style_text))
            f.write("</OGRVRTDataSource>\n")

    def _vrt_layer_block(self, layer: QgsVectorLayer, style_text: str) -> str:
        """Return the VRT XML block for a single layer."""
        source = layer.source().split("|layername=")[0]
        return (
            f'    <OGRVRTLayer name={quoteattr(self._layer_name(layer))}>\n'
            f'        <SrcDataSource>{escape(source)}</SrcDataSource>\n'
            f'        <LayerSRS>EPSG:{_EPSG_CRS}</LayerSRS>\n'
            f'        <GeometryType>wkbUnknown</GeometryType>\n'
            f'{self._vrt_fields(source, style_text)}'
            f'    </OGRVRTLayer>\n'
        )

    @staticmethod
    def _is_tiled_field(field_name: str, style_text: str) -> bool:
        """Whether a dataset field belongs in the tiles.

        q2vt_orig_id only serves the rules export, and data-defined property
        fields matter only if the style references them.
        """
        if field_name in (f"{_FIELD_PREFIX}_orig_id", f"{_FIELD_PREFIX}_biggest"):
            return False
        if f"{_FIELD_PREFIX}_property_" in field_name:
            return field_name in style_text
        return True

    def _vrt_fields(self, source: str, style_text: str) -> str:
        """<Field> elements selecting the tiled fields of ``source``.

        Selecting fields in the VRT keeps the others out of the tiles without
        rewriting the dataset. Returns "" (all fields) if it can't be read.
        """
        ds = ogr.Open(source)
        if ds is None:
            return ""
        defn = ds.GetLayer(0).GetLayerDefn()
        elements = []
        for i in range(defn.GetFieldCount()):
            field = defn.GetFieldDefn(i)
            if not self._is_tiled_field(field.GetName(), style_text):
                continue
            subtype = ""
            if field.GetSubType() != ogr.OFSTNone:
                subtype = f" subtype={quoteattr(ogr.GetFieldSubTypeName(field.GetSubType()))}"
            elements.append(
                f'        <Field name={quoteattr(field.GetName())} '
                f'type={quoteattr(ogr.GetFieldTypeName(field.GetType()))}{subtype}/>\n'
            )
        ds = None
        return "".join(elements)

    # --- ogr2ogr execution ---

    def _ogr2ogr_command(self, run: _Run) -> List[str]:
        """The ogr2ogr command of one run, writing its VRT and CONF next to
        the temporary files."""
        lo, hi = run.lo, run.hi
        run_layers = self._run_layers(lo, hi)
        tag = f"{lo}_{hi}_{uuid4().hex}"
        vrt_path = join(QgsProcessingUtils.tempFolder(), f"layers_{tag}.vrt")
        conf_path = join(QgsProcessingUtils.tempFolder(), f"layers_conf_{tag}.json")
        self._build_vrt(vrt_path, [layer for layer, _, _ in run_layers.values()])
        self._build_layer_conf(conf_path, run_layers)
        # The temporary database is written heavily: in memory if it fits,
        # else on local disk, not next to the output (which may be a synced
        # or network folder).
        if run.in_memory:
            temporary_db = f"/vsimem/tiles_{tag}.temp.db"
        else:
            temporary_db = join(QgsProcessingUtils.tempFolder(), f"tiles_{tag}.temp.db")
        # The driver applies SIMPLIFICATION_MAX_ZOOM at a run's last zoom:
        # only the tiles' real last zoom is left unsimplified.
        last_zoom_simplification = (
            _SIMPLIFICATION_MAX_ZOOM if hi >= getattr(self, "_last_zoom", hi) else _SIMPLIFICATION
        )
        return [
            "ogr2ogr", "-f", "MBTiles", run.output, vrt_path,
            # MAXZOOM also becomes the MBTiles metadata's maxzoom, from which
            # QGIS enlarges the tiles beyond it.
            "-dsco", f"MINZOOM={lo}",
            "-dsco", f"MAXZOOM={hi}",
            "-dsco", f"CONF={conf_path}",
            "-dsco", f"TEMPORARY_DB={temporary_db}",
            "-t_srs", f"EPSG:{_EPSG_CRS}",
            "-dsco", f"MAX_SIZE={self._max_tile_size(hi)}",
            "-dsco", f"MAX_FEATURES={_TILE_MAX_FEATURES}",
            "-dsco", f"SIMPLIFICATION={_SIMPLIFICATION}",
            "-dsco", f"SIMPLIFICATION_MAX_ZOOM={last_zoom_simplification}",
            *self._strip_filter(run),
        ]

    def _run_ogr2ogr_runs(self, runs: List[_Run]):
        """Run ogr2ogr once per run, all at once, each on one thread; raises
        RuntimeError if any fails, and stops them all if the conversion is
        cancelled. A run that fails with its temporary database in memory
        (e.g. out of memory) is run again with it on disk."""
        failed = self._run_processes(runs)
        retry = [run for run, _ in failed if run.in_memory]
        if retry:
            for run in retry:
                run.in_memory = False
                self._remove(run.output)
            failed = self._run_processes(retry)
        if failed:
            error_msg = f"ogr2ogr failed.\nError: {failed[0][1]}"
            if self.feedback:
                self.feedback.reportError(error_msg)
            raise RuntimeError(error_msg)

    def _run_processes(self, runs: List[_Run]) -> List[Tuple[_Run, str]]:
        """Run the runs' ogr2ogr processes at once; the failed runs with the
        end of their error output."""
        env = os.environ.copy()
        env["GDAL_NUM_THREADS"] = "1"
        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            creationflags = _CREATE_NO_WINDOW
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0

        processes = []
        failed = []
        try:
            for run in runs:
                stderr_path = f"{run.output}.log"
                with open(stderr_path, "w", encoding="utf-8") as stderr:
                    processes.append((run, subprocess.Popen(
                        self._ogr2ogr_command(run), env=env,
                        stdout=subprocess.DEVNULL, stderr=stderr,
                        startupinfo=startupinfo, creationflags=creationflags,
                    ), stderr_path))
            while any(process.poll() is None for _, process, _ in processes):
                if self._cancelled.is_set() or (
                        self.feedback is not None and self.feedback.isCanceled()):
                    raise RuntimeError("Tile generation cancelled.")
                time.sleep(0.25)
            for run, process, stderr_path in processes:
                if process.returncode != 0:
                    with open(stderr_path, encoding="utf-8", errors="replace") as f:
                        failed.append((run, f.read()[-4000:]))
        finally:
            for _, process, stderr_path in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                self._remove(stderr_path)
        return failed

    # --- Merging the runs' MBTiles ---

    @classmethod
    def _merge(cls, output: str, runs: List[_Run], first: int, last: int):
        """Merge the runs' MBTiles (disjoint zoom levels, or disjoint tile
        columns of one) into ``output``: each run's own tiles, and metadata
        covering all zooms and layers. A strip's run also writes partial
        tiles beside its columns, which are left out."""
        cls._remove(output)
        shutil.copyfile(runs[0].output, output)
        conn = sqlite3.connect(output)
        try:
            if runs[0].columns is not None:
                conn.execute(
                    "DELETE FROM tiles WHERE tile_column < ? OR tile_column > ?", runs[0].columns
                )
                conn.commit()
            metadata = [dict(conn.execute("SELECT name, value FROM metadata"))]
            # Attached one at a time, always under the same name.
            for run in runs[1:]:
                conn.execute("ATTACH DATABASE ? AS part", (run.output,))
                if run.columns is None:
                    conn.execute("INSERT INTO tiles SELECT * FROM part.tiles")
                else:
                    conn.execute(
                        "INSERT INTO tiles SELECT * FROM part.tiles "
                        "WHERE tile_column BETWEEN ? AND ?", run.columns
                    )
                metadata.append(dict(conn.execute("SELECT name, value FROM part.metadata")))
                conn.commit()
                conn.execute("DETACH DATABASE part")
            merged = cls._merged_metadata(metadata, first, last)
            conn.execute("DELETE FROM metadata")
            conn.executemany("INSERT INTO metadata (name, value) VALUES (?, ?)", merged.items())
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _merged_metadata(parts: List[dict], first: int, last: int) -> dict:
        """One metadata table for the merged tiles: the zoom range, the
        bounds and the vector layers (zoom ranges, fields) of all runs."""
        merged = dict(parts[0])
        merged["name"] = "tiles"
        merged["minzoom"] = str(first)
        merged["maxzoom"] = str(last)
        boxes = []
        for meta in parts:
            try:
                boxes.append([float(v) for v in meta["bounds"].split(",")])
            except (KeyError, ValueError):
                continue
        if boxes:
            west, south = min(b[0] for b in boxes), min(b[1] for b in boxes)
            east, north = max(b[2] for b in boxes), max(b[3] for b in boxes)
            merged["bounds"] = f"{west},{south},{east},{north}"
            merged["center"] = f"{(west + east) / 2},{(south + north) / 2},{first}"
        layers: Dict[str, dict] = {}
        for meta in parts:
            try:
                vector_layers = json.loads(meta.get("json", "{}")).get("vector_layers", [])
            except ValueError:
                continue
            for layer in vector_layers:
                known = layers.get(layer.get("id"))
                if known is None:
                    layers[layer.get("id")] = dict(layer)
                    continue
                known["minzoom"] = min(known.get("minzoom", first), layer.get("minzoom", first))
                known["maxzoom"] = max(known.get("maxzoom", last), layer.get("maxzoom", last))
                known.setdefault("fields", {}).update(layer.get("fields", {}))
        merged["json"] = json.dumps({"vector_layers": list(layers.values())})
        return merged

    # --- Helpers ---

    @staticmethod
    def _remove(path: str):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    @staticmethod
    def _layer_name(layer: QgsVectorLayer) -> str:
        """Tile layer name: the dataset's file name without extension."""
        return basename(layer.source()).split(".")[0]

    def _parse_layer_zoom(self, layer: QgsVectorLayer, marker: str) -> int:
        """Extract a zoom level from the layer filename using the given marker character."""
        return int(self._layer_name(layer).split(marker)[1][:2])

    def _get_global_min_zoom(self) -> int:
        zooms = (self._parse_layer_zoom(layer, "o") for layer in self.layers)
        return min(zooms, default=0)

    def _get_global_max_zoom(self) -> int:
        zooms = (self._parse_layer_zoom(layer, "i") for layer in self.layers)
        return max(zooms, default=14)
