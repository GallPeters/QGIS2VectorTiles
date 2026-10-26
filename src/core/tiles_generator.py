"""
tiles_generator.py

GDALTilesGenerator — builds a multi-layer OGR VRT from the exported
datasets and calls ogr2ogr to produce MVT (Mapbox Vector Tiles) in MBTiles
format, then copies them into the export's single PMTiles file
(pmtiles_writer).

The MVT driver works a run's zoom levels one after another and gains
nothing from GDAL_NUM_THREADS (measured slower), so the zoom levels are
split into parallel ogr2ogr runs, one process each, whose MBTiles are then
merged into one (zoom levels never share a tile). The runs are balanced by
work: the lowest zoom levels share a run, and the costliest are split by
area into strips of tile columns: a strip's run reads only the features
near it, clipped to it, and keeps only its own tiles.

Each run's temporary database, which the driver writes every feature of
every tile to, is kept in memory while the runs' estimated sizes fit in
half the free memory (several times faster than on disk), else on disk.

Depends on: config (for constants only — no custom class dependencies)
"""

import ctypes
import json
import math
import os
import re
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
from ..utils.flattened_rule import CLUSTER_SIZE_FIELD, TILE_LAYERS_KEY
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

# A tile's width in screen pixels (MapLibre's zoom levels).
_TILE_PIXELS = 512

# Polygons smaller than this share of a screen pixel are left out of the
# tiles below the highest zoom: dense small polygons (parcels, buildings)
# make the lowest zooms' tiles the heaviest, for specks hardly drawn.
_MIN_POLYGON_PIXELS = 0.25

# Low-zoom generalization (optional, see GDALTilesGenerator): below the
# tiles' highest zoom, polygons smaller than this share of a screen pixel are
# left out, one point is kept per rule and grid cell of this many screen
# pixels (the one whose label has the highest priority), lines and
# polygons are simplified to a screen pixel (8 tile units) and tiles are
# limited to half the size.
_GENERALIZED_POLYGON_PIXELS = 1.0
_GENERALIZED_POINT_CELL_PIXELS = 16
_GENERALIZED_SIMPLIFICATION = 8

# A zoom level's work per byte of its data: 1, plus this much times the
# square root of its tile count, as features are cut into more tiles
# (fitted on Natural Earth z0-z8 and a city z0-z15, see _plan_runs).
_TILE_COST = 0.017

# ogr2ogr runs per worker (see _plan_runs). More, smaller runs even out a
# wrong estimate, but each reads and clips its features again: measured on
# Natural Earth and a city, three per worker were slower than one.
_RUNS_PER_WORKER = 1


@dataclass
class _Run:
    """One ogr2ogr run: zoom levels lo..hi, of all tile columns or of
    columns[0]..columns[1] only, written to ``output``."""
    lo: int
    hi: int
    columns: Optional[Tuple[int, int]] = None
    output: str = ""
    in_memory: bool = False
    # Estimated work (see GDALTilesGenerator._plan_runs).
    cost: float = 0.0


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
        generalize: bool = False,
    ):
        self.layers = layers
        self.style = style
        self.output_dir = output_dir
        self.extent = extent
        self.cpu_percent = cpu_percent
        self.feedback = feedback
        # Generalize the tiles below the highest zoom more (smaller, faster
        # to draw, less detailed); off by default.
        self.generalize = generalize
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
        runs = self._plan_runs(first, last, workers)
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
        size = max(_TILE_MAX_SIZE, min(_TILE_MAX_SIZE_FEW, _TILE_VIEW_SIZE // self._tile_count(zoom)))
        return size // 2 if self._generalized(zoom) else size

    def _generalized(self, last_zoom: Optional[int]) -> bool:
        """Whether a run whose last zoom level is ``last_zoom`` is
        generalized: when asked, below the tiles' highest zoom."""
        return (self.generalize and last_zoom is not None
                and last_zoom < getattr(self, "_last_zoom", last_zoom))

    # --- Planning the parallel runs ---

    def _plan_runs(self, first: int, last: int, workers: int) -> List[_Run]:
        """The ogr2ogr runs, about _RUNS_PER_WORKER per worker, of about
        equal work each, by estimate (_run_processes starts the largest
        first, and the others as workers finish).

        A zoom level's work is estimated from its datasets' size, more per
        byte the more tiles it has (_TILE_COST): a city's zoom levels cost
        about the same (its features are small next to the tiles, each
        written about once), the world's grow with the zoom (an ocean is
        cut into every tile). The lowest zoom levels share runs, and a zoom level
        costing more than a run's share is split into strips of tile
        columns. The estimate is rough: measured, a byte costs from one to
        twenty times more, with the features' size. The old one run per
        zoom level left most workers idle while Natural Earth's z8 took
        four minutes.
        """
        zooms = [z for z in range(first, last + 1) if self._run_layers(z, z)]
        if workers <= 1 or len(zooms) == 0:
            return [_Run(first, last)]
        cost = {}
        for z in zooms:
            data = sum(self._dataset_size(layer) for layer, _, _ in self._run_layers(z, z).values())
            cost[z] = max(data, 1) * (1 + _TILE_COST * math.sqrt(self._tile_count(z)))
        share = sum(cost.values()) / (workers * _RUNS_PER_WORKER)
        plan: List[list] = []  # [lo, hi, strips, cost]
        group: List[int] = []

        def close_group():
            if group:
                plan.append([group[0], group[-1], 1, sum(cost[z] for z in group)])
                group.clear()

        for z in zooms:
            if cost[z] >= share:
                close_group()
                c0, c1, _, _ = self._tile_range(z)
                plan.append([z, z, max(1, min(c1 - c0 + 1, round(cost[z] / share))), cost[z]])
                continue
            if group and sum(cost[g] for g in group) + cost[z] > share:
                close_group()
            group.append(z)
        close_group()
        runs = []
        for lo, hi, strips, work in plan:
            if strips == 1:
                runs.append(_Run(lo, hi, cost=work))
                continue
            c0, c1, _, _ = self._tile_range(hi)
            total = c1 - c0 + 1
            start = c0
            for i in range(strips):
                width = total // strips + (1 if i < total % strips else 0)
                runs.append(_Run(lo, hi, (start, start + width - 1), cost=work * width / total))
                start += width
        return runs

    def _strip_filter(self, run: _Run) -> List[str]:
        """ogr2ogr's spatial filter for a strip: its columns, widened by a
        tile on each side so that every feature reaching into its tiles
        (or their buffer) is read; nothing for a whole run. Features are
        also clipped to it: the driver writes a feature to every tile it
        reaches, so an ocean or a continent read whole would be cut into
        every tile of the world by every strip."""
        if run.columns is None:
            return []
        tile = 2 * _WORLD_HALF / 2 ** run.hi
        west = -_WORLD_HALF + run.columns[0] * tile - tile
        east = -_WORLD_HALF + (run.columns[1] + 1) * tile + tile
        return ["-spat", repr(west), repr(-_WORLD_HALF), repr(east), repr(_WORLD_HALF),
                "-clipsrc", "spat_extent"]

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

    def _build_layer_conf(self, conf_path: str, run_layers: dict, extra: Optional[dict] = None):
        """Write the MVT driver's CONF file: each layer's own zoom range
        (and ``extra`` entries, e.g. a clustered layer's zoom levels, see
        _cluster_blocks).

        Without it every layer is written to every zoom level of the
        dataset, although each style only shows it within its own range.
        """
        shared = self._shared_tile_layers()
        conf = {
            name: {"minzoom": minzoom, "maxzoom": maxzoom}
            for name, (_, minzoom, maxzoom) in run_layers.items()
        }
        for name, entry in conf.items():
            if name in shared:
                entry["target_name"] = shared[name]
        conf.update(extra or {})
        with open(conf_path, "w", encoding="utf-8") as f:
            json.dump(conf, f)

    def _shared_tile_layers(self) -> Dict[str, str]:
        """By dataset, the tile layer of another dataset it is written into
        (the style draws both as one layer: share_tile_layers)."""
        return (self.style.get("metadata") or {}).get(TILE_LAYERS_KEY, {})

    # --- VRT construction ---

    def _build_vrt(self, vrt_path: str, layers: List[QgsVectorLayer] = None, last_zoom: int = None,
                   extra: str = ""):
        """Write an OGR VRT containing one entry per layer, for a run whose
        last zoom level is ``last_zoom`` (see _min_area), and ``extra``
        entries (see _cluster_blocks)."""
        style_text = str(self.style)
        min_area = self._min_area(last_zoom)
        with open(vrt_path, "w", encoding="utf-8") as f:
            f.write("<OGRVRTDataSource>\n")
            for layer in (self.layers if layers is None else layers):
                f.write(self._vrt_layer_block(
                    layer, style_text, min_area, self._point_cell(last_zoom), last_zoom))
            f.write(extra)
            f.write("</OGRVRTDataSource>\n")

    # --- Point clusters ---

    def _cluster_sources(self) -> Dict[str, dict]:
        """The datasets drawn with a point cluster symbol (their style
        layers' "q2vt:cluster" metadata: the clustering distance in pixels
        or metres), by name."""
        clusters = {}
        for layer_def in self.style.get("layers", []):
            distance = (layer_def.get("metadata") or {}).get("q2vt:cluster")
            if distance and layer_def.get("source-layer"):
                clusters[layer_def["source-layer"]] = distance
        return clusters

    @staticmethod
    def _layer_prefix(name: str) -> str:
        """A dataset name's source layer and rule type ("l12t00")."""
        match = re.match(r"(l\d+t\d\d)", name)
        return match.group(1) if match else name

    def _cluster_of(self, name: str, clusters: Dict[str, dict]) -> Optional[str]:
        """The cluster dataset clustering a dataset's points: its own name
        for a cluster dataset, that of its source layer's for the layer's
        other point datasets (its points drawn alone), else None."""
        if name in clusters:
            return name
        if re.search(r"c00", name):
            prefix = self._layer_prefix(name)
            return next((c for c in clusters if self._layer_prefix(c) == prefix), None)
        return None

    def _cluster_blocks(self, run_layers: dict, clusters: Dict[str, dict], tag: str) -> Tuple[str, dict]:
        """VRT entries and CONF entries of a run's clustered datasets: per
        zoom level, a cluster dataset as one point per cluster (the members'
        mean position, their count in q2vt_cluster_size), the other point
        datasets of its layer as the points alone in their cluster. Points
        cluster in a grid of cells the clustering distance wide at that
        zoom (QGIS clusters by distance from each point); each zoom's entry
        writes the dataset's own tile layer (CONF target_name)."""
        by_name = {self._layer_name(layer): layer for layer in self.layers}
        blocks, conf = [], {}
        for name, (layer, minzoom, maxzoom) in run_layers.items():
            cluster = self._cluster_of(name, clusters)
            if cluster is None or cluster not in by_name:
                continue
            source = layer.source().split("|layername=")[0]
            cluster_source = by_name[cluster].source().split("|layername=")[0]
            ds = ogr.Open(source)
            cds = ogr.Open(cluster_source)
            if ds is None or cds is None:
                continue
            layer_name = ds.GetLayer(0).GetName().replace('"', '""')
            cluster_layer = cds.GetLayer(0).GetName()
            ds = cds = None
            distance = clusters[cluster]
            for zoom in range(minzoom, maxzoom + 1):
                cell = distance.get("m") or distance.get("px", 10) * 2 * _WORLD_HALF / (_TILE_PIXELS * 2 ** zoom)
                key_x = f'CAST(floor(ST_X(GEOMETRY) / {cell!r}) AS INTEGER)'
                key_y = f'CAST(floor(ST_Y(GEOMETRY) / {cell!r}) AS INTEGER)'
                entry = f"{name}__z{zoom:02d}"
                conf[entry] = {"target_name": name, "minzoom": zoom, "maxzoom": zoom}
                if name == cluster:
                    sql = (f"SELECT MakePoint(AVG(ST_X(GEOMETRY)), AVG(ST_Y(GEOMETRY)), {_EPSG_CRS}) AS GEOMETRY, "
                           f'COUNT(*) AS "{CLUSTER_SIZE_FIELD}" FROM "{layer_name}" '
                           f"GROUP BY {key_x}, {key_y} HAVING COUNT(*) > 1")  # nosec B608
                    blocks.append(
                        f'    <OGRVRTLayer name={quoteattr(entry)}>\n'
                        f'        <SrcDataSource>{escape(source)}</SrcDataSource>\n'
                        f'        <SrcSQL dialect="SQLITE">{escape(sql)}</SrcSQL>\n'
                        f'        <LayerSRS>EPSG:{_EPSG_CRS}</LayerSRS>\n'
                        f'        <GeometryType>wkbPoint</GeometryType>\n'
                        f'        <Field name="{CLUSTER_SIZE_FIELD}" type="Integer"/>\n'
                        f'    </OGRVRTLayer>\n')
                    continue
                # The points alone in their cell, the cells counted over every
                # point (the cluster dataset's), in one data source.
                pair = join(QgsProcessingUtils.tempFolder(), f"cluster_{tag}_{len(blocks)}.vrt")
                with open(pair, "w", encoding="utf-8") as f:
                    f.write(
                        "<OGRVRTDataSource>"
                        f'<OGRVRTLayer name="pts"><SrcDataSource>{escape(source)}</SrcDataSource>'
                        f"<SrcLayer>{escape(layer_name.replace(chr(34) * 2, chr(34)))}</SrcLayer></OGRVRTLayer>"
                        f'<OGRVRTLayer name="allp"><SrcDataSource>{escape(cluster_source)}</SrcDataSource>'
                        f"<SrcLayer>{escape(cluster_layer)}</SrcLayer></OGRVRTLayer>"
                        "</OGRVRTDataSource>")
                pts_x = key_x.replace("GEOMETRY", "pts.GEOMETRY")
                pts_y = key_y.replace("GEOMETRY", "pts.GEOMETRY")
                # (OGR finds the tables an SQLite query reads in its text:
                # not inside a derived table, but in an IN subquery.)
                sql = (f"SELECT pts.* FROM pts WHERE ({pts_x}, {pts_y}) IN "
                       f"(SELECT {key_x}, {key_y} FROM allp GROUP BY 1, 2 HAVING COUNT(*) = 1)")  # nosec B608
                blocks.append(
                    f'    <OGRVRTLayer name={quoteattr(entry)}>\n'
                    f'        <SrcDataSource>{escape(pair)}</SrcDataSource>\n'
                    f'        <SrcSQL dialect="SQLITE">{escape(sql)}</SrcSQL>\n'
                    f'        <LayerSRS>EPSG:{_EPSG_CRS}</LayerSRS>\n'
                    f'        <GeometryType>wkbUnknown</GeometryType>\n'
                    f'{self._vrt_fields(source, str(self.style))}'
                    f'    </OGRVRTLayer>\n')
        return "".join(blocks), conf

    def _min_area(self, last_zoom: Optional[int]) -> Optional[float]:
        """The smallest polygon (EPSG:3857 square metres) a run's tiles
        keep: _MIN_POLYGON_PIXELS of a screen pixel at its last zoom level,
        below which a polygon shows as no more than a faint speck. None at
        the tiles' highest zoom, whose tiles are enlarged beyond it."""
        if last_zoom is None or last_zoom >= getattr(self, "_last_zoom", last_zoom):
            return None
        pixel = 2 * _WORLD_HALF / (_TILE_PIXELS * 2 ** last_zoom)
        pixels = _GENERALIZED_POLYGON_PIXELS if self._generalized(last_zoom) else _MIN_POLYGON_PIXELS
        return pixels * pixel * pixel

    def _point_cell(self, last_zoom: Optional[int]) -> Optional[float]:
        """The grid cell (EPSG:3857 metres) of a generalized run's point
        thinning; None when the run isn't generalized."""
        if not self._generalized(last_zoom):
            return None
        return _GENERALIZED_POINT_CELL_PIXELS * 2 * _WORLD_HALF / (_TILE_PIXELS * 2 ** last_zoom)

    def _vrt_layer_block(self, layer: QgsVectorLayer, style_text: str,
                         min_area: Optional[float] = None, point_cell: Optional[float] = None,
                         last_zoom: Optional[int] = None) -> str:
        """Return the VRT XML block for a single layer; a polygon layer's
        smaller polygons than ``min_area`` are left out, and a point layer
        keeps one point per rule and grid cell of ``point_cell``: the one
        whose label MapLibre places first (see _label_sort_key), else its
        first."""
        source = layer.source().split("|layername=")[0]
        geometry = re.search(r"c(\d\d)", self._layer_name(layer))
        geometry = geometry.group(1) if geometry else None
        src_sql = ""
        if (min_area and geometry == "02") or (point_cell and geometry == "00"):
            ds = ogr.Open(source)
            if ds is not None:
                # SQL over the plugin's own dataset; the names are quoted.
                source_layer = ds.GetLayer(0)
                name = source_layer.GetName().replace('"', '""')
                if geometry == "02":
                    sql = f'SELECT * FROM "{name}" WHERE OGR_GEOM_AREA >= {min_area!r}'  # nosec B608
                    src_sql = f'        <SrcSQL dialect="OGRSQL">{escape(sql)}</SrcSQL>\n'
                else:
                    # Run by ogr2ogr: SQLite SQL never runs in QGIS's process.
                    column = (source_layer.GetGeometryColumn() or "GEOMETRY").replace('"', '""')
                    keys = [f'CAST(floor(ST_X("{column}") / {point_cell!r}) AS INTEGER)',
                            f'CAST(floor(ST_Y("{column}") / {point_cell!r}) AS INTEGER)']
                    if source_layer.GetLayerDefn().GetFieldIndex(f"{_FIELD_PREFIX}_rule") >= 0:
                        keys.insert(0, f'"{_FIELD_PREFIX}_rule"')
                    defn = source_layer.GetLayerDefn()
                    fields = {defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())}
                    sort_key = self._label_sort_key(self._layer_name(layer), fields, last_zoom)
                    # SQLite takes a bare column (rowid) from the row whose
                    # min() it returns.
                    kept = (f'SELECT r FROM (SELECT rowid AS r, min(coalesce({sort_key}, 1e300)) '
                            f'FROM "{name}" GROUP BY {", ".join(keys)})' if sort_key else
                            f'SELECT min(rowid) FROM "{name}" GROUP BY {", ".join(keys)}')
                    # "+rowid": SQLite would otherwise fetch each kept row
                    # by its FID, a scan of the file each (minutes).
                    sql = f'SELECT * FROM "{name}" WHERE +rowid IN ({kept})'  # nosec B608
                    src_sql = f'        <SrcSQL dialect="SQLITE">{escape(sql)}</SrcSQL>\n'
                ds = None
        return (
            f'    <OGRVRTLayer name={quoteattr(self._layer_name(layer))}>\n'
            f'        <SrcDataSource>{escape(source)}</SrcDataSource>\n'
            f'{src_sql}'
            f'        <LayerSRS>EPSG:{_EPSG_CRS}</LayerSRS>\n'
            f'        <GeometryType>wkbUnknown</GeometryType>\n'
            f'{self._vrt_fields(source, style_text)}'
            f'    </OGRVRTLayer>\n'
        )

    def _label_sort_key(self, source_layer: str, fields: set, zoom: Optional[int]) -> Optional[str]:
        """The SQLite expression of a dataset's label sort key (MapLibre's
        symbol-sort-key, from the QGIS label priority: lower is placed
        first), when its labels have one varying by feature; else None.

        One expression is needed: the style's label layers of the dataset
        share one, or each draws one rule (q2vt_rule), whose keys are then
        chosen by rule, or else (several labels of a feature) the lowest of
        their keys, its best placed label."""
        keyed = []
        # A dataset sharing another's tile layer is drawn by that one's layers.
        source_layer = self._shared_tile_layers().get(source_layer, source_layer)
        for layer_def in self.style.get("layers", []):
            layout = layer_def.get("layout", {})
            if (layer_def.get("source-layer") != source_layer or "text-field" not in layout
                    or "symbol-sort-key" not in layout):
                continue
            if zoom is not None and not (layer_def.get("minzoom", 0) < zoom + 1
                                         and zoom < layer_def.get("maxzoom", 99)):
                continue
            keyed.append((layer_def.get("filter"), layout["symbol-sort-key"]))
        if not keyed or all(not isinstance(key, list) for _, key in keyed):
            return None
        try:
            if all(key == keyed[0][1] for _, key in keyed):
                return self._sql_of(keyed[0][1], fields, zoom)
            rule_field = f"{_FIELD_PREFIX}_rule"
            if rule_field in fields and all(
                    isinstance(flt, list) and len(flt) == 3 and flt[0] == "=="
                    and flt[1] == ["get", rule_field] and isinstance(flt[2], int)
                    for flt, _ in keyed):
                cases = " ".join(
                    f"WHEN {flt[2]} THEN {self._sql_of(key, fields, zoom)}" for flt, key in keyed)
                return f'(CASE "{rule_field}" {cases} END)'
            keys = []
            for _, key in keyed:
                sql = f"coalesce({self._sql_of(key, fields, zoom)}, 1e300)"
                if sql not in keys:
                    keys.append(sql)
            return f"min({', '.join(keys)})"
        except ValueError:
            return None

    @classmethod
    def _sql_of(cls, expression, fields: set, zoom: Optional[int]) -> str:
        """A sort key expression, as the style writes them, in SQLite;
        raises ValueError for anything else."""
        def sql(e) -> str:
            return cls._sql_of(e, fields, zoom)

        if isinstance(expression, bool):
            raise ValueError(expression)
        if isinstance(expression, (int, float)):
            return repr(float(expression))
        if isinstance(expression, str):
            return "'" + expression.replace("'", "''") + "'"
        if not isinstance(expression, list) or not expression:
            raise ValueError(expression)
        op, args = expression[0], expression[1:]
        if op == "get" and len(args) == 1 and isinstance(args[0], str):
            return '"' + args[0].replace('"', '""') + '"' if args[0] in fields else "NULL"
        if op == "literal" and len(args) == 1:
            return sql(args[0])
        if op == "zoom" and zoom is not None:
            return repr(float(zoom))
        if op == "-" and len(args) in (1, 2):
            return f"(-{sql(args[0])})" if len(args) == 1 else f"({sql(args[0])} - {sql(args[1])})"
        if op in ("+", "*") and args:
            return "(" + f" {op} ".join(sql(a) for a in args) + ")"
        if op in ("min", "max", "coalesce") and args:
            items = [sql(a) for a in args]
            return items[0] if len(items) == 1 else f"{op}({', '.join(items)})"
        if op == "to-number" and args:
            # SQLite's CAST makes any text a number (0 if not one).
            return "coalesce(" + ", ".join(
                f"CAST(NULLIF({sql(a)}, '') AS REAL)" for a in args) + ", 0)"
        if op == "match" and len(args) >= 4 and len(args) % 2 == 0:
            whens = []
            for labels, output in zip(args[1:-1:2], args[2:-1:2]):
                for label in labels if isinstance(labels, list) else [labels]:
                    whens.append(f"WHEN {sql(label)} THEN {sql(output)}")
            return f"(CASE {sql(args[0])} {' '.join(whens)} ELSE {sql(args[-1])} END)"
        if op == "step" and len(args) >= 2 and args[0] == ["zoom"] and zoom is not None \
                and len(args) % 2 == 0:
            result = args[1]
            for stop, output in zip(args[2::2], args[3::2]):
                if zoom < stop:
                    break
                result = output
            return sql(result)
        raise ValueError(expression)

    @staticmethod
    def _is_tiled_field(field_name: str, style_text: str) -> bool:
        """Whether a dataset field belongs in the tiles.

        q2vt_orig_id only serves the rules export, q2vt_description is in
        the style's metadata (one per rule, not per feature), and
        data-defined property fields matter only if the style references
        them.
        """
        if field_name in (f"{_FIELD_PREFIX}_orig_id", f"{_FIELD_PREFIX}_biggest",
                          f"{_FIELD_PREFIX}_description"):
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
        clusters = self._cluster_sources()
        plain = {name: entry for name, entry in run_layers.items()
                 if self._cluster_of(name, clusters) is None}
        cluster_xml, cluster_conf = self._cluster_blocks(run_layers, clusters, tag) if clusters else ("", {})
        self._build_vrt(vrt_path, [layer for layer, _, _ in plain.values()], hi, cluster_xml)
        self._build_layer_conf(conf_path, plain, cluster_conf)
        # The temporary database is written heavily: in memory if it fits,
        # else on local disk, not next to the output (which may be a synced
        # or network folder).
        if run.in_memory:
            temporary_db = f"/vsimem/tiles_{tag}.temp.db"
        else:
            temporary_db = join(QgsProcessingUtils.tempFolder(), f"tiles_{tag}.temp.db")
        # The driver applies SIMPLIFICATION_MAX_ZOOM at a run's last zoom:
        # only the tiles' real last zoom is left unsimplified.
        simplification = _GENERALIZED_SIMPLIFICATION if self._generalized(hi) else _SIMPLIFICATION
        last_zoom_simplification = (
            _SIMPLIFICATION_MAX_ZOOM if hi >= getattr(self, "_last_zoom", hi) else simplification
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
            "-dsco", f"SIMPLIFICATION={simplification}",
            "-dsco", f"SIMPLIFICATION_MAX_ZOOM={last_zoom_simplification}",
            *self._strip_filter(run),
        ]

    def _run_ogr2ogr_runs(self, runs: List[_Run]):
        """Run ogr2ogr once per run, a worker's each, on one thread; raises
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
        """Run the runs' ogr2ogr processes, as many at once as there are
        workers, the largest estimated first; the failed runs with the end
        of their error output."""
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
        waiting = sorted(runs, key=lambda r: -r.cost)
        workers = self._worker_count()
        # Per run: when it started, and how long it took.
        started: Dict[int, float] = {}
        took: Dict[int, float] = {}
        try:
            while waiting or any(process.poll() is None for _, process, _ in processes):
                if self._cancelled.is_set() or (
                        self.feedback is not None and self.feedback.isCanceled()):
                    raise RuntimeError("Tile generation cancelled.")
                for run, process, _ in processes:
                    if process.poll() is not None and id(run) not in took:
                        took[id(run)] = time.perf_counter() - started[id(run)]
                while waiting and len(processes) - len(took) < workers:
                    run = waiting.pop(0)
                    stderr_path = f"{run.output}.log"
                    with open(stderr_path, "w", encoding="utf-8") as stderr:
                        started[id(run)] = time.perf_counter()
                        processes.append((run, subprocess.Popen(
                            self._ogr2ogr_command(run), env=env,
                            stdout=subprocess.DEVNULL, stderr=stderr,
                            startupinfo=startupinfo, creationflags=creationflags,
                        ), stderr_path))
                time.sleep(0.25)
            if self.feedback is not None:
                for run, _, _ in processes:
                    columns = f" columns {run.columns[0]}-{run.columns[1]}" if run.columns else ""
                    seconds = took.get(id(run), time.perf_counter() - started[id(run)])
                    self.feedback.pushDebugInfo(
                        f"Tiles z{run.lo}-{run.hi}{columns}: {seconds:.1f} s"
                        f"{' (database in memory)' if run.in_memory else ''}")
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
