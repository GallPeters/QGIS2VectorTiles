"""
tiles_generator.py

GDALTilesGenerator — builds a multi-layer OGR VRT from the exported
GeoParquet datasets and calls ogr2ogr to produce MVT (Mapbox Vector Tiles)
in MBTiles format.

Depends on: config (for constants only — no custom class dependencies)
"""

import json
import os
import subprocess
from os import cpu_count
from os.path import join, basename
from typing import List, Tuple
from xml.sax.saxutils import escape, quoteattr
from osgeo import ogr
from qgis.core import QgsVectorLayer, QgsProcessingFeedback, QgsProcessingUtils

from ..utils.config import _EPSG_CRS, _FIELD_PREFIX, _SIMPLIFICATION, _SIMPLIFICATION_MAX_ZOOM


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

    def generate(self) -> Tuple[str, int]:
        """Build VRT, run ogr2ogr, return (mbtiles URI, min_zoom)."""
        output, uri = self._prepare_output_paths()
        vrt_path = join(QgsProcessingUtils.tempFolder(), "layers.vrt")
        conf_path = join(QgsProcessingUtils.tempFolder(), "layers_conf.json")

        min_zoom = self._get_global_min_zoom()
        max_zoom = self._get_global_max_zoom()

        self._build_vrt(vrt_path)
        self._build_layer_conf(conf_path)
        self._run_ogr2ogr(vrt_path, conf_path, output, min_zoom, max_zoom)

        return uri, min_zoom

    # --- Per-layer zoom configuration ---

    def _build_layer_conf(self, conf_path: str):
        """Write the MVT driver's CONF file: each layer's own zoom range.

        Without it every layer is written to every zoom level of the
        dataset, although each style only shows it within its own range.
        """
        conf = {
            self._layer_name(layer): {
                "minzoom": self._parse_layer_zoom(layer, "o"),
                "maxzoom": min(self._parse_layer_zoom(layer, "i"), 16),
            }
            for layer in self.layers
        }
        with open(conf_path, "w", encoding="utf-8") as f:
            json.dump(conf, f)

    # --- VRT construction ---

    def _build_vrt(self, vrt_path: str):
        """Write an OGR VRT containing one entry per layer."""
        style_text = str(self.style)
        with open(vrt_path, "w", encoding="utf-8") as f:
            f.write("<OGRVRTDataSource>\n")
            for layer in self.layers:
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
        if field_name == f"{_FIELD_PREFIX}_orig_id":
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

    def _run_ogr2ogr(
        self, vrt_path: str, conf_path: str, output: str, min_zoom: int, max_zoom: int
    ):
        """Execute ogr2ogr to convert the VRT to MBTiles."""
        cpu_num = str(max(1, int(cpu_count() * self.cpu_percent / 100)))
        env = os.environ.copy()
        env["GDAL_NUM_THREADS"] = cpu_num

        cmd = [
            "ogr2ogr", "-f", "MBTiles", output, vrt_path,
            "-dsco", f"MINZOOM={min_zoom}",
            "-dsco", f"MAXZOOM={max_zoom}",
            "-dsco", f"CONF={conf_path}",
            "-t_srs", f"EPSG:{_EPSG_CRS}",
            "-dsco", "MAX_SIZE=5000000",
            "-dsco", "MAX_FEATURES=2000000",
            "-dsco", f"SIMPLIFICATION={_SIMPLIFICATION}",
            "-dsco", f"SIMPLIFICATION_MAX_ZOOM={_SIMPLIFICATION_MAX_ZOOM}"
            
        ]

        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            creationflags = 0x08000000  # CREATE_NO_WINDOW
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0

        try:
            subprocess.run(
                cmd, env=env, check=True, capture_output=True, text=True,
                startupinfo=startupinfo, creationflags=creationflags,
            )
        except subprocess.CalledProcessError as e:
            error_msg = f"ogr2ogr failed.\nError: {e.stderr}"
            if self.feedback:
                self.feedback.reportError(error_msg)
            raise RuntimeError(error_msg) from e

    # --- Helpers ---

    def _prepare_output_paths(self) -> Tuple[str, str]:
        output = join(self.output_dir, "tiles.mbtiles")
        return output, f"type=mbtiles&url={output}"

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
