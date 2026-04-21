"""
tiles_generator.py

GDALTilesGenerator — builds a multi-layer OGR VRT from the exported
GeoParquet datasets and calls ogr2ogr to produce MVT (Mapbox Vector Tiles)
in MBTiles format.

Depends on: config (for constants only — no custom class dependencies)
"""

import os
import subprocess
from os import cpu_count
from os.path import join, basename
from typing import List, Tuple

from qgis.core import QgsVectorLayer, QgsProcessingFeedback, QgsProcessingUtils

from ..utils.config import _EPSG_CRS


class GDALTilesGenerator:
    """Generate MBTiles vector tiles using GDAL CLI with an OGR VRT intermediary."""

    def __init__(
        self,
        layers: List[QgsVectorLayer],
        output_dir: str,
        extent,
        cpu_percent: int,
        feedback: QgsProcessingFeedback,
    ):
        self.layers = layers
        self.output_dir = output_dir
        self.extent = extent
        self.cpu_percent = cpu_percent
        self.feedback = feedback

    def generate(self) -> Tuple[str, int]:
        """Build VRT, run ogr2ogr, return (mbtiles URI, min_zoom)."""
        output, uri = self._prepare_output_paths()
        vrt_path = join(QgsProcessingUtils.tempFolder(), "layers.vrt")

        min_zoom = self._get_global_min_zoom()
        max_zoom = self._get_global_max_zoom()

        self._build_vrt(vrt_path)
        self._run_ogr2ogr(vrt_path, output, min_zoom, max_zoom)

        return uri, min_zoom

    # --- VRT construction ---

    def _build_vrt(self, vrt_path: str):
        """Write an OGR VRT containing one entry per layer with per-zoom configuration."""
        with open(vrt_path, "w", encoding="utf-8") as f:
            f.write("<OGRVRTDataSource>\n")
            for layer in self.layers:
                f.write(self._vrt_layer_block(layer))
            f.write("</OGRVRTDataSource>\n")

    def _vrt_layer_block(self, layer: QgsVectorLayer) -> str:
        """Return the VRT XML block for a single layer."""
        name = basename(layer.source()).split(".")[0]
        min_zoom = self._parse_layer_zoom(layer, "o")
        max_zoom = self._parse_layer_zoom(layer, "i")
        source = layer.source().split("|layername=")[0]
        return (
            f'    <OGRVRTLayer name="{name}">\n'
            f'        <SrcDataSource>{source}</SrcDataSource>\n'
            f'        <LayerSRS>EPSG:{_EPSG_CRS}</LayerSRS>\n'
            f'        <GeometryType>wkbUnknown</GeometryType>\n'
            f'        <LayerCreationOption name="MINZOOM" value="{min_zoom}"/>\n'
            f'        <LayerCreationOption name="MAXZOOM" value="{max_zoom}"/>\n'
            f'    </OGRVRTLayer>\n'
        )

    # --- ogr2ogr execution ---

    def _run_ogr2ogr(self, vrt_path: str, output: str, min_zoom: int, max_zoom: int):
        """Execute ogr2ogr to convert the VRT to MBTiles."""
        cpu_num = str(max(1, int(cpu_count() * self.cpu_percent / 100)))
        env = os.environ.copy()
        env["GDAL_NUM_THREADS"] = cpu_num

        cmd = [
            "ogr2ogr", "-f", "MBTiles", output, vrt_path,
            "-dsco", f"MINZOOM={min_zoom}",
            "-dsco", f"MAXZOOM={max_zoom}",
            "-t_srs", f"EPSG:{_EPSG_CRS}",
            "-dsco", "MAX_SIZE=1000000",
            "-dsco", "MAX_FEATURES=400000",
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

    def _parse_layer_zoom(self, layer: QgsVectorLayer, marker: str) -> int:
        """Extract a zoom level from the layer filename using the given marker character."""
        name = basename(layer.source()).split(".")[0]
        return int(name.split(marker)[1][:2])

    def _get_global_min_zoom(self) -> int:
        zooms = (self._parse_layer_zoom(layer, "o") for layer in self.layers)
        return min(zooms, default=0)

    def _get_global_max_zoom(self) -> int:
        zooms = (self._parse_layer_zoom(layer, "i") for layer in self.layers)
        return max(zooms, default=14)
