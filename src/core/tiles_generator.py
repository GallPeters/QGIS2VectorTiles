"""
tiles_generator.py

GDALTilesGenerator — builds a multi-layer OGR VRT from the exported
GeoParquet datasets and calls ogr2ogr to produce MVT (Mapbox Vector Tiles)
in XYZ directory layout.

Depends on: config (for constants only — no custom class dependencies)
"""

import os
import subprocess
from os.path import join, basename
from os import cpu_count
from typing import List, Tuple
from qgis.core import QgsVectorLayer, QgsProcessingFeedback, QgsProcessingUtils
from ..utils.config import _EPSG_CRS


class GDALTilesGenerator:
    """Generate XYZ tiles using GDAL CLI + VRT (fast, multi-layer, per-zoom)."""

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
        """Main entry point."""
        output, uri = self._prepare_output_paths()
        vrt_path = join(QgsProcessingUtils.tempFolder(), "layers.vrt")

        min_zoom = self._get_global_min_zoom()
        max_zoom = self._get_global_max_zoom()

        self._build_vrt(vrt_path)
        self._run_ogr2ogr(vrt_path, output, min_zoom, max_zoom)

        return uri, min_zoom

    # VRT creation

    def _build_vrt(self, vrt_path: str):
        """Build VRT with per-layer zoom configuration."""
        with open(vrt_path, "w", encoding="utf-8") as f:
            f.write("<OGRVRTDataSource>\n")

            for layer in self.layers:
                layer_name = basename(layer.source()).split(".")[0]

                # Extract zooms from naming convention
                min_zoom = int(layer_name.split("o")[1][:2])
                max_zoom = int(layer_name.split("i")[1][:2])

                source = layer.source().split("|layername=")[0]

                f.write(f"""
    <OGRVRTLayer name="{layer_name}">
        <SrcDataSource>{source}</SrcDataSource>
        <LayerSRS>EPSG:{_EPSG_CRS}</LayerSRS>
        <GeometryType>wkbUnknown</GeometryType>
        <LayerCreationOption name="MINZOOM" value="{min_zoom}"/>
        <LayerCreationOption name="MAXZOOM" value="{max_zoom}"/>
    </OGRVRTLayer>
""")

            f.write("</OGRVRTDataSource>\n")

    # ogr2ogr execution

    def _run_ogr2ogr(self, vrt_path: str, output: str, min_zoom: int, max_zoom: int):
        """Run single ogr2ogr command."""

        cpu_num = str(max(1, int(cpu_count() * self.cpu_percent / 100)))
        env = os.environ.copy()
        env["GDAL_NUM_THREADS"] = cpu_num

        cmd = [
            "ogr2ogr",
            "-f",
            "MBTiles",
            output,
            vrt_path,
            "-dsco",
            f"MINZOOM={min_zoom}",
            "-dsco",
            f"MAXZOOM={max_zoom}",
            "-t_srs",
            f"EPSG:{_EPSG_CRS}",
            "-dsco",
            "MAX_SIZE=1000000",
            "-dsco",
            "MAX_FEATURES=400000"
        ]

        # Windows: hide console window
        startupinfo = None
        creationflags = 0

        if os.name == "nt":
            creationflags = 0x08000000  # CREATE_NO_WINDOW
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0

        try:
            subprocess.run(
                cmd,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                startupinfo=startupinfo,
                creationflags=creationflags,
            )
        except subprocess.CalledProcessError as e:
            error_msg = f"ogr2ogr failed.\nError: {e.stderr}"
            if self.feedback:
                self.feedback.reportError(error_msg)
            raise RuntimeError(error_msg) from e

    # Helpers

    def _prepare_output_paths(self) -> Tuple[str, str]:
        output = join(self.output_dir, "tiles.mbtiles")
        uri = f"type=mbtiles&url={output}"
        return output, uri

    def _get_global_min_zoom(self) -> int:
        min_zoom = float("inf")
        for layer in self.layers:
            name = basename(layer.source()).split(".")[0]
            zoom = int(name.split("o")[1][:2])
            min_zoom = min(min_zoom, zoom)
        return int(min_zoom) if min_zoom != float("inf") else 0

    def _get_global_max_zoom(self) -> int:
        max_zoom = 0
        for layer in self.layers:
            name = basename(layer.source()).split(".")[0]
            zoom = int(name.split("i")[1][:2])
            max_zoom = max(max_zoom, zoom)
        return max_zoom if max_zoom > 0 else 14
