"""QGIS Processing Algorithms for QGIS2VectorTiles plugin."""

import os
from os.path import dirname, join
from typing import Optional
from uuid import uuid4
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterNumber,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFolderDestination,
    QgsCoordinateReferenceSystem,
    QgsProcessingParameterEnum,
    QgsProcessingUtils,
    QgsProject,
)
from qgis.utils import iface
from ..qgis2vectortiles import QGIS2VectorTiles
from ..core.server_initializer import ServerInitializer
from ..utils.config import _PLUGIN_DIR, _EPSG_CRS
from .subprocess_runner import python_executable, run_in_subprocess

# The plugin package folder (…/QGIS2VectorTiles), for the worker process.
_PACKAGE_DIR = dirname(dirname(dirname(__file__)))
# Providers a separate process can't read (their data lives in QGIS's memory).
_IN_PROCESS_ONLY_PROVIDERS = frozenset({"memory"})

_ICON = QIcon(join(_PLUGIN_DIR, "icon.png"))


class QGIS2VectorTilesAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for generating styled tiles from project layers.
    This wrapper provides a user interface for the tiles generation process
    through the QGIS Processing Toolbox.
    """

    # Parameter names (constants for consistency)
    MIN_ZOOM = "MIN_ZOOM"
    MAX_ZOOM = "MAX_ZOOM"
    EXTENT = "EXTENT"
    CPU_PERCENT = "CPU_PERCENT"
    OUTPUT_DIR = "OUTPUT_DIR"
    REQUIRED_FIELDS_ONLY = "FIELDS_INCLUDED"
    OUTPUT_TYPE = "OUTPUT_TYPE"
    POLYGONS_LABELS_BASE = "POLYGONS_LABELS_BASE"
    VIEWER = "VIEWER"
    BACKGROUND_TYPE = "BACKGROUND_TYPE"

    def __init__(self):
        """Initialize the algorithm"""
        super().__init__()
        self._runner = None          # In-process run, finished in postProcessAlgorithm.
        self._finish_args = None     # Separate-process run, likewise.
        self._project_path = None    # Saved project the worker process reads.
        self._project_is_copy = False

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        """
        Returns a new instance of the algorithm. Required by QGIS Processing framework.
        """
        return QGIS2VectorTilesAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm.
        This string should be fixed for the algorithm, and must not be localized.
        """
        return "QGIS2VectorTiles_action"

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr("QGIS2VectorTiles")

    def group(self):
        """
        Returns the name of the group this algorithm belongs to.
        """
        return None  # No inner group as requested

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to.
        """
        return None  # No inner group as requested

    def icon(self):
        """
        Returns the algorithm icon.
        """
        return _ICON

    def shortHelpString(self):
        """
        Returns a localised short helper string for the algorithm.
        """
        return self.tr(
            "QGIS2VectorTiles converts a QGIS project into a vector tile package with a single vector tile source, a web style matching the original QGIS styling, and a ready-to-use web viewer.\nIt enables fast client-side rendering, lightweight publishing, and easy sharing - without servers or third-party libraries.\nMore information can be found at: https://gallpeters.github.io/QGIS2VectorTiles"
        )

    def initAlgorithm(self, config=None):  # pylint: disable=W0613
        """
        Define the inputs and outputs of the algorithm.
        """

        # Minimum zoom level parameter
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_ZOOM,
                self.tr("Minimum Zoom (Inclusive)"),
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=0,
                minValue=0,
                maxValue=22,
            )
        )

        # Maximum zoom level parameter
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_ZOOM,
                self.tr("Maximum Zoom (Inclusive)"),
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=10,
                minValue=0,
                maxValue=22,
            )
        )

        # Extent parameter - defaults to current map canvas extent
        extent_param = QgsProcessingParameterExtent(
            self.EXTENT, self.tr("Tiles Extent"), optional=False
        )
        # Set default to current map canvas extent if available
        if iface and iface.mapCanvas():
            extent_param.setDefaultValue(iface.mapCanvas().extent())
        self.addParameter(extent_param)

        # CPU Percent parameter
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CPU_PERCENT,
                self.tr("CPU Usage Limit (%)"),
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=100,
                minValue=0,
                maxValue=100,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.REQUIRED_FIELDS_ONLY,
                self.tr("Included Fields"),
                options=["Required Fields Only", "All Fields"],
                defaultValue=0,  # Default to Required Fields Only
                optional=False,
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.POLYGONS_LABELS_BASE,
                self.tr("Polygon Labels Base"),
                options=["Whole Polygon", "Visible Polygon"],
                defaultValue=0,  # Default to Required Fields Only
                optional=False,
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.BACKGROUND_TYPE,
                self.tr("Background"),
                options=["OpenStreetMap", "NASA's BlueMarble Imagery", "Project Background Color"],
                defaultValue=0,  # Default to Required Fields Only
                optional=False,
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.VIEWER,
                self.tr("Output Viewer"),
                options=["MapLibre (Recommended)", "OpenLayers"],
                defaultValue=0,  # Default to Required Fields Only
                optional=False,
            )
        )

        # Output directory parameter
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_DIR, self.tr("Output Directory"), optional=False
            )
        )

    def checkParameterValues(self, parameters, context):
        """
        Validate parameter values before processing.
        Returns tuple (is_valid, error_message)
        """
        min_zoom = self.parameterAsInt(parameters, self.MIN_ZOOM, context)
        max_zoom = self.parameterAsInt(parameters, self.MAX_ZOOM, context)

        # Check that min_zoom <= max_zoom
        if min_zoom > max_zoom:
            return False, self.tr(
                "Minimum zoom level must be less than or equal to maximum zoom level"
            )

        return super().checkParameterValues(parameters, context)

    def prepareAlgorithm(self, parameters, context, feedback):
        """Runs on QGIS's main thread before processAlgorithm."""
        QGIS2VectorTiles.clear_project()
        self._project_path = self._project_for_worker(feedback)
        return super().prepareAlgorithm(parameters, context, feedback)

    def _project_for_worker(self, feedback) -> Optional[str]:
        """A saved project the worker process can read, or None to run the
        conversion inside QGIS (main thread only: it may save a copy)."""
        if python_executable() is None:
            return None
        project = QgsProject.instance()
        root = project.layerTreeRoot()
        for layer in project.mapLayers().values():
            node = root.findLayer(layer.id())
            if (node is not None and node.isVisible()
                    and layer.providerType() in _IN_PROCESS_ONLY_PROVIDERS):
                feedback.pushInfo(
                    f'. The "{layer.name()}" layer is a temporary (memory) layer; '
                    "running inside QGIS."
                )
                return None
        if project.fileName() and not project.isDirty():
            self._project_is_copy = False
            return project.fileName()
        # Save the current state to a copy without changing the open project.
        path = join(QgsProcessingUtils.tempFolder(), f"q2vt_project_{uuid4().hex}.qgz")
        file_name, dirty = project.fileName(), project.isDirty()
        try:
            saved = project.write(path)
        finally:
            project.setFileName(file_name)
            project.setDirty(dirty)
        self._project_is_copy = saved
        return path if saved else None

    def processAlgorithm(self, parameters, context, feedback):
        """
        Main processing method. This is where your existing vector tiles generation logic
        should be called.

        Args:
            parameters: Dictionary containing parameter values
            context: QgsProcessingContext object
            feedback: QgsProcessingFeedback object for progress reporting

        Returns:
            Dictionary with results (can be empty for this use case)
        """

        # Extract parameter values
        min_zoom = self.parameterAsInt(parameters, self.MIN_ZOOM, context)
        max_zoom = self.parameterAsInt(parameters, self.MAX_ZOOM, context)
        extent = self.parameterAsExtent(
            parameters, self.EXTENT, context, QgsCoordinateReferenceSystem(f"EPSG:{_EPSG_CRS}")
        )
        cpu_percent = self.parameterAsInt(parameters, self.CPU_PERCENT, context)
        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)
        include_required_fields_only = self.parameterAsBool(
            parameters, self.REQUIRED_FIELDS_ONLY, context
        )
        polygon_labels_base = self.parameterAsInt(parameters, self.POLYGONS_LABELS_BASE, context)
        background_type = self.parameterAsInt(parameters, self.BACKGROUND_TYPE, context)
        viewer = self.parameterAsInt(parameters, self.VIEWER, context)

        if self._project_path:
            return self._process_in_worker(
                feedback, extent, viewer,
                {
                    "min_zoom": min_zoom,
                    "max_zoom": max_zoom,
                    "extent": [extent.xMinimum(), extent.yMinimum(),
                               extent.xMaximum(), extent.yMaximum()],
                    "cpu_percent": cpu_percent,
                    "output_dir": output_dir,
                    "include_required_fields_only": include_required_fields_only,
                    "cent_source": polygon_labels_base,
                    "background_type": background_type,
                    "viewer": viewer,
                },
            )

        try:
            # Your existing vector tile generator class would be called here
            tiles_generator = QGIS2VectorTiles(
                min_zoom=min_zoom,
                max_zoom=max_zoom,
                extent=extent,
                cpu_percent=cpu_percent,
                output_dir=output_dir,
                include_required_fields_only=include_required_fields_only,
                cent_source=polygon_labels_base,
                background_type=background_type,
                viewer=viewer,
                feedback=feedback,
            )

            # Run the generation process (in the processing thread; the
            # project is changed afterwards, in postProcessAlgorithm).
            self._runner = tiles_generator
            tiles_generator.convert_project_to_vector_tiles()
            feedback.pushInfo(". Vector tiles package generation completed successfully")

        except (NameError, ValueError, AttributeError, TypeError) as e:
            feedback.reportError(f"Error during Vector tiles package generation: {str(e)}")
            return {}

        # Return empty results dictionary (modify as needed for your use case)
        return {}

    def _process_in_worker(self, feedback, extent, viewer, params: dict) -> dict:
        """Run the conversion in a separate process (see subprocess_runner)."""
        feedback.pushInfo(". Running the conversion in a separate process...")
        try:
            result = run_in_subprocess(self._project_path, _PACKAGE_DIR, params, feedback)
        finally:
            if self._project_is_copy:
                try:
                    os.remove(self._project_path)
                except OSError:
                    pass
        if result.get("canceled") or not result.get("temp_dir"):
            return {}
        # The tile server touches no project state, so it starts here.
        ServerInitializer(extent, result["min_zoom"], viewer, result["temp_dir"]).serve_tiles()
        self._finish_args = (extent, result["min_zoom"], viewer, result["temp_dir"])
        feedback.pushInfo(". Vector tiles package generation completed successfully")
        return {}

    def postProcessAlgorithm(self, context, feedback):
        """Runs on QGIS's main thread after processAlgorithm.

        Adding the tiles layer here, not in the processing thread, keeps the
        project's layer tree owned by the main thread.
        """
        if self._finish_args is not None:
            QGIS2VectorTiles.finish(*self._finish_args)
            self._finish_args = None
        elif self._runner is not None:
            self._runner.finish_in_main_thread()
            self._runner = None
        return {}
