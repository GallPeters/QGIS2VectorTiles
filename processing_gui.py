"""
QGIS Processing Plugin Wrapper for MBTiles Generation
Generates styled MBTiles from project layers with identical styling
"""

import os
import inspect
import tempfile
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterNumber,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFolderDestination,
    QgsCoordinateReferenceSystem,
    QgsProcessingParameterEnum,
)
from qgis.utils import iface
from qgis_vector_tiles_adapter import QGISVectorTilesAdapter


class GenerateStyledMBTilesAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for generating styled MBTiles from project layers.
    This wrapper provides a user interface for the MBTiles generation process
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

    def __init__(self):
        """Initialize the algorithm"""
        super().__init__()

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        """
        Returns a new instance of the algorithm. Required by QGIS Processing framework.
        """
        return GenerateStyledMBTilesAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm.
        This string should be fixed for the algorithm, and must not be localized.
        """
        return "generate_styled_mbtiles"

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr("Generate Styled MBTiles From Project Layers")

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
        cmd_folder = os.path.split(inspect.getfile(inspect.currentframe()))[0]
        icon_path = os.path.join(os.path.join(cmd_folder, "icon.svg"))
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return super().icon()

    def shortHelpString(self):
        """
        Returns a localised short helper string for the algorithm.
        """
        return self.tr(
            "Generates MBTiles or XYZ tiles from all visible project layers while preserving "
            "their original styling. The generated tiles are automatically loaded "
            "back into the project with identical appearance to the source layers.\n\n"
            "Parameters:\n"
            "• Output Type: Choose between MBTiles or XYZ format\n"
            "• Min/Max Zoom: Define the zoom level range (0-23)\n"
            "• Extent: Area to generate tiles for (default: current map canvas)\n"
            "• CPU Percent: Maximum CPU usage percentage (0-100)\n"
            "• Output Directory: Where to save the tiles\n"
            "• All Fields: Include all layer fields in tiles (affects file size)"
        )

    def initAlgorithm(self, config=None):
        """
        Define the inputs and outputs of the algorithm.
        """

        # Output type parameter
        self.addParameter(
            QgsProcessingParameterEnum(
                self.OUTPUT_TYPE,
                self.tr("Output Type"),
                options=["XYZ", "MBTiles"],
                defaultValue=0,  # Default to XYZ
                optional=False,
            )
        )

        # Minimum zoom level parameter
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_ZOOM,
                self.tr("Minimum Zoom"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=0,
                minValue=0,
                maxValue=22,
            )
        )

        # Maximum zoom level parameter
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_ZOOM,
                self.tr("Maximum Zoom"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
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
                self.tr("CPU Percent"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=50,
                minValue=0,
                maxValue=100,
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.REQUIRED_FIELDS_ONLY,
                self.tr("Included Fields"),
                options=["All Fields", "Required Fields Only"],
                defaultValue=1,  # Default to Required Fields Only
                optional=False,
            )
        )

        # Output directory parameter
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_DIR,
                self.tr("Output Directory"),
                optional=False,
                defaultValue=tempfile.gettempdir(),
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

    def processAlgorithm(self, parameters, context, feedback):
        """
        Main processing method. This is where your existing MBTiles generation logic
        should be called.

        Args:
            parameters: Dictionary containing parameter values
            context: QgsProcessingContext object
            feedback: QgsProcessingFeedback object for progress reporting

        Returns:
            Dictionary with results (can be empty for this use case)
        """

        # Extract parameter values
        output_type_index = self.parameterAsInt(parameters, self.OUTPUT_TYPE, context)
        output_type = ["XYZ", "MBTiles"][output_type_index]
        min_zoom = self.parameterAsInt(parameters, self.MIN_ZOOM, context)
        max_zoom = self.parameterAsInt(parameters, self.MAX_ZOOM, context)
        extent = self.parameterAsExtent(
            parameters, self.EXTENT, context, QgsCoordinateReferenceSystem("EPSG:3857")
        )
        cpu_percent = self.parameterAsInt(parameters, self.CPU_PERCENT, context)
        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)
        include_required_fields_only = self.parameterAsBool(
            parameters, self.REQUIRED_FIELDS_ONLY, context
        )

        try:
            # Your existing MBTiles generator class would be called here
            mbtiles_generator = QGISVectorTilesAdapter(
                min_zoom=min_zoom,
                max_zoom=max_zoom,
                extent=extent,
                cpu_percent=cpu_percent,
                output_dir=output_dir,
                include_required_fields_only=include_required_fields_only,
                output_type=output_type,
                feedback=feedback
            )

            # Run the generation process
            mbtiles_generator.convert_project_to_vector_tiles()
            feedback.pushInfo(f"{output_type} generation completed successfully")

        except Exception as e:
            feedback.reportError(f"Error during {output_type} generation: {str(e)}")
            return {}

        # Return empty results dictionary (modify as needed for your use case)
        return {}


class MBTilesToolbarButton:
    """
    Class to handle the toolbar button functionality
    """

    def __init__(self):
        self.action = None
        self.toolbar = None

    def initGui(self):
        """Initialize the toolbar button"""
        if not iface:
            return

        # Create the action
        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()

        self.action = QAction(icon, "Generate Styled Tiles", iface.mainWindow())
        self.action.setObjectName("generate_styled_tiles_action")
        self.action.setWhatsThis(
            "Generate styled MBTiles or XYZ tiles from project layers"
        )
        self.action.setStatusTip(
            "Generate styled MBTiles or XYZ tiles from project layers"
        )
        self.action.triggered.connect(self.run)

        # Add to plugins toolbar
        self.toolbar = iface.pluginToolBar()
        self.toolbar.addAction(self.action)

    def unload(self):
        """Remove the toolbar button"""
        if self.action and self.toolbar:
            self.toolbar.removeAction(self.action)

    def run(self):
        """Run the processing algorithm"""
        try:
            import processing
            from qgis.PyQt.QtWidgets import QDockWidget

            # Open processing toolbox if not already open
            processing_dock = iface.mainWindow().findChild(
                QDockWidget, "ProcessingToolbox"
            )
            if processing_dock:
                processing_dock.show()
                processing_dock.raise_()

            # Run the algorithm
            processing.execAlgorithmDialog("mbtiles_provider:generate_styled_mbtiles")

        except Exception as e:
            if iface:
                iface.messageBar().pushMessage(
                    "Error", f"Failed to open processing tool: {str(e)}", level=2
                )


# Global toolbar button instance
toolbar_button = MBTilesToolbarButton()


# Test block for running from QGIS Python console
if __name__ == "__console__":
    """
    Test block for running the algorithm from QGIS Python console.
    Usage in QGIS Python Console:

    exec(open('/path/to/this/file.py').read())

    Or copy-paste this entire file into the console.
    """
    import processing
    from qgis.core import QgsApplication, QgsProcessingProvider

    # Create a proper temporary provider class
    class MBTilesProvider(QgsProcessingProvider):
        def __init__(self):
            super().__init__()

        def id(self):
            return "mbtiles_provider"

        def name(self):
            return "MBTiles Provider"

        def icon(self):
            """Provider icon"""
            cmd_folder = os.path.split(inspect.getfile(inspect.currentframe()))[0]
            icon_path = os.path.join(os.path.join(cmd_folder, "icon.svg"))
            if os.path.exists(icon_path):
                return QIcon(icon_path)
            return super().icon()

        def loadAlgorithms(self):
            self.addAlgorithm(GenerateStyledMBTilesAlgorithm())

    # Create algorithm instance for testing
    alg = GenerateStyledMBTilesAlgorithm()

    # Register the provider for processing
    try:
        provider = MBTilesProvider()
        QgsApplication.processingRegistry().addProvider(provider)

        print("Algorithm registered successfully in Processing Toolbox")

        # Initialize toolbar button
        toolbar_button.initGui()
        print("Toolbar button added successfully")

        # # Test algorithm execution with default parameters
        # if iface and iface.mapCanvas():
        #     result = processing.run(
        #         "mbtiles_provider:generate_styled_mbtiles",
        #         {
        #             "OUTPUT_TYPE": 0,  # MBTiles
        #             "MIN_ZOOM": 0,
        #             "MAX_ZOOM": 1,
        #             "EXTENT": iface.mapCanvas().extent(),
        #             "CPU_PERCENT": 50,
        #             "OUTPUT_DIR": tempfile.gettempdir(),
        #             "FIELDS_INCLUDED": 1,
        #         },
        #     )
        #     print("Algorithm test completed")

    except Exception as e:
        print(f"Registration or test failed: {e}")
        print("You can still manually test the algorithm by creating an instance")


# Cleanup function for when script is reloaded
def cleanup():
    """Clean up resources when script is reloaded"""
    global toolbar_button
    if toolbar_button:
        toolbar_button.unload()


# Register cleanup
import atexit

atexit.register(cleanup)
