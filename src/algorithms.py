import os
from os.path import join, exists
import inspect
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterNumber,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFolderDestination,
    QgsCoordinateReferenceSystem,
    QgsProcessingParameterEnum,
)
from qgis.utils import iface

from .qgis2styledtiles import QGIS2StyledTiles
from .qgis2sprites import QGIS2Sprites
from .settings import _ICON



class QGIS2StyledTilesAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for generating styled MBTiles from project layers.
    This wrapper provides a user interface for the MBTiles generation process
    through the QGIS Processing Toolbox.
    """

    # Parameter names (constants for consistency)
    OUTPUT_CONTENT = 'OUTPUT_CONTENT'
    MIN_ZOOM = "MIN_ZOOM"
    MAX_ZOOM = "MAX_ZOOM"
    EXTENT = "EXTENT"
    CPU_PERCENT = "CPU_PERCENT"
    OUTPUT_DIR = "OUTPUT_DIR"
    REQUIRED_FIELDS_ONLY = "FIELDS_INCLUDED"
    OUTPUT_TYPE = "OUTPUT_TYPE"
    POLYGONS_LABELS_BASE = "POLYGONS_LABELS_BASE"

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
        return QGIS2StyledTilesAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm.
        This string should be fixed for the algorithm, and must not be localized.
        """
        return "QGIS2StyledTiles_action"

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr("QGIS2StyledTiles")

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
            "Generates MBTiles or XYZ Directory which contains vector tiles from all visible project layers while preserving "
            "their original styling. The generated tiles are automatically loaded "
            "back into the project with identical appearance to the source layers." 
            "\nMore information can be found at: https://github.com/GallPeters/QGIS2VectorTiles"
        )

    def initAlgorithm(self, config=None):
        """
        Define the inputs and outputs of the algorithm.
        """

        self.addParameter(
            QgsProcessingParameterEnum(
                self.OUTPUT_CONTENT,
                self.tr("Output Content"),
                options=["Style and Tiles", "Style Only"],
                defaultValue=0,  # Default to XYZ
                optional=False,
            )
        )

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
                self.tr("CPU Percent"),
                type=QgsProcessingParameterNumber.Integer,
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
                self.tr("Polygons Labels Base"),
                options=["Whole Polygon", "Visible Polygon"],
                defaultValue=0,  # Default to Required Fields Only
                optional=False,
            )
        )

        # Output directory parameter
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_DIR,
                self.tr("Output Directory"),
                optional=False
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
        output_content_index =  self.parameterAsInt(parameters, self.OUTPUT_CONTENT, context)
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
        POLYGONS_LABELS_BASE = self.parameterAsInt(parameters, self.POLYGONS_LABELS_BASE, context)

        try:
            # Your existing MBTiles generator class would be called here
            tiles_generator = QGIS2StyledTiles(
                min_zoom=min_zoom,
                max_zoom=max_zoom,
                extent=extent,
                cpu_percent=cpu_percent,
                output_dir=output_dir,
                include_required_fields_only=include_required_fields_only,
                output_type=output_type,
                output_content = output_content_index,
                cent_source=POLYGONS_LABELS_BASE,
                feedback=feedback
            )

            # Run the generation process
            tiles_generator.convert_project_to_vector_tiles()
            feedback.pushInfo(f"{output_type} generation completed successfully")

        except Exception as e:
            feedback.reportError(f"Error during {output_type} generation: {str(e)}")
            return {}

        # Return empty results dictionary (modify as needed for your use case)
        return {}




class QGIS2SpritesAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for generating styled MBTiles from project layers.
    This wrapper provides a user interface for the MBTiles generation process
    through the QGIS Processing Toolbox.
    """

    # Parameter names (constants for consistency)
    OUTPUT_DIR = "OUTPUT_DIR"

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
        return QGIS2SpritesAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm.
        This string should be fixed for the algorithm, and must not be localized.
        """
        return "QGIS2Sprites_action"

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr("QGIS2Sprites")

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
            "Generates a MapLibre sprite from all marker symbols in the currenr project." \
            "The sprite will include all renderer and labeling background marker symbols of all current project's visible vector layers."
            "\nMore information can be found at: https://github.com/GallPeters/QGIS2VectorTiles"
        )

    def initAlgorithm(self, config=None):
        """
        Define the inputs and outputs of the algorithm.
        """
        # Output directory parameter
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_DIR,
                self.tr("Output Directory"),
                optional=False            )
        )

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
        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)

        try:
            # Your existing MBTiles generator class would be called here
            sprite_generator = QGIS2Sprites(output_dir)
            sprite_generator.generate_sprite()

            # Run the generation process
            feedback.pushInfo(f"sprite generation completed successfully")

        except Exception as e:
            feedback.reportError(f"Error during sprite generation: {str(e)}")
            return {}
        
        # Return empty results dictionary (modify as needed for your use case)
        return {}

