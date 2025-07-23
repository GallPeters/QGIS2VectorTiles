"""
QGIS Processing Plugin Wrapper for MBTiles Generation
Generates styled MBTiles from project layers with identical styling
"""

import tempfile
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterNumber,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterBoolean,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsMessageLog,
    Qgis
)
from qgis.utils import iface


class GenerateStyledMBTilesAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for generating styled MBTiles from project layers.
    This wrapper provides a user interface for the MBTiles generation process
    through the QGIS Processing Toolbox.
    """
    
    # Parameter names (constants for consistency)
    MIN_ZOOM = 'MIN_ZOOM'
    MAX_ZOOM = 'MAX_ZOOM'
    EXTENT = 'EXTENT'
    OUTPUT_DIR = 'OUTPUT_DIR'
    CPU_PERCENT = 'CPU_PERCENT'
    ALL_FIELDS = 'ALL_FIELDS'

    def __init__(self):
        """Initialize the algorithm"""
        super().__init__()

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

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
        return 'generate_styled_mbtiles'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('Generate Styled MBTiles From Project Layers')

    def group(self):
        """
        Returns the name of the group this algorithm belongs to.
        """
        return self.tr('Tile Generation')

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to.
        """
        return 'tile_generation'

    def shortHelpString(self):
        """
        Returns a localised short helper string for the algorithm.
        """
        return self.tr(
            "Generates MBTiles from all visible project layers while preserving "
            "their original styling. The generated tiles are automatically loaded "
            "back into the project with identical appearance to the source layers.\n\n"
            "Parameters:\n"
            "• Min/Max Zoom: Define the zoom level range (0-23)\n"
            "• Extent: Area to generate tiles for (default: current map canvas)\n"
            "• Output Directory: Where to save the MBTiles files\n"
            "• CPU Usage: Percentage of CPU cores to use (1-100%)\n"
            "• All Fields: Include all layer fields in tiles (affects file size)"
        )

    def initAlgorithm(self, config=None):
        """
        Define the inputs and outputs of the algorithm.
        """
        
        # Minimum zoom level parameter
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_ZOOM,
                self.tr('Minimum Zoom Level'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=0,
                minValue=0,
                maxValue=23
            )
        )

        # Maximum zoom level parameter
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_ZOOM,
                self.tr('Maximum Zoom Level'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=23,
                minValue=0,
                maxValue=23
            )
        )

        # Extent parameter - defaults to current map canvas extent
        extent_param = QgsProcessingParameterExtent(
            self.EXTENT,
            self.tr('Tile Generation Extent'),
            optional=False
        )
        # Set default to current map canvas extent if available
        if iface and iface.mapCanvas():
            extent_param.setDefaultValue(iface.mapCanvas().extent())
        self.addParameter(extent_param)

        # Output directory parameter
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_DIR,
                self.tr('Output Directory'),
                optional=False,
                defaultValue=tempfile.gettempdir()
            )
        )

        # CPU usage percentage parameter
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CPU_PERCENT,
                self.tr('CPU Usage Percentage'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=70,
                minValue=1,
                maxValue=100
            )
        )

        # All fields boolean parameter
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ALL_FIELDS,
                self.tr('Include All Layer Fields'),
                defaultValue=False
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
            return False, self.tr('Minimum zoom level must be less than or equal to maximum zoom level')
        
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
        min_zoom = self.parameterAsInt(parameters, self.MIN_ZOOM, context)
        max_zoom = self.parameterAsInt(parameters, self.MAX_ZOOM, context)
        extent = self.parameterAsExtent(parameters, self.EXTENT, context)
        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)
        cpu_percent = self.parameterAsInt(parameters, self.CPU_PERCENT, context)
        all_fields = self.parameterAsBool(parameters, self.ALL_FIELDS, context)
        
        # Log the parameters for debugging
        feedback.pushInfo(f"Parameters received:")
        feedback.pushInfo(f"  Min Zoom: {min_zoom}")
        feedback.pushInfo(f"  Max Zoom: {max_zoom}")
        feedback.pushInfo(f"  Extent: {extent.toString()}")
        feedback.pushInfo(f"  Output Directory: {output_dir}")
        feedback.pushInfo(f"  CPU Percentage: {cpu_percent}")
        feedback.pushInfo(f"  Include All Fields: {all_fields}")
        
        # TODO: Replace this placeholder with your actual MBTiles generation logic
        # Example call to your existing processing class:
        """
        try:
            # Your existing MBTiles generator class would be called here
            # mbtiles_generator = YourMBTilesGenerator(
            #     min_zoom=min_zoom,
            #     max_zoom=max_zoom,
            #     extent=extent,
            #     output_dir=output_dir,
            #     cpu_percent=cpu_percent,
            #     all_fields=all_fields
            # )
            # 
            # # Set up progress reporting
            # def progress_callback(current, total):
            #     if feedback.isCanceled():
            #         return False
            #     feedback.setProgress(int((current / total) * 100))
            #     return True
            # 
            # # Run the generation process
            # mbtiles_generator.process_layers(progress_callback=progress_callback)
            # 
            # feedback.pushInfo("MBTiles generation completed successfully!")
            
        except Exception as e:
            feedback.reportError(f"Error during MBTiles generation: {str(e)}")
            return {}
        """
        
        # Placeholder implementation
        feedback.pushInfo("🔧 PLACEHOLDER: Your MBTiles generation logic will be called here")
        feedback.pushInfo("📁 Ready to process project layers with the specified parameters")
        
        # Simulate some progress for testing
        for i in range(101):
            if feedback.isCanceled():
                break
            feedback.setProgress(i)
            
        feedback.pushInfo(" Processing wrapper is ready - integrate your MBTiles logic here")
        
        # Return empty results dictionary (modify as needed for your use case)
        return {}


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
    class TempMBTilesProvider(QgsProcessingProvider):
        def __init__(self):
            super().__init__()
        
        def id(self):
            return 'temp_mbtiles_provider'
        
        def name(self):
            return 'Temporary MBTiles Provider'
        
        def loadAlgorithms(self):
            self.addAlgorithm(GenerateStyledMBTilesAlgorithm())
    
    # Create algorithm instance for testing
    alg = GenerateStyledMBTilesAlgorithm()
    
    # Print algorithm information
    print(f"Algorithm Name: {alg.name()}")
    print(f"Display Name: {alg.displayName()}")
    print(f"Group: {alg.group()}")
    print(f"Help: {alg.shortHelpString()}")
    
    # Register the provider temporarily for testing
    try:
        provider = TempMBTilesProvider()
        QgsApplication.processingRegistry().addProvider(provider)
        
        print("\n Algorithm registered successfully!")
        print("🔍 Look for 'Generate Styled MBTiles From Project Layers' in Processing Toolbox")
        print("📂 Under 'Tile Generation' group")
        
        # Test algorithm execution with default parameters
        if iface and iface.mapCanvas():
            print("\n🚀 Testing algorithm execution...")
            result = processing.run(
                "temp_mbtiles_provider:generate_styled_mbtiles",
                {
                    'MIN_ZOOM': 0,
                    'MAX_ZOOM': 10,
                    'EXTENT': iface.mapCanvas().extent(),
                    'OUTPUT_DIR': tempfile.gettempdir(),
                    'CPU_PERCENT': 50,
                    'ALL_FIELDS': False
                }
            )
            print(" Algorithm test completed!")
        else:
            print("  No map canvas available - skipping test execution")
            
    except Exception as e:
        print(f"⚠️  Registration or test failed: {e}")
        print("💡 You can still manually test the algorithm by creating an instance:")
        print("    alg = GenerateStyledMBTilesAlgorithm()")
        print("    # Then call alg.processAlgorithm() directly with parameters")