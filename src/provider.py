"""QGIS2VectorTiles provider implementation."""

from qgis.core import QgsProcessingProvider
from .algorithms import QGIS2StyledTilesAlgorithm, _ICON


# Create a proper temporary provider class
class QGIS2VectorTilesPorvider(QgsProcessingProvider):
    """Provider for QGIS2VectorTiles plugin, integrating the QGIS2StyledTilesAlgorithm
    into the QGIS Processing framework."""

    def __init__(self):
        super().__init__()

    def id(self):
        """Returns the unique ID of the provider."""
        return "QGIS2VectorTiles"

    def name(self):
        """Returns the display name of the provider."""
        return "QGIS2VectorTiles"

    def icon(self):
        """Returns the provider icon."""
        return _ICON

    def loadAlgorithms(self):
        """Loads the algorithms provided by this provider."""
        self.addAlgorithm(QGIS2StyledTilesAlgorithm())
