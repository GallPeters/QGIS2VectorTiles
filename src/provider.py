from os.path import join, exists
from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon
from .algorithms import QGIS2StyledTilesAlgorithm, QGIS2SpritesAlgorithm
from .settings import _ICON


    
# Create a proper temporary provider class
class QGIS2VectorTilesPorvider(QgsProcessingProvider):
    def __init__(self):
        super().__init__()

    def id(self):
        return "QGIS2VectorTiles"

    def name(self):
        return "QGIS2VectorTiles"

    def icon(self):
        """Provider icon"""
        return _ICON

    def loadAlgorithms(self):
        self.addAlgorithm(QGIS2StyledTilesAlgorithm())
        self.addAlgorithm(QGIS2SpritesAlgorithm())


