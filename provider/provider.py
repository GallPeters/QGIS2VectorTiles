
import os
import inspect
from qgis.core import QgsApplication, QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon
from .alg import QGIS2VectorTilesAlgorithm



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
        cmd_folder = os.path.split(inspect.getfile(inspect.currentframe()))[0]
        icon_path = os.path.join(os.path.join(cmd_folder, "icon.svg"))
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return super().icon()

    def loadAlgorithms(self):
        self.addAlgorithm(QGIS2VectorTilesAlgorithm())


