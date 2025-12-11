from qgis.core import QgsApplication
from .provider.provider import QGIS2VectorTilesPorvider

class QGIS2VectorTiles:

    def __init__(self, iface):
        self.provider = None
        self.iface = iface

    def initProcessing(self):
        self.provider = QGIS2VectorTilesPorvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

    def unload(self):
        QgsApplication.processingRegistry().removeProvider(self.provider)