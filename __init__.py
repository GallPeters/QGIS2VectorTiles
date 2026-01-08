from qgis.core import QgsApplication
from .src.provider import QGIS2VectorTilesPorvider


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

def classFactory(iface):
    """invoke plugin"""
    return QGIS2VectorTiles(iface)