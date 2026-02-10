"""QGIS2VectorTiles plugin for QGIS"""

from qgis.core import QgsApplication
from .src.provider import QGIS2VectorTilesPorvider


class QGIS2VectorTiles:
    """QGIS2VectorTiles main class"""

    def __init__(self, iface):
        """Constructor"""
        self.provider = None
        self.iface = iface

    def initProcessing(self):
        """Initialize processing provider"""
        self.provider = QGIS2VectorTilesPorvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        """Initialize GUI"""
        self.initProcessing()

    def unload(self):
        """Unload plugin"""
        QgsApplication.processingRegistry().removeProvider(self.provider)


def classFactory(iface):
    """invoke plugin"""
    return QGIS2VectorTiles(iface)
