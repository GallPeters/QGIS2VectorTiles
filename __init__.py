"""QGIS2VectorTiles plugin for QGIS"""

from qgis.core import QgsApplication, QgsExpression
from .src.processing.provider import QGIS2VectorTilesPorvider


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
        # The expression functions of the rules export (src/core/symbol_geometry),
        # registered if it ran in this process.
        for name in ("q2vt_arrow", "q2vt_offset", "q2vt_callout"):
            QgsExpression.unregisterFunction(name)


def classFactory(iface):
    """invoke plugin"""
    return QGIS2VectorTiles(iface)
