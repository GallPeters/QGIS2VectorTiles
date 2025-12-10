def classFactory(iface):
    """invoke plugin"""
    from .qgis2vectortiles import QGIS2VectorTiles

    return QGIS2VectorTiles(iface)