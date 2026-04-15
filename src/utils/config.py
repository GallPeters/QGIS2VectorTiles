"""
config.py

Global constants, configuration loading, and PyQt version-specific imports.
All other modules import shared symbols from here to avoid duplication.
"""

from os.path import join

from qgis.PyQt.QtCore import qVersion
from qgis.core import QgsApplication

_PLUGIN_DIR = join(QgsApplication.qgisSettingsDirPath(), "python", "plugins", "QGIS2VectorTiles")
_RESOURCES = join(_PLUGIN_DIR, "resources")
_VIEWER = join(_RESOURCES, "maplibre_viewer.html")
_MAPLIBRE = join(_RESOURCES, "maplibre-gl")
_SERVER = join(_RESOURCES, "mbtiles_server.py")
_BAT = join(_RESOURCES, "activate_server.bat")
_SH = join(_RESOURCES, "activate_server.sh")
_PORT = 9000
_EPSG_CRS = 3857
_DATA_SIMPLIFICATION_TOLERANCE = 1  # CRS Units
_REMOVE_DUPLICATES_DISTANCE = 300  # Points
_TOP_SCALE = 419311712

# PyQt version guard — import the right Qt5 / Qt6 symbols once, re-export

if int(qVersion()[0]) == 5:
    from PyQt5.QtXml import QDomDocument
    from PyQt5.QtCore import QVariant
    from PyQt5 import sip
else:
    from PyQt6.QtXml import QDomDocument
    from PyQt6.QtCore import QVariant
    from PyQt6 import sip

__all__ = [
    "_RESOURCES",
    "_BAT",
    "_SH",
    "_SERVER",
    "_TOP_SCALE",
    "_EPSG_CRS",
    "_DATA_SIMPLIFICATION_TOLERANCE",
    "_REMOVE_DUPLICATES_DISTANCE",
    "_VIEWER",
    "_MAPLIBRE",
    "_PORT",
    "_PLUGIN_DIR",
    "QDomDocument",
    "QVariant",
    "sip"
]
