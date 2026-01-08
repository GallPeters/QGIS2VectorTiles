from os.path import exists, join
from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsApplication

_PLUGIN_DIR = r'C:\tests' if __name__ == "__console__" else join(QgsApplication.qgisSettingsDirPath(), r'python/plugins/QGIS2VectorTiles')
_RESOURCES = join(_PLUGIN_DIR, 'resources')
_CONF = join(_RESOURCES, 'conf.toml')
_ICON_PATH = join(_RESOURCES, 'icon.svg')
_ICON = QIcon(_ICON_PATH) if exists(_ICON_PATH) else super().icon()