"""
config.py

Global constants, configuration loading, and PyQt version-specific imports.
All other modules import shared symbols from here to avoid duplication.
"""

from os.path import join

from qgis.PyQt.QtCore import qVersion
from qgis.core import QgsApplication


# =====================================================================
# PLUGIN PATHS
# Computed from the live QGIS install. Do not hardcode replacements —
# these must stay derived from QgsApplication at import time.
# =====================================================================
_PLUGIN_DIR = join(QgsApplication.qgisSettingsDirPath(), "python", "plugins", "QGIS2VectorTiles")
_RESOURCES = join(_PLUGIN_DIR, "resources")
_SERVER = join(_RESOURCES, "tiles_server.py")

# =====================================================================
# SERVER / NETWORK SETTINGS
# Safe to adjust if you need a different local port.
# =====================================================================
_PORT = 9000


# =====================================================================
# DATA PROCESSING SETTINGS
# Safe to adjust to tune output quality/size vs. processing speed.
# =====================================================================
_EPSG_CRS = 3857                          # Output projection (Web Mercator)
_DATA_SIMPLIFICATION_TOLERANCE = 1        # Geometry simplification, in CRS units
_REMOVE_DUPLICATES_DISTANCE = 300         # Minimum spacing between points, in points
_TOP_SCALE = 419311712                    # Max zoomed-out map scale
_SPRITE_QUALITY = 3
_FIELD_PREFIX = 'q2vt'                       # Field name prefix


# =====================================================================
# MAPLIBRE GLYPH (SDF) GENERATION
# =====================================================================
# --- Format constants — fixed by the MapLibre/Mapbox glyph PBF spec. ---
# These are NOT tunable settings. MapLibre GL JS's client-side glyph
# parser hardcodes these exact values; changing them WILL break glyph
# rendering (mismatched image size errors, square/clipped halos, blurry
# text). Do not edit this block.
_GLYPH_RANGE_SIZE = 256        # Codepoints per .pbf range file (spec-fixed)
_MAX_UNICODE = 65535           # Highest codepoint MapLibre's glyph protocol supports
_MAPLIBRE_GLYPH_BORDER = 3     # MapLibre client's fixed glyph border (never changes)
_REFERENCE_EM = 24.0           # MapLibre's internal reference font size, in px
_REFERENCE_BUFFER = 3.0        # Mapbox/MapLibre reference generator (tiny-sdf) buffer default
_REFERENCE_RADIUS = 8.0        # Mapbox/MapLibre reference generator (tiny-sdf) radius default;
                                # also hardcoded client-side as the shader's SDF_PX constant

# --- Derived generation parameters — DO NOT edit individually. ---
# _SDF_RADIUS and _BUFFER are mathematically dependent on _REFERENCE_RADIUS,
# _REFERENCE_BUFFER, and _FONT_RENDER_SIZE above. Changing one without
# recalculating the others reproduces bugs already solved (mismatched
# image size, blurry/rectangular halos). If _FONT_RENDER_SIZE ever needs
# to change, _SDF_RADIUS and _BUFFER must be recomputed using the
# formulas in the comments below — do not just edit the numbers.
_FONT_RENDER_SIZE = 24         # Source rasterization point size
_SDF_CUTOFF = 0.25             # MapLibre/Mapbox spec: outline lands at byte ~192
_SUPERSAMPLE = 4               # Internal antialiasing supersample factor
_SDF_RADIUS = 8.0              # = _REFERENCE_RADIUS * (_FONT_RENDER_SIZE / _REFERENCE_EM)
_BUFFER = 10                   # = ceil(max(_REFERENCE_BUFFER * scale, _SDF_RADIUS + 2, _MAPLIBRE_GLYPH_BORDER))
_SDF_COVERAGE_THRESHOLD = 127  # AA coverage midpoint used to binarize the glyph mask
_MAPLIBRE_LABELS_FACTOR = 1.4   # Factor to decrease label size to match MapLibre GL JS's rendering to the original QGIS project.

# =====================================================================
# PyQt VERSION GUARD
# Imports the right Qt5 / Qt6 symbols once, re-exported below.
# =====================================================================
if int(qVersion()[0]) == 5:
    from PyQt5.QtXml import QDomDocument
    from PyQt5.QtCore import QVariant, Qt
    from PyQt5 import sip
else:
    from PyQt6.QtXml import QDomDocument
    from PyQt6.QtCore import QVariant, Qt
    from PyQt6 import sip


__all__ = [
    # Plugin paths
    "_PLUGIN_DIR",
    "_RESOURCES",
    "_SERVER",
    # Server / network
    "_PORT",
    # Data processing
    "_EPSG_CRS",
    "_DATA_SIMPLIFICATION_TOLERANCE",
    "_REMOVE_DUPLICATES_DISTANCE",
    "_TOP_SCALE",
    "_SPRITE_QUALITY",
    "_FIELD_PREFIX",
    # Glyph generation
    "_GLYPH_RANGE_SIZE",
    "_MAX_UNICODE",
    "_MAPLIBRE_GLYPH_BORDER",
    "_REFERENCE_EM",
    "_REFERENCE_BUFFER",
    "_REFERENCE_RADIUS",
    "_FONT_RENDER_SIZE",
    "_SDF_CUTOFF",
    "_SUPERSAMPLE",
    "_SDF_RADIUS",
    "_BUFFER",
    "_SDF_COVERAGE_THRESHOLD",
    "_MAPLIBRE_LABELS_FACTOR",
    # PyQt re-exports
    "Qt",
    "QDomDocument",
    "QVariant",
    "sip",
]