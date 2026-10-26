"""
zoom_levels.py

ZoomLevels — utility class for converting between map scales and web tile
zoom levels based on the configured tiling scheme.
"""

import math
from typing import Optional

from .config import _TOP_SCALE

# Tolerance for treating a zoom computed from a scale as a whole zoom level.
_ZOOM_EPSILON = 1e-9


class ZoomLevels:
    """Manages zoom level scales and conversions for web mapping standards.

    SCALES[z] is the QGIS map scale that shows what web zoom z shows. Web
    Mercator stretches the ground by 1 / cos(latitude), while QGIS measures
    a project in a geographic or local projected CRS on the ground: at the
    data's latitude its scales are that much smaller (see configure()).
    """

    SCALES = [_TOP_SCALE / (2**zoom) for zoom in range(23)]
    # EPSG:3857 metres per ground metre at the data's latitude (1/cos).
    GROUND_STRETCH = 1.0

    @classmethod
    def configure(cls, scale_factor: float = 1.0) -> None:
        """Scale every zoom's map scale by ``scale_factor`` (the cosine of
        the data's latitude for projects not in Web Mercator, else 1)."""
        cls.SCALES = [_TOP_SCALE * scale_factor / (2**zoom) for zoom in range(23)]

    @classmethod
    def fractional_zoom(cls, scale: float) -> float:
        """The exact (fractional) web zoom that shows ``scale``, in 0..22."""
        if not scale or scale <= 0:
            return 0.0
        zoom = min(max(math.log2(cls.SCALES[0] / scale), 0.0), len(cls.SCALES) - 1.0)
        # A zoom level's own scale gives that whole zoom, not one a hair off.
        return float(round(zoom)) if abs(zoom - round(zoom)) < 1e-6 else zoom

    @classmethod
    def scale_to_zoom(cls, scale: float, edge: str) -> int:
        """The whole zoom level a scale limit puts its edge of a rule at.

        "o" (a minimum scale: visible from that scale in): the zoom level
        whose tiles first show the rule, the floor of the exact zoom, so
        its data is in the tiles of the zoom level it starts within.
        "i" (a maximum scale: visible until that scale): the last zoom
        level whose tiles show it, below the exact zoom (exclusive).
        A scale of 0 is no limit. The style draws the rule from and until
        the exact zooms (see TilesStyler).
        """
        if scale in [0, 0.0]:
            return 0 if edge == "o" else len(cls.SCALES) - 1
        zoom = cls.fractional_zoom(scale)
        if edge == "o":
            return int(math.floor(zoom + _ZOOM_EPSILON))
        return max(0, int(math.ceil(zoom - _ZOOM_EPSILON)) - 1)

    @classmethod
    def zoom_to_scale(cls, zoom: int) -> Optional[float]:
        """Convert zoom level to scale."""
        if 0 <= zoom < len(cls.SCALES):
            return cls.SCALES[zoom]
        return None

    @classmethod
    def set_ground_stretch(cls, extent_3857) -> None:
        """Set GROUND_STRETCH from the export extent's latitude."""
        try:
            y = extent_3857.center().y()
            latitude = math.degrees(math.atan(math.sinh(y / 6378137.0)))
            cls.GROUND_STRETCH = 1.0 / max(math.cos(math.radians(latitude)), 0.01)
        except (AttributeError, RuntimeError):
            cls.GROUND_STRETCH = 1.0

    @staticmethod
    def latitude_scale_factor(project_crs, extent_3857) -> float:
        """configure()'s factor for a project CRS and the export extent (EPSG:3857)."""
        try:
            if project_crs.authid() in ("EPSG:3857", "EPSG:900913", "EPSG:3785", "EPSG:102100"):
                return 1.0
            y = extent_3857.center().y()
        except (AttributeError, RuntimeError):
            return 1.0
        latitude = math.degrees(math.atan(math.sinh(y / 6378137.0)))
        return max(math.cos(math.radians(latitude)), 0.01)
