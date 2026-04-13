"""
zoom_levels.py

ZoomLevels — utility class for converting between map scales and web tile
zoom levels based on the configured tiling scheme.
"""

from typing import Optional

from .config import _TOP_SCALE


class ZoomLevels:
    """Manages zoom level scales and conversions for web mapping standards."""

    SCALES = [_TOP_SCALE / (2**zoom) for zoom in range(23)]

    @classmethod
    def scale_to_zoom(cls, scale: float, edge: str) -> str:
        """Convert scale to zero-padded zoom level string."""
        if scale in [0, 0.0]:
            scale = cls.SCALES[0 if edge == "o" else -1]
        for zoom, zoom_scale in enumerate(cls.SCALES):
            if scale >= zoom_scale and edge == "o":
                return zoom
        if scale > cls.SCALES[0]:
            return 0
        for zoom, zoom_scale in sorted(enumerate(cls.SCALES), reverse=True):
            if scale <= zoom_scale:
                return zoom
        return len(cls.SCALES) - 1

    @classmethod
    def zoom_to_scale(cls, zoom: int) -> Optional[float]:
        """Convert zoom level to scale."""
        if 0 <= zoom < len(cls.SCALES):
            return cls.SCALES[zoom]
        return None
