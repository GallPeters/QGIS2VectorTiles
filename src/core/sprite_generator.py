"""
sprite_generator.py

Generates MapLibre-compatible sprite sheets from QGIS marker symbols.
Output includes sprite.png / sprite.json (pixel ratio 1) and
sprite@2x.png / sprite@2x.json (pixel ratio 2), each drawn by QGIS at its
own resolution rather than scaled from the other.

scale_factor multiplies symbol sizes before rendering:
  - 1: native size    - 2: 2× larger    - 4: 4× larger
Larger scale_factor → larger sprites and larger file size (quadratically).

Rendered images are cached between runs (see _RenderCache), keyed by the
symbol's definition, so unchanged symbols are not drawn again.
"""

import hashlib
import os
import tempfile
import time
from io import BytesIO
from dataclasses import dataclass, field
from json import dumps
from math import sqrt, ceil
from os import makedirs
from os.path import join
from typing import Optional, TypeAlias, Union


from PIL import Image
from qgis.core import (
    Qgis, QgsFillSymbol, QgsLineSymbol, QgsRenderContext, QgsSymbol, QgsSymbolLayer,
    QgsSymbolLayerUtils,
)
from qgis.PyQt.QtCore import qVersion

from ..utils.config import _CACHE_DIR_NAME, _CACHE_MAX_AGE_DAYS, _CACHE_MAX_MB

_QT_VERSION = int(qVersion()[0])
if _QT_VERSION == 5:
    from PyQt5.QtCore import QSize, QBuffer, QIODevice, QPointF, Qt
    from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPolygonF
    _IO_READ_WRITE = QIODevice.ReadWrite
else:
    from PyQt6.QtCore import QSize, QBuffer, QIODevice, QIODeviceBase as QIODevice, QPointF, Qt
    from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPolygonF
    _IO_READ_WRITE = QIODevice.OpenModeFlag.ReadWrite

Img: TypeAlias = Image.Image
MatrixShape: TypeAlias = tuple[int, int]
ImgCoord: TypeAlias = tuple[float, float, float, float]

_PX_PER_MM = 96 / 25.4
_PATTERN_RENDER_PX = 1024  # Largest repeat drawn before downscaling to the tile
# Pixel ratios of the sprite sheets written, and their file name suffixes.
_SHEETS = ((1, ""), (2, "@2x"))
# Transparent pixels between images on the sheet: enough to keep a
# neighbour from bleeding in when an icon is scaled.
_SPRITE_GAP = 2
# Bump when rendering changes, so cached images drawn the old way are not reused.
_RENDER_VERSION = "2"


@dataclass
class FillPattern:
    """A pattern fill symbol layer and its repeat (tile) size in pixels."""

    symbol_layer: QgsSymbolLayer
    width: float
    height: float


@dataclass
class LinePattern:
    """A raster line's image, drawn along the line at ``height`` pixels
    (the line's width)."""

    path: str
    height: float


@dataclass
class HashMark:
    """One stroke of a hashed line: its line sub-symbol drawn ``length``
    pixels long, across the line (vertical in the image, whose x axis
    follows the line)."""

    symbol: QgsLineSymbol
    length: float


@dataclass
class BrushPattern:
    """A simple fill's Qt brush pattern (hatching, crossing, dense dots) in
    a colour: one repeat of it, as Qt paints it (8 x 8 screen pixels)."""

    color: str        # #AARRGGBB
    style: int        # Qt.BrushStyle value
    size: int = 8


SpriteSource = Union[QgsSymbol, FillPattern, LinePattern, HashMark, BrushPattern]


class _RenderCache:
    """Rendered sprite images kept between runs, in the system temp folder,
    keyed by a hash of what is drawn and how."""

    def __init__(self):
        self.directory = join(tempfile.gettempdir(), _CACHE_DIR_NAME, "sprites")
        self.enabled = _CACHE_MAX_MB > 0
        if self.enabled:
            try:
                makedirs(self.directory, exist_ok=True)
                self._prune()
            except OSError:
                self.enabled = False

    def get(self, key: Optional[str]) -> Optional[Img]:
        if not self.enabled or key is None:
            return None
        path = join(self.directory, f"{key}.png")
        try:
            with Image.open(path) as img:
                img.load()
                os.utime(path)
                return img.copy()
        except (OSError, ValueError):
            return None

    def put(self, key: Optional[str], img: Img) -> None:
        if not self.enabled or key is None:
            return
        path = join(self.directory, f"{key}.png")
        temp = f"{path}.{os.getpid()}.tmp"
        try:
            img.save(temp, "PNG")
            os.replace(temp, path)
        except (OSError, ValueError):
            try:
                os.remove(temp)
            except OSError:
                pass

    def _prune(self) -> None:
        """Remove images unused for _CACHE_MAX_AGE_DAYS."""
        limit = time.time() - _CACHE_MAX_AGE_DAYS * 86400
        for entry in os.scandir(self.directory):
            try:
                if entry.is_file() and entry.stat().st_mtime < limit:
                    os.remove(entry.path)
            except OSError:
                continue


def _source_key(source: SpriteSource, scale_factor: float) -> Optional[str]:
    """A hash identifying what ``source`` draws at ``scale_factor``; None if
    it can't be identified (then it is drawn every time)."""
    try:
        if isinstance(source, FillPattern):
            definition = "pattern" + QgsSymbolLayerUtils.symbolProperties(
                QgsFillSymbol([source.symbol_layer.clone()])
            ) + f"{source.width!r}x{source.height!r}"
        elif isinstance(source, LinePattern):
            stat = os.stat(source.path)
            definition = f"line{source.path}|{stat.st_mtime_ns}|{stat.st_size}|{source.height!r}"
        elif isinstance(source, HashMark):
            definition = "hash" + QgsSymbolLayerUtils.symbolProperties(source.symbol) + repr(source.length)
        elif isinstance(source, BrushPattern):
            definition = f"brush{source.color}|{source.style}|{source.size}"
        else:
            definition = "symbol" + QgsSymbolLayerUtils.symbolProperties(source)
    except (OSError, RuntimeError, AttributeError, TypeError):
        return None
    text = f"{_RENDER_VERSION}|{scale_factor!r}|{definition}"
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()


# What an image source draws, as a key: equal keys draw equal images.
sprite_key = _source_key


@dataclass
class SymbolImage:
    """Render a QGIS symbol (or pattern) as a cropped PIL image at the given
    scale factor (the sheet's pixel ratio times the sprite's base scale)."""

    symbol: SpriteSource
    name: str
    scale_factor: float = 1
    cache: Optional[_RenderCache] = None
    img: Img = field(init=False)
    width: int = field(init=False)
    height: int = field(init=False)

    def __post_init__(self):
        key = _source_key(self.symbol, self.scale_factor) if self.cache else None
        cached = self.cache.get(key) if self.cache else None
        if cached is not None:
            self.img = cached
        else:
            if isinstance(self.symbol, FillPattern):
                self._render_pattern()
            elif isinstance(self.symbol, LinePattern):
                self._render_line_pattern()
            elif isinstance(self.symbol, HashMark):
                self._render_hash()
            elif isinstance(self.symbol, BrushPattern):
                self._render_brush()
            else:
                self._render()
            if self.cache:
                self.cache.put(key, self.img)
        self.width = self.img.width
        self.height = self.img.height

    def _extract_markers(self, symbol: QgsSymbol, markers: list):
        """Recursively extract marker symbols from a symbol object"""
        if symbol.type() == 0:
            markers.append(symbol)
        else:
            for sym_layer in symbol.symbolLayers():
                sub_symbol = sym_layer.subSymbol()
                if sub_symbol:
                    self._extract_markers(sub_symbol, markers)

    def _set_symbol_size(self, marker, scale_factor):
        """Set marker symbol size including outlines and buffer width"""
        # Set marker size
        if hasattr(marker, "size") and hasattr(marker, "setSize"):
            marker.setSize(marker.size() * scale_factor)
        # Set outline size
        for sym_layer in marker.symbolLayers():
            if hasattr(sym_layer, 'strokeWidth') and hasattr(sym_layer, 'setStrokeWidth'):
                sym_layer.setStrokeWidth(sym_layer.strokeWidth() * scale_factor)
        # Set buffer size
        if hasattr(marker, "bufferSettings") and marker.bufferSettings():
            marker.bufferSettings().setSize(marker.bufferSettings().size() * scale_factor)

    def _render(self):
        """Render symbol at 1000 px, scale its size, then crop transparent borders.

        The crop is centred on the point the marker is drawn at, so an
        offset or lopsided marker keeps its place when MapLibre centres it.
        """
        try:
            symbol = self.symbol.clone()
            markers = []
            self._extract_markers(symbol, markers)
            for marker in markers:
                self._set_symbol_size(marker, self.scale_factor)
            qt_img = symbol.asImage(QSize(1000, 1000))
            self.img = self._crop_centred(self._qt_to_pil(qt_img))
        except (RuntimeError, AttributeError):
            self.img = Image.new("RGBA", (10, 10), (255, 255, 255, 0))

    @staticmethod
    def _crop_centred(pil_img: Img) -> Img:
        """Crop transparent borders, keeping the image's centre at the centre."""
        bbox = pil_img.getbbox()
        if not bbox:
            return pil_img
        cx, cy = pil_img.width // 2, pil_img.height // 2
        half_w = max(cx - bbox[0], bbox[2] - cx, 1)
        half_h = max(cy - bbox[1], bbox[3] - cy, 1)
        return pil_img.crop((cx - half_w, cy - half_h, cx + half_w, cy + half_h))

    def _render_pattern(self):
        """Draw one repeat of the pattern as a tile that joins seamlessly.

        The pattern fills a 3x3-repeat area at up to 8x scale and the middle
        repeat is cut out, so the tile edges continue into each other; it is
        then scaled to its pixel size (times the scale factor).
        """
        pattern = self.symbol
        try:
            tile_w = max(1, round(pattern.width * self.scale_factor))
            tile_h = max(1, round(pattern.height * self.scale_factor))
            scale = max(1.0, min(8.0, _PATTERN_RENDER_PX / max(pattern.width, pattern.height)))
            repeat_w, repeat_h = max(1, round(pattern.width * scale)), max(1, round(pattern.height * scale))
            area_w, area_h = repeat_w * 3, repeat_h * 3

            qt_img = QImage(area_w, area_h, QImage.Format.Format_ARGB32_Premultiplied)
            qt_img.fill(0)
            painter = QPainter(qt_img)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            context = QgsRenderContext.fromQPainter(painter)
            context.setScaleFactor(_PX_PER_MM * scale)
            context.setFlag(Qgis.RenderContextFlag.Antialiasing, True)
            symbol = QgsFillSymbol([pattern.symbol_layer.clone()])
            area = QPolygonF([QPointF(0, 0), QPointF(area_w, 0), QPointF(area_w, area_h),
                              QPointF(0, area_h), QPointF(0, 0)])
            symbol.startRender(context)
            symbol.renderPolygon(area, None, None, context)
            symbol.stopRender(context)
            painter.end()

            repeat = self._qt_to_pil(qt_img).crop((repeat_w, repeat_h, repeat_w * 2, repeat_h * 2))
            self.img = self._resize_tile(repeat, tile_w, tile_h)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            self.img = Image.new("RGBA", (10, 10), (255, 255, 255, 0))

    def _render_line_pattern(self):
        """A raster line's image at the line's width (times the scale factor)."""
        try:
            with Image.open(self.symbol.path) as source:
                source = source.convert("RGBA")
                height = max(1, round(self.symbol.height * self.scale_factor))
                width = max(1, round(source.width * height / max(1, source.height)))
                self.img = source.resize((width, height), Image.Resampling.LANCZOS)
        except (OSError, ValueError):
            self.img = Image.new("RGBA", (10, 10), (255, 255, 255, 0))

    def _render_hash(self):
        """One stroke of a hashed line, drawn vertically and cropped centred."""
        mark = self.symbol
        try:
            length = mark.length * self.scale_factor
            size = int(ceil(length)) + 64
            qt_img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
            qt_img.fill(0)
            painter = QPainter(qt_img)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            context = QgsRenderContext.fromQPainter(painter)
            context.setScaleFactor(_PX_PER_MM * self.scale_factor)
            context.setFlag(Qgis.RenderContextFlag.Antialiasing, True)
            centre = size / 2
            line = QPolygonF([QPointF(centre, centre - length / 2), QPointF(centre, centre + length / 2)])
            symbol = mark.symbol.clone()
            symbol.startRender(context)
            symbol.renderPolyline(line, None, context)
            symbol.stopRender(context)
            painter.end()
            self.img = self._crop_centred(self._qt_to_pil(qt_img))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            self.img = Image.new("RGBA", (10, 10), (255, 255, 255, 0))

    def _render_brush(self):
        """One repeat of a Qt brush pattern, scaled by the scale factor
        (each pattern pixel becomes a block of that many pixels)."""
        brush = self.symbol
        try:
            scale = max(1, round(self.scale_factor))
            size = brush.size * scale
            qt_img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
            qt_img.fill(0)
            painter = QPainter(qt_img)
            painter.scale(scale, scale)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(brush.color), Qt.BrushStyle(brush.style)))
            painter.drawRect(0, 0, brush.size, brush.size)
            painter.end()
            self.img = self._qt_to_pil(qt_img)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            self.img = Image.new("RGBA", (8, 8), (255, 255, 255, 0))

    @staticmethod
    def _resize_tile(tile: Img, width: int, height: int) -> Img:
        """Resize a repeating tile, sampling across its edges as if it were repeated."""
        mosaic = Image.new("RGBA", (tile.width * 3, tile.height * 3))
        for col in range(3):
            for row in range(3):
                mosaic.paste(tile, (col * tile.width, row * tile.height))
        mosaic = mosaic.resize((width * 3, height * 3), Image.Resampling.LANCZOS)
        return mosaic.crop((width, height, width * 2, height * 2))

    @staticmethod
    def _qt_to_pil(qt_img: QImage) -> Img:
        """Convert a QImage to a PIL Image."""
        try:
            buffer = QBuffer()
            buffer.open(_IO_READ_WRITE)
            qt_img.save(buffer, "PNG")
            bio = BytesIO(buffer.data())
            buffer.close()
            bio.seek(0)
            return Image.open(bio)
        except (OSError, RuntimeError):
            return Image.new("RGBA", (10, 10), (255, 255, 255, 0))


@dataclass
class SpriteMatrix:
    """Arrange SymbolImages in an optimal grid layout."""

    imgs: list[SymbolImage]
    ratio: tuple[int, int] = (3, 4)
    pixelspace: int = _SPRITE_GAP
    shape: MatrixShape = field(init=False)
    imgsmatrix: list[list[SymbolImage]] = field(init=False)
    imgsrows: list[tuple] = field(init=False)

    def __post_init__(self):
        self.calculate_shape()
        self.generate_imgs_matrix()
        self.get_matrix_rows()

    def calculate_shape(self):
        """Compute the optimal (height, width) grid from image count and ratio."""
        count = len(self.imgs)
        base = sqrt(count / (self.ratio[0] * self.ratio[1]))
        height = ceil(base * self.ratio[0])
        width = ceil(base * self.ratio[1])
        if width * (height - 1) >= count:
            height -= 1
        self.shape = (height, width)

    def generate_imgs_matrix(self):
        """Distribute images into rows according to the calculated grid shape."""
        _, w = self.shape
        symbols = list(self.imgs)
        self.imgsmatrix = [symbols[w * r: w * (r + 1)] for r in range(self.shape[0])]

    def get_matrix_rows(self):
        """Compute (width, height, images) for each row including pixel spacing."""
        self.imgsrows = []
        for row in self.imgsmatrix:
            if not row:
                continue
            w = sum(img.width + self.pixelspace for img in row) - self.pixelspace
            h = max(img.height for img in row) + self.pixelspace
            self.imgsrows.append((w, h, row))


@dataclass
class SpriteImage:
    """Composite a sprite sheet from a SpriteMatrix and track image coordinates."""

    matrix: SpriteMatrix
    pixelspace: int = _SPRITE_GAP
    img: Img = field(init=False)
    imgscoords: dict = field(init=False)

    def __post_init__(self):
        self._build()
        self._populate()

    def _build(self):
        """Create a blank RGBA canvas sized from the matrix rows."""
        if not self.matrix.imgsrows:
            self.img = Image.new("RGBA", (100, 100), (255, 255, 255, 0))
            return
        widths, heights, _ = zip(*self.matrix.imgsrows)
        w = max(widths) + self.pixelspace * 2
        h = sum(heights) + self.pixelspace * 2
        self.img = Image.new("RGBA", (w, h), (255, 255, 255, 0))

    def _populate(self):
        """Paste symbols into the sheet and record their (x, y, w, h) coordinates."""
        self.imgscoords = {}
        y = self.pixelspace
        for row_width, row_height, row in self.matrix.imgsrows:
            x_start = round((self.img.width - row_width) / 2)
            self.imgscoords.update(self._paste_row(row, x_start, row_height, y))
            y += row_height

    def _paste_row(
        self, row: list[SymbolImage], left_x: float, row_height: float, current_y: float
    ) -> dict:
        """Paste a single row of images; return {name: (x, y, w, h)} mapping."""
        coords = {}
        x = left_x
        for img in row:
            y_offset = round((row_height - self.matrix.pixelspace - img.height) / 2)
            img_y = current_y + y_offset
            self.img.paste(img.img, (int(x), int(img_y)))
            coords[img.name] = (int(x), int(img_y), img.width, img.height)
            x += img.width + self.matrix.pixelspace
        return coords

    def save(self, output_dir: str, suffix: str):
        """Save the sheet as sprite{suffix}.png in output_dir."""
        try:
            makedirs(output_dir, exist_ok=True)
            self.img.save(join(output_dir, f"sprite{suffix}.png"))
        except (OSError, RuntimeError):
            pass


@dataclass
class SpriteJSON:
    """MapLibre JSON metadata for one sprite sheet."""

    spriteimg: SpriteImage
    pixel_ratio: int = 1
    jsondict: dict = field(init=False)

    def __post_init__(self):
        self.jsondict = {
            name: {"x": int(x), "y": int(y), "width": int(w), "height": int(h),
                   "pixelRatio": self.pixel_ratio}
            for name, (x, y, w, h) in self.spriteimg.imgscoords.items()
        }

    def save(self, output_dir: str, suffix: str):
        """Save the metadata as sprite{suffix}.json in output_dir."""
        try:
            makedirs(output_dir, exist_ok=True)
            with open(join(output_dir, f"sprite{suffix}.json"), "w", encoding="utf8") as f:
                f.write(dumps(self.jsondict, indent=2))
        except (OSError, RuntimeError):
            pass


class SpriteGenerator:
    """Orchestrate the full sprite pipeline: render → arrange → JSON → save,
    once per sheet (pixel ratio 1 and 2)."""

    def __init__(
        self,
        symbols_dict: dict[str, SpriteSource],
        output_dir: str,
        scale_factor: int = 1,
        test_mode: bool = False,
    ):
        self.symbols_dict = symbols_dict
        self.output_dir = output_dir
        self.scale_factor = scale_factor
        self.test_mode = test_mode

    def generate(self) -> Optional[str]:
        """Run the sprite generation pipeline; return output directory or None on failure."""
        if not self.symbols_dict:
            return None
        try:
            cache = _RenderCache()
            sprite_dir = join(self.output_dir, "sprite")
            for pixel_ratio, suffix in _SHEETS:
                imgs = [
                    SymbolImage(sym, name, self.scale_factor * pixel_ratio, cache)
                    for name, sym in self.symbols_dict.items()
                ]
                sprite_img = SpriteImage(SpriteMatrix(imgs))
                sprite_json = SpriteJSON(sprite_img, pixel_ratio * self.scale_factor)
                sprite_img.save(sprite_dir, suffix)
                sprite_json.save(sprite_dir, suffix)
                if self.test_mode:
                    self._verify_coordinates(sprite_img, sprite_json, suffix or "@1x")
            return self.output_dir
        except (RuntimeError, AttributeError, KeyError, ValueError, OSError):
            return None

    def _verify_coordinates(self, sprite_img: SpriteImage, sprite_json: SpriteJSON, res_name: str):
        """Extract and save individual symbol crops to verify sprite coordinates."""
        sprite, coords = sprite_img.img, sprite_json.jsondict
        test_dir = join(self.output_dir, f"test_{res_name}")
        makedirs(test_dir, exist_ok=True)
        for name, c in coords.items():
            try:
                x, y, w, h = int(c["x"]), int(c["y"]), int(c["width"]), int(c["height"])
                if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= sprite.width and y + h <= sprite.height:
                    crop = sprite.crop((x, y, x + w, y + h))
                    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
                    crop.save(join(test_dir, f"{safe_name}.png"))
            except (KeyError, ValueError, OSError):
                pass


if __name__ == "__console__":
    pass
