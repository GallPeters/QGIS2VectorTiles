"""
sprite_generator.py

Generates MapLibre-compatible sprite sheets from QGIS marker symbols.
Output includes sprite.png, sprite.json and high-resolution (@2x) versions.

scale_factor multiplies symbol sizes before rendering:
  - 1: native size    - 2: 2× larger    - 4: 4× larger
Larger scale_factor → larger sprites and larger file size (quadratically).
"""

from io import BytesIO
from dataclasses import dataclass, field
from json import dumps
from math import sqrt, ceil
from os import makedirs
from os.path import join
from typing import Optional, TypeAlias
from datetime import datetime
from gc import collect

from PIL import Image
from qgis.core import (
    QgsProcessingUtils,
    QgsSymbol,
    QgsRuleBasedRenderer,
    QgsRuleBasedLabeling,
    QgsProject,
    QgsVectorLayerSimpleLabeling,
    QgsSingleSymbolRenderer,
    QgsCategorizedSymbolRenderer,
    QgsGraduatedSymbolRenderer,
)
from qgis.PyQt.QtCore import qVersion

_QT_VERSION = int(qVersion()[0])
if _QT_VERSION == 5:
    from PyQt5.QtCore import QSize, QBuffer, QIODevice
    from PyQt5.QtGui import QImage
    _IO_READ_WRITE = QIODevice.ReadWrite
else:
    from PyQt6.QtCore import QSize, QBuffer, QIODevice, QIODeviceBase as QIODevice
    from PyQt6.QtGui import QImage
    _IO_READ_WRITE = QIODevice.OpenModeFlag.ReadWrite

Img: TypeAlias = Image.Image
MatrixShape: TypeAlias = tuple[int, int]
ImgCoord: TypeAlias = tuple[float, float, float, float]


@dataclass
class SymbolImage:
    """Render a QGIS symbol as a cropped PIL image at the given scale factor."""

    symbol: QgsSymbol
    name: str
    scale_factor: int = 1
    img: Img = field(init=False)
    width: int = field(init=False)
    height: int = field(init=False)

    def __post_init__(self):
        self._render()

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
        """Render symbol at 1000 px, scale its size, then crop transparent borders."""
        try:
            symbol = self.symbol.clone()
            markers = []
            self._extract_markers(symbol, markers)
            for marker in markers:
                self._set_symbol_size(marker, self.scale_factor)
            qt_img = symbol.asImage(QSize(1000, 1000))
            pil_img = self._qt_to_pil(qt_img)
            bbox = pil_img.getbbox()
            self.img = pil_img.crop(bbox) if bbox else pil_img
            self.img.name = self.name
        except (RuntimeError, AttributeError):
            self.img = Image.new("RGBA", (10, 10), (255, 255, 255, 0))

        self.width = self.img.width
        self.height = self.img.height

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
    pixelspace: int = 20
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
    pixelspace: int = 20
    lowerfactor: int = 2
    scale_factor: int = 1
    img: Img = field(init=False)
    lowerimg: Img = field(init=False)
    imgscoords: dict = field(init=False)

    def __post_init__(self):
        self._build()
        self._populate()
        self._build_highres()

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
            coords[img.name] = (x, img_y, img.width, img.height)
            x += img.width + self.matrix.pixelspace
        return coords

    def _build_highres(self):
        """Generate a high-resolution version by upscaling with LANCZOS."""
        new_w = int(self.img.width * self.lowerfactor)
        new_h = int(self.img.height * self.lowerfactor)
        self.lowerimg = self.img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def save(self, output_dir: str):
        """Save base and high-res sprite PNGs to output_dir."""
        try:
            makedirs(output_dir, exist_ok=True)
            self.img.save(join(output_dir, "sprite.png"))
            self.lowerimg.save(join(output_dir, f"sprite@{self.lowerfactor}x.png"))
        except (OSError, RuntimeError):
            pass


@dataclass
class SpriteJSON:
    """Generate MapLibre JSON metadata for 1x and Nx sprite sheets."""

    spriteimg: SpriteImage
    lowerfactor: int = 2
    scale_factor: int = 1
    jsondict: dict = field(init=False)
    lowerjsondict: dict = field(init=False)

    def __post_init__(self):
        self._generate()

    def _generate(self):
        """Build coordinate dicts for both base and high-res sprite sheets."""
        self.jsondict = {
            name: {"x": int(x), "y": int(y), "width": int(w), "height": int(h), "pixelRatio": 1}
            for name, (x, y, w, h) in self.spriteimg.imgscoords.items()
        }
        self.lowerjsondict = {
            name: {
                "x": int(c["x"] * self.lowerfactor),
                "y": int(c["y"] * self.lowerfactor),
                "width": int(c["width"] * self.lowerfactor),
                "height": int(c["height"] * self.lowerfactor),
                "pixelRatio": self.lowerfactor,
            }
            for name, c in self.jsondict.items()
        }

    def save(self, output_dir: str):
        """Save base and high-res sprite JSON files to output_dir."""
        try:
            makedirs(output_dir, exist_ok=True)
            with open(join(output_dir, "sprite.json"), "w", encoding="utf8") as f:
                f.write(dumps(self.jsondict, indent=2))
            with open(join(output_dir, f"sprite@{self.lowerfactor}x.json"), "w", encoding="utf8") as f:
                f.write(dumps(self.lowerjsondict, indent=2))
        except (OSError, RuntimeError):
            pass


class SpriteGenerator:
    """Orchestrate the full sprite pipeline: render → arrange → JSON → save."""

    def __init__(
        self,
        symbols_dict: dict[str, QgsSymbol],
        output_dir: str,
        scale_factor: int = 1,
        test_mode: bool = False,
    ):
        self.symbols_dict = symbols_dict
        self.output_dir = output_dir
        self.scale_factor = scale_factor
        self.lower_factor = 2
        self.test_mode = test_mode

    def generate(self) -> Optional[str]:
        """Run the sprite generation pipeline; return output directory or None on failure."""
        if not self.symbols_dict:
            return None
        try:
            imgs = [
                SymbolImage(sym, name, self.scale_factor)
                for name, sym in self.symbols_dict.items()
            ]
            matrix = SpriteMatrix(imgs)
            sprite_img = SpriteImage(matrix, lowerfactor=self.lower_factor, scale_factor=self.scale_factor)
            sprite_json = SpriteJSON(sprite_img, lowerfactor=self.lower_factor, scale_factor=self.scale_factor)

            sprite_dir = join(self.output_dir, "sprite")
            sprite_img.save(sprite_dir)
            sprite_json.save(sprite_dir)

            if self.test_mode:
                self._verify_coordinates(sprite_img, sprite_json)

            return self.output_dir
        except (RuntimeError, AttributeError, KeyError, ValueError, OSError):
            return None

    def _verify_coordinates(self, sprite_img: SpriteImage, sprite_json: SpriteJSON):
        """Extract and save individual symbol crops to verify sprite coordinates."""
        for res_name, sprite, coords in [
            ("1x", sprite_img.img, sprite_json.jsondict),
            (f"{self.lower_factor}x", sprite_img.lowerimg, sprite_json.lowerjsondict),
        ]:
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
