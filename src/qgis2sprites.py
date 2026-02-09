"""
MapLibre Sprite Generator for QGIS - Safe Version for Linux/Windows

Generates MapLibre-compatible sprite sheets from QGIS marker symbols.
Output includes sprite.png, sprite.json and high-resolution versions.

The sprite generator supports scaling symbols to different sizes with a scale_factor parameter.
scale_factor controls how large symbols appear in the output:
  - scale_factor=1: Normal size, renders at 1000×1000, pixelRatio=1
  - scale_factor=2: 2× larger symbols, renders at 2000×2000, pixelRatio=2
  - scale_factor=3: 3× larger symbols, renders at 3000×3000, pixelRatio=3
  - etc.

Higher scale_factor values produce sharper symbols due to high-resolution rendering.
The JSON coordinates are automatically scaled down and pixelRatio is set accordingly,
so MapLibre displays symbols at the correct size while using the higher-resolution image.
The sprite sheet size scales proportionally (2× scale_factor = 4× file size).
"""
import zipfile
from dataclasses import dataclass, field
from json import dumps
from math import sqrt, ceil
from os import makedirs, listdir
from os.path import join, basename
from io import BytesIO
from typing import Optional, TypeAlias
from datetime import datetime
from gc import collect

from qgis.core import (
    QgsSymbol, QgsRuleBasedRenderer, QgsRuleBasedLabeling, QgsPalLayerSettings,
    QgsProject, QgsVectorLayerSimpleLabeling, QgsSingleSymbolRenderer,
    QgsCategorizedSymbolRenderer, QgsGraduatedSymbolRenderer
)
from qgis.PyQt.QtCore import qVersion
qt_version = int(qVersion()[0])
if qt_version == 5:
    from PyQt5.QtCore import QSize, QBuffer, QIODevice
    from PyQt5.QtGui import QImage
else:
    from PyQt6.QtCore import QSize, QBuffer, QIODevice
    from PyQt6.QtGui import QImage
from PIL import Image

Img: TypeAlias = Image.Image
MatrixShape: TypeAlias = tuple[int, int]
ImgCoord: TypeAlias = tuple[float, float, float, float]


@dataclass
class SymbolImage:
    """Wrapper for QGIS symbol rendered as PIL image.

    Renders a QGIS symbol at high resolution based on scale_factor,
    then crops transparent borders.

    Attributes:
        symbol: QgsSymbol to render
        name: Unique symbol identifier
        scale_factor: Scaling multiplier (1=render at 1000px, 2=render at 2000px, etc.)
                     Symbols render at 1000 × scale_factor pixels. JSON coordinates
                     are automatically scaled down so MapLibre displays at correct size.
    """
    symbol: QgsSymbol
    name: str
    scale_factor: int = 1
    img: Img = field(init=False)
    width: int = field(init=False)
    height: int = field(init=False)

    def __post_init__(self):
        self.generate_symbol_img()

    def generate_symbol_img(self):
        """Render symbol to PIL image at high resolution, crop transparent borders.

        Renders the symbol at 1000 × scale_factor pixels, then crops any
        transparent borders. Higher scale values produce sharper symbols.
        """
        try:
            # Render at higher resolution for sharpness
            render_size = 1000 * self.scale_factor
            qt_img = self.symbol.asImage(QSize(render_size, render_size))
            pil_img = self._qt_to_pil(qt_img)
            bbox = pil_img.getbbox()
            self.img = pil_img.crop(bbox) if bbox else pil_img
            self.img.name = self.name
        except (RuntimeError, AttributeError):
            # Fallback: create empty image if rendering fails
            self.img = Image.new("RGBA", (10, 10), (255, 255, 255, 0))

        # Set dimensions
        self.width = self.img.width
        self.height = self.img.height

    @staticmethod
    def _qt_to_pil(qt_img: QImage) -> Img:
        """Convert Qt QImage to PIL Image."""
        try:
            buffer = QBuffer()
            buffer.open(QIODevice.ReadWrite)
            qt_img.save(buffer, "PNG")
            bio = BytesIO(buffer.data())
            buffer.close()
            bio.seek(0)
            return Image.open(bio)
        except Exception:
            return Image.new("RGBA", (10, 10), (255, 255, 255, 0))


@dataclass
class SpriteMatrix:
    """Arranges symbol images in optimal grid layout."""
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
        """Calculate grid dimensions based on symbol count and aspect ratio."""
        count = len(self.imgs)
        ratio_prod = self.ratio[0] * self.ratio[1]
        base_size = sqrt(count / ratio_prod)
        
        height = ceil(base_size * self.ratio[0])
        width = ceil(base_size * self.ratio[1])
        
        if width * (height - 1) >= count:
            height -= 1
        self.shape = (height, width)

    def generate_imgs_matrix(self):
        """Distribute images into matrix rows."""
        h, w = self.shape
        symbols = list(self.imgs)
        self.imgsmatrix = [symbols[w * r : w * (r + 1)] for r in range(h)]

    def get_matrix_rows(self):
        """Calculate row dimensions (width, height, images)."""
        self.imgsrows = []
        for row in self.imgsmatrix:
            if not row:
                continue
            w = sum(img.width + self.pixelspace for img in row) - self.pixelspace
            h = max(img.height for img in row) + self.pixelspace
            self.imgsrows.append((w, h, row))


@dataclass
class SpriteImage:
    """Creates final sprite sheet with positioned symbols.

    Arranges symbol images into a sprite sheet, records their coordinates,
    and generates both 1× and 2× resolution versions.

    Attributes:
        matrix: SpriteMatrix containing arranged symbols
        pixelspace: Padding between symbols (default 20)
        lowerfactor: Scale multiplier for high-res version (default 2 for @2x)
        scale_factor: Symbol scaling multiplier from SymbolImage rendering
    """
    matrix: SpriteMatrix
    pixelspace: int = 20
    lowerfactor: int = 2
    scale_factor: int = 1
    img: Img = field(init=False)
    lowerimg: Img = field(init=False)
    imgscoords: dict = field(init=False)

    def __post_init__(self):
        self.construct_img()
        self.populate_img()
        self.generate_lowerimg()

    def construct_img(self):
        """Create blank sprite sheet with calculated dimensions."""
        if not self.matrix.imgsrows:
            self.img = Image.new("RGBA", (100, 100), (255, 255, 255, 0))
            return

        widths, heights, _ = zip(*self.matrix.imgsrows)
        w = max(widths) + self.pixelspace * 2
        h = sum(heights) + self.pixelspace * 2
        self.img = Image.new("RGBA", (w, h), (255, 255, 255, 0))

    def populate_img(self):
        """Paste symbols into sprite sheet, record coordinates."""
        self.imgscoords = {}
        y = self.pixelspace

        for row_width, row_height, row in self.matrix.imgsrows:
            x_align = round((self.img.width - row_width) / 2)
            row_coords = self._populate_row(row, x_align, row_height, y)
            self.imgscoords.update(row_coords)
            y += row_height

    def _populate_row(self, row: list[SymbolImage], left_x: float,
                      row_height: float, current_y: float) -> dict:
        """Paste images in row, return coordinate mapping."""
        coords = {}
        x = left_x

        for img in row:
            y_align = round((row_height - self.matrix.pixelspace - img.height) / 2)
            img_y = current_y + y_align

            self.img.paste(img.img, (int(x), int(img_y)))
            coords[img.name] = (x, img_y, img.width, img.height)
            x += img.width + self.matrix.pixelspace

        return coords

    def generate_lowerimg(self):
        """Generate high-resolution version (2x scale)."""
        new_w = int(self.img.width * self.lowerfactor)
        new_h = int(self.img.height * self.lowerfactor)
        self.lowerimg = self.img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def save(self, output_zip: str):
        """Save sprite images to zip."""
        with zipfile.ZipFile(output_zip, "w") as zf:
            bio1 = BytesIO()
            self.img.save(bio1, format='PNG')
            bio2 = BytesIO()
            self.lowerimg.save(bio2, format='PNG')
            zf.writestr("sprite.png", bio1.getvalue())
            zf.writestr(f"sprite@{self.lowerfactor}x.png", bio2.getvalue())


@dataclass
class SpriteJSON:
    """Generates MapLibre-compatible JSON metadata for sprites.

    Creates JSON files for both 1× and 2× sprite versions. Each JSON file
    contains coordinate mappings for all symbols in the sprite sheet,
    plus pixelRatio for MapLibre display scaling.

    Attributes:
        spriteimg: SpriteImage containing symbol positions
        lowerfactor: Scale multiplier for @2x version (default 2)
        scale_factor: Symbol scaling multiplier (controls final symbol size)
    """
    spriteimg: SpriteImage
    lowerfactor: int = 2
    scale_factor: int = 1
    jsondict: dict = field(init=False)
    lowerjsondict: dict = field(init=False)

    def __post_init__(self):
        self.generate_json()

    def generate_json(self):
        """Create JSON metadata for sprite coordinates.
        
        When scale_factor > 1, symbols render at higher resolution.
        The JSON width/height are divided by scale_factor so MapLibre
        displays them at the intended size, with pixelRatio indicating
        the higher resolution.
        """
        self.jsondict = {}
        for name, (x, y, w, h) in self.spriteimg.imgscoords.items():
            # Divide coordinates by scale_factor to get display size
            # pixelRatio tells MapLibre this is a high-resolution sprite
            self.jsondict[name] = {
                "x": int(x / self.scale_factor),
                "y": int(y / self.scale_factor),
                "width": int(w / self.scale_factor),
                "height": int(h / self.scale_factor),
                "pixelRatio": self.scale_factor
            }

        # High-resolution version (scaled by lowerfactor)
        # This is an additional scaling on top of scale_factor scaling
        self.lowerjsondict = {}
        for name, coords in self.jsondict.items():
            self.lowerjsondict[name] = {
                "x": int(coords["x"] * self.lowerfactor),
                "y": int(coords["y"] * self.lowerfactor),
                "width": int(coords["width"] * self.lowerfactor),
                "height": int(coords["height"] * self.lowerfactor),
                "pixelRatio": self.scale_factor * self.lowerfactor
            }

    def save(self, output_zip: str):
        """Save JSON metadata to zip."""
        with zipfile.ZipFile(output_zip, "a") as zf:
            zf.writestr("sprite.json", dumps(self.jsondict, indent=2))
            zf.writestr(f"sprite@{self.lowerfactor}x.json", 
                       dumps(self.lowerjsondict, indent=2))


class SpriteGenerator:
    """Main generator for MapLibre sprites.

    Orchestrates the sprite generation process:
    1. Renders symbols at scaled resolution
    2. Arranges them in optimal grid layout
    3. Generates coordinate JSON metadata
    4. Saves PNG and JSON files to zip

    Attributes:
        symbols_dict: Dictionary of symbol names to QgsSymbol objects
        output_dir: Output directory for generated files
        scale_factor: Scaling multiplier for symbols (1=normal, 2=2× larger, etc.)
        test_mode: Enable test coordinate extraction if True
    """

    def __init__(self, symbols_dict: dict[str, QgsSymbol], output_dir: str,
                 scale_factor: int = 1, test_mode: bool = False):
        self.symbols_dict = symbols_dict
        self.output_dir = output_dir
        self.scale_factor = scale_factor
        self.lower_factor = 2
        self.test_mode = test_mode

    def generate(self) -> Optional[str]:
        """Generate sprite sheet and metadata.

        Processes all symbols through the sprite generation pipeline
        and returns the output directory path.

        Returns:
            Output directory path if successful, None if symbols_dict is empty
        """
        if not self.symbols_dict:
            return None

        try:
            imgs = [SymbolImage(sym, name, self.scale_factor)
                   for name, sym in self.symbols_dict.items()]
            matrix = SpriteMatrix(imgs)
            sprite_img = SpriteImage(matrix, lowerfactor=self.lower_factor,
                                    scale_factor=self.scale_factor)
            sprite_json = SpriteJSON(sprite_img, lowerfactor=self.lower_factor,
                                    scale_factor=self.scale_factor)

            self._save_files(sprite_img, sprite_json)
            if self.test_mode:
                self._test_coordinates(sprite_img, sprite_json)

            return self.output_dir
        except Exception as e:
            print(f"Sprite generation error: {e}")
            return None

    def _save_files(self, sprite_img: SpriteImage, sprite_json: SpriteJSON):
        """Save sprite images and JSON to zip."""
        zip_path = f'{self.output_dir}.zip' if not self.test_mode else \
                   join(self.output_dir, f'{basename(self.output_dir)}.zip')
        sprite_img.save(zip_path)
        sprite_json.save(zip_path)

    def _test_coordinates(self, sprite_img: SpriteImage, sprite_json: SpriteJSON):
        """Extract individual symbols to verify coordinates."""
        for res_name, sprite, coords in [
            ("1x", sprite_img.img, sprite_json.jsondict),
            (f"{self.lower_factor}x", sprite_img.lowerimg, sprite_json.lowerjsondict)
        ]:
            test_dir = join(self.output_dir, f"test_{res_name}")
            makedirs(test_dir, exist_ok=True)
            
            for name, c in coords.items():
                try:
                    x, y, w, h = int(c["x"]), int(c["y"]), int(c["width"]), int(c["height"])
                    if x >= 0 and y >= 0 and w > 0 and h > 0 and \
                       x + w <= sprite.width and y + h <= sprite.height:
                        symbol_img = sprite.crop((x, y, x + w, y + h))
                        safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in name)
                        symbol_img.save(join(test_dir, f"{safe_name}.png"))
                except Exception:
                    pass


class QGIS2Sprites:
    """Safe sprite collector with Linux crash fixes.

    Main entry point for sprite generation. Collects all symbols from the
    current QGIS project's visible vector layers and generates a sprite sheet.

    Symbols are collected from:
    - Renderer symbols (single, categorized, graduated, rule-based)
    - Label background markers

    Attributes:
        base_output_dir: Base directory for output files
        scale_factor: Scaling multiplier for all symbols (1=normal, 2=2× larger, etc.)
        test_mode: If True, extracts individual symbols for verification

    Example:
        collector = QGIS2Sprites('/path/to/output', scale_factor=2)
        output = collector.generate_sprite()
    """

    def __init__(self, output_dir: str, scale_factor: int = 1, test_mode: bool = False):
        self.base_output_dir = output_dir
        self.scale_factor = scale_factor
        self.test_mode = test_mode
        self.symbols_dict = {}
        self.name_counter = {}

    def generate_sprite(self) -> Optional[str]:
        """Collect symbols from project and generate sprite sheet.

        Scans all visible vector layers, collects their symbols, and creates
        sprite sheet with PNG and JSON metadata.

        Returns:
            Output directory path if successful, None if no symbols found
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = join(self.base_output_dir, f"sprite_{timestamp}")
        if self.test_mode:
            makedirs(output_dir, exist_ok=True)

        self._collect_all_symbols()

        if not self.symbols_dict:
            return None

        generator = SpriteGenerator(self.symbols_dict, output_dir,
                                   self.scale_factor, self.test_mode)
        return generator.generate()

    def _collect_all_symbols(self):
        """Collect symbols from all visible vector layers."""
        try:
            project = QgsProject.instance()
            if not project:
                return
            
            layers = project.mapLayers().values()
            if not layers:
                return
            
            for layer_idx, layer in enumerate(layers):
                if not self._is_valid_layer(layer):
                    continue
                
                layer_name = layer.name() or f"layer_{layer_idx}"
                
                # Collect renderer symbols
                try:
                    if layer.renderer():
                        self._collect_renderer_symbols(layer.renderer(), layer_name, layer_idx)
                except (RuntimeError, AttributeError):
                    pass
                
                # Collect labeling symbols
                try:
                    if hasattr(layer, 'labeling') and layer.labeling():
                        self._collect_labeling_symbols(layer.labeling(), layer_name, layer_idx)
                except (RuntimeError, AttributeError):
                    pass
        
        except (RuntimeError, AttributeError, TypeError):
            pass
        finally:
            collect()

    def _is_valid_layer(self, layer) -> bool:
        """Check if layer is valid vector layer."""
        try:
            is_vector = layer.type() == 0 and layer.geometryType() != 4
            layer_node = QgsProject.instance().layerTreeRoot().findLayer(layer.id())
            is_visible = layer_node.isVisible() if layer_node else False
            return is_vector and is_visible
        except (RuntimeError, AttributeError):
            return False

    def _collect_renderer_symbols(self, renderer, layer_name: str, layer_idx: int):
        """Safely collect symbols from renderer."""
        if not renderer:
            return
        
        try:
            if isinstance(renderer, QgsSingleSymbolRenderer):
                self._collect_single_symbol(renderer.symbol(), layer_name, layer_idx)
            elif isinstance(renderer, QgsCategorizedSymbolRenderer):
                self._collect_categorized(renderer.categories(), layer_name, layer_idx)
            elif isinstance(renderer, QgsGraduatedSymbolRenderer):
                self._collect_graduated(renderer.ranges(), layer_name, layer_idx)
            elif isinstance(renderer, QgsRuleBasedRenderer):
                try:
                    root = renderer.rootRule()
                    if root:
                        self._collect_rule_symbols(root.children(), layer_name, layer_idx)
                except (RuntimeError, AttributeError):
                    pass
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _collect_single_symbol(self, symbol, layer_name: str, layer_idx: int):
        """Collect single symbol safely."""
        try:
            marker = self._get_marker_symbol(symbol)
            if marker:
                unique_name = self._get_unique_name(layer_name, layer_name, layer_idx)
                self.symbols_dict[unique_name] = marker.clone()
        except (RuntimeError, AttributeError):
            pass

    def _collect_categorized(self, categories, layer_name: str, layer_idx: int):
        """Collect categorized symbols safely."""
        if not categories:
            return
        
        for cat_idx, category in enumerate(categories):
            try:
                if category and category.renderState():
                    symbol = category.symbol()
                    if symbol:
                        marker = self._get_marker_symbol(symbol)
                        if marker:
                            label = category.label() or f"{layer_name}_{cat_idx}"
                            unique_name = self._get_unique_name(label, layer_name, layer_idx, cat_idx)
                            self.symbols_dict[unique_name] = marker.clone()
            except (RuntimeError, AttributeError, TypeError):
                pass

    def _collect_graduated(self, ranges, layer_name: str, layer_idx: int):
        """Collect graduated symbols safely."""
        if not ranges:
            return
        
        for range_idx, range_item in enumerate(ranges):
            try:
                if range_item and range_item.renderState():
                    symbol = range_item.symbol()
                    if symbol:
                        marker = self._get_marker_symbol(symbol)
                        if marker:
                            label = range_item.label() or f"{layer_name}_{range_idx}"
                            unique_name = self._get_unique_name(label, layer_name, layer_idx, range_idx)
                            self.symbols_dict[unique_name] = marker.clone()
            except (RuntimeError, AttributeError, TypeError):
                pass

    def _collect_rule_symbols(self, rules, layer_name: str, layer_idx: int, 
                             parent_path: str = ""):
        """Recursively collect rule-based symbols safely."""
        if not rules:
            return
        
        for rule_idx, rule in enumerate(rules):
            try:
                if rule and rule.active():
                    symbol = rule.symbol()
                    if symbol:
                        marker = self._get_marker_symbol(symbol)
                        if marker:
                            label = rule.label() or f"{layer_name}_{rule_idx}"
                            name = f"{parent_path}_{label}" if parent_path else label
                            unique_name = self._get_unique_name(name, layer_name, layer_idx, rule_idx)
                            self.symbols_dict[unique_name] = marker.clone()
                
                # Process children recursively
                try:
                    children = rule.children()
                    if children:
                        label = rule.label() or f"rule_{rule_idx}"
                        new_path = f"{parent_path}_{label}" if parent_path else label
                        self._collect_rule_symbols(children, layer_name, layer_idx, new_path)
                except (RuntimeError, AttributeError):
                    pass
            except (RuntimeError, AttributeError, TypeError):
                pass

    def _collect_labeling_symbols(self, labeling, layer_name: str, layer_idx: int):
        """Safely collect labeling symbols."""
        if not labeling:
            return
        
        try:
            if isinstance(labeling, QgsVectorLayerSimpleLabeling):
                try:
                    settings = labeling.settings()
                    if settings and self._has_marker_background(settings):
                        marker = self._safe_get_marker_from_settings(settings)
                        if marker:
                            unique_name = self._get_unique_name(f"{layer_name}_label", layer_name, layer_idx)
                            self.symbols_dict[unique_name] = marker.clone()
                except (RuntimeError, AttributeError):
                    pass
            
            elif isinstance(labeling, QgsRuleBasedLabeling):
                try:
                    root = labeling.rootRule()
                    if root:
                        self._collect_labeling_rules(root.children(), layer_name, layer_idx)
                except (RuntimeError, AttributeError):
                    pass
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _collect_labeling_rules(self, rules, layer_name: str, layer_idx: int, 
                               parent_path: str = ""):
        """Recursively collect labeling rule symbols safely."""
        if not rules:
            return
        
        for rule_idx, rule in enumerate(rules):
            try:
                if rule and rule.active():
                    settings = rule.settings()
                    if settings and self._has_marker_background(settings):
                        marker = self._safe_get_marker_from_settings(settings)
                        if marker:
                            label = rule.description() or f"{layer_name}_label_{rule_idx}"
                            name = f"{parent_path}_{label}" if parent_path else label
                            unique_name = self._get_unique_name(name, layer_name, layer_idx, rule_idx)
                            self.symbols_dict[unique_name] = marker.clone()
                
                # Process children
                try:
                    children = rule.children()
                    if children:
                        label = rule.description() or f"rule_{rule_idx}"
                        new_path = f"{parent_path}_{label}" if parent_path else label
                        self._collect_labeling_rules(children, layer_name, layer_idx, new_path)
                except (RuntimeError, AttributeError):
                    pass
            except (RuntimeError, AttributeError, TypeError):
                pass

    def _get_marker_symbol(self, symbol) -> Optional[QgsSymbol]:
        """Safely extract marker symbol from symbol."""
        if not symbol:
            return None
        
        try:
            # Check if already marker
            if hasattr(symbol, 'type') and symbol.type() == QgsSymbol.SymbolType.Marker:
                return symbol
            
            # Check symbol layers
            if hasattr(symbol, 'symbolLayers'):
                try:
                    layers = symbol.symbolLayers()
                    if layers:
                        for layer in layers:
                            if layer and hasattr(layer, 'subSymbol'):
                                try:
                                    subsym = layer.subSymbol()
                                    if subsym and hasattr(subsym, 'type') and \
                                       subsym.type() == QgsSymbol.SymbolType.Marker:
                                        return subsym
                                except (RuntimeError, AttributeError):
                                    continue
                except (RuntimeError, AttributeError):
                    pass
        except (RuntimeError, AttributeError, TypeError):
            pass
        
        return None

    def _has_marker_background(self, settings) -> bool:
        """Safely check if settings have marker background."""
        if not settings:
            return False
        
        try:
            if not hasattr(settings, 'format'):
                return False
            fmt = settings.format()
            if not fmt or not hasattr(fmt, 'background'):
                return False
            
            bg = fmt.background()
            if not bg or not hasattr(bg, 'enabled'):
                return False
            
            if not bg.enabled():
                return False
            
            if not hasattr(bg, 'markerSymbol'):
                return False
            
            marker = bg.markerSymbol()
            if marker and hasattr(marker, 'type'):
                return marker.type() == QgsSymbol.SymbolType.Marker
        except (RuntimeError, AttributeError, TypeError):
            pass
        
        return False

    def _safe_get_marker_from_settings(self, settings) -> Optional[QgsSymbol]:
        """Safely extract marker from label settings."""
        try:
            if hasattr(settings, 'format'):
                fmt = settings.format()
                if fmt and hasattr(fmt, 'background'):
                    bg = fmt.background()
                    if bg and hasattr(bg, 'markerSymbol'):
                        return bg.markerSymbol()
        except (RuntimeError, AttributeError):
            pass
        
        return None

    def _get_unique_name(self, name: str, layer_name: str, layer_idx: int, 
                        item_idx: int = None) -> str:
        """Generate unique symbol name."""
        name = (name or "").strip()
        
        if name in self.symbols_dict:
            suffix = f"{layer_name}_{item_idx}" if item_idx else layer_name or f"layer_{layer_idx}"
            name = f"{name}_{suffix}"
        
        if not name or name in self.symbols_dict:
            base = name or (layer_name or f"layer_{layer_idx}") + "_symbol"
            if base not in self.name_counter:
                self.name_counter[base] = 0
            self.name_counter[base] += 1
            name = f"{base}_{self.name_counter[base]}"
        
        return name


# Example usage for QGIS Python console
if __name__ == "__console__":
    # Set scale_factor to control symbol size:
    # 1 = normal size, 2 = 2× larger, 3 = 3× larger, etc.
    scale_factor = 2
    collector = QGIS2Sprites(output_dir=QgsProcessingUtils.tempFolder(), scale_factor=scale_factor)
    collector.generate_sprite()