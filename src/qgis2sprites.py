"""
MapLibre Sprite Generator for QGIS - Safe Version for Linux/Windows

Generates MapLibre-compatible sprite sheets from QGIS marker symbols.
Output includes sprite.png, sprite.json and high-resolution versions.

The sprite generator supports scaling symbols to different sizes with a scale_factor parameter.
scale_factor multiplies all symbol sizes (width, height, stroke width) by the factor:
  - scale_factor=1: Normal size (e.g., symbol size 2 renders as 2 units)
  - scale_factor=2: 2x larger symbols (symbol size 2 renders as 4 units)  
  - scale_factor=4: 4x larger symbols (symbol size 2 renders as 8 units)
  - etc.

Larger scale_factor produces larger symbols in the final sprite and in maps.
The sprite sheet size grows quadratically (2x scale_factor = 4x file size).
"""
import zipfile
from dataclasses import dataclass, field
from json import dumps
from math import sqrt, ceil
from os import makedirs
from os.path import join, basename
from io import BytesIO
from typing import Optional, TypeAlias
from datetime import datetime
from gc import collect

from qgis.core import (
    QgsProcessingUtils, QgsSymbol, QgsRuleBasedRenderer, QgsRuleBasedLabeling, QgsPalLayerSettings,
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
    """Render QGIS symbol as PIL image at scale_factor resolution, crop transparent borders."""
    symbol: QgsSymbol
    name: str
    scale_factor: int = 1
    img: Img = field(init=False)
    width: int = field(init=False)
    height: int = field(init=False)

    def __post_init__(self):
        self.generate_symbol_img()

    def generate_symbol_img(self):
        """Render at 1000×scale_factor px, crop transparent borders for sharp display."""
        try:
            # Clone symbol and scale its size by scale_factor
            symbol = self.symbol.clone()
            for layer_idx in range(symbol.symbolLayerCount()):
                layer = symbol.symbolLayer(layer_idx)
                if layer and hasattr(layer, 'setSize'):
                    current_size = layer.size() if hasattr(layer, 'size') else 0
                    if current_size and current_size > 0:
                        layer.setSize(current_size * self.scale_factor)
                # Scale stroke width if available
                if layer and hasattr(layer, 'setStrokeWidth'):
                    try:
                        current_width = layer.strokeWidth() if hasattr(layer, 'strokeWidth') else 0
                        if current_width and current_width > 0:
                            layer.setStrokeWidth(current_width * self.scale_factor)
                    except Exception:
                        pass
            
            # Render at fixed size with scaled symbol
            render_size = 1000
            qt_img = symbol.asImage(QSize(render_size, render_size))
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
        """Convert Qt QImage to PIL Image with fallback to empty RGBA."""
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
    """Arrange symbol images in grid layout with aspect ratio and pixelspace."""
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
        """Calculate optimal (height, width) grid based on symbol count and ratio."""
        count = len(self.imgs)
        ratio_prod = self.ratio[0] * self.ratio[1]
        base_size = sqrt(count / ratio_prod)
        
        height = ceil(base_size * self.ratio[0])
        width = ceil(base_size * self.ratio[1])
        
        if width * (height - 1) >= count:
            height -= 1
        self.shape = (height, width)

    def generate_imgs_matrix(self):
        """Split images into rows based on calculated grid shape."""
        h, w = self.shape
        symbols = list(self.imgs)
        self.imgsmatrix = [symbols[w * r : w * (r + 1)] for r in range(h)]

    def get_matrix_rows(self):
        """Compute (width, height, images) tuple for each row with pixelspace."""
        self.imgsrows = []
        for row in self.imgsmatrix:
            if not row:
                continue
            w = sum(img.width + self.pixelspace for img in row) - self.pixelspace
            h = max(img.height for img in row) + self.pixelspace
            self.imgsrows.append((w, h, row))


@dataclass
class SpriteImage:
    """Build sprite sheet from matrix, compute coordinates, generate 1× and 2× versions."""
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
        """Create blank RGBA sprite sheet at calculated (w, h) from matrix rows."""
        if not self.matrix.imgsrows:
            self.img = Image.new("RGBA", (100, 100), (255, 255, 255, 0))
            return

        widths, heights, _ = zip(*self.matrix.imgsrows)
        w = max(widths) + self.pixelspace * 2
        h = sum(heights) + self.pixelspace * 2
        self.img = Image.new("RGBA", (w, h), (255, 255, 255, 0))

    def populate_img(self):
        """Paste symbols into sheet and save (name→x,y,w,h) coordinate mapping."""
        self.imgscoords = {}
        y = self.pixelspace

        for row_width, row_height, row in self.matrix.imgsrows:
            x_align = round((self.img.width - row_width) / 2)
            row_coords = self._populate_row(row, x_align, row_height, y)
            self.imgscoords.update(row_coords)
            y += row_height

    def _populate_row(self, row: list[SymbolImage], left_x: float,
                      row_height: float, current_y: float) -> dict:
        """Paste row images centered, return {name: (x, y, w, h)} dict."""
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
        """Upscale sprite by lowerfactor using LANCZOS for high-res @Nx version."""
        new_w = int(self.img.width * self.lowerfactor)
        new_h = int(self.img.height * self.lowerfactor)
        self.lowerimg = self.img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def save(self, output_zip: str):
        """Save sprite images to zip."""
        try:
            with zipfile.ZipFile(output_zip, "w") as zf:
                bio1 = BytesIO()
                self.img.save(bio1, format='PNG')
                zf.writestr("sprite.png", bio1.getvalue())
                
                bio2 = BytesIO()
                self.lowerimg.save(bio2, format='PNG')
                zf.writestr(f"sprite@{self.lowerfactor}x.png", bio2.getvalue())
                print(f"    ✓ Sprite images written to {output_zip}")
        except Exception as e:
            print(f"    ✗ Error saving sprite images: {e}")


@dataclass
class SpriteJSON:
    """Generate MapLibre coordinate JSON for sprites with pixelRatio (1× and Nx versions)."""
    spriteimg: SpriteImage
    lowerfactor: int = 2
    scale_factor: int = 1
    jsondict: dict = field(init=False)
    lowerjsondict: dict = field(init=False)

    def __post_init__(self):
        self.generate_json()

    def generate_json(self):
        """Coordinates reflect scale_factor-multiplied symbol sizes; pixelRatio=1 for base sprites."""
        self.jsondict = {}
        for name, (x, y, w, h) in self.spriteimg.imgscoords.items():
            # Coordinates already include scale_factor effect (scaled symbols = larger coordinates)
            # pixelRatio=1 since coordinates are in final display pixels
            self.jsondict[name] = {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "pixelRatio": 1
            }

        # High-resolution version (scaled by lowerfactor for @2x, @3x, etc.)
        # Multiply base coordinates by lowerfactor; pixelRatio indicates rendered resolution
        self.lowerjsondict = {}
        for name, coords in self.jsondict.items():
            self.lowerjsondict[name] = {
                "x": int(coords["x"] * self.lowerfactor),
                "y": int(coords["y"] * self.lowerfactor),
                "width": int(coords["width"] * self.lowerfactor),
                "height": int(coords["height"] * self.lowerfactor),
                "pixelRatio": self.lowerfactor
            }

    def save(self, output_zip: str):
        """Save JSON metadata to zip."""
        try:
            with zipfile.ZipFile(output_zip, "a") as zf:
                zf.writestr("sprite.json", dumps(self.jsondict, indent=2))
                zf.writestr(f"sprite@{self.lowerfactor}x.json", 
                           dumps(self.lowerjsondict, indent=2))
                print(f"    ✓ JSON coordinates written to {output_zip}")
        except Exception as e:
            print(f"    ✗ Error saving JSON: {e}")


class SpriteGenerator:
    """Orchestrate sprite generation: render symbols, arrange grid, generate JSON, save zip."""

    def __init__(self, symbols_dict: dict[str, QgsSymbol], output_dir: str,
                 scale_factor: int = 1, test_mode: bool = False):
        self.symbols_dict = symbols_dict
        self.output_dir = output_dir
        self.scale_factor = scale_factor
        self.lower_factor = 2
        self.test_mode = test_mode

    def generate(self) -> Optional[str]:
        """Process symbols through pipeline and return output directory path or None."""
        if not self.symbols_dict:
            print(f"  ✗ No symbols in symbols_dict")
            return None

        try:
            print(f"  Creating SymbolImage objects for {len(self.symbols_dict)} symbols...")
            imgs = [SymbolImage(sym, name, self.scale_factor)
                   for name, sym in self.symbols_dict.items()]
            print(f"  ✓ Created {len(imgs)} SymbolImage objects")
            for img in imgs:
                print(f"    - {img.name}: {img.width}x{img.height}px")
            
            print(f"  Creating SpriteMatrix...")
            matrix = SpriteMatrix(imgs)
            print(f"  ✓ Matrix shape: {matrix.shape}")
            
            print(f"  Creating SpriteImage...")
            sprite_img = SpriteImage(matrix, lowerfactor=self.lower_factor,
                                    scale_factor=self.scale_factor)
            print(f"  ✓ Sprite image: {sprite_img.img.width}x{sprite_img.img.height}px")
            
            print(f"  Creating SpriteJSON...")
            sprite_json = SpriteJSON(sprite_img, lowerfactor=self.lower_factor,
                                    scale_factor=self.scale_factor)
            print(f"  ✓ JSON coordinates: {len(sprite_json.jsondict)} entries")

            print(f"  Saving sprite files to zip...")
            self._save_files(sprite_img, sprite_json)
            print(f"  ✓ Files saved")
            
            if self.test_mode:
                self._test_coordinates(sprite_img, sprite_json)

            print(f"✓ Sprites generated (scale_factor={self.scale_factor}): {self.output_dir}.zip")
            return self.output_dir
        except Exception as e:
            import traceback
            print(f"✗ Sprite generation error: {e}")
            traceback.print_exc()
            return None

    def _save_files(self, sprite_img: SpriteImage, sprite_json: SpriteJSON):
        """Save sprite images and JSON to zip."""
        zip_path = f'{self.output_dir}.zip' if not self.test_mode else \
                   join(self.output_dir, f'{basename(self.output_dir)}.zip')
        print(f"    Creating zip at: {zip_path}")
        sprite_img.save(zip_path)
        sprite_json.save(zip_path)
        print(f"    ✓ Zip saved")

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
    """Main entry: collect symbols from project layers, generate sprite sheet with scale_factor."""

    def __init__(self, output_dir: str, scale_factor: int = 1, test_mode: bool = False):
        self.base_output_dir = output_dir
        self.scale_factor = scale_factor
        self.test_mode = test_mode
        self.symbols_dict = {}
        self.name_counter = {}

    def generate_sprite(self) -> Optional[str]:
        """Scan layers, collect symbols, generate sprites; return path or None."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = join(self.base_output_dir, f"sprite_{timestamp}")
        if self.test_mode:
            makedirs(output_dir, exist_ok=True)

        self._collect_all_symbols()

        if not self.symbols_dict:
            return None

        generator = SpriteGenerator(self.symbols_dict, output_dir,
                                   self.scale_factor, self.test_mode)
        result = generator.generate()
        if result:
            print(f"✓ Sprite collection complete: {result}.zip")
        return result

    def _collect_all_symbols(self):
        """Scan project layers and collect renderer + labeling symbols; handle errors safely."""
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
        """Return True if layer is visible, vector, and non-point."""
        try:
            is_vector = layer.type() == 0 and layer.geometryType() != 4
            layer_node = QgsProject.instance().layerTreeRoot().findLayer(layer.id())
            is_visible = layer_node.isVisible() if layer_node else False
            return is_vector and is_visible
        except (RuntimeError, AttributeError):
            return False

    def _collect_renderer_symbols(self, renderer, layer_name: str, layer_idx: int):
        """Extract symbols from Single, Categorized, Graduated, or RuleBased renderers."""
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
        """Extract marker from symbol and add to dict with unique name."""
        try:
            marker = self._get_marker_symbol(symbol)
            if marker:
                unique_name = self._get_unique_name(layer_name, layer_name, layer_idx)
                self.symbols_dict[unique_name] = marker.clone()
        except (RuntimeError, AttributeError):
            pass

    def _collect_categorized(self, categories, layer_name: str, layer_idx: int):
        """Add marker from each category if enabled and symbol present."""
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
        """Add marker from each range if enabled and symbol present."""
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
        """Recursively add markers from active rules and their children."""
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
        """Extract marker from simple or rule-based label backgrounds."""
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
        """Recursively add markers from active labeling rules and their children."""
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
        """Extract marker from symbol or its layers' subsymbols; return None if not found."""
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
        """Return True if label settings has enabled background marker."""
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
        """Return marker from label background or None if unavailable."""
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
        """Generate collision-free name using suffix or counter."""
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
    scale_factor = 4
    collector = QGIS2Sprites(output_dir=QgsProcessingUtils.tempFolder(), scale_factor=scale_factor)
    collector.generate_sprite()