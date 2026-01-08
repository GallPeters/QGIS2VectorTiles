"""
MapLibre Sprite Generator for QGIS

This module generates MapLibre-compatible sprite sheets from QGIS marker symbols.
It creates a sprite atlas (PNG image + JSON metadata) with symbol coordinates for
use in MapLibre map rendering.

The output includes:
- sprite.png: Standard resolution sprite sheet
- sprite.json: Metadata with symbol coordinates
- sprite@2x.png: High resolution sprite sheet (2x scale)
- sprite@2x.json: Metadata for high resolution sprites
"""

from dataclasses import dataclass, field
from json import dump
from math import sqrt, ceil
from os import makedirs
from os.path import join
from io import BytesIO
from typing import Optional, TypeAlias
from datetime import datetime

from qgis.core import (
    QgsSymbol, 
    QgsRuleBasedRenderer,
    QgsRuleBasedLabeling,
    QgsCategorizedSymbolRenderer,
    QgsGraduatedSymbolRenderer,
    QgsSingleSymbolRenderer,
    QgsPalLayerSettings,
    QgsProject
)
from PyQt5.QtCore import QSize, QBuffer, QIODevice
from PyQt5.QtGui import QImage
from PIL import Image

# Type aliases for clarity
Img: TypeAlias = Image.Image
MatrixShape: TypeAlias = tuple[int, int]
MatrixRow: TypeAlias = tuple[int, int, list['SymbolImage']]
ImgCoord: TypeAlias = tuple[float, float, float, float]
ImgsCoords: TypeAlias = dict[str, ImgCoord]


@dataclass
class SymbolImage:
    """
    Wrapper for QGIS symbol that generates and stores a PIL image representation.
    
    Attributes:
        symbol: QGIS symbol object to render
        name: Identifier for the symbol in the sprite sheet
        img: Generated PIL image (created automatically)
    """
    symbol: QgsSymbol
    name: str
    img: Img = field(init=False)

    def __post_init__(self):
        """Generate the symbol image after initialization."""
        self.generate_symbol_img()

    def __getattr__(self, name):
        """Delegate attribute access to the underlying PIL image."""
        return getattr(self.img, name)

    def generate_symbol_img(self):
        """
        Render the QGIS symbol to a PIL image and crop to content bounds.
        
        The symbol is rendered at 1000x1000px, then cropped to remove transparent
        borders, optimizing space in the sprite sheet.
        """
        qt_img = self.symbol.asImage(QSize(1000, 1000))
        pil_img = self._qt_img_to_pil(qt_img)
        bbox = pil_img.getbbox()
        cropped_img = pil_img.crop(bbox) if bbox else pil_img
        cropped_img.name = self.name
        self.img = cropped_img

    def _qt_img_to_pil(self, qt_img: QImage) -> Img:
        """
        Convert Qt QImage to PIL Image.
        
        Args:
            qt_img: Qt image object
            
        Returns:
            PIL Image object
        """
        buffer = QBuffer()
        buffer.open(QIODevice.ReadWrite)
        qt_img.save(buffer, "PNG")
        strio = BytesIO()
        strio.write(buffer.data())
        buffer.close()
        strio.seek(0)
        return Image.open(strio)


@dataclass
class SpriteMatrix:
    """
    Arranges symbol images in a grid matrix for optimal sprite sheet layout.
    
    Attributes:
        imgs: List of symbol images to arrange
        ratio: Aspect ratio for the matrix (height_factor, width_factor)
        pixelspace: Spacing between symbols in pixels
        shape: Calculated matrix dimensions (height, width) in symbols
        imgsmatrix: 2D list of images arranged in rows
        imgsrows: Row metadata (width, height, images)
    """
    imgs: list[SymbolImage]
    ratio: tuple[int, int] = (3, 4)
    pixelspace: int = 20
    shape: MatrixShape = field(init=False)
    imgsmatrix: list[list[SymbolImage]] = field(init=False)
    imgsrows: list[MatrixRow] = field(init=False)

    def __post_init__(self):
        """Calculate layout and generate matrix structure."""
        self.calculate_shape()
        self.generate_imgs_matrix()
        self.get_matrix_rows()

    def calculate_shape(self):
        """
        Calculate optimal matrix dimensions based on symbol count and aspect ratio.
        
        Uses the ratio to determine a grid that accommodates all symbols while
        maintaining the desired aspect ratio. Optimizes by removing empty rows.
        """
        imgs_count = len(self.imgs)
        ratio_product = self.ratio[0] * self.ratio[1]
        base_size = sqrt(imgs_count / ratio_product)
        
        height = ceil(base_size * self.ratio[0])
        width = ceil(base_size * self.ratio[1])
        
        # Remove unnecessary rows
        if width * (height - 1) >= imgs_count:
            height -= 1
            
        self.shape = (height, width)

    def generate_imgs_matrix(self):
        """Distribute symbol images into matrix rows."""
        height, width = self.shape
        symbols = list(self.imgs)
        self.imgsmatrix = [
            symbols[width * row_num : width * (row_num + 1)] 
            for row_num in range(height)
        ]

    def get_matrix_rows(self):
        """
        Calculate dimensions for each row in the matrix.
        
        Each row's width is the sum of its image widths plus spacing.
        Each row's height is the maximum image height in that row.
        """
        matrix_rows = []
        for row in self.imgsmatrix:
            if not row:
                continue
            width = sum(img.width + self.pixelspace for img in row) - self.pixelspace
            height = max(img.height for img in row) + self.pixelspace
            matrix_rows.append((width, height, row))
        self.imgsrows = matrix_rows


@dataclass
class SpriteImage:
    """
    Creates the final sprite sheet image with all symbols positioned correctly.
    
    Attributes:
        matrix: Symbol matrix with layout information
        pixelspace: Padding around edges and between symbols
        lowerfactor: Scale factor for high-res version (default 2x)
        img: Generated sprite sheet image
        lowerimg: High resolution version of sprite sheet
        imgscoords: Dictionary mapping symbol names to coordinates
    """
    matrix: SpriteMatrix
    pixelspace: int = 20
    lowerfactor: int = 2
    img: Img = field(init=False)
    lowerimg: Img = field(init=False)
    imgscoords: ImgsCoords = field(init=False)

    def __post_init__(self):
        """Construct and populate the sprite sheet."""
        self.construct_img()
        self.populate_img()
        self.generate_lowerimg()

    def construct_img(self):
        """
        Create blank sprite sheet with calculated dimensions.
        
        Size is determined by the widest row and total height of all rows,
        plus edge padding.
        """
        dimensions = [(width, height) for width, height, _ in self.matrix.imgsrows]
        if not dimensions:
            self.img = Image.new("RGBA", (100, 100), (255, 255, 255, 0))
            return
            
        rows_widths, rows_heights = zip(*dimensions)
        edges_buffer = self.pixelspace * 2
        sprite_width = max(rows_widths) + edges_buffer
        sprite_height = sum(rows_heights) + edges_buffer
        
        # Transparent background (alpha=0)
        self.img = Image.new("RGBA", (sprite_width, sprite_height), (255, 255, 255, 0))

    def populate_img(self):
        """
        Paste all symbol images into the sprite sheet and record coordinates.
        
        Symbols are centered horizontally in the sprite and vertically within
        their row. Coordinates are stored in MapLibre format (x, y from top-left).
        """
        imgs_coords = {}
        current_y = self.pixelspace
        
        for row_width, row_height, row in self.matrix.imgsrows:
            # Center row horizontally
            horizontal_align = round((self.img.width - row_width) / 2)
            img_left_x = horizontal_align
            
            row_coords = self._populate_row(
                row, img_left_x, row_height, current_y
            )
            imgs_coords.update(row_coords)
            current_y += row_height
            
        self.imgscoords = imgs_coords

    def _populate_row(
        self, 
        row_imgs: list[SymbolImage], 
        left_x: float, 
        row_height: float, 
        current_y: float
    ) -> dict[str, ImgCoord]:
        """
        Paste images in a single row and record their coordinates.
        
        Args:
            row_imgs: Symbol images in this row
            left_x: Starting x position for the row
            row_height: Height allocated for this row
            current_y: Y position of the row top
            
        Returns:
            Dictionary mapping symbol names to (x, y, width, height) tuples
        """
        row_imgs_coords = {}
        current_x = left_x
        
        for img in row_imgs:
            # Center image vertically in row
            vertical_align = round((row_height - self.matrix.pixelspace - img.height) / 2)
            img_upper_y = current_y + vertical_align
            
            # Paste image
            self.img.paste(img.img, (int(current_x), int(img_upper_y)))
            
            # Store coordinates (x, y from top-left, width, height)
            row_imgs_coords[img.name] = (
                current_x,
                img_upper_y,
                img.width,
                img.height
            )
            
            current_x += img.width + self.matrix.pixelspace
            
        return row_imgs_coords

    def generate_lowerimg(self):
        """Generate high-resolution version by scaling up the sprite sheet."""
        new_size = (
            int(self.img.width * self.lowerfactor),
            int(self.img.height * self.lowerfactor)
        )
        self.lowerimg = self.img.resize(new_size, Image.Resampling.LANCZOS)

    def save(self, output_dir: str):
        """
        Save sprite sheet images to disk.
        
        Args:
            output_dir: Directory to save files
        """
        img_path = join(output_dir, "sprite")
        self.img.save(f"{img_path}.png")
        
        lowerimg_path = f"{img_path}@{self.lowerfactor}x.png"
        self.lowerimg.save(lowerimg_path)


@dataclass
class SpriteJSON:
    """
    Generates MapLibre-compatible JSON metadata for sprite sheets.
    
    The JSON format specifies each symbol's position and dimensions in the sprite,
    following MapLibre's sprite specification.
    
    Attributes:
        spriteimg: Sprite image with coordinate data
        jsondict: Standard resolution metadata
        lowerfactor: Scale factor for high-res version
        lowerjsondict: High resolution metadata
    """
    spriteimg: SpriteImage
    jsondict: dict = field(init=False)
    lowerfactor: int = 2
    lowerjsondict: dict = field(init=False)

    def __post_init__(self):
        """Generate JSON metadata for both resolutions."""
        self.generate_json()
        self.generate_lowerjsondict()

    def generate_json(self):
        """
        Create MapLibre sprite JSON with symbol coordinates.
        
        Format for each symbol:
        {
            "symbol_name": {
                "width": <int>,
                "height": <int>,
                "x": <int>,
                "y": <int>,
                "pixelRatio": 1
            }
        }
        """
        sprite_json = {}
        
        for name, (x, y, width, height) in self.spriteimg.imgscoords.items():
            sprite_json[name] = {
                "width": int(width),
                "height": int(height),
                "x": int(x),
                "y": int(y),
                "pixelRatio": 1
            }
            
        self.jsondict = sprite_json

    def generate_lowerjsondict(self):
        """
        Create high-res JSON by scaling coordinates.
        
        All dimensions and positions are multiplied by lowerfactor,
        and pixelRatio is updated accordingly.
        """
        lowerjsondict = {}
        
        for name, coords in self.jsondict.items():
            lowerjsondict[name] = {
                "width": int(coords["width"] * self.lowerfactor),
                "height": int(coords["height"] * self.lowerfactor),
                "x": int(coords["x"] * self.lowerfactor),
                "y": int(coords["y"] * self.lowerfactor),
                "pixelRatio": self.lowerfactor
            }
            
        self.lowerjsondict = lowerjsondict

    def save(self, output_dir: str):
        """
        Save JSON metadata files to disk.
        
        Args:
            output_dir: Directory to save files
        """
        json_path = join(output_dir, "sprite")
        
        with open(f"{json_path}.json", "w", encoding="utf8") as output:
            dump(self.jsondict, output, indent=2)
            
        lower_json_path = f"{json_path}@{self.lowerfactor}x"
        with open(f"{lower_json_path}.json", "w", encoding="utf8") as output_lower:
            dump(self.lowerjsondict, output_lower, indent=2)


class SpriteGenerator:
    """
    Main class for generating MapLibre sprite sheets from QGIS symbols.
    
    This generator extracts marker symbols from QGIS rules (both renderer and
    labeling), arranges them in an optimal sprite sheet layout, and outputs
    MapLibre-compatible sprite files.
    """

    def __init__(self, symbols_dict: dict[str, QgsSymbol], output_dir: str):
        """
        Initialize sprite generator.
        
        Args:
            symbols_dict: Dictionary mapping symbol names to QGIS symbols
            output_dir: Directory for output files
        """
        self.symbols_dict = symbols_dict
        self.output_dir = output_dir
        self.lower_factor = 2

    def generate(self, test_mode: bool = False) -> Optional[str]:
        """
        Generate sprite sheet and metadata files.
        
        Args:
            test_mode: If True, extract individual symbols from sprite to verify
                      coordinate accuracy
        
        Returns:
            Output directory path on success, None if no symbols found
        """
        imgs = self._create_symbol_images()
        
        if not imgs:
            return None
            
        matrix = SpriteMatrix(imgs)
        sprite_img = SpriteImage(matrix, lowerfactor=self.lower_factor)
        sprite_json = SpriteJSON(sprite_img, lowerfactor=self.lower_factor)
        
        self._save_files(sprite_img, sprite_json)
        
        if test_mode:
            self._test_coordinates(sprite_img, sprite_json)
            
        return self.output_dir

    def _create_symbol_images(self) -> list[SymbolImage]:
        """
        Create SymbolImage objects from the symbols dictionary.
        
        Returns:
            List of SymbolImage objects ready for sprite sheet
        """
        imgs = []
        for name, symbol in self.symbols_dict.items():
            symbol_img = SymbolImage(symbol, name)
            imgs.append(symbol_img)
        return imgs

    def _save_files(self, sprite_img: SpriteImage, sprite_json: SpriteJSON):
        """
        Save sprite images and JSON metadata.
        
        Args:
            sprite_img: Sprite sheet images
            sprite_json: Sprite metadata
        """
        sprite_img.save(self.output_dir)
        sprite_json.save(self.output_dir)

    def _test_coordinates(self, sprite_img: SpriteImage, sprite_json: SpriteJSON):
        """
        Test coordinate accuracy by extracting individual symbols.
        
        Creates test directories with extracted symbols at both resolutions.
        This helps verify that JSON coordinates correctly map to sprite positions.
        
        Args:
            sprite_img: Sprite sheet images
            sprite_json: Sprite metadata
        """
        test_dir_1x = join(self.output_dir, "test_1x")
        test_dir_2x = join(self.output_dir, f"test_{self.lower_factor}x")
        makedirs(test_dir_1x, exist_ok=True)
        makedirs(test_dir_2x, exist_ok=True)
        
        self._extract_test_symbols(
            sprite_img.img,
            sprite_json.jsondict,
            test_dir_1x,
            "1x"
        )
        
        self._extract_test_symbols(
            sprite_img.lowerimg,
            sprite_json.lowerjsondict,
            test_dir_2x,
            f"{self.lower_factor}x"
        )

    def _extract_test_symbols(
        self,
        sprite: Img,
        coords_dict: dict,
        output_dir: str,
        resolution: str
    ):
        """
        Extract individual symbols from sprite using JSON coordinates.
        
        Args:
            sprite: Sprite sheet image
            coords_dict: Symbol coordinate metadata
            output_dir: Directory to save extracted symbols
            resolution: Resolution label for logging
        """
        for name, coords in coords_dict.items():
            try:
                x = int(coords["x"])
                y = int(coords["y"])
                width = int(coords["width"])
                height = int(coords["height"])
                
                if x < 0 or y < 0 or width <= 0 or height <= 0:
                    continue
                
                if x + width > sprite.width or y + height > sprite.height:
                    continue
                
                box = (x, y, x + width, y + height)
                symbol_img = sprite.crop(box)
                
                safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)
                output_path = join(output_dir, f"{safe_name}.png")
                symbol_img.save(output_path)
                
            except Exception:
                continue


class QGIS2Sprites:
    """
    Collects all marker symbols from QGIS project layers.
    
    Extracts symbols from:
    - Single symbol renderers
    - Categorized renderers
    - Graduated renderers
    - Rule-based renderers
    - Single labeling (background markers)
    - Rule-based labeling (background markers)
    """

    def __init__(self, output_dir: str):
        """
        Initialize symbol collector.
        
        Args:
            output_dir: Base directory for output files
        """
        self.base_output_dir = output_dir
        self.symbols_dict = {}
        self.name_counter = {}

    def generate_sprite(self, test_mode: bool = False) -> Optional[str]:
        """
        Collect all symbols from project layers and generate sprite sheet.
        
        Args:
            test_mode: If True, generate test extraction folders
            
        Returns:
            Output directory path on success, None if no symbols found
        """
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = join(self.base_output_dir, f"sprite_{timestamp}")
        makedirs(output_dir, exist_ok=True)
        
        # Collect symbols from all layers
        self._collect_all_symbols()
        
        if not self.symbols_dict:
            return None
        
        # Generate sprite sheet
        generator = SpriteGenerator(self.symbols_dict, output_dir)
        return generator.generate(test_mode=test_mode)

    def _collect_all_symbols(self):
        """Collect symbols from all visible vector layers in the project."""
        project = QgsProject.instance()
        layers = project.mapLayers().values()
        
        for layer_idx, layer in enumerate(layers):
            # Skip non-vector layers
            if not hasattr(layer, 'renderer'):
                continue
                
            # Skip invisible layers
            if not layer.isSpatial() or not layer.isValid():
                continue
            
            layer_name = layer.name() if layer.name() else f"layer_{layer_idx}"
            
            # Collect renderer symbols
            if layer.renderer():
                self._collect_renderer_symbols(layer.renderer(), layer_name, layer_idx)
            
            # Collect labeling symbols
            if hasattr(layer, 'labeling') and layer.labeling():
                self._collect_labeling_symbols(layer.labeling(), layer_name, layer_idx)

    def _collect_renderer_symbols(self, renderer, layer_name: str, layer_idx: int):
        """
        Collect marker symbols from layer renderer.
        
        Args:
            renderer: QGIS renderer object
            layer_name: Name of the layer
            layer_idx: Index of the layer
        """
        renderer_type = type(renderer).__name__
        
        if isinstance(renderer, QgsSingleSymbolRenderer):
            symbol = renderer.symbol()
            if self._is_marker_symbol(symbol):
                name = layer_name
                unique_name = self._get_unique_name(name, layer_name, layer_idx)
                self.symbols_dict[unique_name] = symbol.clone()
                
        elif isinstance(renderer, QgsCategorizedSymbolRenderer):
            for cat_idx, category in enumerate(renderer.categories()):
                symbol = category.symbol()
                if self._is_marker_symbol(symbol):
                    name = category.label() if category.label() else f"{layer_name}_{cat_idx}"
                    unique_name = self._get_unique_name(name, layer_name, layer_idx, cat_idx)
                    self.symbols_dict[unique_name] = symbol.clone()
                    
        elif isinstance(renderer, QgsGraduatedSymbolRenderer):
            for range_idx, range_item in enumerate(renderer.ranges()):
                symbol = range_item.symbol()
                if self._is_marker_symbol(symbol):
                    name = range_item.label() if range_item.label() else f"{layer_name}_{range_idx}"
                    unique_name = self._get_unique_name(name, layer_name, layer_idx, range_idx)
                    self.symbols_dict[unique_name] = symbol.clone()
                    
        elif isinstance(renderer, QgsRuleBasedRenderer):
            root_rule = renderer.rootRule()
            if root_rule:
                self._collect_rule_symbols(root_rule.children(), layer_name, layer_idx)

    def _collect_rule_symbols(self, rules, layer_name: str, layer_idx: int, parent_path: str = ""):
        """
        Recursively collect symbols from rule-based renderer.
        
        Args:
            rules: List of renderer rules
            layer_name: Name of the layer
            layer_idx: Index of the layer
            parent_path: Path of parent rules for nested rules
        """
        for rule_idx, rule in enumerate(rules):
            symbol = rule.symbol()
            if symbol and self._is_marker_symbol(symbol):
                rule_label = rule.label() if rule.label() else f"{layer_name}_{rule_idx}"
                name = f"{parent_path}_{rule_label}" if parent_path else rule_label
                unique_name = self._get_unique_name(name, layer_name, layer_idx, rule_idx)
                self.symbols_dict[unique_name] = symbol.clone()
            
            # Recursively process child rules
            if rule.children():
                new_path = f"{parent_path}_{rule_label}" if parent_path else rule_label
                self._collect_rule_symbols(rule.children(), layer_name, layer_idx, new_path)

    def _collect_labeling_symbols(self, labeling, layer_name: str, layer_idx: int):
        """
        Collect marker symbols from layer labeling.
        
        Args:
            labeling: QGIS labeling object
            layer_name: Name of the layer
            layer_idx: Index of the layer
        """
        # Handle simple labeling
        if hasattr(labeling, 'settings'):
            settings = labeling.settings()
            if self._has_marker_background(settings):
                marker = settings.format().background().markerSymbol()
                name = f"{layer_name}_label"
                unique_name = self._get_unique_name(name, layer_name, layer_idx)
                self.symbols_dict[unique_name] = marker.clone()
        
        # Handle rule-based labeling
        elif isinstance(labeling, QgsRuleBasedLabeling):
            root_rule = labeling.rootRule()
            if root_rule:
                self._collect_labeling_rule_symbols(root_rule.children(), layer_name, layer_idx)

    def _collect_labeling_rule_symbols(self, rules, layer_name: str, layer_idx: int, parent_path: str = ""):
        """
        Recursively collect symbols from rule-based labeling.
        
        Args:
            rules: List of labeling rules
            layer_name: Name of the layer
            layer_idx: Index of the layer
            parent_path: Path of parent rules for nested rules
        """
        for rule_idx, rule in enumerate(rules):
            settings = rule.settings()
            if settings and self._has_marker_background(settings):
                marker = settings.format().background().markerSymbol()
                rule_desc = rule.description() if rule.description() else f"{layer_name}_label_{rule_idx}"
                name = f"{parent_path}_{rule_desc}" if parent_path else rule_desc
                unique_name = self._get_unique_name(name, layer_name, layer_idx, rule_idx)
                self.symbols_dict[unique_name] = marker.clone()
            
            # Recursively process child rules
            if rule.children():
                new_path = f"{parent_path}_{rule_desc}" if parent_path else rule_desc
                self._collect_labeling_rule_symbols(rule.children(), layer_name, layer_idx, new_path)

    def _is_marker_symbol(self, symbol) -> bool:
        """
        Check if symbol is a marker type.
        
        Args:
            symbol: QGIS symbol object
            
        Returns:
            True if symbol is marker type
        """
        if not symbol:
            return False
            
        if symbol.type() == QgsSymbol.SymbolType.Marker:
            return True
            
        # Check symbol layers
        symbol_layers = symbol.symbolLayers()
        if symbol_layers:
            for layer in symbol_layers:
                if layer.type() == QgsSymbol.SymbolType.Marker:
                    return True
                    
                subsymbol = layer.subSymbol() if hasattr(layer, 'subSymbol') else None
                if subsymbol and subsymbol.type() == QgsSymbol.SymbolType.Marker:
                    return True
        
        return False

    def _has_marker_background(self, settings) -> bool:
        """
        Check if label settings have marker background enabled.
        
        Args:
            settings: QgsPalLayerSettings object
            
        Returns:
            True if background is enabled and is marker type
        """
        if not settings:
            return False
            
        try:
            background = settings.format().background()
            if not background.enabled():
                return False
                
            marker = background.markerSymbol()
            return marker is not None and marker.type() == QgsSymbol.SymbolType.Marker
        except Exception:
            return False

    def _get_unique_name(self, name: str, layer_name: str, layer_idx: int, item_idx: int = None) -> str:
        """
        Generate unique name for symbol, handling duplicates.
        
        Args:
            name: Preferred symbol name
            layer_name: Name of the layer
            layer_idx: Index of the layer
            item_idx: Index of the item/rule (optional)
            
        Returns:
            Unique symbol name
        """
        # Clean name
        name = name.strip() if name else ""
        
        # If name exists, append layer info
        if name in self.symbols_dict:
            if item_idx is not None:
                suffix = f"{layer_name}_{item_idx}" if layer_name else f"layer_{layer_idx}_{item_idx}"
            else:
                suffix = layer_name if layer_name else f"layer_{layer_idx}"
            name = f"{name}_{suffix}"
        
        # If still exists or empty, use counter
        if not name or name in self.symbols_dict:
            base_name = name if name else f"{layer_name}_symbol" if layer_name else f"layer_{layer_idx}_symbol"
            if base_name not in self.name_counter:
                self.name_counter[base_name] = 0
            self.name_counter[base_name] += 1
            name = f"{base_name}_{self.name_counter[base_name]}"
        
        return name


# Example usage for QGIS Python console
if __name__ == "__console__":
    output_dir = r'C:\Users\P0026701\OneDrive - Ness Israel\Desktop\ScratchWorkspace'
    collector = QGIS2Sprites(output_dir)
    collector.generate_sprite(test_mode=True)