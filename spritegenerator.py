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

from qgis.core import QgsSymbol, QgsRuleBasedRenderer
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

    def __init__(self, rules: list, output_dir: str):
        """
        Initialize sprite generator.
        
        Args:
            rules: List of QGIS rules containing symbols
            output_dir: Directory for output files
        """
        self.rules = rules
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
        self._prepare_rules()
        imgs = self._extract_symbol_images()
        
        if not imgs:
            print("No marker symbols found in rules")
            return None
            
        matrix = SpriteMatrix(imgs)
        sprite_img = SpriteImage(matrix, lowerfactor=self.lower_factor)
        sprite_json = SpriteJSON(sprite_img, lowerfactor=self.lower_factor)
        
        self._save_files(sprite_img, sprite_json)
        
        if test_mode:
            self._test_coordinates(sprite_img, sprite_json)
            
        return self.output_dir

    def _prepare_rules(self):
        """
        Add properties to rules for uniform handling.
        
        Rules can be either renderer rules (type 0) or labeling rules (type 1).
        This method normalizes access to their symbols.
        """
        for rule in self.rules:
            if isinstance(rule, QgsRuleBasedRenderer.Rule):
                rule.type = 0
                rule.rulesymbol = rule.symbol()
            else:
                rule.type = 1
                # Extract marker symbol from label background
                rule.rulesymbol = rule.settings().format().background().markerSymbol()

    def _extract_symbol_images(self) -> list[SymbolImage]:
        """
        Extract and render symbol images from rules.
        
        Returns:
            List of SymbolImage objects ready for sprite sheet
        """
        imgs = []
        name_counter = {}
        
        for rule in self.rules:
            if self._rule_has_marker_symbol(rule):
                # Get rule name or generate unique name
                name = rule.description() if rule.description() else None
                
                # Generate unique name if empty or duplicate
                if not name or name in name_counter:
                    base_name = name if name else "symbol"
                    if base_name not in name_counter:
                        name_counter[base_name] = 0
                    name_counter[base_name] += 1
                    name = f"{base_name}_{name_counter[base_name]}"
                else:
                    name_counter[name] = 0
                
                symbol_img = SymbolImage(rule.rulesymbol, name)
                imgs.append(symbol_img)
        return imgs

    @staticmethod
    def _rule_has_marker_symbol(rule) -> bool:
        """
        Check if a rule contains a marker symbol suitable for sprites.
        
        Args:
            rule: QGIS rule object
            
        Returns:
            True if rule contains a marker symbol
        """
        if not hasattr(rule, 'rulesymbol') or rule.rulesymbol is None:
            return False
            
        # Labeling rules always have marker symbols
        if rule.type == 1:
            return True
            
        # Check renderer symbol layers
        symbol_layers = rule.rulesymbol.symbolLayers()
        if not symbol_layers:
            return False
            
        symbol_lyr = symbol_layers[0]
        
        # Direct marker layer
        if symbol_lyr.type() == QgsSymbol.SymbolType.Marker:
            return True
            
        # Marker in subsymbol
        subsymbol = symbol_lyr.subSymbol()
        if subsymbol and subsymbol.type() == QgsSymbol.SymbolType.Marker:
            return True
            
        return False

    def _save_files(self, sprite_img: SpriteImage, sprite_json: SpriteJSON):
        """
        Save sprite images and JSON metadata.
        
        Args:
            sprite_img: Sprite sheet images
            sprite_json: Sprite metadata
        """
        sprite_img.save(self.output_dir)
        sprite_json.save(self.output_dir)
        print(f"Sprite files saved to: {self.output_dir}")

    def _test_coordinates(self, sprite_img: SpriteImage, sprite_json: SpriteJSON):
        """
        Test coordinate accuracy by extracting individual symbols.
        
        Creates test directories with extracted symbols at both resolutions.
        This helps verify that JSON coordinates correctly map to sprite positions.
        
        Args:
            sprite_img: Sprite sheet images
            sprite_json: Sprite metadata
        """
        print("\n" + "="*60)
        print("Running coordinate verification test...")
        print("="*60)
        
        # Create test directories
        test_dir_1x = join(self.output_dir, "test_1x")
        test_dir_2x = join(self.output_dir, f"test_{self.lower_factor}x")
        makedirs(test_dir_1x, exist_ok=True)
        makedirs(test_dir_2x, exist_ok=True)
        
        # Debug info
        print(f"\nSprite dimensions:")
        print(f"  1x: {sprite_img.img.width}x{sprite_img.img.height}")
        print(f"  {self.lower_factor}x: {sprite_img.lowerimg.width}x{sprite_img.lowerimg.height}")
        print(f"\nTotal symbols in JSON: {len(sprite_json.jsondict)}")
        
        # Test standard resolution
        print(f"\nExtracting 1x symbols to: {test_dir_1x}")
        self._extract_test_symbols(
            sprite_img.img,
            sprite_json.jsondict,
            test_dir_1x,
            "1x"
        )
        
        # Test high resolution
        print(f"\nExtracting {self.lower_factor}x symbols to: {test_dir_2x}")
        self._extract_test_symbols(
            sprite_img.lowerimg,
            sprite_json.lowerjsondict,
            test_dir_2x,
            f"{self.lower_factor}x"
        )
        
        print("\n" + "="*60)
        print("Test complete!")
        print("="*60)

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
        success_count = 0
        error_count = 0
        
        for name, coords in coords_dict.items():
            try:
                x = int(coords["x"])
                y = int(coords["y"])
                width = int(coords["width"])
                height = int(coords["height"])
                
                # Validate coordinates
                if x < 0 or y < 0 or width <= 0 or height <= 0:
                    print(f"  ⚠️ Invalid coordinates for '{name}': x={x}, y={y}, w={width}, h={height}")
                    error_count += 1
                    continue
                
                if x + width > sprite.width or y + height > sprite.height:
                    print(f"  ⚠️ Coordinates out of bounds for '{name}': box=({x},{y},{x+width},{y+height}), sprite_size=({sprite.width},{sprite.height})")
                    error_count += 1
                    continue
                
                # Extract symbol region
                box = (x, y, x + width, y + height)
                symbol_img = sprite.crop(box)
                
                # Save with sanitized filename
                safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)
                output_path = join(output_dir, f"{safe_name}.png")
                symbol_img.save(output_path)
                success_count += 1
                
            except Exception as e:
                print(f"  ⚠️ Error extracting '{name}': {e}")
                error_count += 1
            
        print(f"  ✓ Extracted {success_count}/{len(coords_dict)} symbols at {resolution} resolution")
        if error_count > 0:
            print(f"  ✗ Failed to extract {error_count} symbols")


# Example usage for QGIS Python console
if __name__ == "__console__":
    output_dir = r'C:\Users\P0026701\OneDrive - Ness Israel\Desktop\ScratchWorkspace\New folder (2)'
    rules = []

    def fetch_rules(rule):
        """Recursively collect all rules from layer."""
        rules.append(rule)
        if rule.children():
            for child in rule.children():
                fetch_rules(child)

    # Extract rules from active layer
    layer = iface.activeLayer()
    
    # Get renderer rules
    if hasattr(layer, 'renderer') and layer.renderer():
        root_rule = layer.renderer().rootRule()
        if root_rule:
            for child in root_rule.children():
                fetch_rules(child)
    
    # Get labeling rules
    if hasattr(layer, 'labeling') and layer.labeling():
        labeling_root = layer.labeling().rootRule()
        if labeling_root:
            for child in labeling_root.children():
                fetch_rules(child)
    
    # Generate sprites with coordinate testing enabled
    generator = SpriteGenerator(rules, output_dir)
    result = generator.generate(test_mode=True)
    
    if result:
        print(f"\nSprite generation complete!")
        print(f"Output directory: {result}")
    else:
        print("\nNo sprites generated - no marker symbols found")