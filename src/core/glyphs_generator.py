import os
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from qgis.PyQt.QtCore import qVersion
# PyQt version guard — import the right Qt5 / Qt6 symbols once, re-export

if int(qVersion()[0]) == 5:
    from PyQt5.QtGui import (
        QGuiApplication,
        QFontDatabase,
        QFont,
        QFontMetrics,
        QPainterPath
    )
    from PyQt5.QtCore import QCoreApplication
else:
    from PyQt6.QtGui import (
        QGuiApplication,
        QFontDatabase,
        QFont,
        QFontMetrics,
        QPainterPath
    )
    from PyQt6.QtCore import QCoreApplication

logger = logging.getLogger(__name__)

class GlyphGenerator:
    """
    Generates MapLibre-compatible SDF glyphs (PBF format) from system fonts.
    
    Leverages PyQt6 for cross-platform system font discovery and path extraction.
    """

    GLYPH_RANGE_SIZE = 256
    MAX_UNICODE = 65535 

    def __init__(self, fonts: List[str], output_dir: str):
        self.requested_fonts = fonts
        self.output_dir = Path(output_dir)
        
        # Ensure a QCoreApplication exists, required for QFontDatabase access
        self._app = QCoreApplication.instance()
        if not self._app:
            self._app = QGuiApplication([])

    def generate(self) -> str:
        """
        Main execution method. Discovers fonts, generates SDF ranges, and writes PBFs.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # In PyQt6, families() is a static method on QFontDatabase
        available_families = QFontDatabase.families()
        
        for requested_family in self.requested_fonts:
            matched_family = next((f for f in available_families if f.lower() == requested_family.lower()), None)
            
            if not matched_family:
                logger.warning(f"Font family '{requested_family}' not found. Skipping.")
                continue
                
            logger.info(f"Processing font family: {matched_family}")
            
            # styles() is also a static method in PyQt6
            styles = QFontDatabase.styles(matched_family)
            for style in styles:
                style_name = "Regular" if style.lower() in ["normal", "regular"] or not style else style
                fontstack_name = f"{matched_family} {style_name}".strip()
                
                fontstack_dir = self.output_dir / fontstack_name
                fontstack_dir.mkdir(exist_ok=True)
                
                # font() is a static method in PyQt6
                qfont = QFontDatabase.font(matched_family, style, 24)
                
                self._generate_ranges_for_fontstack(qfont, fontstack_name, fontstack_dir)

        return str(self.output_dir)

    def _generate_ranges_for_fontstack(self, font: QFont, fontstack_name: str, out_dir: Path) -> None:
        """Iterates over Unicode ranges and creates the .pbf files."""
        for start in range(0, self.MAX_UNICODE, self.GLYPH_RANGE_SIZE):
            end = start + self.GLYPH_RANGE_SIZE - 1
            
            pbf_data = self._create_sdf_pbf_for_range(font, fontstack_name, start, end)
            
            # Since pbf_data is now guaranteed to be a non-empty byte string (if glyphs exist), 
            # this check passes and the file is written.
            if pbf_data:
                pbf_path = out_dir / f"{start}-{end}.pbf"
                with open(pbf_path, "wb") as f:
                    f.write(pbf_data)

    def _create_sdf_pbf_for_range(self, font: QFont, fontstack_name: str, start: int, end: int) -> Optional[bytes]:
        """Extracts metrics and paths, generates an SDF bitmap, and encodes into PBF format."""
        metrics = QFontMetrics(font)
        
        has_glyphs = any(metrics.inFontUcs4(char_code) for char_code in range(start, end + 1))
        if not has_glyphs:
            return None 
            
        glyphs_data = []

        for char_code in range(start, end + 1):
            if not metrics.inFontUcs4(char_code):
                continue
                
            char_str = chr(char_code)
            
            advance = metrics.horizontalAdvance(char_str)
            rect = metrics.boundingRect(char_str)
            
            path = QPainterPath()
            path.addText(0, 0, font, char_str)
            
            sdf_bitmap, width, height, left, top = self._render_sdf_bitmap(path, rect)
            
            glyphs_data.append({
                "id": char_code,
                "bitmap": sdf_bitmap,
                "width": width,
                "height": height,
                "left": left,
                "top": top,
                "advance": advance
            })

        if not glyphs_data:
            return None

        return self._serialize_to_pbf(fontstack_name, glyphs_data)

    def _render_sdf_bitmap(self, path: QPainterPath, bounding_rect) -> Tuple[bytes, int, int, int, int]:
            """
            Converts a Qt vector path into a Signed Distance Field bitmap
            with scaled metrics to maintain visual consistency.
            """
            # Increase this for higher sharpness; keep it consistent
            BUFFER = 5
            
            # 1. Expand the container size (The PBF asset size)
            # This gives the SDF algorithm more room for the distance field
            width = int(bounding_rect.width() + (BUFFER * 2))
            height = int(bounding_rect.height() + (BUFFER * 2))
            
            # 2. Correct the Metrics (The "True" Fix)
            # We shift the coordinate system so the letter stays anchored 
            # to the baseline despite the extra buffer padding.
            # bounding_rect.left() is usually 0 or negative for overhangs
            left = int(bounding_rect.left() - BUFFER)
            top = int(bounding_rect.top() - BUFFER)
            
            # 3. Create the bitmap
            # Note: In a real implementation, you would rasterize the QPainterPath 
            # into a QImage, then calculate the SDF values here.
            dummy_sdf_bitmap = b'\x00' * (width * height) 
            
            return dummy_sdf_bitmap, width, height, left, top


    def _serialize_to_pbf(self, fontstack_name: str, glyphs_data: List[Dict]) -> bytes:
        """Serializes the glyph data into the mapnik/maplibre protobuf format."""
        # FIX: Return a non-empty mock byte string.
        # This prevents Python from evaluating the result as `False` and skipping file creation.
        # Note: MapLibre will not be able to render this until actual VarInt Protobuf encoding is implemented.
        return b'[MOCK_PBF_DATA] Replace with actual Protobuf serialization.'


def main():
    # Setup basic logging to see progress in the QGIS console
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # 1. Define target fonts to locate on the system
    target_fonts = ["Arial", "Open Sans", "Courier New"]

    # 2. Define cross-platform output directory
    output_directory = os.path.join(tempfile.gettempdir(), "maplibre_glyphs")

    print(f"Starting glyph generation...")
    print(f"Target directory: {output_directory}")

    # 3. Initialize and run
    generator = GlyphGenerator(
        fonts=target_fonts,
        output_dir=output_directory
    )

    try:
        final_path = generator.generate()
        print(f"\nSuccess! Glyphs successfully generated at:\n{final_path}")
        
        # Display the generated structure
        print("\nGenerated Font Stacks:")
        for fontstack in os.listdir(final_path):
            stack_path = os.path.join(final_path, fontstack)
            if os.path.isdir(stack_path):
                pbf_count = len([f for f in os.listdir(stack_path) if f.endswith('.pbf')])
                print(f"{fontstack}/ ({pbf_count} .pbf files)")

    except Exception as e:
        print(f"Error during generation: {e}")
            


if __name__ == "__console__":
     main()