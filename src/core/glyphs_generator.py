import os
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from qgis.PyQt.QtCore import qVersion
# PyQt version guard — import the right Qt5 / Qt6 symbols once, re-export

from qgis.PyQt.QtCore import qVersion

if int(qVersion()[0]) == 5:
    from PyQt5.QtGui import (
        QGuiApplication, QFontDatabase, QFont, QFontMetrics, 
        QPainterPath, QImage, QPainter, QColor
    )
    from PyQt5.QtCore import QCoreApplication, Qt
else:
    from PyQt6.QtGui import (
        QGuiApplication, QFontDatabase, QFont, QFontMetrics, 
        QPainterPath, QImage, QPainter, QColor
    )
    from PyQt6.QtCore import QCoreApplication, Qt

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

    @staticmethod
    def _encode_varint(value: int) -> bytes:
        """Encode an integer as a protobuf varint."""
        out = bytearray()
        while True:
            byte = value & 0x7f
            value >>= 7
            if value:
                out.append(byte | 0x80)
            else:
                out.append(byte)
                break
        return bytes(out)

    @staticmethod
    def _encode_svarint(value: int) -> bytes:
        """ZigZag encode a signed integer (used for negative offsets)."""
        zigzag = (value << 1) ^ (value >> 31)
        return GlyphGenerator._encode_varint(zigzag)

    @staticmethod
    def _encode_tag(field_number: int, wire_type: int) -> bytes:
        return GlyphGenerator._encode_varint((field_number << 3) | wire_type)

    @staticmethod
    def _encode_string(field_number: int, value: str) -> bytes:
        encoded = value.encode('utf-8')
        return GlyphGenerator._encode_tag(field_number, 2) + GlyphGenerator._encode_varint(len(encoded)) + encoded

    @staticmethod
    def _encode_bytes(field_number: int, value: bytes) -> bytes:
        return GlyphGenerator._encode_tag(field_number, 2) + GlyphGenerator._encode_varint(len(value)) + value

    @staticmethod
    def _encode_uint(field_number: int, value: int) -> bytes:
        return GlyphGenerator._encode_tag(field_number, 0) + GlyphGenerator._encode_varint(value)

    @staticmethod
    def _encode_sint(field_number: int, value: int) -> bytes:
        return GlyphGenerator._encode_tag(field_number, 0) + GlyphGenerator._encode_svarint(value)

    @staticmethod
    def _encode_message(field_number: int, message_bytes: bytes) -> bytes:
        return GlyphGenerator._encode_tag(field_number, 2) + GlyphGenerator._encode_varint(len(message_bytes)) + message_bytes
    
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
            BUFFER = 3
            glyph_width = int(bounding_rect.width())
            glyph_height = int(bounding_rect.height())

            if glyph_width == 0 or glyph_height == 0:
                return b'', 0, 0, 0, 0

            bitmap_width = glyph_width + (BUFFER * 2)
            bitmap_height = glyph_height + (BUFFER * 2)
            
            # 1. Create the image
            image = QImage(bitmap_width, bitmap_height, QImage.Format.Format_Grayscale8)
            image.fill(0)
            
            # 2. Draw
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QColor(255, 255, 255))
            painter.setPen(Qt.PenStyle.NoPen if hasattr(Qt, 'PenStyle') else Qt.NoPen)
            
            # Adjust for buffer
            painter.translate(-bounding_rect.left() + BUFFER, -bounding_rect.top() + BUFFER)
            painter.drawPath(path)
            painter.end()

            # 3. FIX: Strip padding bytes row-by-row
            # We need a flat array of exactly bitmap_width * bitmap_height
            raw_data = bytearray()
            bytes_per_line = image.bytesPerLine()
            
            # Access the raw bits
            ptr = image.constBits()
            ptr.setsize(image.sizeInBytes())
            
            # Copy only the valid pixels, skipping padding
            for y in range(bitmap_height):
                # Calculate the start of the row in the QImage buffer
                row_start = y * bytes_per_line
                # Slice exactly the width we need
                row_data = ptr[row_start : row_start + bitmap_width]
                raw_data.extend(row_data)

            return bytes(raw_data), glyph_width, glyph_height, int(bounding_rect.left()), int(bounding_rect.top())

    def _serialize_to_pbf(self, fontstack_name: str, glyphs_data: List[Dict]) -> bytes:
            """Serializes the glyph data into the MapLibre protobuf format."""
            fontstack_bytes = bytearray()
            
            # Field 1: Fontstack Name
            fontstack_bytes += self._encode_string(1, fontstack_name)
            # Field 2: Range (Optional but good practice)
            fontstack_bytes += self._encode_string(2, "0-255") 
            
            for glyph in glyphs_data:
                glyph_bytes = bytearray()
                
                # MapLibre expects these exact field indices
                glyph_bytes += self._encode_uint(1, glyph["id"])
                if glyph.get("bitmap"):
                    glyph_bytes += self._encode_bytes(2, glyph["bitmap"])
                glyph_bytes += self._encode_uint(3, glyph["width"])
                glyph_bytes += self._encode_uint(4, glyph["height"])
                glyph_bytes += self._encode_sint(5, glyph["left"])
                glyph_bytes += self._encode_sint(6, glyph["top"])
                glyph_bytes += self._encode_uint(7, glyph["advance"])
                
                # Field 3 in Fontstack is the repeated Glyph message
                fontstack_bytes += self._encode_message(3, bytes(glyph_bytes))
                
            # Field 1 in the Root message is the repeated Fontstack message
            return self._encode_message(1, bytes(fontstack_bytes))


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
