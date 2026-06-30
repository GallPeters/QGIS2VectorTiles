import os
import math
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set

import numpy as np
from scipy.ndimage import distance_transform_edt

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

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

from ..utils.config import (
    _GLYPH_RANGE_SIZE,
    _MAX_UNICODE,
    _MAPLIBRE_GLYPH_BORDER,
    _REFERENCE_EM,
    _REFERENCE_BUFFER,
    _REFERENCE_RADIUS,
    _SDF_COVERAGE_THRESHOLD,
    _FONT_RENDER_SIZE,
    _SDF_CUTOFF,
    _SUPERSAMPLE,
    _SDF_RADIUS,
    _BUFFER,
)

logger = logging.getLogger(__name__)


def _edt(mask: np.ndarray) -> np.ndarray:
    """Distance from each True pixel to the nearest False pixel.
    DIST_MASK_PRECISE is required for true circular (not octagonal) iso-
    distance contours — an approximate mask mode produces visibly square
    halos at larger radii. Falls back to scipy if cv2 is unavailable."""
    if _HAS_CV2:
        src = (mask.astype(np.uint8)) * 255
        return cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return distance_transform_edt(mask)


class GlyphGenerator:
    """
    Generates MapLibre-compatible SDF glyphs (PBF format) from system fonts,
    scoped to only the characters that actually appear in a given field
    across one or more datasets per font (read via GDAL/OGR — works with
    Parquet, GeoPackage, or anything else OGR can open).

    Generation parameters are fixed (see ..utils.config) and not
    configurable per call, to keep output predictable and avoid
    accidentally regenerating with mismatched settings.
    """

    def __init__(
        self,
        fonts_datasets: Dict[str, List[str]],
        field_name: str,
        output_dir: str,
    ):
        """
        fonts_datasets: dict mapping a combined "Family + Style" string
            (e.g. "Open Sans Extra Bold", "David Bold") to a list of
            dataset paths (Parquet, GeoPackage, etc. — anything OGR can
            open) whose `field_name` field contains text that must be
            renderable in that font.
        field_name: the attribute field to scan for characters, in every
            dataset, for every font.
        """
        self.fonts_datasets = fonts_datasets
        self.field_name = field_name
        self.output_dir = Path(output_dir)

        self.font_render_size = _FONT_RENDER_SIZE
        self.sdf_cutoff = _SDF_CUTOFF
        self.supersample = _SUPERSAMPLE
        self.sdf_radius = _SDF_RADIUS
        self.buffer = _BUFFER

        if self.buffer < _MAPLIBRE_GLYPH_BORDER:
            logger.warning(
                "buffer=%d < MapLibre's fixed %dpx border; narrow glyphs may "
                "get invalid (negative) width/height metadata.",
                self.buffer, _MAPLIBRE_GLYPH_BORDER
            )
        if self.buffer < self.sdf_radius:
            logger.warning(
                "buffer=%d < sdf_radius=%.2f. Edges near the glyph border "
                "may show flat/clipped artifacts.",
                self.buffer, self.sdf_radius
            )
        if not _HAS_CV2:
            logger.info("cv2 not found — using slower scipy distance transform "
                        "and numpy downsampling. Install opencv-python-headless "
                        "for a significant speed boost.")

        self._app = QCoreApplication.instance()
        if not self._app:
            self._app = QGuiApplication([])

    # ------------------------------------------------------------------
    # Font key resolution: "Open Sans Extra Bold" -> ("Open Sans", "Extra Bold")
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_font_key(font_key: str, available_families: List[str]) -> Optional[Tuple[str, str]]:
        """
        Resolves a combined 'Family + Style' string into a (family, style)
        tuple matching what QFontDatabase actually has installed, via
        longest-prefix family matching validated against that family's own
        real style list (so 'Open Sans Extra Bold' isn't mis-split by a
        shorter family name like 'Open').
        """
        key = font_key.strip()
        for family in sorted(available_families, key=len, reverse=True):
            if not key.lower().startswith(family.lower()):
                continue
            remainder = key[len(family):].strip()
            styles = QFontDatabase.styles(family)
            if remainder == "":
                default_style = next((s for s in styles if s.lower() in ("regular", "normal")), None)
                return family, default_style or (styles[0] if styles else "Regular")
            for style in styles:
                if remainder.lower() == style.lower():
                    return family, style
        return None

    # ------------------------------------------------------------------
    # Character discovery via GDAL/OGR
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_unique_chars(dataset_paths: List[str], field_name: str) -> Set[str]:
        """
        Reads each dataset via GDAL/OGR (osgeo) — not pyqgis vector layers,
        lower overhead for a read-only, single-field scan — and accumulates
        the set of unique characters found in `field_name` across all of
        them. Uses column-projection push-down (ExecuteSQL selecting only
        the needed field) where the driver supports it. Works with
        Parquet, GeoPackage, or any other OGR-readable format.
        """
        try:
            from osgeo import ogr, gdal
            gdal.UseExceptions()
        except ImportError:
            logger.error("GDAL/OGR Python bindings (osgeo) not available in this environment.")
            return set()

        unique_chars: Set[str] = set()

        for path in dataset_paths:
            ds = None
            result_layer = None
            try:
                ds = ogr.Open(path)
                if ds is None:
                    logger.warning(f"Could not open dataset '{path}' with GDAL/OGR. Skipping.")
                    continue

                base_layer = ds.GetLayer(0)
                if base_layer is None:
                    logger.warning(f"No layers found in dataset '{path}'. Skipping.")
                    continue

                if base_layer.GetLayerDefn().GetFieldIndex(field_name) < 0:
                    logger.warning(f"Field '{field_name}' not found in dataset '{path}'. Skipping.")
                    continue

                layer_name = base_layer.GetName()
                read_layer = base_layer
                try:
                    projected = ds.ExecuteSQL(f'SELECT "{field_name}" FROM "{layer_name}"')
                    if projected is not None:
                        result_layer = projected
                        read_layer = projected
                except Exception:
                    read_layer = base_layer

                read_layer.ResetReading()
                field_idx = read_layer.GetLayerDefn().GetFieldIndex(field_name)
                feat = read_layer.GetNextFeature()
                while feat is not None:
                    if field_idx >= 0 and feat.IsFieldSetAndNotNull(field_idx):
                        val = feat.GetFieldAsString(field_idx)
                        if val:
                            unique_chars.update(val)
                    feat = read_layer.GetNextFeature()

            except Exception as e:
                logger.warning(f"Error reading dataset '{path}': {e}")
            finally:
                if result_layer is not None and ds is not None:
                    try:
                        ds.ReleaseResultSet(result_layer)
                    except Exception:
                        pass

        return unique_chars

    # ------------------------------------------------------------------
    # Main generation flow
    # ------------------------------------------------------------------
    def generate(self) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        available_families = QFontDatabase.families()

        for font_key, dataset_paths in self.fonts_datasets.items():
            resolved = self._resolve_font_key(font_key, available_families)
            if not resolved:
                logger.warning(f"Could not resolve font '{font_key}' to an installed family+style. Skipping.")
                continue
            family, style = resolved
            logger.info(f"Processing font '{font_key}' -> family='{family}', style='{style}'")

            needed_chars = self._extract_unique_chars(dataset_paths, self.field_name)
            if not needed_chars:
                logger.warning(f"No characters found in field '{self.field_name}' for font '{font_key}'. Skipping.")
                continue

            codepoints = sorted({ord(c) for c in needed_chars if ord(c) <= _MAX_UNICODE})
            logger.info(
                f"Font '{font_key}': {len(codepoints)} unique codepoints required "
                f"across {len(dataset_paths)} dataset(s)."
            )

            style_name = "Regular" if style.lower() in ("normal", "regular") or not style else style
            fontstack_name = f"{family} {style_name}".strip()
            fontstack_dir = self.output_dir / fontstack_name
            fontstack_dir.mkdir(exist_ok=True)

            qfont = QFontDatabase.font(family, style, self.font_render_size)
            hi_font = QFont(qfont)
            hi_font.setPointSizeF(qfont.pointSizeF() * self.supersample)
            renderer = _GlyphRenderer(self, qfont, hi_font)

            self._generate_blocks_for_fontstack(renderer, fontstack_name, fontstack_dir, codepoints)

        return str(self.output_dir)

    def _generate_blocks_for_fontstack(self, renderer: "_GlyphRenderer", fontstack_name: str,
                                        out_dir: Path, codepoints: List[int]) -> None:
        """Groups required codepoints into MapLibre's 256-wide range blocks
        and generates a PBF only for blocks that actually contain a needed
        codepoint, containing only those codepoints (not the whole block)."""
        blocks: Dict[int, List[int]] = {}
        for cp in codepoints:
            block_start = (cp // _GLYPH_RANGE_SIZE) * _GLYPH_RANGE_SIZE
            blocks.setdefault(block_start, []).append(cp)

        for block_start, cps_in_block in sorted(blocks.items()):
            block_end = block_start + _GLYPH_RANGE_SIZE - 1
            pbf_data = self._create_sdf_pbf_for_codepoints(renderer, fontstack_name, cps_in_block)
            if pbf_data:
                pbf_path = out_dir / f"{block_start}-{block_end}.pbf"
                with open(pbf_path, "wb") as f:
                    f.write(pbf_data)

    def _create_sdf_pbf_for_codepoints(self, renderer: "_GlyphRenderer", fontstack_name: str,
                                        codepoints: List[int]) -> Optional[bytes]:
        metrics = renderer.metrics
        glyphs_data = []

        for char_code in codepoints:
            if not metrics.inFontUcs4(char_code):
                continue

            char_str = chr(char_code)
            advance = metrics.horizontalAdvance(char_str)
            rect = metrics.boundingRect(char_str)

            sdf_bitmap, width, height, left, top = renderer.render(char_str, rect)

            glyphs_data.append({
                "id": char_code, "bitmap": sdf_bitmap, "width": width,
                "height": height, "left": left, "top": top, "advance": advance
            })

        if not glyphs_data:
            return None
        return self._serialize_to_pbf(fontstack_name, glyphs_data)

    # ------------------------------------------------------------------
    # PBF encoding (protobuf, hand-rolled — MapLibre glyph spec)
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_varint(value: int) -> bytes:
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

    def _serialize_to_pbf(self, fontstack_name: str, glyphs_data: List[Dict]) -> bytes:
        fontstack_bytes = bytearray()
        fontstack_bytes += self._encode_string(1, fontstack_name)
        fontstack_bytes += self._encode_string(2, "0-255")

        for glyph in glyphs_data:
            glyph_bytes = bytearray()
            glyph_bytes += self._encode_uint(1, glyph["id"])
            if glyph.get("bitmap"):
                glyph_bytes += self._encode_bytes(2, glyph["bitmap"])
            glyph_bytes += self._encode_uint(3, glyph["width"])
            glyph_bytes += self._encode_uint(4, glyph["height"])
            glyph_bytes += self._encode_sint(5, glyph["left"])
            glyph_bytes += self._encode_sint(6, glyph["top"])
            glyph_bytes += self._encode_uint(7, glyph["advance"])
            fontstack_bytes += self._encode_message(3, bytes(glyph_bytes))

        return self._encode_message(1, bytes(fontstack_bytes))


class _GlyphRenderer:
    """
    Holds a single reused QImage scratch canvas + QPainter for an entire
    fontstack, so per-glyph rendering only does a cheap fill(0) instead of
    allocating a new QImage/QPainter per glyph. Falls back to a one-off
    allocation for the rare glyph too large for the scratch canvas.
    """

    def __init__(self, gen: GlyphGenerator, font: QFont, hi_font: QFont):
        self.gen = gen
        self.font = font
        self.hi_font = hi_font
        self.metrics = QFontMetrics(font)
        self.ss = gen.supersample
        self.buffer = gen.buffer

        fm = self.metrics
        max_w = int(fm.maxWidth() * 1.5) + self.buffer * 2
        max_h = int((fm.ascent() + fm.descent()) * 1.5) + self.buffer * 2
        self.scratch_w = max(max_w * self.ss, 16)
        self.scratch_h = max(max_h * self.ss, 16)

        self.scratch_image = QImage(self.scratch_w, self.scratch_h, QImage.Format.Format_Grayscale8)
        self.painter = QPainter()

    def render(self, char_str: str, bounding_rect) -> Tuple[bytes, int, int, int, int]:
        render_buffer = self.buffer
        ss = self.ss
        glyph_width = int(bounding_rect.width())
        glyph_height = int(bounding_rect.height())

        if glyph_width == 0 or glyph_height == 0:
            return b'', 0, 0, 0, 0

        bitmap_width = glyph_width + (render_buffer * 2)
        bitmap_height = glyph_height + (render_buffer * 2)
        hi_w, hi_h = bitmap_width * ss, bitmap_height * ss

        if hi_w <= self.scratch_w and hi_h <= self.scratch_h:
            image = self.scratch_image
            image.fill(0)
        else:
            image = QImage(hi_w, hi_h, QImage.Format.Format_Grayscale8)
            image.fill(0)

        hi_path = QPainterPath()
        hi_path.addText(0, 0, self.hi_font, char_str)

        self.painter.begin(image)
        self.painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.painter.setBrush(QColor(255, 255, 255))
        self.painter.setPen(Qt.PenStyle.NoPen if hasattr(Qt, 'PenStyle') else Qt.NoPen)
        self.painter.translate((-bounding_rect.left() + render_buffer) * ss,
                                (-bounding_rect.top() + render_buffer) * ss)
        self.painter.drawPath(hi_path)
        self.painter.end()

        bytes_per_line = image.bytesPerLine()
        ptr = image.constBits()
        ptr.setsize(image.sizeInBytes())
        full = np.frombuffer(ptr, dtype=np.uint8).reshape(image.height(), bytes_per_line)
        coverage = full[:hi_h, :hi_w]

        inside_mask = coverage >= _SDF_COVERAGE_THRESHOLD

        if not inside_mask.any():
            logger.warning("No pixels above threshold (%dx%d); blank SDF.", glyph_width, glyph_height)
            sdf_hi = np.zeros((hi_h, hi_w), dtype=np.float32)
        else:
            outer = _edt(~inside_mask)
            inner = _edt(inside_mask)
            signed_distance = (outer - inner) / ss
            sdf_hi = np.clip(
                255.0 - 255.0 * (signed_distance / self.gen.sdf_radius + self.gen.sdf_cutoff),
                0, 255
            ).astype(np.float32)

        if ss > 1:
            if _HAS_CV2:
                sdf_bitmap = cv2.resize(sdf_hi, (bitmap_width, bitmap_height), interpolation=cv2.INTER_AREA)
                sdf_bitmap = np.clip(sdf_bitmap, 0, 255).astype(np.uint8)
            else:
                sdf_bitmap = sdf_hi.reshape(bitmap_height, ss, bitmap_width, ss).mean(axis=(1, 3)).astype(np.uint8)
        else:
            sdf_bitmap = sdf_hi.astype(np.uint8)

        width_field = max(0, bitmap_width - 2 * _MAPLIBRE_GLYPH_BORDER)
        height_field = max(0, bitmap_height - 2 * _MAPLIBRE_GLYPH_BORDER)

        return sdf_bitmap.tobytes(), width_field, height_field, int(bounding_rect.left()), int(bounding_rect.top())


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_directory = os.path.join(os.path.expanduser("~"), "maplibre_glyphs")

    # Maps each "Family + Style" font key to the datasets (Parquet,
    # GeoPackage, etc.) whose text field must be coverable by that font.
    fonts_datasets = {
        "Open Sans Extra Bold": [
            "/path/to/dataset_labels_a.gpkg",
            "/path/to/dataset_labels_b.gpkg",
        ],
        "David Bold": [
            "/path/to/dataset_hebrew_labels.gpkg",
        ],
    }

    field_name = "label"

    print("Starting glyph generation...")
    print(f"Target directory: {output_directory}")

    # Generation parameters are fixed (see ..utils.config) — no quality
    # presets to choose from.
    generator = GlyphGenerator(
        fonts_datasets=fonts_datasets,
        field_name=field_name,
        output_dir=output_directory,
    )

    try:
        final_path = generator.generate()
        print(f"\nSuccess! Glyphs generated at:\n{final_path}")
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