import os
import math
import hashlib
import logging
import base64
import json
import tempfile
import time
import unicodedata
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
    qfontdb = QFontDatabase()
else:
    from PyQt6.QtGui import (
        QGuiApplication, QFontDatabase, QFont, QFontMetrics,
        QPainterPath, QImage, QPainter, QColor
    )
    from PyQt6.QtCore import QCoreApplication, Qt
    qfontdb = QFontDatabase
    

from ..utils.config import (
    _CACHE_DIR_NAME,
    _CACHE_MAX_AGE_DAYS,
    _CACHE_MAX_MB,
    _GLYPH_RANGE_SIZE,
    _MAX_UNICODE,
    _MAPLIBRE_GLYPH_BORDER,
    _SDF_COVERAGE_THRESHOLD,
    _FONT_RENDER_SIZE,
    _SDF_CUTOFF,
    _SUPERSAMPLE,
    _SDF_RADIUS,
    _BUFFER,
)

logger = logging.getLogger(__name__)

_SHAPING_OFFSET = 17  # MapLibre's SHAPING_DEFAULT_OFFSET (-17), in px of a 24 px em

# Arabic script blocks. MapLibre's RTL text plugin joins Arabic letters by
# replacing them with their presentation forms, whose glyphs are then
# requested: text in these blocks needs those forms too.
_ARABIC_BLOCKS = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF))
_ARABIC_PRESENTATION_FORMS = ((0xFB50, 0xFDFF), (0xFE70, 0xFEFF))

# Fonts tried first for characters the label's font lacks (then any font
# that has them), as Qt falls back to other fonts when QGIS draws a label.
_FALLBACK_FAMILIES = (
    "Arial", "Segoe UI", "Tahoma", "Noto Sans", "DejaVu Sans", "Liberation Sans",
    "Segoe UI Symbol", "Arial Unicode MS",
)

# Bump when glyph rendering changes, so glyphs cached the old way are not reused.
_GLYPH_CACHE_VERSION = "1"


def _with_presentation_forms(codepoints: Set[int]) -> Set[int]:
    """The codepoints plus the Arabic presentation forms of their letters:
    each form (initial, medial, final, isolated, ligature) whose Unicode
    decomposition uses only letters the text has."""
    if not any(lo <= cp <= hi for cp in codepoints for lo, hi in _ARABIC_BLOCKS):
        return codepoints
    forms = set()
    for lo, hi in _ARABIC_PRESENTATION_FORMS:
        for cp in range(lo, hi + 1):
            parts = unicodedata.decomposition(chr(cp)).split()
            letters = [int(p, 16) for p in parts if not p.startswith("<")]
            if letters and all(letter in codepoints for letter in letters):
                forms.add(cp)
    return codepoints | forms


class _GlyphCache:
    """A fontstack's rendered glyphs, kept between runs in the system temp
    folder (keyed by the font and the generation parameters).

    Stored as JSON (bitmaps as base64), never pickled: reading the cache
    can't run code, whatever the file holds.
    """

    # The fields of a glyph and their types; entries not matching are dropped.
    _FIELDS = {"id": int, "width": int, "height": int, "left": int, "top": int, "advance": int}

    def __init__(self, family: str, style: str):
        self.enabled = _CACHE_MAX_MB > 0
        directory = os.path.join(tempfile.gettempdir(), _CACHE_DIR_NAME, "glyphs")
        key = "|".join(str(v) for v in (
            _GLYPH_CACHE_VERSION, family, style, _FONT_RENDER_SIZE, _SDF_CUTOFF, _SUPERSAMPLE,
            _SDF_RADIUS, _BUFFER, _SDF_COVERAGE_THRESHOLD, _MAPLIBRE_GLYPH_BORDER,
        ))
        digest = hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()
        self.path = os.path.join(directory, digest + ".json")
        self.glyphs: Dict[int, Optional[dict]] = {}
        self._dirty = False
        if not self.enabled:
            return
        try:
            os.makedirs(directory, exist_ok=True)
            self._prune(directory)
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self.glyphs = {
                    int(codepoint): self._decode(glyph) for codepoint, glyph in loaded.items()
                    if glyph is None or self._decode(glyph) is not None
                }
        except (OSError, ValueError, TypeError):
            self.glyphs = {}

    @classmethod
    def _decode(cls, glyph) -> Optional[dict]:
        """A glyph read from JSON, or None if it isn't a valid glyph."""
        if not isinstance(glyph, dict) or not isinstance(glyph.get("bitmap"), str):
            return None
        if not all(isinstance(glyph.get(k), t) for k, t in cls._FIELDS.items()):
            return None
        try:
            bitmap = base64.b64decode(glyph["bitmap"], validate=True)
        except ValueError:
            return None
        return {**{k: glyph[k] for k in cls._FIELDS}, "bitmap": bitmap}

    @classmethod
    def _encode(cls, glyph: Optional[dict]) -> Optional[dict]:
        if glyph is None:
            return None
        return {**{k: int(glyph[k]) for k in cls._FIELDS},
                "bitmap": base64.b64encode(glyph["bitmap"]).decode("ascii")}

    def put(self, codepoint: int, glyph: Optional[dict]) -> None:
        self.glyphs[codepoint] = glyph
        self._dirty = True

    def save(self) -> None:
        if not self.enabled or not self._dirty:
            return
        temp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(temp, "w", encoding="utf-8") as f:
                json.dump({str(cp): self._encode(g) for cp, g in self.glyphs.items()}, f,
                          separators=(",", ":"))
            os.replace(temp, self.path)
        except (OSError, TypeError, ValueError, KeyError):
            try:
                os.remove(temp)
            except OSError:
                pass

    @staticmethod
    def _prune(directory: str) -> None:
        limit = time.time() - _CACHE_MAX_AGE_DAYS * 86400
        for entry in os.scandir(directory):
            try:
                if entry.is_file() and entry.stat().st_mtime < limit:
                    os.remove(entry.path)
            except OSError:
                continue


def _edt(mask: np.ndarray) -> np.ndarray:
    """Distance from each True pixel to the nearest False pixel.
    DIST_MASK_PRECISE is required for true circular (not octagonal) iso-
    distance contours — an approximate mask mode produces visibly square
    halos at larger radii. Falls back to scipy if cv2 is unavailable."""
    if _HAS_CV2:
        src = (mask.astype(np.uint8)) * 255
        return cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return distance_transform_edt(mask)


# A fonts_datasets "dataset" of this prefix, with no field, gives its
# characters themselves (text the tiles hold but no dataset does, e.g. the
# counts of point clusters).
CHARS_PREFIX = "chars:"


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
        fonts_datasets: Dict[str, List[Tuple[str, str]]],
        output_dir: str,
    ):
        """
        fonts_datasets: dict mapping a combined "Family + Style" string
            (e.g. "Open Sans Extra Bold", "David Bold") to a list of
            (dataset path, field name) pairs (anything OGR can open) whose
            field contains text that must be renderable in that font.
        """
        self.fonts_datasets = fonts_datasets
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
        tuple matching what qfontdb actually has installed, via
        longest-prefix family matching validated against that family's own
        real style list (so 'Open Sans Extra Bold' isn't mis-split by a
        shorter family name like 'Open').
        """
        key = font_key.strip()
        for family in sorted(available_families, key=len, reverse=True):
            if not key.lower().startswith(family.lower()):
                continue
            remainder = key[len(family):].strip()
            styles = qfontdb.styles(family)
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
    def _extract_unique_chars(datasets: List[Tuple[str, str]]) -> Set[str]:
        """
        Reads each (dataset, field) via GDAL/OGR (osgeo) — not pyqgis vector
        layers, lower overhead for a read-only, single-field scan — and
        accumulates the set of unique characters found in the fields.
        
        Uses SetIgnoredFields to achieve column-projection push-down and bypass
        geometry parsing entirely. Works with Parquet, GeoPackage, or any other
        OGR-readable format.
        """
        try:
            from osgeo import ogr, gdal
            gdal.UseExceptions()
        except ImportError:
            # logger.error("GDAL/OGR Python bindings (osgeo) not available in this environment.")
            return set()

        unique_chars: Set[str] = set()

        for path, field_name in datasets:
            if field_name is None:  # Characters given as they are (see CHARS_PREFIX).
                if path.startswith(CHARS_PREFIX):
                    unique_chars.update(path[len(CHARS_PREFIX):])
                continue
            ds = None
            try:
                ds = ogr.Open(path)
                if ds is None:
                    # logger.warning(f"Could not open dataset '{path}' with GDAL/OGR. Skipping.")
                    continue

                base_layer = ds.GetLayer(0)
                if base_layer is None:
                    # logger.warning(f"No layers found in dataset '{path}'. Skipping.")
                    continue

                layer_defn = base_layer.GetLayerDefn()
                field_idx = layer_defn.GetFieldIndex(field_name)
                
                if field_idx < 0:
                    # logger.warning(f"Field '{field_name}' not found in dataset '{path}'. Skipping.")
                    continue

                # --- BYPASS SECURITY SCANNER & OPTIMIZE ---
                # Build a list of all fields to ignore, plus the geometry
                ignored_fields = ["OGR_GEOMETRY"]
                for i in range(layer_defn.GetFieldCount()):
                    if i != field_idx:
                        ignored_fields.append(layer_defn.GetFieldDefn(i).GetName())

                # Push the projection down to the driver (identical to SELECT field_name)
                base_layer.SetIgnoredFields(ignored_fields)

                base_layer.ResetReading()
                feat = base_layer.GetNextFeature()
                
                while feat is not None:
                    if feat.IsFieldSetAndNotNull(field_idx):
                        val = feat.GetFieldAsString(field_idx)
                        if val:
                            unique_chars.update(val)
                    feat = base_layer.GetNextFeature()

            except Exception as e:
                # logger.warning(f"Error reading dataset '{path}': {e}")
                pass
            finally:
                # Cleanup the dataset reference
                ds = None

        return unique_chars

    # ------------------------------------------------------------------
    # Main generation flow
    # ------------------------------------------------------------------
    def generate(self) -> Dict[str, str]:
        """Write the glyphs; returns each generated font key's fontstack name.

        The fontstack name is the folder the glyphs are in, which the style's
        text-font must use: it names the font found (e.g. "Arial Regular" for
        a key without a style), not the key as given.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        available_families = qfontdb.families()
        fontstacks: Dict[str, str] = {}

        # Font keys naming one font (e.g. "Open Sans Regular" and "Open Sans "
        # for a font without a style name) share its fontstack folder: their
        # labels' characters are gathered and its glyphs written once, so no
        # key's range files overwrite another's.
        stacks: Dict[str, Tuple[str, str, List[Tuple[str, str]], List[str]]] = {}
        for font_key, dataset_paths in self.fonts_datasets.items():
            resolved = self._resolve_font_key(font_key, available_families)
            if not resolved:
                logger.warning(f"Could not resolve font '{font_key}' to an installed family+style. Skipping.")
                continue
            family, style = resolved
            style_name = "Regular" if style.lower() in ("normal", "regular") or not style else style
            fontstack_name = f"{family} {style_name}".strip()
            entry = stacks.setdefault(fontstack_name, (family, style, [], []))
            entry[2].extend(d for d in dataset_paths if d not in entry[2])
            entry[3].append(font_key)

        for fontstack_name, (family, style, dataset_paths, font_keys) in stacks.items():
            logger.info(f"Processing fontstack '{fontstack_name}' for {font_keys}")

            needed_chars = self._extract_unique_chars(dataset_paths)
            if not needed_chars:
                logger.warning(f"No label characters found for fontstack '{fontstack_name}'. Skipping.")
                continue

            codepoints = sorted(_with_presentation_forms(
                {ord(c) for c in needed_chars if ord(c) <= _MAX_UNICODE}
            ))
            logger.info(
                f"Fontstack '{fontstack_name}': {len(codepoints)} unique codepoints required "
                f"across {len(dataset_paths)} dataset field(s)."
            )

            fontstack_dir = self.output_dir / fontstack_name
            fontstack_dir.mkdir(exist_ok=True)

            # MapLibre scales glyphs as if drawn with a 24 px em: a pixel
            # size, not a point size (24 pt is 32 px).
            qfont = qfontdb.font(family, style, self.font_render_size)
            qfont.setPixelSize(self.font_render_size)
            hi_font = QFont(qfont)
            hi_font.setPixelSize(self.font_render_size * self.supersample)
            renderer = _GlyphRenderer(self, qfont, hi_font)
            self._cache = _GlyphCache(family, style)
            self._fallbacks: Dict[int, Optional["_GlyphRenderer"]] = {}
            self._fallback_renderers: Dict[str, "_GlyphRenderer"] = {}
            self._style = style

            self._generate_blocks_for_fontstack(renderer, fontstack_name, fontstack_dir, codepoints)
            self._cache.save()
            for font_key in font_keys:
                fontstacks[font_key] = fontstack_name

        return fontstacks

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
        glyphs_data = []

        for char_code in codepoints:
            if char_code in self._cache.glyphs:
                glyph = self._cache.glyphs[char_code]
            else:
                glyph = self._render_glyph(renderer, char_code)
                self._cache.put(char_code, glyph)
            if glyph is not None:
                glyphs_data.append(glyph)

        if not glyphs_data:
            return None
        return self._serialize_to_pbf(fontstack_name, glyphs_data)

    def _render_glyph(self, renderer: "_GlyphRenderer", char_code: int) -> Optional[dict]:
        """One glyph, from the label's font or, if it lacks the character,
        from a fallback font; None if no font has it."""
        if not renderer.metrics.inFontUcs4(char_code):
            renderer = self._fallback_for(char_code, renderer)
            if renderer is None:
                return None
        metrics = renderer.metrics
        char_str = chr(char_code)
        advance = metrics.horizontalAdvance(char_str)
        rect = metrics.boundingRect(char_str)
        sdf_bitmap, width, height, left, top = renderer.render(char_str, rect)
        return {
            "id": char_code, "bitmap": sdf_bitmap, "width": width,
            "height": height, "left": left, "top": top, "advance": advance
        }

    def _fallback_for(self, char_code: int, primary: "_GlyphRenderer") -> Optional["_GlyphRenderer"]:
        """A renderer of a font that has the character (preferred families
        first), drawn on the label font's baseline; None if none has it."""
        if char_code in self._fallbacks:
            return self._fallbacks[char_code]
        families = list(_FALLBACK_FAMILIES) + [
            f for f in qfontdb.families() if f not in _FALLBACK_FAMILIES
        ]
        found = None
        for family in families:
            fallback = self._fallback_renderer(family, primary)
            if fallback is not None and fallback.metrics.inFontUcs4(char_code):
                found = fallback
                break
        self._fallbacks[char_code] = found
        return found

    def _fallback_renderer(self, family: str, primary: "_GlyphRenderer") -> Optional["_GlyphRenderer"]:
        if family in self._fallback_renderers:
            return self._fallback_renderers[family]
        renderer = None
        try:
            styles = qfontdb.styles(family)
            if styles:
                style = self._style if self._style in styles else next(
                    (s for s in styles if s.lower() in ("regular", "normal")), styles[0])
                qfont = qfontdb.font(family, style, self.font_render_size)
                qfont.setPixelSize(self.font_render_size)
                hi_font = QFont(qfont)
                hi_font.setPixelSize(self.font_render_size * self.supersample)
                renderer = _GlyphRenderer(self, qfont, hi_font)
                # On the label font's baseline, as Qt draws fallback glyphs.
                renderer.top_shift = primary.top_shift
        except (RuntimeError, TypeError, AttributeError):
            renderer = None
        self._fallback_renderers[family] = renderer
        return renderer

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

        # MapLibre puts a glyph's top at -17 - top in its 24 px line box, which
        # is centred on the anchor. This top centres the font's ascent and
        # descent there, as QGIS centres a label's text box on its point.
        self.top_shift = _SHAPING_OFFSET + (fm.ascent() - fm.descent()) / 2

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

        # MapLibre places the bitmap assuming a 3 px border around the glyph;
        # the extra border moves left/top back so the glyph lands in place.
        extra_border = render_buffer - _MAPLIBRE_GLYPH_BORDER
        left = int(bounding_rect.left()) - extra_border
        top = round(-bounding_rect.top() - self.top_shift) + extra_border
        return sdf_bitmap.tobytes(), width_field, height_field, left, top


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_directory = os.path.join(os.path.expanduser("~"), "maplibre_glyphs")

    # Maps each "Family + Style" font key to the (dataset, field) pairs
    # (GeoPackage, FlatGeobuf, etc.) whose text must be coverable by that font.
    fonts_datasets = {
        "Open Sans Extra Bold": [
            ("/path/to/dataset_labels_a.gpkg", "label"),
            ("/path/to/dataset_labels_b.gpkg", "label"),
        ],
        "David Bold": [
            ("/path/to/dataset_hebrew_labels.gpkg", "label"),
        ],
    }

    print("Starting glyph generation...")
    print(f"Target directory: {output_directory}")

    # Generation parameters are fixed (see ..utils.config) — no quality
    # presets to choose from.
    generator = GlyphGenerator(
        fonts_datasets=fonts_datasets,
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