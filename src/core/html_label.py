"""
html_label.py

Labels with HTML formatting ("Allow HTML formatting" in a text format):
parts of one label in their own font family, weight, size, colour and
background colour, in paragraphs.

QGIS reads the HTML with QgsTextDocument; so does the export, to the
same runs of text. A label's runs go into the tiles as slots, one per run
(RichFields): its text, and where it differs from the label's own format
its font (a font key), size (a scale of the label's) and colour. A run
with a background colour is drawn as an image, a "badge" of its text on
that colour (MapLibre has no background per part of a text): its slot
holds the badge's name, which describes it (badge_name), so that the web
map can draw a badge its sprite lacks (new data in a live map).

The label's own field keeps the plain text (search, glyphs, popups).

Depends on: config
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from qgis.core import QgsTextDocument, QgsTextFormat, QgsUnitTypes
from qgis.PyQt.QtGui import QColor, QFont

from ..utils.config import _FIELD_PREFIX

# Most runs a label's slots hold; beyond, the rest joins the last slot. A
# dataset has as many slots as its longest label needs (a sign of 85 bus
# lines, each on its own badge: about 190).
MAX_SLOTS = 256
# Pixels per point (96 DPI), as label sizes are converted.
PX_PER_PT = 96 / 72
BADGE_PREFIX = "q2vtb|"
# A 1 px transparent image filling a slot without a badge, in exports
# before v4.5 (the viewer still draws it); a slot without one is now empty.
EMPTY_IMAGE = "q2vt-none"


@dataclass(frozen=True)
class BaseFormat:
    """A label's own text format, as its runs are compared with."""
    family: str
    weight: int
    italic: bool
    size_pt: float
    color: str            # #RRGGBBAA

    @classmethod
    def of(cls, text_format: QgsTextFormat) -> "BaseFormat":
        font = text_format.font()
        size = _points(text_format.size(), text_format.sizeUnit())
        weight = _weight_int(font.weight())
        if text_format.forcedBold():
            weight = 700
        return cls(font.family(), weight, bool(font.italic() or text_format.forcedItalic()),
                   float(size or 10), _hex(text_format.color()))

    @property
    def font_key(self) -> str:
        return font_key(self.family, self.weight, self.italic)


@dataclass(frozen=True)
class Run:
    """A part of a label in one format: None where it is the label's own."""
    text: str
    font: Optional[str] = None
    scale: Optional[float] = None
    color: Optional[str] = None
    background: Optional[str] = None


def _points(size: float, unit) -> float:
    """A label size in points (map units, which have none, as points)."""
    name = str(getattr(unit, "name", unit)).lower()
    if "millimeter" in name:
        return size * 72 / 25.4
    if "pixel" in name:
        return size / PX_PER_PT
    if "inch" in name:
        return size * 72
    return size


def _weight_int(weight) -> int:
    value = int(getattr(weight, "value", weight))
    # Qt 5 weights (0-99) to CSS / Qt 6 ones (100-900).
    if value < 100:
        value = {25: 300, 50: 400, 57: 500, 63: 600, 75: 700, 81: 800, 87: 900}.get(value, 400)
    return value


def _hex(color: QColor) -> str:
    return f"#{color.red():02X}{color.green():02X}{color.blue():02X}{color.alpha():02X}"


def font_key(family: str, weight: int, italic: bool) -> str:
    """The installed style of ``family`` nearest a weight and slant, as the
    "Family Style" key labels' fonts are named by."""
    qfont = QFont(family)
    try:
        qfont.setWeight(QFont.Weight(weight))
    except (TypeError, ValueError):
        qfont.setBold(weight >= 600)
    qfont.setItalic(italic)
    from .glyphs_generator import qfontdb  # pylint: disable=import-outside-toplevel  (Qt 5 and 6)
    style = qfontdb.styleString(qfont)
    styles = qfontdb.styles(family)
    if styles and style not in styles:
        # The family's installed style nearest in weight, of the same slant
        # (Qt names 600 "Demi Bold" where a font names it "SemiBold").
        def distance(name: str) -> tuple:
            slanted = qfontdb.italic(family, name)
            return (slanted != italic, abs(_weight_int(qfontdb.weight(family, name)) - weight))
        style = min(styles, key=distance)
    return f"{family} {style}".strip()


def runs_of(html: str, base: BaseFormat, families: Optional[set] = None) -> List[Run]:
    """The runs of a label's HTML, as QGIS reads it; paragraphs end in a
    line break. Adjacent runs in one format are joined."""
    if html is None:
        return []
    document = QgsTextDocument.fromHtml([str(html)])
    runs: List[Run] = []
    for b in range(document.size()):
        if b:
            runs.append(Run("\n"))
        block = document.at(b)
        for i in range(block.size()):
            fragment = block.at(i)
            fmt = fragment.characterFormat()
            family = fmt.family() or base.family
            if families is not None:
                families.add(family)
            weight = fmt.fontWeight() if fmt.fontWeight() not in (-1, None) else base.weight
            italic_set = getattr(getattr(fmt.italic(), "name", ""), "lower", lambda: "")()
            italic = base.italic if "notset" in italic_set else "true" in italic_set
            key = font_key(family, int(weight), italic)
            size = fmt.fontPointSize() if fmt.fontPointSize() and fmt.fontPointSize() > 0 else None
            if size is None and fmt.fontPercentageSize() and fmt.fontPercentageSize() > 0:
                size = base.size_pt * fmt.fontPercentageSize()
            scale = round(size / base.size_pt, 4) if size and base.size_pt else None
            color = _hex(fmt.textColor()) if fmt.textColor().isValid() else None
            background = None
            if fmt.hasBackground() and fmt.backgroundBrush().color().isValid():
                background = _hex(fmt.backgroundBrush().color())
            runs.append(Run(
                fragment.text(),
                None if key == base.font_key else key,
                None if scale in (None, 1, 1.0) else scale,
                None if color in (None, base.color) else color,
                background,
            ))
    merged: List[Run] = []
    for run in runs:
        if merged and run.background is None and merged[-1].background is None and (
                merged[-1].font, merged[-1].scale, merged[-1].color) == (run.font, run.scale, run.color):
            merged[-1] = Run(merged[-1].text + run.text, run.font, run.scale, run.color)
        else:
            merged.append(run)
    return kept_edges([r for r in merged if r.text])


# White space a line starts or ends with, which the web map would trim
# (QGIS draws it): kept behind a word joiner, as plain labels' is.
_SPACES = " \u00a0\t"
_JOINER = "\u2060"


def kept_edges(runs: List[Run]) -> List[Run]:
    """``runs`` with a word joiner before the spaces starting a line and
    after those ending one (rules_exporter's own labels do the same)."""
    texts = [r.text for r in runs]
    at_start = True
    for k, text in enumerate(texts):
        out = []
        for ch in text:
            if ch == "\n":
                last = out[-1] if out else (texts[k - 1][-1] if k and texts[k - 1] else "")
                if last and last in _SPACES:
                    if out:
                        out.append(_JOINER)
                    else:
                        texts[k - 1] += _JOINER
                at_start = True
            elif at_start and ch in _SPACES:
                out.append(_JOINER)
                at_start = False
            else:
                at_start = False
            out.append(ch)
        texts[k] = "".join(out)
    if texts and texts[-1] and texts[-1][-1] in _SPACES:
        texts[-1] += _JOINER
    return [Run(t, r.font, r.scale, r.color, r.background) for t, r in zip(texts, runs)]


def plain_text(runs: List[Run]) -> str:
    return "".join(r.text for r in runs)


# MapLibre lays out a text two ways (shaping.ts, align): when every line's
# largest font-scale is 1, each line's glyphs centred in its line height;
# otherwise each line's glyphs 17 - line height / 2 (in 24ths of an em)
# lower. So that one text-translate (a layer's, see maplibre_converter)
# puts every label's glyphs where QGIS draws them, every HTML label takes
# the second way: one with no line larger than 1 gets an invisible
# character a hair larger (a zero width space), at the end of a line of
# scale 1 (its height 0.01% more).
MIXED_MARK = "​"
MIXED_SCALE = 1.0001


def line_scales(runs: List[Run]) -> List[Optional[float]]:
    """Each line's largest font-scale, as MapLibre takes it (a badge, an
    image, as 1); None for an empty line."""
    lines: List[Optional[float]] = [None]
    for run in runs:
        scale = 1.0 if run.background else (run.scale or 1.0)
        for ch in run.text:
            if ch == "\n":
                lines.append(None)
            else:
                lines[-1] = scale if lines[-1] is None else max(lines[-1], scale)
    return lines


def mixed_shaping(runs: List[Run]) -> List[Run]:
    """``runs``, laid out by MapLibre the way a label with sizes is: a
    MIXED_MARK added at the end of the first line of scale 1 where no line
    is larger."""
    scales = [s for s in line_scales(runs) if s is not None]
    if not scales or max(scales) != 1.0:
        return runs
    target = line_scales(runs).index(1.0)
    line = 0
    for k, run in enumerate(runs):
        for pos, ch in enumerate(run.text):
            if ch != "\n":
                continue
            if line == target:
                head, tail = run.text[:pos], run.text[pos:]
                mark = Run(MIXED_MARK, scale=MIXED_SCALE)
                parts = ([Run(head, run.font, run.scale, run.color, run.background)] if head else []) + [
                    mark, Run(tail, run.font, run.scale, run.color, run.background)]
                return runs[:k] + parts + runs[k + 1:]
            line += 1
    return runs + [Run(MIXED_MARK, scale=MIXED_SCALE)]


def mixed_shift(line_height: float) -> float:
    """How much lower than QGIS (in ems of the label's size) MapLibre draws
    the glyphs of a label laid out the mixed way (mixed_shaping): 17/24 em
    less half its line height (text-line-height, ems)."""
    return 17 / 24 - line_height / 2


def maplibre_box(runs: List[Run], line_height: float, baseline: float,
                 badge_height) -> Tuple[float, float]:
    """(first baseline, height) of the box MapLibre lays a badged label's
    ``runs`` out in (mixed_shaping), in ems of the label's size, its text
    aligned to the top of each line ("vertical-align": "top").

    MapLibre's line is its largest scale S tall (badges count as 1) times
    ``line_height``, plus how much its tallest badge is taller than S; a
    text of scale s has its baseline ``baseline`` * s below the line's
    top (the glyphs' own top: GlyphGenerator's top_shift / 24).
    ``badge_height(run)`` is a badge's height in ems."""
    lines: List[List[Run]] = [[]]
    for run in runs:
        parts = run.text.split("\n")
        for k, part in enumerate(parts):
            if k:
                lines.append([])
            if part:
                lines[-1].append(Run(part, run.font, run.scale, run.color, run.background))
    height, first = 0.0, None
    for line in lines:
        if not line:
            height += line_height
            continue
        text_scales = [run.scale or 1.0 for run in line if not run.background]
        badges = [badge_height(run) for run in line if run.background]
        scale = max(text_scales + [1.0] * bool(badges))
        if first is None:
            first = height + baseline * max(text_scales or [scale])
        height += line_height * scale + max([0.0] + [b - scale for b in badges])
    return (first or 0.0), height


def badge_name(run: Run, base: BaseFormat) -> str:
    """A badge's image name, describing it: font key, size (pt), text and
    background colours, text."""
    size = base.size_pt * (run.scale or 1)
    return (f"{BADGE_PREFIX}{run.font or base.font_key}|{size:.2f}|{run.color or base.color}|"
            f"{run.background}|{run.text}")


def parse_badge_name(name: str) -> Optional[Tuple[str, float, str, str, str]]:
    """(font key, size pt, text colour, background colour, text) of a badge."""
    if not name.startswith(BADGE_PREFIX):
        return None
    parts = name[len(BADGE_PREFIX):].split("|", 4)
    if len(parts) != 5:
        return None
    try:
        return parts[0], float(parts[1]), parts[2], parts[3], parts[4]
    except ValueError:
        return None


class RichFields:
    """The slot fields of a label field (q2vt_label or q2vt_label_NNNN)."""

    def __init__(self, label_field: str):
        suffix = label_field[len(f"{_FIELD_PREFIX}_label"):]
        self.label = label_field
        self.prefix = f"{_FIELD_PREFIX}_rich{suffix}"

    def text(self, i: int) -> str:
        return f"{self.prefix}_t{i}"

    def font(self, i: int) -> str:
        return f"{self.prefix}_f{i}"

    def scale(self, i: int) -> str:
        return f"{self.prefix}_s{i}"

    def color(self, i: int) -> str:
        return f"{self.prefix}_c{i}"

    def image(self, i: int) -> str:
        return f"{self.prefix}_i{i}"

    def all(self, slots: int) -> List[Tuple[str, str]]:
        """(name, OGR type) of the fields of ``slots`` slots."""
        out = []
        for i in range(slots):
            out += [(self.text(i), "String"), (self.font(i), "String"), (self.scale(i), "Real"),
                    (self.color(i), "String"), (self.image(i), "String")]
        return out

    @staticmethod
    def slots_in(names) -> dict:
        """{prefix: slot count} of the rich fields among ``names``."""
        found = {}
        for name in names:
            match = re.fullmatch(rf"({_FIELD_PREFIX}_rich(?:_\d+)?)_t(\d+)", name)
            if match:
                found[match.group(1)] = max(found.get(match.group(1), 0), int(match.group(2)) + 1)
        return found


def slot_values(runs: List[Run], base: BaseFormat, slots: int) -> List[dict]:
    """Per slot: {text, font, scale, color, image} of a label's runs (the
    runs beyond the slots join the last one's text)."""
    if len(runs) > slots:
        tail = "".join(r.text for r in runs[slots - 1:])
        runs = runs[:slots - 1] + [Run(tail)]
    values = []
    for run in runs:
        if run.background:
            values.append({"text": run.text, "image": badge_name(run, base)})
        else:
            values.append({"text": run.text, "font": run.font, "scale": run.scale, "color": run.color})
    return values


def sql_base(base: BaseFormat) -> dict:
    """A label's own format, as q2vt.html_label (live_sql_library.py) reads it."""
    return {"family": base.family, "weight": base.weight, "italic": base.italic,
            "size": base.size_pt, "color": base.color, "key": base.font_key}


def sql_fonts(families) -> dict:
    """{"family|weight|italic": font key} of ``families``' installed styles,
    for q2vt.html_label: SQL has no font database."""
    table = {}
    for family in sorted({f for f in families if f}):
        for weight in range(100, 1000, 100):
            for italic in (False, True):
                table[f"{family.lower()}|{weight}|{'true' if italic else 'false'}"] = font_key(family, weight, italic)
    return table
