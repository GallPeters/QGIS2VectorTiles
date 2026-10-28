"""Convert QGIS Vector Tile Layer styles to MapLibre GL JSON style format."""

import copy
import json
import math
import os
import re
from os.path import join
from typing import Any, Dict, List, Optional, Tuple, Union

from qgis.PyQt.QtGui import QColor, QFont, QFontMetrics, QFontMetricsF
from ..utils.config import Qt
from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsCsException,
    QgsUnitTypes,
    QgsVectorTileLayer,
    QgsVectorTileBasicRenderer,
    QgsVectorTileBasicLabeling,
    QgsSymbol,
    QgsPalLayerSettings,
    QgsSimpleLineSymbolLayer,
    QgsSimpleFillSymbolLayer,
    QgsProcessingUtils,
    QgsExpression,
    QgsProperty,
    QgsSymbolLayer,
    QgsSymbolLayerUtils,
    QgsTextFormat,
    QgsTextDocument,
    QgsTextDocumentMetrics,
    QgsRenderContext,
    QgsProject,
    QgsTextBackgroundSettings,
    QgsMapUnitScale,
    QgsFontMarkerSymbolLayer,
)
from qgis.utils import iface
from .glyphs_generator import CHARS_PREFIX, GlyphGenerator, qfontdb
from .sprite_generator import (
    BalloonShape, BrushPattern, FillPattern, HashMark, LabelShape, LinePattern, SpriteGenerator, sprite_key,
)
from .rules_exporter import _TEMP_RULE_FORMAT, RULE_FIELD, color_hex
from ..utils.flattened_rule import (
    CLUSTER_SIZE_FIELD, DISPLACEMENT_ANCHOR_FIELD, DISPLACEMENT_RADIUS_FIELD, HEATMAP_PROPERTY,
    OFFSET_PROPERTY, TILE_LAYERS_KEY,
    cluster_settings, displacement_label_settings, displacement_settings, pinned_field,
)
from ..utils.config import _SPRITE_QUALITY, _TILES_MAX_ZOOM
from ..utils.zoom_levels import ZoomLevels


def _mix_colors(*colors: QColor) -> QColor:
    """The mean of colours (with their alpha), e.g. a gradient's two ends."""
    valid = [c for c in colors if c is not None and c.isValid()]
    if not valid:
        return QColor(0, 0, 0, 0)
    n = len(valid)
    return QColor(
        round(sum(c.red() for c in valid) / n), round(sum(c.green() for c in valid) / n),
        round(sum(c.blue() for c in valid) / n), round(sum(c.alpha() for c in valid) / n),
    )


def _ramp_or_colors(symbol_layer, two_color_type: int) -> QColor:
    """A gradient or shapeburst fill's mean colour: of its two colours, or
    its colour ramp sampled along its length."""
    try:
        if PropertyExtractor.enum_int(symbol_layer.gradientColorType()
                                      if hasattr(symbol_layer, "gradientColorType")
                                      else symbol_layer.colorType()) == two_color_type:
            return _mix_colors(symbol_layer.color(), symbol_layer.color2())
        ramp = symbol_layer.colorRamp()
        if ramp is not None:
            return _mix_colors(*(ramp.color(i / 4) for i in range(5)))
    except (AttributeError, RuntimeError, TypeError):
        pass
    return symbol_layer.color()


# QgsTextBackgroundSettings shape types: drawn as a marker symbol, as a
# shape (see LabelShape) and among those as an ellipse; its fixed size type;
# the percentage render unit.
_BACKGROUND_MARKER = 5
_BACKGROUND_SHAPES = frozenset({0, 1, 2, 3, 4})  # Rectangle, square, ellipse, circle, SVG.
_BACKGROUND_SVG = 4
_BACKGROUND_ELLIPSES = frozenset({2, 3})
_BACKGROUND_SIZE_FIXED = 1
_PERCENTAGE_UNIT = int(getattr(Qgis.RenderUnit.Percentage, "value", Qgis.RenderUnit.Percentage))

# Fonts a label's data-defined font family can pick from, at most (each
# one's glyphs are written).
_MAX_LABEL_FONTS = 16

# Qt brush styles a simple fill draws as a repeated pattern: Dense1-7,
# horizontal, vertical, cross, diagonal and diagonal cross hatching.
_BRUSH_PATTERNS = frozenset(range(2, 15))

# QgsPointDisplacementRenderer.Placement.Grid
_DISPLACEMENT_GRID = 2

# A style filter selecting one rule's rows of a shared dataset (TilesStyler).
_RULE_FILTER = re.compile(r'"' + re.escape(RULE_FIELD) + r'" = (\d+)')

# Layout properties MapLibre evaluates per feature, which a merged layer can
# give per rule; the other layout properties must be equal to merge.
_DATA_DRIVEN_LAYOUT = frozenset({
    "symbol-sort-key", "line-sort-key", "fill-sort-key", "icon-image", "icon-size",
    "icon-rotate", "icon-offset", "icon-anchor", "text-field", "text-size", "text-rotate",
    "text-offset", "text-letter-spacing", "text-max-width", "text-justify", "text-anchor",
    "text-transform", "text-radial-offset",
})
# Paint properties MapLibre can't evaluate per feature.
_CONSTANT_PAINT = frozenset({
    "fill-antialias", "fill-translate", "fill-translate-anchor", "fill-pattern",
    "line-translate", "line-translate-anchor", "line-dasharray", "line-pattern",
    "icon-translate", "icon-translate-anchor", "text-translate", "text-translate-anchor",
})


# Fill symbol layers whose offset() is an x/y offset of the whole fill (a
# pattern fill's offset is its pattern's).
_OFFSET_FILLS = frozenset({"QgsSimpleFillSymbolLayer", "QgsGradientFillSymbolLayer",
                           "QgsShapeburstFillSymbolLayer"})


def fill_offset(symbol_layer) -> Optional[List]:
    """A fill symbol layer's x/y offset as MapLibre ``*-translate``; None
    if it has none."""
    if symbol_layer is None or type(symbol_layer).__name__ not in _OFFSET_FILLS:
        return None
    try:
        offset = symbol_layer.offset()
        return PropertyExtractor.offset_to_translate(
            offset.x(), offset.y(), symbol_layer.offsetUnit(),
            PropertyExtractor.map_unit_scale(symbol_layer, "offsetMapUnitScale"))
    except (AttributeError, RuntimeError, TypeError):
        return None


def outline_offset(symbol_layer) -> Optional[List]:
    """The fill offset an outline line layer carries (see OFFSET_PROPERTY),
    as MapLibre ``line-translate``; None if it carries none."""
    try:
        prop = symbol_layer.dataDefinedProperties().property(OFFSET_PROPERTY)
        if prop is None or prop.isActive() or not isinstance(symbol_layer, QgsSimpleLineSymbolLayer):
            return None
        data = json.loads(prop.staticValue())
        scale = QgsMapUnitScale(data["min_scale"], data["max_scale"])
        scale.minSizeMMEnabled, scale.minSizeMM = data["min_mm_on"], data["min_mm"]
        scale.maxSizeMMEnabled, scale.maxSizeMM = data["max_mm_on"], data["max_mm"]
        return PropertyExtractor.offset_to_translate(data["x"], data["y"], data["unit"], scale)
    except (AttributeError, RuntimeError, ValueError, KeyError, TypeError):
        return None


def _rule_of(layer_def: dict) -> Optional[int]:
    """The rule number a layer is filtered to, if it is a rule's layer."""
    flt = layer_def.get("filter")
    if (isinstance(flt, list) and len(flt) == 3 and flt[0] == "=="
            and flt[1] == ["get", RULE_FIELD] and isinstance(flt[2], int)):
        return flt[2]
    return None


def _mergeable(first: dict, other: dict) -> bool:
    """Whether two rule layers of one dataset can be drawn as one layer."""
    if _rule_of(first) is None or _rule_of(other) is None:
        return False
    for key in ("type", "source", "source-layer", "minzoom", "maxzoom", "_enabled", "_only", "metadata"):
        if first.get(key) != other.get(key):
            return False
    for section, varying in (("layout", _DATA_DRIVEN_LAYOUT), ("paint", None)):
        a, b = first.get(section, {}), other.get(section, {})
        if a.keys() != b.keys():
            return False
        for key, value in a.items():
            if value == b[key]:
                continue
            allowed = key in varying if varying is not None else key not in _CONSTANT_PAINT
            if (not allowed or PropertyExtractor.uses_zoom(value)
                    or PropertyExtractor.uses_zoom(b[key])):
                return False
    return True


def _match_value(value):
    """A value as a ``match`` output: literal arrays need ``literal``."""
    if isinstance(value, list) and (not value or not isinstance(value[0], str)):
        return ["literal", value]
    return value


# A layer id's parts that differ between the rules of one dataset: rule
# depth and number, and the renderer rule a label was split by.
_RULE_ID_PARTS = re.compile(r"[drf]\d\d")


def _slot(layer_def: dict) -> tuple:
    """What a layer draws regardless of its rule: its type and its id
    without the rule's own parts (the symbol layer, the geometry) or the
    _N that keeps ids unique (a fill and its outline share a rule's name)."""
    name = re.sub(r"_\d+$", "", _RULE_ID_PARTS.sub("", layer_def.get("id", "")))
    return layer_def.get("type"), name


def group_rule_layers(layers: List[dict]) -> List[dict]:
    """Order each run of consecutive layers drawn from one shared dataset by
    what they draw, keeping the rules' order within each: the fills of all
    categories, then all their outlines (or all road casings, then all road
    fills), as QGIS's symbol levels draw them. Rule by rule (category 1's
    fill and outline, then category 2's...) no two layers of a kind are
    next to each other, so none could merge.
    """
    result: List[dict] = []
    block: List[dict] = []

    def flush():
        order: Dict[tuple, int] = {}
        for layer_def in block:
            order.setdefault(_slot(layer_def), len(order))
        result.extend(sorted(block, key=lambda l: order[_slot(l)]))
        block.clear()

    for layer_def in layers:
        if (block and _rule_of(layer_def) is not None
                and layer_def.get("source-layer") == block[0].get("source-layer")):
            block.append(layer_def)
            continue
        flush()
        if _rule_of(layer_def) is not None:
            block.append(layer_def)
        else:
            result.append(layer_def)
    flush()
    return result


def merge_rule_layers(layers: List[dict]) -> List[dict]:
    """Merge consecutive layers of the rules sharing a dataset that differ
    only in properties MapLibre evaluates per feature (typically the
    categories of a categorized renderer, or labels split by category) into
    one layer, the varying properties as ``match`` expressions on the rule
    number. Fewer layers render and place labels faster; features are then
    drawn in their own order, as QGIS draws them (without symbol levels).

    A run holds each rule once: two layers of one rule (e.g. a road's
    casing and fill) would need two values for one ``match`` label.
    """
    runs: List[List[dict]] = []
    for layer_def in layers:
        if (runs and _mergeable(runs[-1][0], layer_def)
                and _rule_of(layer_def) not in {_rule_of(l) for l in runs[-1]}):
            runs[-1].append(layer_def)
        else:
            runs.append([layer_def])
    merged = []
    for run in runs:
        if len(run) == 1:
            merged.append(run[0])
            continue
        rules = [_rule_of(layer_def) for layer_def in run]
        layer_def = copy.deepcopy(run[0])
        layer_def["filter"] = ["match", ["get", RULE_FIELD], rules, True, False]
        for section in ("layout", "paint"):
            for key in layer_def.get(section, {}):
                values = [member[section][key] for member in run]
                if all(v == values[0] for v in values):
                    continue
                expression = ["match", ["get", RULE_FIELD]]
                for rule, value in zip(rules, values):
                    expression += [rule, _match_value(value)]
                expression.append(_match_value(values[-1]))
                layer_def[section][key] = expression
        merged.append(layer_def)
    return merged


def _band_key(layer_def: dict) -> tuple:
    """What a zoom band's layer draws regardless of its zooms: two layers
    with equal keys are bands of one rule (and symbol layer)."""
    name = re.sub(r"_\d+$", "", re.sub(r"o\d\di\d\d", "", layer_def.get("id", "")))
    return (layer_def.get("type"), layer_def.get("source-layer"), name,
            json.dumps(layer_def.get("filter"), sort_keys=True),
            json.dumps(layer_def.get("_enabled"), sort_keys=True),
            json.dumps(layer_def.get("_only"), sort_keys=True))


def _steppable(run: List[dict]) -> bool:
    """Whether layers can be one layer whose differing values are zoom
    steps: the same properties, none differing that takes no expression
    (visibility) or already depends on the zoom (a step can't hold one)."""
    for section in ("layout", "paint"):
        keys = run[0].get(section, {}).keys()
        if any(layer_def.get(section, {}).keys() != keys for layer_def in run):
            return False
        for key in keys:
            values = [layer_def[section][key] for layer_def in run]
            if all(v == values[0] for v in values):
                continue
            if key == "visibility" or any(PropertyExtractor.uses_zoom(v) for v in values):
                return False
    return True


def _group_bands(layers: List[dict]) -> List[dict]:
    """In each run of consecutive layers drawn from one dataset, put the
    layers of one rule's zoom bands next to each other, in zoom order: a
    band's fill and outline come in turn (fill 17, outline 17, fill 16...),
    and bands of one kind never overlap in zoom, so their order doesn't
    matter. Kinds keep the order they first appear in (fills, then
    outlines, as each band drew them)."""
    result: List[dict] = []
    block: List[dict] = []

    def flush():
        order: Dict[tuple, int] = {}
        for layer_def in block:
            order.setdefault(_band_key(layer_def), len(order))
        result.extend(sorted(block, key=lambda l: (order[_band_key(l)], l.get("minzoom") or 0)))
        block.clear()

    for layer_def in layers:
        if block and layer_def.get("source-layer") and \
                layer_def.get("source-layer") == block[0].get("source-layer"):
            block.append(layer_def)
            continue
        flush()
        if layer_def.get("source-layer"):
            block.append(layer_def)
        else:
            result.append(layer_def)
    flush()
    return result


def merge_zoom_bands(layers: List[dict]) -> List[dict]:
    """Merge the consecutive layers of one rule's zoom bands (a rule split
    by its @map_scale expressions, drawn from one dataset) into one layer:
    a value that differs between bands becomes ["step", ["zoom"], ...],
    switching at each band's first zoom, as the band layers switched.
    Fewer layers draw faster, with the same result at every zoom.
    """
    merged: List[dict] = []
    run: List[dict] = []
    layers = _group_bands(layers)

    def flush():
        if run:
            merged.append(_merged_bands(run))
        run.clear()

    for layer_def in layers:
        if (run and "source-layer" in layer_def
                and _band_key(run[0]) == _band_key(layer_def)
                and run[-1].get("maxzoom") is not None and layer_def.get("minzoom") is not None
                and abs(run[-1]["maxzoom"] - layer_def["minzoom"]) < 1e-6
                and _steppable(run + [layer_def])):
            run.append(layer_def)
            continue
        flush()
        run.append(layer_def)
    flush()
    return merged


def _merged_bands(run: List[dict]) -> dict:
    """One layer drawing a run of zoom bands (in zoom order): a value that
    differs between them becomes ["step", ["zoom"], ...], switching at each
    band's first zoom."""
    if len(run) == 1:
        return run[0]
    layer_def = copy.deepcopy(run[0])
    layer_def["maxzoom"] = run[-1].get("maxzoom")
    for section in ("layout", "paint"):
        for key in layer_def.get(section, {}):
            values = [member[section][key] for member in run]
            if all(v == values[0] for v in values):
                continue
            expression = ["step", ["zoom"], _match_value(values[0])]
            previous = values[0]
            for member, value in zip(run[1:], values[1:]):
                if value != previous:  # A stop only where the value changes.
                    expression += [member.get("minzoom", 0), _match_value(value)]
                    previous = value
            layer_def[section][key] = expression
    return layer_def


# The zoom range part of a dataset name (the tiles' zooms it is written to).
_DATASET_ZOOMS = re.compile(r"o\d\di\d\d")


def _dataset_band_key(layer_def: dict) -> Optional[tuple]:
    """What a zoom band's layer draws regardless of its zooms and of its
    dataset's: two layers with equal keys are bands of one rule drawn from
    datasets of their own (a geometry made for each zoom's scale)."""
    source_layer = layer_def.get("source-layer")
    if not source_layer or not _DATASET_ZOOMS.search(source_layer):
        return None
    meta = layer_def.get("metadata") or {}
    if meta.get("q2vt:cluster") or meta.get("q2vt:displacement") or layer_def.get("type") == "heatmap":
        return None
    if RULE_FIELD in json.dumps(layer_def.get("filter")):
        # A dataset shared by rules: merge_rule_layers draws its rules as one.
        return None
    name = re.sub(r"_\d+$", "", _DATASET_ZOOMS.sub("", layer_def.get("id", "")))
    return (layer_def.get("type"), _DATASET_ZOOMS.sub("", source_layer), name,
            json.dumps(layer_def.get("filter"), sort_keys=True),
            json.dumps(layer_def.get("_enabled"), sort_keys=True),
            json.dumps(layer_def.get("_only"), sort_keys=True),
            json.dumps(layer_def.get("metadata"), sort_keys=True))


def _dataset_zooms(layer_def: dict) -> Tuple[int, int]:
    """The tile zooms a layer's dataset is written to (its name's oNNiNN)."""
    zooms = _DATASET_ZOOMS.search(layer_def.get("source-layer", "")).group()
    return int(zooms[1:3]), int(zooms[4:6])


def _mergeable_bands(run: List[dict]) -> bool:
    """Whether a rule's zoom band layers (in zoom order) can be one layer:
    they follow each other, their datasets in tiles of their own zooms (a
    band switching within a zoom shares its tiles, and bands beyond the
    tiles' last zoom are all written into its tiles), with steppable values."""
    return all(
        a.get("maxzoom") is not None and b.get("minzoom") is not None
        and abs(a["maxzoom"] - b["minzoom"]) < 1e-6
        and min(_dataset_zooms(a)[1], _TILES_MAX_ZOOM) < min(_dataset_zooms(b)[0], _TILES_MAX_ZOOM)
        for a, b in zip(run, run[1:])) and _steppable(run)


def merge_dataset_bands(layers: List[dict]) -> Tuple[List[dict], Dict[str, str]]:
    """Merge the layers of one rule's zoom bands that each draw a dataset of
    their own (a rule whose geometry follows the scale: arrows, trims,
    offsets or callouts sized on screen) into one layer, as merge_zoom_bands
    merges bands sharing a dataset. Their datasets are written into one tile
    layer, the first band's, each still at its own zooms only
    (GDALTilesGenerator), so every zoom draws its own band.

    A run of consecutive band layers of one rule (a fill band and its
    outline band in turn) becomes one layer per kind, kinds in the order
    they first appear (each zoom still draws its fill, then its outline),
    only if every kind merges; otherwise it is left as it is.

    Returns the layers and, by dataset, the tile layer it is written to.
    """
    result: List[dict] = []
    shared: Dict[str, str] = {}
    i = 0
    while i < len(layers):
        key = _dataset_band_key(layers[i])
        j = i + 1
        while (key is not None and j < len(layers) and _dataset_band_key(layers[j]) is not None
               and _dataset_band_key(layers[j])[1] == key[1]):
            j += 1
        block = layers[i:j]
        kinds: Dict[tuple, List[dict]] = {}
        for layer_def in block:
            if key is not None:
                kinds.setdefault(_dataset_band_key(layer_def), []).append(layer_def)
        runs = [sorted(run, key=lambda l: l.get("minzoom") or 0) for run in kinds.values()]
        if key is not None and any(len(run) > 1 for run in runs) and all(map(_mergeable_bands, runs)):
            for run in runs:
                for member in run[1:]:
                    shared[member["source-layer"]] = run[0]["source-layer"]
                result.append(_merged_bands(run))
        else:
            result.extend(block)
        i = j
    return result, shared


def bridge_zoom_bands(layers: List[dict], archive_max_zoom: int) -> List[dict]:
    """Widen each data layer's zoom range by one zoom on each side where the
    tiles there don't hold its data, so the map never goes blank between zooms.

    While a new zoom's tiles load, MapLibre draws the previous zoom's tiles in
    their place; a layer shown only from the new zoom has no data in them. Each
    dataset is written only into its own zooms' tiles (tiles_generator CONF),
    so the extra zoom only ever draws it from those stand-in tiles, never twice.
    Beyond the archive's last zoom its tiles are enlarged and do hold the data,
    so ranges are not widened there.

    Only whole-zoom edges are widened: an edge within a zoom level (a scale
    limit's exact zoom) has the data in that zoom's tiles on both sides of
    it, and widening it would draw the layer beyond its scale range.
    """
    for layer_def in layers:
        if "source-layer" not in layer_def:
            continue  # background
        low = layer_def.get("minzoom")
        high = layer_def.get("maxzoom")
        if low is not None and 0 < low <= archive_max_zoom and float(low).is_integer():
            layer_def["minzoom"] = low - 1
        if high is not None and high <= archive_max_zoom and float(high).is_integer():
            layer_def["maxzoom"] = high + 1
    return layers


# MapLibre's default values (style specification) of the properties the
# converter writes: a property at its default is left out of style.json.
_SPEC_DEFAULTS = {
    "visibility": "visible",
    "fill-antialias": True, "fill-opacity": 1, "fill-translate": [0, 0],
    "fill-translate-anchor": "map",
    "line-blur": 0, "line-gap-width": 0, "line-offset": 0, "line-opacity": 1,
    "line-translate": [0, 0], "line-translate-anchor": "map", "line-width": 1,
    "line-cap": "butt", "line-join": "miter", "line-miter-limit": 2, "line-round-limit": 1.05,
    "icon-opacity": 1, "icon-halo-width": 0, "icon-halo-blur": 0, "icon-translate": [0, 0],
    "icon-translate-anchor": "map",
    "text-opacity": 1, "text-halo-width": 0, "text-halo-blur": 0, "text-translate": [0, 0],
    "text-translate-anchor": "map",
    "symbol-placement": "point", "symbol-spacing": 250, "symbol-avoid-edges": False,
    "symbol-z-order": "auto",
    "icon-allow-overlap": False, "icon-ignore-placement": False, "icon-optional": False,
    "icon-size": 1, "icon-rotate": 0, "icon-padding": 2, "icon-keep-upright": False,
    "icon-offset": [0, 0], "icon-anchor": "center", "icon-text-fit": "none",
    "icon-rotation-alignment": "auto", "icon-pitch-alignment": "auto",
    "text-allow-overlap": False, "text-ignore-placement": False, "text-optional": False,
    "text-size": 16, "text-max-width": 10, "text-line-height": 1.2, "text-letter-spacing": 0,
    "text-justify": "center", "text-radial-offset": 0, "text-anchor": "center",
    "text-max-angle": 45, "text-rotate": 0, "text-padding": 2, "text-keep-upright": True,
    "text-transform": "none", "text-offset": [0, 0],
    "text-rotation-alignment": "auto", "text-pitch-alignment": "auto",
}

# Expression operators reading the feature: a sort key without them is the
# same for every feature of its layer.
_FEATURE_OPERATORS = frozenset({"get", "has", "properties", "feature-state", "geometry-type", "id"})


def _reads_feature(value) -> bool:
    """Whether a style value is an expression reading the feature."""
    if not isinstance(value, list) or not value:
        return False
    if isinstance(value[0], str) and value[0] in _FEATURE_OPERATORS:
        return True
    return any(_reads_feature(item) for item in value)


def strip_defaults(layers: List[dict]) -> List[dict]:
    """Leave out of each layer the properties at MapLibre's default value
    (a smaller style, parsed faster; drawn the same).

    A sort key equal for all of a layer's features orders nothing, but
    MapLibre still sorts each tile's features by it: it is left out too, a
    symbol layer then drawing its symbols in their own order, as the key
    did ("source"; MapLibre would otherwise sort them by their position on
    screen). A halo of no width draws nothing, whatever its colour.
    """
    for layer_def in layers:
        layout = layer_def.get("layout", {})
        for key in ("line-sort-key", "fill-sort-key", "symbol-sort-key"):
            if key in layout and not _reads_feature(layout[key]):
                del layout[key]
                if key == "symbol-sort-key" and layout.get("symbol-z-order", "auto") == "auto":
                    layout["symbol-z-order"] = "source"
        for section in ("paint", "layout"):
            props = layer_def.get(section)
            if not props:
                continue
            for key in [k for k, v in props.items() if k in _SPEC_DEFAULTS
                        and v == _SPEC_DEFAULTS[k]
                        and isinstance(v, bool) == isinstance(_SPEC_DEFAULTS[k], bool)]:
                del props[key]
            for kind in ("icon", "text"):
                if section == "paint" and f"{kind}-halo-width" not in props:
                    props.pop(f"{kind}-halo-color", None)
                    props.pop(f"{kind}-halo-blur", None)
            if not props:
                del layer_def[section]
    return layers


# A label callout's dataset (RulesFlattener._callout_rules: symbol layer 9N).
_CALLOUT_DATASET = re.compile(r"t00d\d\dr\d\dg\d\dc0[012]o\d\di\d\ds9\d$")


def raise_callouts(layers: List[dict]) -> List[dict]:
    """Put the labels' callouts above every other layer but the labels:
    QGIS draws them with the labels, over the map."""
    callouts = [l for l in layers if _CALLOUT_DATASET.search(l.get("source-layer") or "")]
    if not callouts:
        return layers
    rest = [l for l in layers if not _CALLOUT_DATASET.search(l.get("source-layer") or "")]
    first_label = next((i for i, l in enumerate(rest)
                        if "text-field" in l.get("layout", {})), len(rest))
    return rest[:first_label] + callouts + rest[first_label:]



def share_tile_layers(layers: List[dict]) -> Tuple[List[dict], Dict[str, str]]:
    """Draw consecutive layers that are alike in all but their dataset as
    one layer: their datasets are written into one tile layer (the first's,
    see GDALTilesGenerator), which the layer draws. Fewer layers draw faster.

    Only datasets each drawn by that one layer alone, for the same zooms,
    are shared: the tile layer then holds no feature another layer would
    draw. Returns the layers and, by dataset, the tile layer it is written to.
    """
    refs: Dict[str, int] = {}
    for layer_def in layers:
        if layer_def.get("source-layer"):
            refs[layer_def["source-layer"]] = refs.get(layer_def["source-layer"], 0) + 1

    def alike(first: dict, other: dict) -> bool:
        a, b = first.get("source-layer"), other.get("source-layer")
        if (not a or not b or a == b or refs[a] != 1 or refs[b] != 1
                or first.get("type") not in ("fill", "line", "symbol")
                or (first.get("metadata") or {}).get("q2vt:cluster")
                or (first.get("metadata") or {}).get("q2vt:displacement")):
            return False
        zooms_a, zooms_b = _DATASET_ZOOMS.search(a), _DATASET_ZOOMS.search(b)
        if not zooms_a or not zooms_b or zooms_a.group() != zooms_b.group():
            return False
        skip = ("id", "source-layer")
        return ({k: v for k, v in first.items() if k not in skip}
                == {k: v for k, v in other.items() if k not in skip})

    result: List[dict] = []
    shared: Dict[str, str] = {}
    for layer_def in layers:
        if result and alike(result[-1], layer_def):
            shared[layer_def["source-layer"]] = result[-1]["source-layer"]
            continue
        result.append(layer_def)
    return result, shared


class PropertyExtractor:
    """Utility class for extracting and converting PyQGIS properties to MapLibre format.

    Provides shared helpers used by every specialised extractor:
    - Resolution of static vs. data-defined values into MapLibre expressions.
    - Conversion of ``QColor`` instances into MapLibre ``rgba()`` strings.
    - Conversion of QGIS length units into pixels at a 96 DPI baseline.
    """

    # Conversion factors to pixels at 96 DPI, keyed by unit string keywords.
    _UNIT_KEYWORDS: dict = {
        "millimeter": 3.78, "mm": 3.78,
        "inch": 96.0, "in": 96.0,
        "point": 96.0 / 72.0, "pt": 96.0 / 72.0,
        "pixel": 1.0, "px": 1.0,
    }
    # QgsUnitTypes enum integer values → conversion factors.
    _UNIT_ENUMS: dict = {0: 3.78, 2: 1.0, 4: 96.0 / 72.0, 5: 96.0}
    # Map-based units: a length on the ground, so its pixel size depends on the zoom.
    _MAP_UNITS = 1
    _METERS_IN_MAP_UNITS = 7
    # EPSG:3857 metres per web map pixel at zoom 0 (MapLibre's 512 px tiles).
    _MERCATOR_M_PER_PX_Z0 = 40075016.68557849 / 512
    # EPSG:3857 metres per map unit / per ground metre at the data, set per export.
    map_unit_m = 1.0
    ground_m = 1.0

    @staticmethod
    def get_value_or_expression(value: Any, prop: QgsProperty) -> Union[Any, List]:
        """Return a static value or a MapLibre ``["get", field]`` expression.

        If the supplied ``QgsProperty`` is active and references a tile attribute
        (``q2vt*`` field naming convention), a MapLibre expression is returned so
        the property is data-driven at render time. Otherwise the original
        ``value`` is returned unchanged.

        Args:
            value: The static fallback value to use when no data-defined override applies.
            prop:  The data-defined property descriptor from QGIS.

        Returns:
            Either the original ``value`` or a MapLibre expression list.
        """
        if prop and prop.isActive():
            expression = prop.expressionString()
            qexpr = QgsExpression(expression)
            evaluation = qexpr.evaluate()
            if evaluation is not None and evaluation != "" and not qexpr.hasEvalError():
                return evaluation
            field_name = expression.replace('"', "")
            idx = field_name.find("q2vt")
            if field_name and idx != -1:
                extracted_name = [part for part in field_name.split(' ') if part.startswith('q2vt')][0]
                return ["get", extracted_name.replace(',', '')]
        return value

    @staticmethod
    def convert_qcolor_to_maplibre(color: QColor) -> str:
        """Convert a ``QColor`` into a MapLibre-compatible ``rgba()`` string.

        Args:
            color: The Qt color to convert.

        Returns:
            A string in the form ``"rgba(r, g, b, a)"`` where alpha is normalised
            to the ``[0, 1]`` range expected by MapLibre.
        """
        return f"rgba({color.red()}, {color.green()}, {color.blue()}, {color.alphaF()})"

    @classmethod
    def convert_length_to_pixels(cls, value: float, unit_obj=None) -> float:
        """Convert a length value from a QGIS unit into pixels at 96 DPI.

        The unit can be supplied as either a textual hint (e.g. ``"MM"``,
        ``"Pixels"``) or as a ``QgsUnitTypes`` enum integer. When the unit is
        unknown, the value is treated as millimetres.

        Args:
            value:    The length to convert.
            unit_obj: A unit descriptor; either a string-like or enum integer.

        Returns:
            The equivalent pixel value (float).
        """
        if value is None:
            return value

        try:
            unit_str = str(unit_obj).lower() if unit_obj is not None else ""
        except (OSError, RuntimeError):
            unit_str = ""

        for keyword, factor in cls._UNIT_KEYWORDS.items():
            if keyword in unit_str:
                return value * factor

        try:
            factor = cls._UNIT_ENUMS.get(unit_obj)
            if factor is not None:
                return value * factor
        except (RuntimeError, AttributeError):
            pass

        return value * 3.78  # Default: treat as millimetres

    @staticmethod
    def enum_int(enum_obj) -> Optional[int]:
        """A QGIS/Qt enum value as an integer (Qt5 gives ints, Qt6 enums)."""
        try:
            return int(getattr(enum_obj, "value", enum_obj))
        except (TypeError, ValueError):
            return None

    @classmethod
    def is_map_unit(cls, unit_obj) -> bool:
        """Whether a length is on the ground (map units or metres) rather than on screen."""
        return cls.enum_int(unit_obj) in (cls._MAP_UNITS, cls._METERS_IN_MAP_UNITS)

    @staticmethod
    def map_unit_scale(obj, *getters: str):
        """A symbol layer's (or text format's) QgsMapUnitScale for a length,
        from the first of ``getters`` it has; None if none."""
        for getter in getters:
            method = getattr(obj, getter, None)
            if callable(method):
                try:
                    return method()
                except (RuntimeError, TypeError):
                    return None
        return None

    @classmethod
    def _scale_zooms(cls, scale) -> tuple:
        """The zooms (low, high) between which a map unit scale's scale
        limits let a ground length follow the zoom: beyond them QGIS draws
        it at the size it has at the limit."""
        low, high = 0.0, 24.0
        if scale is None:
            return low, high
        if scale.minScale > 0:  # Zoomed out beyond it: the size at it.
            low = ZoomLevels.fractional_zoom(scale.minScale)
        if scale.maxScale > 0:  # Zoomed in beyond it: the size at it.
            high = ZoomLevels.fractional_zoom(scale.maxScale)
        return low, max(low, high)

    @classmethod
    def _size_limits_px(cls, scale) -> tuple:
        """A map unit scale's size limits in pixels (None where unset)."""
        if scale is None:
            return None, None
        low = cls.convert_length_to_pixels(scale.minSizeMM, "mm") if scale.minSizeMMEnabled else None
        high = cls.convert_length_to_pixels(scale.maxSizeMM, "mm") if scale.maxSizeMMEnabled else None
        return low, high

    @classmethod
    def length_to_style_value(cls, value: float, unit_obj=None, scale=None) -> Union[float, List]:
        """A length as a MapLibre value: pixels, or a zoom expression for ground lengths.

        A ground length doubles in pixels with every zoom level, which an
        exponential base-2 interpolation reproduces exactly. Its map unit
        ``scale`` limits it as QGIS does: to its size at a scale limit
        beyond that scale, and to a minimum and maximum size on screen.
        Both stop it growing (or shrinking) beyond a zoom, where the
        interpolation's stops are put (it is constant beyond its stops:
        MapLibre allows no min/max around a zoom expression).
        """
        if value is None or not cls.is_map_unit(unit_obj):
            return cls.convert_length_to_pixels(value, unit_obj)
        metres = cls.map_unit_m if cls.enum_int(unit_obj) == cls._MAP_UNITS else cls.ground_m
        px_z0 = value * metres / cls._MERCATOR_M_PER_PX_Z0
        low, high = cls._scale_zooms(scale)
        min_px, max_px = cls._size_limits_px(scale)
        if px_z0 > 0:
            if min_px is not None:
                low = max(low, math.log2(min_px / px_z0))
            if max_px is not None:
                high = min(high, math.log2(max_px / px_z0))
        low, high = max(low, 0.0), min(high, 24.0)
        if high <= low:  # The limits leave one size.
            size = px_z0 * 2 ** min(max(low, 0.0), 24.0)
            if min_px is not None:
                size = max(size, min_px)
            if max_px is not None:
                size = min(size, max_px)
            return size
        return ["interpolate", ["exponential", 2], ["zoom"],
                round(low, 4), px_z0 * 2 ** low, round(high, 4), px_z0 * 2 ** high]

    @classmethod
    def scale_length_expression(cls, expression: List, unit_obj=None, scale=None) -> List:
        """A per-feature length (an expression giving a number in ``unit_obj``)
        as a MapLibre pixel value: times the unit's pixel size, or for
        ground lengths, doubled with every zoom level, within the map unit
        ``scale``'s limits (see length_to_style_value: a size limit depends
        on the feature's value, so it limits the value at each whole zoom)."""
        number = ["to-number", expression, 0]
        if not cls.is_map_unit(unit_obj):
            return ["*", number, cls.convert_length_to_pixels(1.0, unit_obj)]
        metres = cls.map_unit_m if cls.enum_int(unit_obj) == cls._MAP_UNITS else cls.ground_m
        factor = metres / cls._MERCATOR_M_PER_PX_Z0
        low, high = cls._scale_zooms(scale)
        min_px, max_px = cls._size_limits_px(scale)

        def at(zoom: float):
            value = ["*", number, factor * 2 ** zoom]
            if min_px is not None:
                value = ["max", min_px, value]
            if max_px is not None:
                value = ["min", max_px, value]
            return value

        zooms = [low, high]
        if min_px is not None or max_px is not None:
            zooms = [low] + list(range(int(low) + 1, int(math.ceil(high)))) + [high]
        if high <= low:
            return at(low)
        expression = ["interpolate", ["exponential", 2], ["zoom"]]
        for zoom in zooms:
            expression += [round(zoom, 4), at(zoom)]
        return expression

    @classmethod
    def length_value_or_expression(
        cls, value: float, unit_obj, prop: QgsProperty, scale=None,
    ) -> Union[float, List]:
        """A length that may be data-defined, as a MapLibre value.

        QGIS reads a data-defined length in the property's unit (mm, points,
        map units...), but MapLibre reads pixels, so the value is converted
        as the static length is: a constant directly, a field through an
        expression. ``scale`` is the length's map unit scale.
        """
        base = cls.length_to_style_value(value, unit_obj, scale)
        resolved = cls.get_value_or_expression(None, prop)
        if resolved is None:
            return base
        if isinstance(resolved, list):
            return cls.scale_length_expression(resolved, unit_obj, scale)
        try:
            return cls.length_to_style_value(float(resolved), unit_obj, scale)
        except (TypeError, ValueError):
            return base

    @classmethod
    def offset_to_translate(cls, x: float, y: float, unit_obj=None, scale=None) -> Union[List, None]:
        """A fill's offset as ``fill-translate`` / ``line-translate``
        ``[x, y]`` in pixels, a zoom interpolation of them for a ground
        offset (within its map unit scale's scale limits; its size limits,
        on the offset's length, are left out); None for no offset."""
        if not x and not y:
            return None
        if not cls.is_map_unit(unit_obj):
            return [cls.convert_length_to_pixels(x, unit_obj), cls.convert_length_to_pixels(y, unit_obj)]
        metres = cls.map_unit_m if cls.enum_int(unit_obj) == cls._MAP_UNITS else cls.ground_m
        x_z0, y_z0 = (v * metres / cls._MERCATOR_M_PER_PX_Z0 for v in (x, y))
        low, high = cls._scale_zooms(scale)
        if high <= low:
            return [x_z0 * 2 ** low, y_z0 * 2 ** low]
        return ["interpolate", ["exponential", 2], ["zoom"],
                round(low, 4), ["literal", [x_z0 * 2 ** low, y_z0 * 2 ** low]],
                round(high, 4), ["literal", [x_z0 * 2 ** high, y_z0 * 2 ** high]]]

    @classmethod
    def opacity_value_or_expression(cls, base_opacity: float, prop: QgsProperty) -> Union[float, List]:
        """An opacity that may be data-defined: QGIS reads a data-defined
        opacity in percent (0-100), MapLibre in 0-1."""
        resolved = cls.get_value_or_expression(None, prop)
        if resolved is None:
            return base_opacity
        if isinstance(resolved, list):
            return ["/", ["to-number", resolved, 100], 100]
        try:
            return max(0.0, min(1.0, float(resolved) / 100.0))
        except (TypeError, ValueError):
            return base_opacity

    @staticmethod
    def color_value_or_expression(base_color: str, prop: QgsProperty) -> Union[str, List]:
        """A colour that may be data-defined: the per-feature colour, or the
        symbol's own where a feature has none (NULL, or no colour QGIS
        reads), as QGIS falls back to it."""
        resolved = PropertyExtractor.get_value_or_expression(base_color, prop)
        if isinstance(resolved, list):
            return ["to-color", resolved, base_color]
        return resolved

    @staticmethod
    def enabled_condition(prop: QgsProperty) -> Optional[Union[bool, List]]:
        """A boolean data-defined property (a symbol layer's "enabled", a
        label's "show label") as a MapLibre filter condition; None if it
        doesn't apply, False if it is always off.

        The export writes its per-feature value to a field; QGIS treats a
        NULL value as on, and MapLibre doesn't see NULL fields at all.
        """
        if not prop or not prop.isActive():
            return None
        resolved = PropertyExtractor.get_value_or_expression(None, prop)
        if isinstance(resolved, list):
            return ["any", ["!", ["has", resolved[1]]], ["to-boolean", resolved]]
        if resolved is None:
            return None
        if isinstance(resolved, str):
            return False if resolved.strip().lower() in ("", "0", "false") else None
        return None if resolved else False

    @staticmethod
    def uses_zoom(value) -> bool:
        """Whether a style value is (or contains) a zoom expression."""
        if isinstance(value, list):
            return value[:1] == ["zoom"] or any(PropertyExtractor.uses_zoom(v) for v in value)
        return False

    @classmethod
    def set_map_units(cls, crs, latitude: float) -> None:
        """Set how many EPSG:3857 metres a map unit and a ground metre are at the data.

        Web Mercator stretches ground lengths by 1/cos(latitude); a project
        in Web Mercator already draws its map units stretched that way.
        """
        stretch = 1.0 / max(math.cos(math.radians(latitude)), 0.01)
        cls.ground_m = stretch
        cls.map_unit_m = stretch
        try:
            if crs.isGeographic():
                cls.map_unit_m = 111319.49079327357  # 3857 metres per degree of longitude
            elif crs.authid() in ("EPSG:3857", "EPSG:900913", "EPSG:3785"):
                cls.map_unit_m = 1.0
            else:
                cls.map_unit_m = stretch * QgsUnitTypes.fromUnitToUnitFactor(
                    crs.mapUnits(), Qgis.DistanceUnit.Meters
                )
        except (AttributeError, RuntimeError, TypeError):
            pass

    @staticmethod
    def get_attribute(obj: Any, *names) -> Any:
        """Fetch the first available attribute from ``obj`` matching any of ``names``.

        Tries each attribute name in turn, returning the resolved value
        (calling it if callable). Returns ``None`` if no attribute exists or
        every access fails.

        Args:
            obj:    The object to inspect.
            *names: Attribute names to try, in order.

        Returns:
            The resolved value, or ``None`` if every name fails.
        """
        for name in names:
            if hasattr(obj, name):
                try:
                    attr = getattr(obj, name)
                    return attr() if callable(attr) else attr
                except (RuntimeError, AttributeError, OSError):
                    continue
        return None


class LinePropertyExtractor:
    """Extract line paint and layout properties from QGIS line symbol layers."""

    @staticmethod
    def get_line_color(symbol_layer: QgsSimpleLineSymbolLayer) -> Union[str, List]:
        """Return ``line-color`` resolving any data-defined override."""
        base_color = PropertyExtractor.convert_qcolor_to_maplibre(symbol_layer.color())
        color_prop = symbol_layer.dataDefinedProperties().property(QgsSymbolLayer.Property.PropertyStrokeColor)
        return PropertyExtractor.color_value_or_expression(base_color, color_prop)

    @staticmethod
    def get_line_width(symbol_layer: QgsSimpleLineSymbolLayer) -> Union[float, List]:
        """Return ``line-width`` in pixels, resolving any data-defined override."""
        width_unit = PropertyExtractor.get_attribute(
            symbol_layer, "widthUnit", "widthUnits", "strokeWidthUnit", "strokeWidthUnits"
        )
        width_prop = symbol_layer.dataDefinedProperties().property(
            QgsSymbolLayer.Property.PropertyStrokeWidth
        )
        return PropertyExtractor.length_value_or_expression(
            symbol_layer.width(), width_unit, width_prop,
            PropertyExtractor.map_unit_scale(symbol_layer, "widthMapUnitScale"),
        )

    @staticmethod
    def get_line_opacity(
        symbol_layer: QgsSimpleLineSymbolLayer, symbol: "QgsSymbol" = None
    ) -> Union[float, List]:
        """Return ``line-opacity`` from the symbol's general Opacity setting.

        Independent of the stroke colour's alpha channel: that alpha is already
        embedded in the ``line-color`` rgba() string, and MapLibre multiplies
        ``line-opacity`` against it at render time. ``symbol.opacity()`` is
        QGIS's own general "Opacity" slider (top of the Symbol panel) - a
        separate multiplier from colour alpha. Honours a data-defined
        ``Property.PropertyOpacity`` override when active.
        """
        try:
            base_opacity = symbol.opacity() if symbol is not None else 1.0
        except (AttributeError, RuntimeError):
            base_opacity = 1.0
        try:
            opacity_prop = symbol_layer.dataDefinedProperties().property(
                QgsSymbolLayer.Property.PropertyOpacity
            )
            return PropertyExtractor.opacity_value_or_expression(base_opacity, opacity_prop)
        except (AttributeError, RuntimeError):
            return base_opacity

    @staticmethod
    def get_line_cap(symbol_layer: QgsSimpleLineSymbolLayer) -> str:
        """Return ``line-cap`` mapped from the Qt pen-cap style."""
        cap_map = {0: "butt", 16: "square", 32: "round"}
        return cap_map.get(PropertyExtractor.enum_int(symbol_layer.penCapStyle()), "round")

    @staticmethod
    def get_line_join(symbol_layer: QgsSimpleLineSymbolLayer) -> str:
        """Return ``line-join`` mapped from the Qt pen-join style."""
        # Qt's SvgMiterJoin (256) is a miter join too.
        join_map = {0: "miter", 64: "bevel", 128: "round", 256: "miter"}
        return join_map.get(PropertyExtractor.enum_int(symbol_layer.penJoinStyle()), "round")

    @staticmethod
    def get_line_miter_limit() -> float:
        """Return ``line-miter-limit`` (MapLibre default: 2.0)."""
        return 2.0

    @staticmethod
    def get_line_round_limit() -> float:
        """Return ``line-round-limit`` (MapLibre default: 1.05)."""
        return 1.05

    @staticmethod
    def get_line_dasharray(
        symbol_layer: QgsSimpleLineSymbolLayer, width_px: float
    ) -> Optional[List[float]]:
        """Return ``line-dasharray`` from a custom dash vector or pen-style preset.

        Custom dash patterns take precedence over Qt's pen-style enum presets.
        MapLibre dash lengths are multiples of the line width, so custom
        lengths are converted to pixels and divided by ``width_px``. When no
        dashing is configured, ``None`` is returned and ``line-dasharray``
        should be omitted.
        """
        dash_vector = None
        try:
            dash_vector = symbol_layer.customDashVector()
        except (RuntimeError, AttributeError):
            pass

        custom_dash_enabled = False
        try:
            for attr in ("useCustomDashPattern", "customDashEnabled", "isCustomDash"):
                if hasattr(symbol_layer, attr):
                    custom_dash_enabled = bool(getattr(symbol_layer, attr)())
                    break
        except (RuntimeError, AttributeError):
            pass

        pen_style = PropertyExtractor.get_attribute(
            symbol_layer, "penStyle", "strokeStyle", "pen_style"
        )

        if custom_dash_enabled and dash_vector:
            dash_unit = PropertyExtractor.get_attribute(
                symbol_layer, "customDashPatternUnit", "customDashVectorUnit"
            )
            width_unit = PropertyExtractor.get_attribute(
                symbol_layer, "widthUnit", "widthUnits", "strokeWidthUnit", "strokeWidthUnits"
            )
            unit = PropertyExtractor.enum_int(dash_unit or width_unit)
            if PropertyExtractor.is_map_unit(width_unit) and unit == PropertyExtractor.enum_int(width_unit):
                # Ground dashes on a ground width: the ratio holds at every zoom.
                width = symbol_layer.width() or 1.0
                return [d / width for d in dash_vector]
            width = width_px if width_px and width_px > 0 else 1.0
            return [
                PropertyExtractor.convert_length_to_pixels(d, dash_unit or width_unit) / width
                for d in dash_vector
            ]

        # Qt's own patterns, as [dash, gap, ...] multiples of the line width.
        fixed_presets = {
            2: [4, 2],             # DashLine
            3: [1, 2],             # DotLine
            4: [4, 2, 1, 2],       # DashDotLine
            5: [4, 2, 1, 2, 1, 2], # DashDotDotLine
        }
        return fixed_presets.get(PropertyExtractor.enum_int(pen_style))

    @staticmethod
    def get_line_offset(symbol_layer: QgsSimpleLineSymbolLayer) -> Union[float, List]:
        """Return ``line-offset`` in pixels (positive values offset to the right).

        Honours a data-defined ``PropertyOffset`` override when active. Note
        that data-defined offsets are emitted in QGIS units (pixels for the
        static path); for field-based overrides the user must supply pixel
        values to match the MapLibre rendering convention.
        """
        try:
            offset = symbol_layer.offset()
            offset_unit = PropertyExtractor.get_attribute(symbol_layer, "offsetUnit", "offsetUnits")
            offset_prop = symbol_layer.dataDefinedProperties().property(
                QgsSymbolLayer.Property.PropertyOffset
            )
            return PropertyExtractor.length_value_or_expression(
                offset, offset_unit, offset_prop,
                PropertyExtractor.map_unit_scale(symbol_layer, "offsetMapUnitScale"))
        except (AttributeError, RuntimeError, TypeError):
            return 0.0

    @staticmethod
    def get_line_blur() -> float:
        """Return ``line-blur`` in pixels (MapLibre default: 0)."""
        return 0

    @staticmethod
    def get_line_gap_width() -> float:
        """Return ``line-gap-width`` in pixels (MapLibre default: 0)."""
        return 0

    @staticmethod
    def get_line_translate(symbol_layer: QgsSymbolLayer = None) -> Union[List, None]:
        """Return ``line-translate``: a fill's offset, for its outline (see
        outline_offset), else ``[0, 0]``."""
        offset = outline_offset(symbol_layer)
        return offset if offset is not None else [0, 0]

    @staticmethod
    def get_line_translate_anchor(symbol_layer: QgsSymbolLayer = None) -> str:
        """Return ``line-translate-anchor``: "viewport" for a fill's offset
        outline (QGIS offsets on screen, not turned with the map), else
        MapLibre's default "map"."""
        return "viewport" if outline_offset(symbol_layer) is not None else "map"

    @staticmethod
    def get_line_sort_key(symbol_layer: QgsSymbolLayer) -> Union[float, List]:
        """Return ``line-sort-key``: 0, features drawn in their own order.

        QGIS symbol layers have no per-feature drawing order (a layer's
        "control feature rendering order" sorts the exported features
        already). The "enabled" property is a filter (see enabled_condition).
        """
        return 0

    @staticmethod
    def is_pattern_line(symbol_layer: QgsSymbolLayer) -> bool:
        """Return ``True`` if this line symbol layer is drawn from an image.

        Detected by class-name inspection so QGIS versions lacking the
        subclass remain compatible.
        """
        return type(symbol_layer).__name__ == "QgsRasterLineSymbolLayer"

    @staticmethod
    def get_line_pattern_path(symbol_layer: QgsSymbolLayer) -> Optional[str]:
        """Return the source image path for a pattern line, or ``None``.

        Used by the converter to register a pattern image with the sprite
        pipeline and to populate ``line-pattern``.
        """
        path = PropertyExtractor.get_attribute(symbol_layer, "path", "imagePath")
        if isinstance(path, str) and path:
            return path
        return None

    @staticmethod
    def is_marker_line(symbol_layer: QgsSymbolLayer) -> bool:
        """Return ``True`` if this line symbol layer decorates the line with markers.

        Detected by class-name inspection (matching the ``is_pattern_line``
        convention above) so QGIS versions lacking the symbol-layer class
        remain compatible.
        """
        return type(symbol_layer).__name__ == "QgsMarkerLineSymbolLayer"

    @staticmethod
    def is_hashed_line(symbol_layer: QgsSymbolLayer) -> bool:
        """Return ``True`` for a hashed line (short strokes across the line)."""
        return type(symbol_layer).__name__ == "QgsHashedLineSymbolLayer"

    @staticmethod
    def _placements(symbol_layer: QgsSymbolLayer):
        try:
            return symbol_layer.placements()
        except (AttributeError, RuntimeError):
            return None

    @staticmethod
    def get_marker_line_symbol_placement(symbol_layer: QgsSymbolLayer) -> str:
        """Map QGIS marker-line placements to MapLibre ``symbol-placement``.

        MapLibre only places evenly-spaced (``"line"``) or once-per-line
        (``"line-center"``) symbols. Vertex placements of marker lines are
        exported as points (see RulesFlattener._split_marker_line); what
        reaches here is interval placement (``"line"``), or segment centres
        and other single placements, approximated by one symbol in the
        middle of the line.
        """
        placements = LinePropertyExtractor._placements(symbol_layer)
        if placements is None or placements & Qgis.MarkerLinePlacement.Interval:
            return "line"
        return "line-center"

    @staticmethod
    def get_marker_line_spacing(symbol_layer: QgsSymbolLayer) -> float:
        """Return ``symbol-spacing`` in pixels from the marker-line interval,
        or MapLibre's default (250 px) when the line has none."""
        try:
            interval = float(symbol_layer.interval())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            interval = 0.0
        if interval > 0:
            unit = PropertyExtractor.get_attribute(
                symbol_layer, "intervalUnit", "intervalUnits"
            )
            return max(1.0, PropertyExtractor.convert_length_to_pixels(interval, unit))
        return 250.0

    @staticmethod
    def get_fallback_line(symbol_layer: QgsSymbolLayer) -> tuple:
        """(colour, width) approximating a line symbol layer MapLibre can't
        draw: its colour (the mean of a burst's two colours, an arrow's fill
        colour) and width, so it shows as a plain line of the right look
        rather than MapLibre's default black hairline."""
        kind = type(symbol_layer).__name__
        color = None
        width = None
        unit = None
        try:
            if kind == "QgsLineburstSymbolLayer":
                color = _mix_colors(symbol_layer.color(), symbol_layer.color2())
            elif kind == "QgsArrowSymbolLayer":
                sub = symbol_layer.subSymbol()
                color = sub.color() if sub is not None else symbol_layer.color()
                width, unit = symbol_layer.arrowWidth(), symbol_layer.arrowWidthUnit()
            else:
                color = symbol_layer.color()
        except (AttributeError, RuntimeError, TypeError):
            color = None
        if width is None:
            width = PropertyExtractor.get_attribute(symbol_layer, "width")
            unit = PropertyExtractor.get_attribute(symbol_layer, "widthUnit")
        if not isinstance(width, (int, float)) or width <= 0:
            width, unit = 0.26, None  # QGIS's default line width, in mm.
        if color is None or not color.isValid():
            color = QColor(0, 0, 0)
        return (PropertyExtractor.convert_qcolor_to_maplibre(color),
                PropertyExtractor.length_to_style_value(width, unit, PropertyExtractor.map_unit_scale(
                    symbol_layer, "arrowWidthUnitScale" if kind == "QgsArrowSymbolLayer" else "widthMapUnitScale")))

    @staticmethod
    def get_marker_line_rotate_symbols(symbol_layer: QgsSymbolLayer) -> bool:
        """Return whether markers should rotate to follow the line bearing.

        Mirrors QGIS ``rotateSymbols()``; defaults to ``True`` (QGIS's own
        default) if the property cannot be read.
        """
        try:
            return bool(symbol_layer.rotateSymbols())
        except (AttributeError, RuntimeError):
            return True

    @staticmethod
    def get_marker_line_offset(symbol_layer: QgsSymbolLayer) -> float:
        """Return the marker-line's perpendicular offset from the line, in pixels.

        Positive QGIS offsets shift markers to the left of the line
        direction; ``0`` is returned (and the caller should omit
        ``icon-offset``) when no offset is configured.
        """
        try:
            offset = float(symbol_layer.offset())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return 0.0
        if offset == 0:
            return 0.0
        unit = PropertyExtractor.get_attribute(symbol_layer, "offsetUnit", "offsetUnits")
        return PropertyExtractor.convert_length_to_pixels(offset, unit)


class FillPropertyExtractor:
    """Extract fill paint and layout properties from QGIS fill symbol layers."""

    @staticmethod
    def get_fill_color(symbol_layer: QgsSimpleFillSymbolLayer) -> Union[str, List]:
        """Return ``fill-color`` resolving any data-defined override."""
        base_color = PropertyExtractor.convert_qcolor_to_maplibre(symbol_layer.color())
        color_prop = symbol_layer.dataDefinedProperties().property(QgsSymbolLayer.Property.PropertyFillColor)
        return PropertyExtractor.color_value_or_expression(base_color, color_prop)

    @staticmethod
    def get_fill_opacity(
        symbol_layer: QgsSimpleFillSymbolLayer, symbol: "QgsSymbol" = None
    ) -> Union[float, List]:
        """Return ``fill-opacity`` from the symbol's general Opacity setting.

        Independent of the fill colour's alpha channel, for the same reason as
        ``get_line_opacity``: colour alpha lives in ``fill-color``, and
        ``symbol.opacity()`` is QGIS's separate general "Opacity" slider.
        Honours a data-defined ``Property.PropertyOpacity`` override when active.
        """
        try:
            base_opacity = symbol.opacity() if symbol is not None else 1.0
        except (AttributeError, RuntimeError):
            base_opacity = 1.0
        try:
            opacity_prop = symbol_layer.dataDefinedProperties().property(
                QgsSymbolLayer.Property.PropertyOpacity
            )
            return PropertyExtractor.opacity_value_or_expression(base_opacity, opacity_prop)
        except (AttributeError, RuntimeError):
            return base_opacity

    @staticmethod
    def get_fill_outline_color(
        symbol_layer: QgsSimpleFillSymbolLayer,
    ) -> Union[str, List, None]:
        """Return ``fill-outline-color`` if the polygon stroke is visible."""
        try:
            stroke_visible = symbol_layer.strokeWidth() > 0 and symbol_layer.strokeStyle() != 0
        except (AttributeError, RuntimeError):
            stroke_visible = False
        if stroke_visible:
            base_color = PropertyExtractor.convert_qcolor_to_maplibre(symbol_layer.strokeColor())
            color_prop = symbol_layer.dataDefinedProperties().property(
                QgsSymbolLayer.Property.PropertyStrokeColor
            )
            return PropertyExtractor.color_value_or_expression(base_color, color_prop)
        return None

    @staticmethod
    def get_fill_antialias() -> bool:
        """Return ``fill-antialias`` (MapLibre default: ``True``)."""
        return True

    @staticmethod
    def get_fill_translate(symbol_layer: QgsSymbolLayer = None) -> Union[List, None]:
        """Return ``fill-translate``: the fill's x/y offset in pixels (a zoom
        interpolation for a ground offset), else ``[0, 0]``."""
        offset = fill_offset(symbol_layer)
        return offset if offset is not None else [0, 0]

    @staticmethod
    def get_fill_translate_anchor(symbol_layer: QgsSymbolLayer = None) -> str:
        """Return ``fill-translate-anchor``: "viewport" for an offset fill
        (QGIS offsets on screen, not turned with the map), else MapLibre's
        default "map"."""
        return "viewport" if fill_offset(symbol_layer) is not None else "map"

    @staticmethod
    def get_fill_sort_key(symbol_layer: QgsSymbolLayer) -> Union[float, List]:
        """Return ``fill-sort-key``: 0 (see LinePropertyExtractor.get_line_sort_key)."""
        return 0

    @staticmethod
    def is_pattern_fill(symbol_layer: QgsSymbolLayer) -> bool:
        """Return ``True`` for pattern-based fill layers (point/line/raster/SVG)."""
        pattern_class_names = {
            "QgsPointPatternFillSymbolLayer",
            "QgsLinePatternFillSymbolLayer",
            "QgsRasterFillSymbolLayer",
            "QgsSVGFillSymbolLayer",
            "QgsRandomMarkerFillSymbolLayer",
        }
        return type(symbol_layer).__name__ in pattern_class_names

    @staticmethod
    def get_fallback_fill_color(symbol_layer: QgsSymbolLayer) -> str:
        """A colour approximating a fill MapLibre can't draw as QGIS does.

        Gradient and shapeburst fills vary across each feature, which a
        MapLibre fill can't; their mean colour is the closest flat colour.
        Other fills use their own colour, never MapLibre's default black.
        """
        kind = type(symbol_layer).__name__
        if kind == "QgsGradientFillSymbolLayer":
            color = _ramp_or_colors(symbol_layer, 0)  # SimpleTwoColor
        elif kind == "QgsShapeburstFillSymbolLayer":
            color = _ramp_or_colors(symbol_layer, 0)
        else:
            try:
                color = symbol_layer.color()
            except (AttributeError, RuntimeError):
                color = QColor(0, 0, 0, 0)
        if color is None or not color.isValid():
            color = QColor(0, 0, 0, 0)
        return PropertyExtractor.convert_qcolor_to_maplibre(color)

    # Tile size of a non-repeating pattern (random markers), in pixels.
    _RANDOM_PATTERN_TILE = 128.0

    @staticmethod
    def get_pattern_tile_size(symbol_layer: QgsSymbolLayer) -> tuple:
        """The pattern's repeat (width, height) in pixels: one tile of ``fill-pattern``.

        A tile of exactly one repeat makes MapLibre's repeated image seamless.
        Point grids repeat every distance (twice that along a row
        displacement); parallel lines every distance / sin or cos of their
        angle; SVG and raster images every image width and its height.
        """
        px = PropertyExtractor.convert_length_to_pixels
        get = PropertyExtractor.get_attribute
        kind = type(symbol_layer).__name__
        try:
            if kind == "QgsPointPatternFillSymbolLayer":
                width = px(symbol_layer.distanceX(), get(symbol_layer, "distanceXUnit"))
                height = px(symbol_layer.distanceY(), get(symbol_layer, "distanceYUnit"))
                if symbol_layer.displacementX():
                    height *= 2
                if symbol_layer.displacementY():
                    width *= 2
                return width, height
            if kind == "QgsLinePatternFillSymbolLayer":
                distance = px(symbol_layer.distance(), get(symbol_layer, "distanceUnit"))
                angle = math.radians(symbol_layer.lineAngle())
                sin, cos = abs(math.sin(angle)), abs(math.cos(angle))
                # A tile edge parallel to the lines can be any length: use the distance.
                width = distance / sin if sin > 1e-3 else distance
                height = distance / cos if cos > 1e-3 else distance
                return width, height
            if kind == "QgsSVGFillSymbolLayer":
                from qgis.PyQt.QtSvg import QSvgRenderer
                width = px(symbol_layer.patternWidth(), get(symbol_layer, "patternWidthUnit"))
                box = QSvgRenderer(symbol_layer.svgFilePath()).viewBoxF()
                ratio = box.height() / box.width() if box.width() > 0 else 1.0
                return width, width * ratio
            if kind == "QgsRasterFillSymbolLayer":
                from qgis.PyQt.QtGui import QImageReader
                size = QImageReader(symbol_layer.imageFilePath()).size()
                width = symbol_layer.width()
                width = px(width, get(symbol_layer, "sizeUnit", "widthUnit")) if width > 0 else size.width()
                ratio = size.height() / size.width() if size.width() > 0 else 1.0
                return width, width * ratio
        except (AttributeError, RuntimeError, TypeError, ValueError, ImportError):
            pass
        return FillPropertyExtractor._RANDOM_PATTERN_TILE, FillPropertyExtractor._RANDOM_PATTERN_TILE

    @staticmethod
    def get_fill_pattern_path(symbol_layer: QgsSymbolLayer) -> Optional[str]:
        """Return the source image path for a raster/SVG fill pattern, or ``None``."""
        path = PropertyExtractor.get_attribute(symbol_layer, "imageFilePath", "svgFilePath", "path")
        if isinstance(path, str) and path:
            return path
        return None


class IconPropertyExtractor:
    """Extract icon paint and layout properties for MapLibre symbol layers."""

    @staticmethod
    def get_icon_image(marker_name: str) -> str:
        """Return the registered sprite name for ``icon-image``."""
        return marker_name

    @staticmethod
    def get_icon_size(
        symbol_layer: QgsSymbolLayer, default_size: float = 1.0
    ) -> Union[float, List]:
        """Return ``icon-size`` honouring any data-defined override.

        The sprite draws the marker at its own size (times _SPRITE_QUALITY).
        A data-defined size is a marker size, not a scale, so the icon is
        scaled by its ratio to the marker's own size: a constant becomes a
        number, a field a MapLibre expression.
        """
        size_prop = symbol_layer.dataDefinedProperties().property(QgsSymbolLayer.Property.PropertySize)
        size_val = PropertyExtractor.get_value_or_expression(default_size, size_prop)
        if size_val is default_size:
            return default_size / _SPRITE_QUALITY
        try:
            base = float(symbol_layer.size())
        except (AttributeError, TypeError, ValueError):
            base = 0.0
        if base <= 0:
            return default_size / _SPRITE_QUALITY
        if isinstance(size_val, list):
            return ["/", ["to-number", size_val, base], base * _SPRITE_QUALITY]
        try:
            return float(size_val) / base / _SPRITE_QUALITY
        except (TypeError, ValueError):
            return default_size / _SPRITE_QUALITY

    @staticmethod
    def get_icon_rotate(
        symbol_layer: QgsSymbolLayer = None,
        background: QgsTextBackgroundSettings = None,
    ) -> Union[float, List]:
        """Return ``icon-rotate`` in degrees, honouring data-defined overrides.

        When given a marker symbol layer, the static rotation is taken from
        ``angle()`` and any active ``PropertyAngle`` data-defined override is
        applied. When given a label background, the background rotation is
        used instead.
        """
        if symbol_layer is not None:
            base_angle = 0.0
            try:
                base_angle = float(symbol_layer.angle())
            except (AttributeError, RuntimeError, TypeError):
                pass
            try:
                angle_prop = symbol_layer.dataDefinedProperties().property(
                    QgsSymbolLayer.Property.PropertyAngle
                )
                return PropertyExtractor.get_value_or_expression(base_angle, angle_prop)
            except (AttributeError, RuntimeError):
                return base_angle
        if background is not None and background.enabled():
            try:
                return float(background.rotation())
            except (AttributeError, RuntimeError, TypeError):
                return 0.0
        return 0.0

    @staticmethod
    def get_icon_padding() -> float:
        """Return ``icon-padding`` in pixels (MapLibre default: 2)."""
        return 1

    @staticmethod
    def get_icon_rotation_alignment() -> str:
        """Return ``icon-rotation-alignment`` (default: ``"map"`` for QGIS markers)."""
        return "map"

    @staticmethod
    def get_icon_pitch_alignment() -> str:
        """Return ``icon-pitch-alignment`` (default: ``"viewport"``)."""
        return "viewport"

    @staticmethod
    def get_icon_anchor() -> str:
        """Return ``icon-anchor`` (default: ``"center"``)."""
        return "center"

    @staticmethod
    def get_icon_allow_overlap(allow_overlap: bool = False) -> bool:
        """Return ``icon-allow-overlap`` (MapLibre default: ``False``)."""
        return allow_overlap

    @staticmethod
    def get_icon_ignore_placement() -> bool:
        """Return ``icon-ignore-placement`` (MapLibre default: ``False``)."""
        return False

    @staticmethod
    def get_icon_optional() -> bool:
        """Return ``icon-optional`` (MapLibre default: ``False``)."""
        return False

    @staticmethod
    def get_icon_keep_upright() -> bool:
        """Return ``icon-keep-upright``.

        Defaults to ``True`` to match QGIS rendering for line-aligned markers.
        """
        return True

    @staticmethod
    def get_icon_text_fit(background: QgsTextBackgroundSettings) -> Optional[str]:
        """Return ``icon-text-fit`` if the background is sized as a text buffer."""
        if background.enabled() and background.sizeType() == 0:
            return "both"
        return None

    @staticmethod
    def get_icon_text_fit_padding(
        background: QgsTextBackgroundSettings,
    ) -> Optional[List[float]]:
        """Return ``icon-text-fit-padding`` ``[top, right, bottom, left]`` from buffer size."""
        if background.enabled() and background.sizeType() == 0:
            buf_px = PropertyExtractor.convert_length_to_pixels(
                background.size().width(), background.sizeUnit()
            )
            buf_px = max(buf_px, 3)  
            return [buf_px * 2, buf_px, buf_px * 2, buf_px]
        return None

    @staticmethod
    def get_icon_offset(background: QgsTextBackgroundSettings = None) -> List[float]:
        """Return ``icon-offset`` ``[x, y]`` in pixels from background offset settings."""
        if background and background.enabled():
            offset = background.offset()
            unit = background.offsetUnit()
            return [
                PropertyExtractor.convert_length_to_pixels(offset.x(), unit),
                PropertyExtractor.convert_length_to_pixels(offset.y(), unit),
            ]
        return [0, 0]

    @staticmethod
    def get_icon_opacity(background: QgsTextBackgroundSettings = None) -> float:
        """Return ``icon-opacity`` from background settings (default ``1.0``)."""
        if background and background.enabled():
            try:
                return background.opacity()
            except (OSError, RuntimeError):
                pass
        return 1.0

    @staticmethod
    def get_icon_color(background: QgsTextBackgroundSettings = None) -> str:
        """Return ``icon-color`` from background fill colour (default white)."""
        if background and background.enabled():
            try:
                return PropertyExtractor.convert_qcolor_to_maplibre(background.fillColor())
            except (OSError, RuntimeError):
                pass
        return "rgb(255, 255, 255)"

    @staticmethod
    def get_icon_halo_color(background: QgsTextBackgroundSettings = None) -> str:
        """Return ``icon-halo-color``.

        When a label background is provided and has a stroke, the stroke
        colour is used; otherwise black is returned to match the MapLibre
        default.
        """
        if background and background.enabled():
            try:
                return PropertyExtractor.convert_qcolor_to_maplibre(background.strokeColor())
            except (OSError, RuntimeError, AttributeError):
                pass
        return "rgb(0, 0, 0)"

    @staticmethod
    def get_icon_halo_width(background: QgsTextBackgroundSettings = None) -> float:
        """Return ``icon-halo-width`` in pixels from the background stroke width."""
        if background and background.enabled():
            try:
                return PropertyExtractor.convert_length_to_pixels(
                    background.strokeWidth(), background.strokeWidthUnit()
                )
            except (OSError, RuntimeError, AttributeError):
                pass
        return 0

    @staticmethod
    def get_icon_halo_blur() -> float:
        """Return ``icon-halo-blur`` in pixels (default 0)."""
        return 0

    @staticmethod
    def get_icon_translate() -> List[float]:
        """Return ``icon-translate`` ``[x, y]`` offset in pixels (default ``[0, 0]``)."""
        return [0, 0]

    @staticmethod
    def get_icon_translate_anchor() -> str:
        """Return ``icon-translate-anchor`` (MapLibre default: ``"map"``)."""
        return "map"

    @staticmethod
    def get_symbol_placement(label_settings: QgsPalLayerSettings = None) -> str:
        """Return ``symbol-placement`` (``"point"``, ``"line"``, or ``"line-center"``).

        Mapped from ``QgsPalLayerSettings.placement``: line/curved/perimeter
        placements follow the line, repeated along it ("line") when QGIS
        repeats them (a repeat distance), else once in its middle
        ("line-center"), as QGIS labels each line once; everything else is
        "point" (outside-polygon labels too: they are placed around the
        polygon's point, see get_text_variable_anchor).
        """
        if label_settings is None:
            return "point"
        try:
            placement = PropertyExtractor.enum_int(label_settings.placement)
        except AttributeError:
            return "point"
        # QgsPalLayerSettings placement enum: 2=Line, 3=Curved, 7=PerimeterCurved
        if placement in TextPropertyExtractor._LINE_PLACEMENTS:
            try:
                repeats = (label_settings.repeatDistance or 0) > 0
            except (AttributeError, RuntimeError):
                repeats = True
            return "line" if repeats else "line-center"
        return "point"

    @staticmethod
    def get_symbol_spacing(label_settings: QgsPalLayerSettings = None) -> float:
        """Return ``symbol-spacing`` in pixels (MapLibre default: 250).

        Sourced from ``label_settings.repeatDistance`` when set; otherwise
        the MapLibre default of 250 px is returned.
        """
        if label_settings is None:
            return 250.0
        try:
            distance = label_settings.repeatDistance
            if distance and distance > 0:
                unit = PropertyExtractor.get_attribute(
                    label_settings, "repeatDistanceUnit", "distUnits"
                )
                return PropertyExtractor.convert_length_to_pixels(distance, unit)
        except (AttributeError, RuntimeError):
            pass
        return 250.0

    @staticmethod
    def get_symbol_avoid_edges() -> bool:
        """Return ``symbol-avoid-edges`` (MapLibre default: ``False``)."""
        return False

    @staticmethod
    def get_symbol_sort_key() -> float:
        """Return ``symbol-sort-key`` (default 0; lower keys render first)."""
        return 0

    @staticmethod
    def get_symbol_z_order() -> str:
        """Return ``symbol-z-order`` (default ``"auto"``)."""
        return "auto"


class TextPropertyExtractor:
    """Extract text paint and layout properties from QGIS label settings."""

    # Widest halo MapLibre draws cleanly, in ems (see get_text_halo_width).
    _HALO_MAX_EM = 0.2

    @staticmethod
    def get_text_field(label_settings: QgsPalLayerSettings) -> Optional[List]:
        """Return ``text-field`` as a MapLibre ``["get", field]`` expression."""
        if label_settings.fieldName:
            return ["get", label_settings.fieldName]
        return None

    @staticmethod
    def get_text_font(text_format: QgsTextFormat) -> List[str]:
        """Return ``text-font`` as a single-element list of ``"family style"``."""
        font = text_format.font()
        font_str = f"{font.family()} {font.styleName()}"
        return font_str

    @staticmethod
    def get_text_size(
        text_format: QgsTextFormat, label_settings: QgsPalLayerSettings
    ) -> Union[float, List]:
        """Return ``text-size`` in pixels, honouring data-defined overrides."""
        size_prop = label_settings.dataDefinedProperties().property(QgsPalLayerSettings.Property.Size)
        return PropertyExtractor.length_value_or_expression(
            text_format.size(), text_format.sizeUnit(), size_prop, text_format.sizeMapUnitScale()
        )

    @staticmethod
    def get_text_size_px(text_format: QgsTextFormat) -> float:
        """The label's font size in pixels, to turn pixel lengths into ems.

        A size in map units has no single pixel size; 16 px stands in for it.
        """
        if PropertyExtractor.is_map_unit(text_format.sizeUnit()):
            return 16.0
        size = PropertyExtractor.convert_length_to_pixels(text_format.size(), text_format.sizeUnit())
        return size if size and size > 0 else 16.0

    @staticmethod
    def get_text_color(
        text_format: QgsTextFormat, label_settings: QgsPalLayerSettings
    ) -> Union[str, List]:
        """Return ``text-color`` honouring any data-defined override."""
        base_color = PropertyExtractor.convert_qcolor_to_maplibre(text_format.color())
        color_prop = label_settings.dataDefinedProperties().property(QgsPalLayerSettings.Property.Color)
        return PropertyExtractor.color_value_or_expression(base_color, color_prop)

    @staticmethod
    def get_text_opacity(
        text_format: QgsTextFormat,
        label_settings: QgsPalLayerSettings = None,
    ) -> Union[float, List]:
        """Return ``text-opacity`` from the text format's general Opacity setting.

        Independent of the font colour's alpha channel, same reasoning as the
        fill/line variants. ``text_format.opacity()`` is QGIS's own general
        text "Opacity" slider. Honours a data-defined ``FontOpacity`` override
        on ``label_settings`` when active. When the property does not exist on
        the running QGIS build, the static opacity value is returned.
        """
        try:
            base_opacity = text_format.opacity()
        except (AttributeError, RuntimeError):
            base_opacity = 1.0
        if label_settings is None:
            return base_opacity
        try:
            opacity_prop = label_settings.dataDefinedProperties().property(
                QgsPalLayerSettings.Property.FontOpacity
            )
            return PropertyExtractor.opacity_value_or_expression(base_opacity, opacity_prop)
        except (AttributeError, RuntimeError):
            return base_opacity

    @staticmethod
    def get_text_halo_color(
        text_format: QgsTextFormat,
        label_settings: QgsPalLayerSettings = None,
    ) -> Union[str, List]:
        """Return ``text-halo-color`` from buffer settings.

        Honours a data-defined ``BufferColor`` override on ``label_settings``
        when active, so per-feature halo colour attributes are emitted as
        MapLibre ``["get", field]`` expressions rather than the hard-coded
        static buffer colour. Falls back to white when the buffer is
        disabled and no override is supplied.
        """
        buffer = text_format.buffer()
        if not buffer.enabled():
            return "rgb(255, 255, 255)"
        base_color = PropertyExtractor.convert_qcolor_to_maplibre(buffer.color())
        if label_settings is None:
            return base_color
        try:
            color_prop = label_settings.dataDefinedProperties().property(
                QgsPalLayerSettings.Property.BufferColor
            )
            return PropertyExtractor.color_value_or_expression(base_color, color_prop)
        except (AttributeError, RuntimeError):
            return base_color

    @staticmethod
    def get_text_halo_width(
        text_format: QgsTextFormat,
        label_settings: QgsPalLayerSettings = None,
    ) -> Union[float, List]:
        """Return ``text-halo-width`` in pixels from the buffer size.

        Honours a data-defined ``BufferSize`` override on ``label_settings``
        when active. Field-based overrides are returned as MapLibre
        expressions; the user is responsible for storing pixel-equivalent
        values in the source attribute since MapLibre expressions cannot
        express the QGIS unit conversion at evaluation time.
        """
        buffer = text_format.buffer()
        if not buffer.enabled():
            return 0
        # MapLibre draws halos up to about a fifth of the text size; wider
        # ones reach the edge of the glyphs' distance field and fill each
        # letter's whole box.
        cap = TextPropertyExtractor._HALO_MAX_EM * TextPropertyExtractor.get_text_size_px(text_format)
        size_prop = None
        if label_settings is not None:
            try:
                size_prop = label_settings.dataDefinedProperties().property(
                    QgsPalLayerSettings.Property.BufferSize
                )
            except (AttributeError, RuntimeError):
                size_prop = None
        # Both are the halo's width outside the letters.
        try:
            width = PropertyExtractor.length_value_or_expression(
                buffer.size(), buffer.sizeUnit(), size_prop, buffer.sizeMapUnitScale()
            )
        except (AttributeError, RuntimeError):
            width = buffer.size()
        if isinstance(width, (int, float)):
            return min(width, cap)
        if not PropertyExtractor.uses_zoom(width):
            return ["min", width, cap]
        return width

    @staticmethod
    def get_text_halo_blur(
        text_format: QgsTextFormat,
        label_settings: QgsPalLayerSettings = None,
    ) -> Union[float, List]:
        """Return ``text-halo-blur`` in pixels.

        QGIS exposes a separate buffer blur radius via ``blurRadius`` when
        available; otherwise half the buffer size is used as a sensible
        approximation. Honours the data-defined ``BufferBlurRadius``
        override on ``label_settings`` when the QGIS build supports it.
        """
        buffer = text_format.buffer()
        if not buffer.enabled():
            return 0
        try:
            blur = buffer.blurRadius()
            if blur is not None:
                base_blur = PropertyExtractor.convert_length_to_pixels(
                    blur, buffer.blurRadiusUnit()
                )
            else:
                base_blur = buffer.size() * 0.5
        except (AttributeError, RuntimeError):
            base_blur = buffer.size() * 0.5

        if label_settings is None:
            return base_blur
        try:
            prop_key = getattr(QgsPalLayerSettings, "BufferBlurRadius")
        except AttributeError:
            return base_blur
        try:
            blur_prop = label_settings.dataDefinedProperties().property(prop_key)
            return PropertyExtractor.get_value_or_expression(base_blur, blur_prop)
        except (AttributeError, RuntimeError):
            return base_blur

    @staticmethod
    def get_text_anchor(label_settings: QgsPalLayerSettings) -> str:
        """Return ``text-anchor`` mapped from the QGIS quadrant offset."""
        anchor_map = {
            0: "bottom-right", 1: "bottom",  2: "bottom-left",
            3: "right",        4: "center",  5: "left",
            6: "top-right",    7: "top",     8: "top-left",
        }
        return anchor_map.get(PropertyExtractor.enum_int(label_settings.quadOffset), "center")

    # QGIS's label alignment at a data-defined position (horizontal: Left,
    # Center, Right; vertical: Bottom, Base, Half, Cap, Top) as MapLibre
    # anchor parts.
    _HALI = {"left": "left", "center": "", "right": "right"}
    _VALI = {"bottom": "bottom", "base": "bottom", "half": "", "cap": "top", "top": "top"}

    @staticmethod
    def _anchor_name(vertical: str, horizontal: str) -> str:
        """A MapLibre anchor from its parts ("" for the middle)."""
        return "-".join(part for part in (vertical, horizontal) if part) or "center"

    @staticmethod
    def get_pinned_anchor(label_settings: QgsPalLayerSettings) -> Union[None, str, List]:
        """``text-anchor`` of labels at a data-defined position (X, Y): the
        label's corner QGIS puts there, from its data-defined alignment
        (horizontal and vertical, static or per feature), by default its
        bottom left; None without a position."""
        props = label_settings.dataDefinedProperties()
        x = props.property(QgsPalLayerSettings.Property.PositionX)
        y = props.property(QgsPalLayerSettings.Property.PositionY)
        if not (x and y and x.isActive() and y.isActive()):
            return None
        parts = []
        for key, names, default in ((QgsPalLayerSettings.Property.Vali, TextPropertyExtractor._VALI, "bottom"),
                                    (QgsPalLayerSettings.Property.Hali, TextPropertyExtractor._HALI, "left")):
            prop = props.property(key)
            value = PropertyExtractor.get_value_or_expression(None, prop) if prop and prop.isActive() else None
            if isinstance(value, list):
                # Per feature: QGIS reads the name in any case; others give the default.
                cases = ["match", ["downcase", ["to-string", ["coalesce", value, ""]]]]
                for name, part in names.items():
                    cases += [name, part]
                parts.append(cases + [default])
            else:
                parts.append(names.get(str(value).lower(), default) if value is not None else default)
        vertical, horizontal = parts
        if isinstance(vertical, str) and isinstance(horizontal, str):
            return TextPropertyExtractor._anchor_name(vertical, horizontal)
        # Every combination of the parts, as a match on them.
        vertical_cases = vertical if isinstance(vertical, list) else ["literal", vertical]
        horizontal_cases = horizontal if isinstance(horizontal, list) else ["literal", horizontal]
        expression = ["match", ["concat", vertical_cases, "|", horizontal_cases]]
        for v in ("bottom", "", "top"):
            for h in ("left", "", "right"):
                expression += [f"{v}|{h}", TextPropertyExtractor._anchor_name(v, h)]
        return expression + ["bottom-left"]

    @staticmethod
    def pinned_anchor_expression(label_settings: QgsPalLayerSettings) -> str:
        """get_pinned_anchor as a QGIS expression (the label's anchor name per
        feature, from the settings' own data-defined alignment)."""
        props = label_settings.dataDefinedProperties()
        parts = []
        for key, names, default in ((QgsPalLayerSettings.Property.Vali, TextPropertyExtractor._VALI, "bottom"),
                                    (QgsPalLayerSettings.Property.Hali, TextPropertyExtractor._HALI, "left")):
            prop = props.property(key)
            if not (prop and prop.isActive()):
                parts.append(QgsExpression.quotedString(default))
                continue
            value = f"lower(to_string(({prop.asExpression()})))"
            cases = " ".join(f"WHEN {value} = {QgsExpression.quotedString(name)} "
                             f"THEN {QgsExpression.quotedString(part)}" for name, part in names.items())
            parts.append(f"CASE {cases} ELSE {QgsExpression.quotedString(default)} END")
        vertical, horizontal = parts
        joined = f"array_to_string(array_filter(array({vertical}, {horizontal}), @element <> ''), '-')"
        return f"coalesce(nullif({joined}, ''), 'center')"

    @staticmethod
    def get_text_justify(label_settings: QgsPalLayerSettings) -> Union[str, List]:
        """Return ``text-justify`` mapped from the multi-line alignment setting.

        Honours a data-defined ``MultiLineAlignment`` override when active.
        For field-based overrides the source attribute is expected to
        contain the literal MapLibre keyword (``"left"``, ``"center"``,
        ``"right"`` or ``"auto"``); QGIS-specific integer/string codes are
        not translated server-side.
        """
        justification = getattr(label_settings, "multilineAlign", 1)
        # Qgis.LabelMultiLineAlignment: left, center, right, follow the
        # placement (MapLibre's auto: the anchor's side), justify.
        justify_map = {0: "left", 1: "center", 2: "right", 3: "auto", 4: "left"}
        base_justify = justify_map.get(PropertyExtractor.enum_int(justification), "left")
        try:
            justify_prop = label_settings.dataDefinedProperties().property(
                QgsPalLayerSettings.Property.MultiLineAlignment
            )
            return PropertyExtractor.get_value_or_expression(base_justify, justify_prop)
        except (AttributeError, RuntimeError):
            return base_justify

    # QGIS's "Offset from point" placement, the only one that uses xOffset/yOffset.
    _OVER_POINT = 1

    # Line label placement flags (Qgis.LabelLinePlacementFlag).
    _ON_LINE, _ABOVE_LINE, _BELOW_LINE = 1, 2, 4
    # Half a line of text, in ems: from a line label's centre to its edge.
    _HALF_LINE_EM = 0.6
    # Placements of labels along lines (Line, Curved, Perimeter curved).
    _LINE_PLACEMENTS = (2, 3, 7)

    @staticmethod
    def get_text_offset(label_settings: QgsPalLayerSettings) -> List[float]:
        """Return ``text-offset`` ``[x, y]`` in ems (y down, as in QGIS).

        "Offset from point" labels use their X / Y offset. Line labels
        placed above or below (not on) the line are moved off it by half a
        line of text plus their distance, as QGIS places them.
        """
        em = TextPropertyExtractor.get_text_size_px(label_settings.format())
        placement = PropertyExtractor.enum_int(label_settings.placement)
        if placement == TextPropertyExtractor._OVER_POINT:
            x_offset = label_settings.xOffset
            y_offset = label_settings.yOffset
            if x_offset != 0 or y_offset != 0:
                offset_unit = PropertyExtractor.get_attribute(label_settings, "offsetUnits")
                return [
                    PropertyExtractor.convert_length_to_pixels(x_offset, offset_unit) / em,
                    PropertyExtractor.convert_length_to_pixels(y_offset, offset_unit) / em,
                ]
            return [0, 0]
        if placement in TextPropertyExtractor._LINE_PLACEMENTS:
            try:
                flags = PropertyExtractor.enum_int(label_settings.lineSettings().placementFlags())
            except (AttributeError, RuntimeError, TypeError):
                return [0, 0]
            if flags is None or flags & TextPropertyExtractor._ON_LINE:
                return [0, 0]
            shift = TextPropertyExtractor._HALF_LINE_EM + TextPropertyExtractor._distance_em(
                label_settings, em)
            if flags & TextPropertyExtractor._ABOVE_LINE:
                return [0, -shift]
            if flags & TextPropertyExtractor._BELOW_LINE:
                return [0, shift]
        return [0, 0]

    @staticmethod
    def _distance_em(label_settings: QgsPalLayerSettings, em: float) -> float:
        """The label's distance setting, in ems of its text."""
        try:
            distance = float(label_settings.dist)
        except (AttributeError, TypeError, ValueError):
            return 0.0
        if not distance:
            return 0.0
        unit = PropertyExtractor.get_attribute(label_settings, "distUnits", "distUnit")
        return PropertyExtractor.convert_length_to_pixels(distance, unit) / em

    @staticmethod
    def get_text_radial_offset(label_settings: QgsPalLayerSettings) -> float:
        """Return ``text-radial-offset`` in ems: the label's distance
        setting (QGIS puts a label around a point with its side or corner
        on the point, whatever the marker, then that far from it)."""
        em = TextPropertyExtractor.get_text_size_px(label_settings.format())
        return TextPropertyExtractor._distance_em(label_settings, em)

    # QGIS's "around point", "cartographic" (ordered positions) and
    # "outside polygons" placements; the last is drawn around the polygon's
    # point (MapLibre can't place a label outside a polygon's outline).
    _AROUND_POINT, _ORDERED_POSITIONS, _OUTSIDE_POLYGONS = 0, 6, 8
    # QGIS's candidate order around a point: above right first.
    _AROUND_POINT_ANCHORS = [
        "bottom-left", "bottom-right", "top-left", "top-right", "left", "right", "bottom", "top",
    ]
    # Anchor (the side of the label at the point) per
    # Qgis.LabelPredefinedPointPosition: TopLeft, TopSlightlyLeft, TopMiddle,
    # TopSlightlyRight, TopRight, MiddleLeft, MiddleRight, BottomLeft,
    # BottomSlightlyLeft, BottomMiddle, BottomSlightlyRight, BottomRight,
    # OverPoint.
    _POSITION_ANCHORS = [
        "bottom-right", "bottom-right", "bottom", "bottom-left", "bottom-left", "right", "left",
        "top-right", "top-right", "top", "top-left", "top-left", "center",
    ]

    @staticmethod
    def get_text_variable_anchor(label_settings: QgsPalLayerSettings = None) -> Optional[List[str]]:
        """Return ``text-variable-anchor`` when QGIS chooses the label's side.

        "Around point" tries every side, above right first; "cartographic"
        tries its own position order. Other placements put the label at a
        fixed place (None: the static ``text-anchor``).
        """
        if label_settings is None:
            return None
        placement = PropertyExtractor.enum_int(label_settings.placement)
        if placement in (TextPropertyExtractor._AROUND_POINT, TextPropertyExtractor._OUTSIDE_POLYGONS):
            return list(TextPropertyExtractor._AROUND_POINT_ANCHORS)
        if placement != TextPropertyExtractor._ORDERED_POSITIONS:
            return None
        anchors = []
        try:
            positions = label_settings.predefinedPositionOrder()
        except (AttributeError, RuntimeError):
            positions = []
        for position in positions:
            index = PropertyExtractor.enum_int(position)
            if index is not None and 0 <= index < len(TextPropertyExtractor._POSITION_ANCHORS):
                anchor = TextPropertyExtractor._POSITION_ANCHORS[index]
                if anchor not in anchors:
                    anchors.append(anchor)
        return anchors or list(TextPropertyExtractor._AROUND_POINT_ANCHORS)

    @staticmethod
    def _positioned(label_settings: QgsPalLayerSettings) -> bool:
        """Whether the label is at a data-defined position (X and Y)."""
        props = label_settings.dataDefinedProperties()
        x = props.property(QgsPalLayerSettings.Property.PositionX)
        y = props.property(QgsPalLayerSettings.Property.PositionY)
        return bool(x and y and x.isActive() and y.isActive())

    @staticmethod
    def fixed_anchor(label_settings: QgsPalLayerSettings) -> tuple:
        """(text-anchor, text-offset in ems) of a point's label at its point:
        the style's, or the first side of one placed around it (a label
        with a callout keeps it: keeps_callout_side). The offset may be
        data-defined (not numbers)."""
        anchor = TextPropertyExtractor.get_text_anchor(label_settings)
        offset = TextPropertyExtractor.get_text_offset(label_settings)
        variable = TextPropertyExtractor.get_text_variable_anchor(label_settings)
        if variable:
            anchor = variable[0]
            offset = TextPropertyExtractor.radial_to_offset(
                anchor, TextPropertyExtractor.get_text_radial_offset(label_settings))
        return anchor, offset

    @staticmethod
    def balloon_icon(label_settings: QgsPalLayerSettings, geometry_type=None) -> bool:
        """Whether the label's balloon callout is its icon, stretched around
        the text (see BalloonShape): a point's label at its point, at a
        fixed offset, without a background shape. MapLibre then fits it to
        the text at every zoom, and hides it with its label as QGIS does."""
        callout = label_settings.callout()
        if (callout is None or not callout.enabled() or callout.type() != "balloon"
                or callout.fillSymbol() is None or label_settings.format().background().enabled()):
            return False
        if geometry_type is None or PropertyExtractor.enum_int(geometry_type) != PropertyExtractor.enum_int(
                Qgis.GeometryType.Point):
            return False
        if TextPropertyExtractor._positioned(label_settings):
            return False
        _, offset = TextPropertyExtractor.fixed_anchor(label_settings)
        return all(isinstance(v, (int, float)) for v in offset)

    @staticmethod
    def has_drawn_callout(label_settings: QgsPalLayerSettings, geometry_type=None) -> bool:
        """Whether the label's callout is drawn in the tiles (RulesFlattener):
        a point's label (but a balloon that is its icon), or one at a
        data-defined position."""
        callout = label_settings.callout()
        if callout is None or not callout.enabled():
            return False
        if TextPropertyExtractor._positioned(label_settings):
            return True
        return (geometry_type is not None and PropertyExtractor.enum_int(geometry_type)
                == PropertyExtractor.enum_int(Qgis.GeometryType.Point)
                and not TextPropertyExtractor.balloon_icon(label_settings, geometry_type))

    @staticmethod
    def keeps_callout_side(label_settings: QgsPalLayerSettings) -> bool:
        """Whether a point's label placed around it keeps its first side:
        it has a callout (drawn in the tiles, see RulesFlattener) and no
        data-defined position, so the label stays where its callout is."""
        callout = label_settings.callout()
        if callout is None or not callout.enabled():
            return False
        props = label_settings.dataDefinedProperties()
        x = props.property(QgsPalLayerSettings.Property.PositionX)
        y = props.property(QgsPalLayerSettings.Property.PositionY)
        return (not (x and y and x.isActive() and y.isActive())
                and PropertyExtractor.enum_int(label_settings.placement) == TextPropertyExtractor._AROUND_POINT)

    @staticmethod
    def radial_to_offset(anchor: str, radial: float) -> List[float]:
        """``text-offset`` [x, y] (ems, y down) placing a label with this
        anchor ``radial`` ems from its point, as MapLibre's radial offset
        does: along the axis for a side, split over both for a corner."""
        x = 1 if "left" in anchor else -1 if "right" in anchor else 0
        y = 1 if "top" in anchor else -1 if "bottom" in anchor else 0
        if x and y:
            radial /= math.sqrt(2)
        return [x * radial, y * radial]

    @staticmethod
    def get_text_sort_key(label_settings: QgsPalLayerSettings) -> Union[float, List]:
        """Return ``symbol-sort-key`` from the label priority.

        QGIS places higher priorities (0-10) first; MapLibre places lower
        sort keys first. A data-defined priority becomes an expression,
        as QGIS reads it: limited to 0-10, the static priority when empty
        or not a number.
        """
        base = 10 - label_settings.priority
        try:
            prop = label_settings.dataDefinedProperties().property(
                QgsPalLayerSettings.Property.Priority
            )
        except (AttributeError, RuntimeError):
            return base
        resolved = PropertyExtractor.get_value_or_expression(None, prop)
        if resolved is None:
            return base
        if isinstance(resolved, list):
            static = label_settings.priority
            value = ["to-number", ["coalesce", resolved, static], static]
            return ["-", 10, ["max", 0, ["min", 10, value]]]
        try:
            return 10 - min(max(float(resolved), 0), 10)
        except (TypeError, ValueError):
            return base

    @staticmethod
    def get_text_max_angle(label_settings: QgsPalLayerSettings) -> float:
        """Return ``text-max-angle`` in degrees for curved-label placement.

        Sourced from ``maxCurvedCharAngleIn`` (with a fallback to
        ``maxCurvedCharAngleOut``); MapLibre defaults to 45° when omitted.
        """
        try:
            angle_in = label_settings.maxCurvedCharAngleIn
            if angle_in is not None:
                return abs(float(angle_in))
        except (AttributeError, RuntimeError, TypeError):
            pass
        try:
            angle_out = label_settings.maxCurvedCharAngleOut
            if angle_out is not None:
                return abs(float(angle_out))
        except (AttributeError, RuntimeError, TypeError):
            pass
        return 45.0

    @staticmethod
    def get_text_allow_overlap(label_settings: QgsPalLayerSettings = None) -> bool:
        """Return ``text-allow-overlap``: QGIS's "Allow overlaps without penalty"."""
        try:
            handling = label_settings.placementSettings().overlapHandling()
            return PropertyExtractor.enum_int(handling) == PropertyExtractor.enum_int(
                Qgis.LabelOverlapHandling.AllowOverlapAtNoCost
            )
        except (AttributeError, RuntimeError):
            return False

    @staticmethod
    def get_text_ignore_placement() -> bool:
        """Return ``text-ignore-placement`` (MapLibre default: ``False``)."""
        return False

    @staticmethod
    def get_text_optional() -> bool:
        """Return ``text-optional`` (MapLibre default: ``False``)."""
        return False

    @staticmethod
    def get_text_padding() -> float:
        """Return ``text-padding`` in pixels (MapLibre default: 2).

        QGIS lets labels touch, so the default keeps about as many labels.
        """
        return 2

    @staticmethod
    def label_box_metrics(text_format: QgsTextFormat) -> Optional[Tuple[float, float, float]]:
        """How QGIS lays out a label's lines (QgsTextDocumentMetrics), in
        ems: (its box's top to the first baseline, the step between
        baselines, the last baseline to the box's bottom); None for a size
        in map units."""
        if text_format is None or PropertyExtractor.is_map_unit(text_format.sizeUnit()):
            return None
        try:
            context = QgsRenderContext()
            context.setScaleFactor(96 / 25.4)
            document = QgsTextDocument.fromTextAndFormat(["H", "H"], text_format)
            metrics = QgsTextDocumentMetrics.calculateMetrics(document, text_format, context)
            mode = Qgis.TextLayoutMode.Labeling
            height = metrics.documentSize(mode, Qgis.TextOrientation.Horizontal).height()
            first, second = metrics.baselineOffset(0, mode), metrics.baselineOffset(1, mode)
            size = TextPropertyExtractor.get_text_size_px(text_format)
            if not size or second <= first:
                return None
            # Baselines are measured from the box's bottom (negative: above it).
            return (height + first) / size, (second - first) / size, -second / size
        except (AttributeError, RuntimeError, TypeError):
            return None

    @staticmethod
    def get_text_line_height(text_format: QgsTextFormat = None) -> float:
        """Return ``text-line-height`` in ems, as QGIS spaces lines: its
        own step between baselines (a multiple of the font's line spacing,
        Open Sans: 1.36 em, or a length)."""
        metrics = TextPropertyExtractor.label_box_metrics(text_format)
        if metrics is not None:
            return round(metrics[1], 4)
        try:
            font = QFont(text_format.font())
            font.setPixelSize(100)
            spacing = QFontMetricsF(font).lineSpacing() / 100
            if PropertyExtractor.enum_int(text_format.lineHeightUnit()) == PropertyExtractor.enum_int(
                Qgis.RenderUnit.Percentage
            ):
                return round(spacing * text_format.lineHeight(), 3)
            length = PropertyExtractor.length_to_style_value(text_format.lineHeight(), text_format.lineHeightUnit())
            size = TextPropertyExtractor.get_text_size_px(text_format)
            if isinstance(length, (int, float)) and size:
                return round(length / size, 3)
            return round(spacing, 3)
        except (AttributeError, RuntimeError, TypeError, ZeroDivisionError):
            return 1.2

    @staticmethod
    def get_text_letter_spacing(text_format: QgsTextFormat = None) -> float:
        """Return ``text-letter-spacing`` in ems, from the font's letter
        spacing (in the text size's unit, or a percentage of normal)."""
        if text_format is None:
            return 0
        try:
            font = text_format.font()
            spacing = font.letterSpacing()
            if PropertyExtractor.enum_int(font.letterSpacingType()) == 1:  # Absolute
                size = text_format.size()
                return spacing / size if size else 0
            return (spacing - 100) / 100 if spacing else 0  # Percentage
        except (AttributeError, RuntimeError, TypeError, ZeroDivisionError):
            return 0

    @staticmethod
    def get_text_transform(label_settings: QgsPalLayerSettings = None) -> str:
        """Return ``text-transform`` (``"none"``, ``"uppercase"``, or ``"lowercase"``).

        Maps from the QGIS capitalisation setting when available.
        """
        if label_settings is None:
            return "none"
        try:
            cap = PropertyExtractor.get_attribute(
                label_settings.format(), "capitalization"
            )
        except (AttributeError, RuntimeError):
            return "none"
        # Qgis.Capitalization: 1=AllUppercase, 2=AllLowercase. The export
        # usually bakes the type case into the label text (RulesExporter),
        # which leaves MixedCase here.
        cap = PropertyExtractor.enum_int(cap)
        if cap == 1:
            return "uppercase"
        if cap == 2:
            return "lowercase"
        return "none"

    @staticmethod
    def get_text_max_width(label_settings: QgsPalLayerSettings) -> float:
        """Return ``text-max-width`` in ems.

        QGIS's ``autoWrapLength`` counts characters; an average character is
        about half an em wide. Without it, a very large value effectively
        disables wrapping.
        """
        if label_settings.autoWrapLength > 0:
            return label_settings.autoWrapLength * 0.5
        return 999

    @staticmethod
    def get_text_keep_upright() -> bool:
        """Return ``text-keep-upright`` (default ``True``)."""
        return True

    @staticmethod
    def get_text_rotate(label_settings: QgsPalLayerSettings) -> Union[float, List]:
        """Return ``text-rotate`` in degrees, honouring data-defined overrides."""
        base_rotation = label_settings.angleOffset if label_settings.angleOffset != 0 else 0
        rotation_prop = label_settings.dataDefinedProperties().property(
            QgsPalLayerSettings.Property.LabelRotation
        )
        return PropertyExtractor.get_value_or_expression(base_rotation, rotation_prop)

    @staticmethod
    def get_text_rotation_alignment() -> str:
        """Return ``text-rotation-alignment`` (default ``"map"``)."""
        return "map"

    @staticmethod
    def get_text_pitch_alignment() -> str:
        """Return ``text-pitch-alignment`` (default ``"viewport"``)."""
        return "viewport"

    @staticmethod
    def get_text_translate() -> List[float]:
        """Return ``text-translate`` ``[x, y]`` offset in pixels (default ``[0, 0]``)."""
        return [0, 0]

    @staticmethod
    def get_text_translate_anchor() -> str:
        """Return ``text-translate-anchor`` (MapLibre default: ``"map"``)."""
        return "map"


class QgisMapLibreStyleExporter:
    """Export QGIS Vector Tile Layer styles to a MapLibre GL style JSON.

    Walks a ``QgsVectorTileLayer``'s renderer and labelling, converting each
    style entry into the appropriate MapLibre layer (``fill``, ``line``,
    ``symbol``, etc.). Marker symbols and label background markers are
    registered with an internal sprite registry so the companion
    ``SpriteGenerator`` can render them as a sprite sheet alongside the
    emitted ``style.json``.
    """

    def __init__(
        self,
        output_dir: str,
        utils_dir,
        layer: Optional[QgsVectorTileLayer] = None,
        minzoom: int = 0,
        maxzoom: int = 14,
        extent=None,
        zoom_ranges: Optional[Dict[str, tuple]] = None,
        descriptions: Optional[Dict[str, Dict[str, str]]] = None,
        raw_colors: Optional[Dict[str, set]] = None,
    ):
        """Initialise the exporter.

        Args:
            output_dir:      Directory where ``style.json`` and the sprite
                             folder will be written.
            layer:           The vector tile layer to export. If ``None``,
                             the active layer in the QGIS interface is used.
        """
        self.output_dir = output_dir
        self.utils_dir = utils_dir
        self.marker_symbols: dict = {}
        self.marker_counter = 0
        # Sprite name per image definition (see _register_sprite).
        self._sprite_names: Dict[str, str] = {}
        self._current_rule: Optional[int] = None
        self.glyphs = {}
        self.minzoom = minzoom
        self.maxzoom = maxzoom
        # The export extent (EPSG:3857), whose latitude sizes ground lengths.
        self.extent = extent
        # Per style name: the exact zooms it is drawn from and until
        # (TilesStyler.zoom_ranges); otherwise its whole zoom levels.
        self.zoom_ranges = zoom_ranges or {}
        # Per dataset: its colour fields holding raw colour values, mapped to
        # colours in the style (_map_raw_colors); others hold #RRGGBBAA.
        self.raw_colors = raw_colors or {}
        self.layer = self._resolve_layer(layer)
        self.source_name = "q2vt_tiles"
        self.style = self._build_style_skeleton()
        if descriptions:
            # The viewers' popups describe a feature by its source-layer and
            # q2vt_rule ("" without), from here rather than from the tiles.
            self.style["metadata"] = {"q2vt:descriptions": descriptions}
        self.style["layers"].append(self._build_background_layer())

    def _resolve_layer(self, layer: Optional[QgsVectorTileLayer]) -> QgsVectorTileLayer:
        """Return ``layer`` or fall back to the active QGIS layer.

        Raises:
            ValueError: If no layer is supplied and no active layer exists,
                        or if the resolved layer is not a ``QgsVectorTileLayer``.
        """
        if layer is None:
            try:
                if iface and iface.activeLayer():
                    layer = iface.activeLayer()
                else:
                    raise ValueError("No active layer found and no layer provided")
            except (ImportError, ValueError) as e:
                raise e
        if not isinstance(layer, QgsVectorTileLayer):
            raise ValueError(f"Layer must be a QgsVectorTileLayer, got {type(layer).__name__}")
        return layer

    def _build_style_skeleton(self) -> dict:
        """Build the empty MapLibre style document with sources and sprite refs.

        Its URLs are relative to style.json, so the export works from any
        web server or static host; the viewers resolve them. The tiles are
        one PMTiles file (the version token, unique per export, is set by
        ServerInitializer).
        """
        return {
            "version": 8,
            "name": f"{self.source_name}_style",
            "sources": {
                self.source_name: {
                    "type": "vector",
                    "url": "pmtiles://../tiles.pmtiles?v=10031993",
                    # The highest zoom with tiles; viewers enlarge its tiles
                    # beyond it rather than request tiles that don't exist.
                    "maxzoom": max(0, min(self.maxzoom, _TILES_MAX_ZOOM)),
                }
            },
            "glyphs": "glyphs/{fontstack}/{range}.pbf",
            "sprite": "sprite/sprite",
            "layers": [],
        }

    def _build_background_layer(self) -> dict:
        """The style's bottom layer: the project's background colour.

        The viewers show a basemap (OpenStreetMap, NASA Blue Marble) above
        it from their own basemap switcher.
        """
        bg_color = PropertyExtractor.convert_qcolor_to_maplibre(
            QgsProject.instance().backgroundColor()
        )
        return {
            "id": "background", "type": "background", "minzoom": 0, "maxzoom": 22,
            "paint": {"background-color": bg_color},
        }

    def export(self) -> Dict[str, Any]:
        """Convert all styles and labelling on the layer and write to disk.

        Returns:
            The complete MapLibre style dictionary.
        """
        self.convert()
        self.save_to_file()
        return self.style

    def convert(self) -> Dict[str, Any]:
        """Convert all styles and labelling on the layer, without writing
        anything (save_to_file() then writes the style, sprites and glyphs).

        Returns:
            The MapLibre style dictionary.
        """
        self.layer.source()
        self._set_map_units()

        renderer = self.layer.renderer()
        if isinstance(renderer, QgsVectorTileBasicRenderer):
            for style in renderer.styles():
                self._convert_renderer_style(style)

        labeling = self.layer.labeling()
        if isinstance(labeling, QgsVectorTileBasicLabeling):
            for style in labeling.styles():
                self._convert_labeling_style(style)

        self._map_raw_colors(self.style["layers"])
        self.style["layers"], band_datasets = merge_dataset_bands(merge_zoom_bands(self.style["layers"]))
        self.style["layers"] = self.finish_filters(merge_rule_layers(group_rule_layers(
            self.style["layers"])))
        # After merging, which compares the layers' zoom ranges.
        archive_max_zoom = self.style["sources"][self.source_name]["maxzoom"]
        self.style["layers"] = bridge_zoom_bands(self.style["layers"], archive_max_zoom)
        self.style["layers"] = raise_callouts(self.style["layers"])
        self.style["layers"], shared = share_tile_layers(self.style["layers"])
        shared.update(band_datasets)
        for dataset in shared:
            # Through each step to the tile layer finally drawn (a band's
            # dataset in its first band's, that one in a like layer's).
            target = shared[dataset]
            while target in shared and shared[target] != target:
                target = shared[target]
            shared[dataset] = target
        if shared:
            # For GDALTilesGenerator; not written to style.json.
            self.style["metadata"] = {TILE_LAYERS_KEY: shared}
        return self.style

    def _set_map_units(self) -> None:
        """Size map-unit lengths for the project's CRS at the tiles' latitude."""
        latitude = 0.0
        try:
            # The tiles don't exist yet when the style is exported, so the
            # tiles layer has no extent: the export extent gives the latitude.
            extent, crs = self.extent, QgsCoordinateReferenceSystem("EPSG:3857")
            if extent is None or extent.isEmpty():
                extent, crs = self.layer.extent(), self.layer.crs()
            if not extent.isEmpty():
                to_wgs84 = QgsCoordinateTransform(
                    crs, QgsCoordinateReferenceSystem("EPSG:4326"), QgsProject.instance()
                )
                latitude = to_wgs84.transform(extent.center()).y()
        except (QgsCsException, RuntimeError, AttributeError):  # Equator sizes then.
            latitude = 0.0
        PropertyExtractor.set_map_units(QgsProject.instance().crs(), latitude)

    def _convert_renderer_style(self, style):
        """Convert a single ``QgsVectorTileBasicRendererStyle`` into MapLibre layer(s)."""
        if not style.isEnabled() or not style.symbol():
            return
        min_zoom, max_zoom = self._zooms_of(style)
        first = len(self.style["layers"])
        # The rule's rows of a shared dataset (read by _value_combinations).
        match = _RULE_FILTER.fullmatch(style.filterExpression() or "")
        self._current_rule = int(match.group(1)) if match else None
        self._convert_symbol(
            style.symbol(), style.styleName(), style.layerName(),
            self.source_name, min_zoom, max_zoom,
        )
        self._apply_rule_filter(style, first)
        # Each symbol layer's "enabled" property: its layer's id is the style
        # name, "_layerN" for the Nth symbol layer of a line symbol.
        symbol = style.symbol()
        for layer_def in list(self.style["layers"][first:]):
            match = re.search(r"_layer(\d+)$", layer_def["id"])
            index = int(match.group(1)) if match else 0
            if index >= symbol.symbolLayerCount():
                continue
            prop = symbol.symbolLayer(index).dataDefinedProperties().property(
                QgsSymbolLayer.Property.PropertyLayerEnabled)
            self._apply_enabled(layer_def, PropertyExtractor.enabled_condition(prop))

    def _convert_labeling_style(self, style):
        """Convert a single ``QgsVectorTileBasicLabelingStyle`` into a MapLibre symbol layer."""
        if not style.isEnabled() or not style.labelSettings():
            return
        min_zoom, max_zoom = self._zooms_of(style)
        first = len(self.style["layers"])
        self._convert_label(
            style.labelSettings(), style.styleName(), style.layerName(),
            self.source_name, min_zoom, max_zoom, style.geometryType(),
        )
        self._apply_rule_filter(style, first)
        settings = style.labelSettings()  # A copy: kept while its properties are read.
        prop = QgsProperty(settings.dataDefinedProperties().property(QgsPalLayerSettings.Property.Show))
        condition = PropertyExtractor.enabled_condition(prop)
        for layer_def in list(self.style["layers"][first:]):
            self._apply_enabled(layer_def, condition)

    def _apply_enabled(self, layer_def: dict, condition) -> None:
        """Keep a "show" condition (enabled_condition) with a layer until
        its filter is final (finish_filters); an always-off layer is removed."""
        if condition is False:
            self.style["layers"].remove(layer_def)
        elif condition is not None:
            layer_def["_enabled"] = condition

    @staticmethod
    def finish_filters(layers: List[dict]) -> List[dict]:
        """Add each layer's "show" condition, and the part of its features a
        layer of its own draws ("_only": labels at their position), to its
        (rule) filter."""
        for layer_def in layers:
            for key in ("_enabled", "_only"):
                condition = layer_def.pop(key, None)
                if condition is None:
                    continue
                existing = layer_def.get("filter")
                layer_def["filter"] = ["all", existing, condition] if existing else condition
        return layers

    def _map_raw_colors(self, layers: List[dict]) -> None:
        """Map raw colour values (RulesExporter.raw_color_fields) to colours:
        each ["get", field] of such a field becomes a match from each value
        the dataset holds to its colour, decoded as QGIS decodes it; other
        values give no colour (the to-color around it then falls back to
        the symbol's own)."""
        cache: Dict[tuple, list] = {}

        def mapped(dataset: str, field: str) -> list:
            key = (dataset, field)
            if key not in cache:
                expression = ["match", ["to-string", ["get", field]]]
                for value in self._distinct_values(dataset, field):
                    color = QgsSymbolLayerUtils.decodeColor(value)
                    if color.isValid():
                        expression += [value, PropertyExtractor.convert_qcolor_to_maplibre(color)]
                cache[key] = expression + [""] if len(expression) > 2 else ""
            return cache[key]

        def replace(value, dataset: str, fields: set):
            if isinstance(value, list):
                if len(value) == 2 and value[0] == "get" and value[1] in fields:
                    return mapped(dataset, value[1])
                return [replace(v, dataset, fields) for v in value]
            return value

        for layer_def in layers:
            fields = self.raw_colors.get(layer_def.get("source-layer"))
            if not fields:
                continue
            for section in ("paint", "layout"):
                for key, value in list(layer_def.get(section, {}).items()):
                    if key != "icon-image":  # Marker variants read raw values themselves.
                        layer_def[section][key] = replace(value, layer_def["source-layer"], fields)

    @staticmethod
    def _cluster_distance(cluster: dict) -> dict:
        """A point cluster renderer's distance, for GDALTilesGenerator: in
        screen pixels ("px") or EPSG:3857 metres ("m")."""
        value = cluster["tolerance"]
        unit = cluster["unit"]
        if PropertyExtractor.is_map_unit(unit):
            metres = (PropertyExtractor.map_unit_m if PropertyExtractor.enum_int(unit) == PropertyExtractor._MAP_UNITS
                      else PropertyExtractor.ground_m)
            return {"m": value * metres}
        return {"px": PropertyExtractor.convert_length_to_pixels(value, unit)}

    def _displacement_meta(self, displacement: dict) -> dict:
        """A point displacement renderer's settings, for GDALTilesGenerator:
        the role of the layer's dataset, the grouping distance, the
        placement and the lengths (screen pixels) it places points by."""
        return {"role": displacement["role"], "distance": self._cluster_distance(displacement),
                "placement": displacement["placement"], "d": displacement["diagonal_px"],
                "cd": displacement["center_px"], "a": displacement["addition_px"],
                "f": displacement.get("label_factor", 0.5)}

    @staticmethod
    def _displacement_circle_layer(layer_def: dict, displacement: dict) -> dict:
        """A point displacement renderer's circle around each group (ring
        placements: a circle of the radius the tiles hold, in screen pixels
        as QGIS draws it) or the lines joining a grid's points."""
        r, g, b, a = displacement["circle_color"]
        color = f"rgba({r}, {g}, {b}, {a})"
        layer_def = dict(layer_def)
        if displacement["placement"] == _DISPLACEMENT_GRID:
            layer_def["type"] = "line"
            layer_def["paint"] = {"line-color": color, "line-width": displacement["circle_px"]}
            layer_def["layout"] = {"visibility": "visible"}
            return layer_def
        layer_def["type"] = "circle"
        layer_def["paint"] = {
            "circle-radius": ["get", DISPLACEMENT_RADIUS_FIELD],
            "circle-color": "rgba(0, 0, 0, 0)",
            "circle-stroke-color": color,
            "circle-stroke-width": displacement["circle_px"],
        }
        layer_def["layout"] = {"visibility": "visible"}
        return layer_def

    def _cluster_count_layer(self, font_marker, layer_def: dict) -> dict:
        """A cluster symbol's font marker as the cluster's count: a label of
        the q2vt_cluster_size field the tiles hold, in the marker's font,
        size and colour."""
        font = font_marker.fontFamily()
        style = font_marker.fontStyle() if hasattr(font_marker, "fontStyle") else ""
        key = self._font_key(font, style or "Regular") or self._font_key(font, "") or "Arial"
        self.glyphs.setdefault(key, []).append((f"{CHARS_PREFIX}0123456789", None))
        layer_def = dict(layer_def)
        layer_def["layout"] = {
            "text-field": ["to-string", ["get", CLUSTER_SIZE_FIELD]],
            "text-font": [key],
            "text-size": PropertyExtractor.length_to_style_value(font_marker.size(), font_marker.sizeUnit()),
            "text-allow-overlap": True,
            "text-ignore-placement": True,
            "visibility": "visible",
        }
        layer_def["paint"] = {
            "text-color": PropertyExtractor.convert_qcolor_to_maplibre(font_marker.color()),
        }
        return layer_def

    def _heatmap_layer(self, symbol_layer: QgsSymbolLayer, layer_def: dict, dataset: str) -> Optional[dict]:
        """A heatmap renderer's placeholder marker (see HEATMAP_PROPERTY) as a
        MapLibre ``heatmap`` layer: its radius, its colour ramp along the
        density, and each point's weight (divided by the maximum value; a
        maximum QGIS works out from the view is taken as the data's largest
        weight). None for other markers."""
        try:
            prop = symbol_layer.dataDefinedProperties().property(HEATMAP_PROPERTY)
            if prop is None or prop.isActive():
                return None
            settings = json.loads(prop.staticValue())
        except (AttributeError, RuntimeError, ValueError, TypeError):
            return None
        scale = QgsMapUnitScale(settings["min_scale"], settings["max_scale"])
        layer_def = dict(layer_def, type="heatmap", layout={"visibility": "visible"})
        paint = {"heatmap-radius": PropertyExtractor.length_to_style_value(
            settings["radius"], settings["unit"], scale)}
        stops = settings["stops"] or [[0, [0, 0, 255, 0.0]], [1, [255, 0, 0, 1.0]]]
        color = ["interpolate", ["linear"], ["heatmap-density"]]
        for t, (r, g, b, a) in stops:
            color += [t, f"rgba({r}, {g}, {b}, {a})"]
        paint["heatmap-color"] = color
        weight = PropertyExtractor.get_value_or_expression(
            None, symbol_layer.dataDefinedProperties().property(QgsSymbolLayer.Property.PropertySize))
        if isinstance(weight, list):
            maximum = settings["max_value"] or self._max_value(dataset, weight[1]) if weight[:1] == ["get"] else 0
            value = ["to-number", weight, 0]
            paint["heatmap-weight"] = ["/", value, maximum] if maximum and maximum > 0 else value
        layer_def["paint"] = paint
        return layer_def

    def _max_value(self, dataset: str, field: str) -> float:
        """The largest value of a dataset's field; 0 if none."""
        from osgeo import ogr  # pylint: disable=import-outside-toplevel
        ds = ogr.Open(join(self.utils_dir, f"{dataset}.{_TEMP_RULE_FORMAT}"))
        if ds is None:
            return 0.0
        try:
            layer = ds.GetLayer(0)
            name = field.replace('"', '""')
            # OGR SQL over the plugin's own dataset; every name is quoted.
            result = ds.ExecuteSQL(f'SELECT MAX("{name}") FROM "{layer.GetName()}"')  # nosec B608
            feature = result.GetNextFeature()
            value = feature.GetFieldAsDouble(0) if feature and feature.IsFieldSetAndNotNull(0) else 0.0
            ds.ReleaseResultSet(result)
            return value
        finally:
            ds = None

    def _label_fonts(self, label_settings: QgsPalLayerSettings, text_format: QgsTextFormat,
                     font: str, dataset: str) -> tuple:
        """(text-font, the font keys whose glyphs it needs) of a label.

        A data-defined font family is each feature's family in the
        format's style (or the family's regular style where it has no such
        style), as a ``match`` on the exported family field over the
        dataset's families (up to _MAX_LABEL_FONTS); a family that isn't
        installed, or a feature without one, keeps the format's font, as
        QGIS draws it.
        """
        try:
            prop = label_settings.dataDefinedProperties().property(QgsPalLayerSettings.Property.Family)
        except (AttributeError, RuntimeError):
            prop = None
        resolved = PropertyExtractor.get_value_or_expression(None, prop) if prop else None
        if resolved is None:
            return [font], [font]
        style = text_format.font().styleName()
        if isinstance(resolved, str):
            key = self._font_key(resolved, style)
            return ([key], [key]) if key else ([font], [font])
        if not (isinstance(resolved, list) and resolved[:1] == ["get"]):
            return [font], [font]
        keys = {}
        for family in self._distinct_values(dataset, resolved[1])[:_MAX_LABEL_FONTS]:
            key = self._font_key(family, style)
            if key and key != font:
                keys[family] = key
        if not keys:
            return [font], [font]
        expression = ["match", ["to-string", resolved]]
        for family, key in keys.items():
            expression += [family, ["literal", [key]]]
        expression.append(["literal", [font]])
        return expression, [font] + sorted(set(keys.values()))

    @staticmethod
    def _font_key(family: str, style: str) -> Optional[str]:
        """The glyph font key of a family in a style: "family style" if the
        family has the style, else the family alone (its regular style);
        None if the family isn't installed."""
        family = (family or "").strip()
        if not family:
            return None
        families = qfontdb.families()
        if GlyphGenerator._resolve_font_key(f"{family} {style}", families):  # pylint: disable=protected-access
            return f"{family} {style}".strip()
        if GlyphGenerator._resolve_font_key(family, families):  # pylint: disable=protected-access
            return family
        return None

    def _distinct_values(self, dataset: str, field: str) -> List[str]:
        """The distinct non-NULL values of a dataset's field, as text."""
        from osgeo import ogr  # pylint: disable=import-outside-toplevel
        ds = ogr.Open(join(self.utils_dir, f"{dataset}.{_TEMP_RULE_FORMAT}"))
        if ds is None:
            return []
        try:
            layer = ds.GetLayer(0)
            name = field.replace('"', '""')
            # OGR SQL over the plugin's own dataset; every name is quoted.
            result = ds.ExecuteSQL(
                f'SELECT DISTINCT "{name}" FROM "{layer.GetName()}"')  # nosec B608
            values = [f.GetFieldAsString(0) for f in result if f.IsFieldSetAndNotNull(0)]
            ds.ReleaseResultSet(result)
            return values
        finally:
            ds = None

    def _zooms_of(self, style) -> tuple:
        """(minzoom, maxzoom) of a style's layers: the exact zooms of its
        rule's scale range, else its whole zoom levels (maxzoom exclusive)."""
        exact = self.zoom_ranges.get(style.styleName())
        if exact is not None:
            return exact
        return style.minZoomLevel(), style.maxZoomLevel() + 1

    def _apply_rule_filter(self, style, first: int) -> None:
        """Filter the layers converted from ``style`` (from index ``first``)
        to its rule's rows, in a dataset shared by several rules."""
        match = _RULE_FILTER.fullmatch(style.filterExpression() or "")
        if not match:
            return
        for layer_def in self.style["layers"][first:]:
            layer_def["filter"] = ["==", ["get", RULE_FIELD], int(match.group(1))]

    def _convert_symbol(
        self,
        symbol: QgsSymbol,
        style_name: str,
        source_layer_name: str,
        source_name: str,
        min_zoom: int = -1,
        max_zoom: int = -1,
    ):
        """Dispatch a QGIS symbol to the correct conversion routine by symbol type."""
        if symbol.symbolLayerCount() == 0:
            return
        symbol_layer = symbol.symbolLayer(0)
        symbol_type = symbol.type()

        if symbol_type == QgsSymbol.SymbolType.Marker:
            self._convert_marker_symbol(
                symbol_layer, symbol, style_name, source_layer_name,
                source_name, min_zoom, max_zoom,
            )
        elif symbol_type == QgsSymbol.SymbolType.Line:
            self._convert_line_symbol(
                symbol, style_name, source_layer_name, source_name, min_zoom, max_zoom
            )
        elif symbol_type == QgsSymbol.SymbolType.Fill:
            self._convert_fill_symbol(
                symbol_layer, symbol, style_name, source_layer_name, source_name, min_zoom, max_zoom
            )

    def _base_layer_def(
        self,
        layer_type: str,
        style_name: str,
        source_layer_name: str,
        source_name: str,
        min_zoom: int,
        max_zoom: int,
    ) -> dict:
        """Build a common MapLibre layer-definition skeleton with id/source/zoom range."""
        layer_def: dict = {
            "id": style_name,
            "type": layer_type,
            "source": source_name,
            "source-layer": source_layer_name,
            "paint": {},
            "layout": {},
        }
        if min_zoom >= 0:
            layer_def["minzoom"] = min_zoom
        if max_zoom >= 0:
            layer_def["maxzoom"] = max_zoom
        return layer_def

    def _register_pattern(self, symbol_or_layer) -> str:
        """Register a pattern image (or sub-symbol) in the sprite registry.

        Used by line/fill pattern conversion. Returns the unique pattern name
        the MapLibre layer should reference via ``line-pattern`` or
        ``fill-pattern``.
        """
        try:
            symbol_or_layer = symbol_or_layer.clone()
        except (AttributeError, RuntimeError):
            pass
        if FillPropertyExtractor.is_pattern_fill(symbol_or_layer):
            width, height = FillPropertyExtractor.get_pattern_tile_size(symbol_or_layer)
            symbol_or_layer = FillPattern(symbol_or_layer, width, height)
        return self._register_sprite(symbol_or_layer, "pattern")

    def _register_sprite(self, source, prefix: str) -> str:
        """The sprite name of an image to draw (a symbol or pattern).

        An image drawn the same way as one already registered (the same
        marker in several rules, categories or zoom bands) reuses its name,
        so the sprite sheet holds it once.
        """
        key = sprite_key(source, 1)
        if key is not None and key in self._sprite_names:
            return self._sprite_names[key]
        name = f"{prefix}_{self.marker_counter}"
        self.marker_counter += 1
        self.marker_symbols[name] = source
        if key is not None:
            self._sprite_names[key] = name
        return name

    def _convert_marker_symbol(
        self,
        symbol_layer: QgsSymbolLayer,
        symbol: QgsSymbol,
        style_name: str,
        source_layer_name: str,
        source_name: str,
        min_zoom: int = -1,
        max_zoom: int = -1,
    ):
        """Convert a QGIS marker symbol into a MapLibre ``symbol`` layer."""
        layer_def = self._base_layer_def(
            "symbol", style_name, source_layer_name, source_name, min_zoom, max_zoom
        )

        heatmap = self._heatmap_layer(symbol_layer, layer_def, source_layer_name)
        if heatmap is not None:
            self.style["layers"].append(heatmap)
            return
        displacement = displacement_settings(symbol)
        if displacement is not None:
            layer_def["metadata"] = {"q2vt:displacement": self._displacement_meta(displacement)}
            if displacement["role"] == "circle":
                self.style["layers"].append(self._displacement_circle_layer(layer_def, displacement))
                return
            # The centre symbol: drawn at every group's centre, as QGIS does.
            layer_def["layout"].update({"icon-allow-overlap": True, "icon-ignore-placement": True})
        cluster = cluster_settings(symbol)
        if cluster is not None:
            layer_def["metadata"] = {"q2vt:cluster": self._cluster_distance(cluster)}
            if isinstance(symbol_layer, QgsFontMarkerSymbolLayer):
                # The cluster's count, as QGIS's default cluster symbol draws
                # it (a font marker of @cluster_size).
                self.style["layers"].append(self._cluster_count_layer(symbol_layer, layer_def))
                return
        sprite_symbol, ground_icon_size = self._sprite_marker(symbol)
        marker_name = self._register_sprite(sprite_symbol, "marker")
        marker_name = self._marker_variants(sprite_symbol, source_layer_name) or marker_name

        if cluster is not None:  # Clusters are drawn, all of them, as QGIS does.
            layer_def["layout"].update({"icon-allow-overlap": True, "icon-ignore-placement": True})
        layer_def["layout"].update({
            "icon-image": IconPropertyExtractor.get_icon_image(marker_name),
            "icon-size": ground_icon_size or IconPropertyExtractor.get_icon_size(symbol_layer, 1.0),
            "icon-rotate": IconPropertyExtractor.get_icon_rotate(symbol_layer=symbol_layer),
            "icon-padding": IconPropertyExtractor.get_icon_padding(),
            "icon-rotation-alignment": IconPropertyExtractor.get_icon_rotation_alignment(),
            "icon-pitch-alignment": IconPropertyExtractor.get_icon_pitch_alignment(),
            "icon-anchor": IconPropertyExtractor.get_icon_anchor(),
            "icon-allow-overlap": IconPropertyExtractor.get_icon_allow_overlap(True),
            "icon-ignore-placement": IconPropertyExtractor.get_icon_ignore_placement(),
            "icon-optional": IconPropertyExtractor.get_icon_optional(),
            "icon-keep-upright": IconPropertyExtractor.get_icon_keep_upright(),
            "symbol-placement": IconPropertyExtractor.get_symbol_placement(),
            "symbol-spacing": IconPropertyExtractor.get_symbol_spacing(),
            "symbol-avoid-edges": IconPropertyExtractor.get_symbol_avoid_edges(),
            "symbol-sort-key": IconPropertyExtractor.get_symbol_sort_key(),
            "symbol-z-order": IconPropertyExtractor.get_symbol_z_order(),
            "visibility": "visible",
        })
        layer_def["paint"].update({
            "icon-opacity": IconPropertyExtractor.get_icon_opacity(),
            "icon-halo-color": IconPropertyExtractor.get_icon_halo_color(),
            "icon-halo-width": IconPropertyExtractor.get_icon_halo_width(),
            "icon-halo-blur": IconPropertyExtractor.get_icon_halo_blur(),
            "icon-translate": IconPropertyExtractor.get_icon_translate(),
            "icon-translate-anchor": IconPropertyExtractor.get_icon_translate_anchor(),
        })

        self.style["layers"].append(layer_def)

    # Properties drawn into a marker's sprite that the style applies itself
    # (icon-size, icon-rotate) or that aren't part of its look.
    _MARKER_STYLE_PROPERTIES = frozenset(PropertyExtractor.enum_int(p) for p in (
        QgsSymbolLayer.Property.PropertySize, QgsSymbolLayer.Property.PropertyAngle,
        QgsSymbolLayer.Property.PropertyLayerEnabled,
    ))
    # Most images one data-defined marker is drawn as; beyond it the marker
    # is drawn with its default look.
    _MAX_MARKER_VARIANTS = 64

    def _marker_variants(self, sprite_symbol: QgsSymbol, source_layer_name: str):
        """``icon-image`` for a marker whose look is data-defined (colour,
        stroke, shape...): one sprite per combination of the exported values
        its features have, picked per feature by a ``match``; None if its
        look is fixed or has too many combinations.
        """
        props = []  # (symbol layer index, property key, expression, fields)
        fields: List[str] = []
        for index, layer in enumerate(sprite_symbol.symbolLayers()):
            collection = layer.dataDefinedProperties()
            for key in collection.propertyKeys():
                prop = collection.property(key)
                if not prop.isActive() or int(key) in self._MARKER_STYLE_PROPERTIES:
                    continue
                expression = prop.asExpression()
                columns = sorted(c for c in QgsExpression(expression).referencedColumns()
                                 if c.startswith("q2vt_property"))
                if not columns:
                    continue
                props.append((index, key, expression, columns))
                fields.extend(c for c in columns if c not in fields)
        if not props:
            return None
        combinations = self._value_combinations(source_layer_name, fields)
        if not combinations or len(combinations) > self._MAX_MARKER_VARIANTS:
            return None
        if len(fields) == 1:
            selector = ["to-string", ["get", fields[0]]]
        else:
            selector = ["concat"]
            for i, name in enumerate(fields):
                selector += (["|"] if i else []) + [["to-string", ["get", name]]]
        expression = ["match", selector]
        labels = set()
        for values in combinations:
            label = "|".join(self._match_text(v) for v in values)
            if label in labels:
                continue
            labels.add(label)
            variant = sprite_symbol.clone()
            # A raw colour value as the #RRGGBBAA the marker's property reads.
            raw = self.raw_colors.get(source_layer_name, set())
            literal = {
                name: QgsExpression.quotedValue(
                    color_hex(str(value)) if name in raw and value is not None else value)
                for name, value in zip(fields, values)
            }
            for index, key, expr, columns in props:
                for name in columns:
                    expr = expr.replace(f'"{name}"', literal[name])
                variant.symbolLayer(index).dataDefinedProperties().setProperty(
                    key, QgsProperty.fromExpression(expr))
            expression += [label, self._register_sprite(variant, "marker")]
        expression.append(self._register_sprite(sprite_symbol, "marker"))
        return expression

    @staticmethod
    def _match_text(value) -> str:
        """A value as MapLibre's to-string writes it (null as "")."""
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def _value_combinations(self, dataset_name: str, fields: List[str]) -> Optional[list]:
        """The distinct combinations of ``fields``' values in a dataset
        (in this rule's rows of a shared one); None if more than
        _MAX_MARKER_VARIANTS or unreadable."""
        from osgeo import ogr  # pylint: disable=import-outside-toplevel
        path = join(self.utils_dir, f"{dataset_name}.{_TEMP_RULE_FORMAT}")
        ds = ogr.Open(path)
        if ds is None:
            return None
        try:
            layer = ds.GetLayer(0)
            defn = layer.GetLayerDefn()
            indices = [defn.GetFieldIndex(name) for name in fields]
            if any(i < 0 for i in indices):
                return None
            if self._current_rule is not None and defn.GetFieldIndex(RULE_FIELD) >= 0:
                layer.SetAttributeFilter(f'"{RULE_FIELD}" = {int(self._current_rule)}')
            layer.SetIgnoredFields([
                defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())
                if i not in indices and defn.GetFieldDefn(i).GetName() != RULE_FIELD
            ] + ["OGR_GEOMETRY"])
            found = {}
            for feature in layer:
                values = tuple(feature.GetField(i) if feature.IsFieldSetAndNotNull(i) else None
                               for i in indices)
                found.setdefault(values, None)
                if len(found) > self._MAX_MARKER_VARIANTS:
                    return None
            return list(found)
        finally:
            ds = None

    # Size a marker sized on the ground is drawn at in the sprite, in mm;
    # the style scales it to its ground size at each zoom.
    _GROUND_MARKER_SPRITE_MM = 6.0

    @classmethod
    def _sprite_marker(cls, symbol: QgsSymbol) -> tuple:
        """(the symbol as the sprite draws it, its icon-size or None).

        The sprite is drawn unrotated: the style rotates it by the first
        symbol layer's angle (icon-rotate), which would otherwise apply
        twice; the other layers keep their angles relative to it.

        A marker sized on the ground (map units or metres) has no pixel
        size: it is drawn at a fixed size, and its icon-size grows with
        the zoom so that it covers its ground size, as in QGIS.
        """
        sprite_symbol = symbol.clone()
        try:
            if sprite_symbol.type() == QgsSymbol.SymbolType.Marker:
                sprite_symbol.setAngle(0)
        except (AttributeError, RuntimeError):
            pass
        try:
            unit = sprite_symbol.sizeUnit()
            size = sprite_symbol.size()
        except (AttributeError, RuntimeError):
            return sprite_symbol, None
        if not PropertyExtractor.is_map_unit(unit) or not size or size <= 0:
            return sprite_symbol, None
        ratio = cls._GROUND_MARKER_SPRITE_MM / size
        for layer in sprite_symbol.symbolLayers():
            width = PropertyExtractor.get_attribute(layer, "strokeWidth")
            stroke_unit = PropertyExtractor.get_attribute(layer, "strokeWidthUnit")
            if isinstance(width, (int, float)) and PropertyExtractor.is_map_unit(stroke_unit):
                layer.setStrokeWidth(width * ratio)
        sprite_symbol.setOutputUnit(Qgis.RenderUnit.Millimeters)
        sprite_symbol.setSize(cls._GROUND_MARKER_SPRITE_MM)
        sprite_px = PropertyExtractor.convert_length_to_pixels(cls._GROUND_MARKER_SPRITE_MM, "mm")
        ground = PropertyExtractor.length_to_style_value(  # Pixels by zoom.
            size, unit, PropertyExtractor.map_unit_scale(symbol, "mapUnitScale"))
        if not isinstance(ground, list):  # Its size limits leave one size.
            return sprite_symbol, ground / sprite_px / _SPRITE_QUALITY
        icon_size = list(ground)
        icon_size[4] = ground[4] / sprite_px / _SPRITE_QUALITY
        icon_size[6] = ground[6] / sprite_px / _SPRITE_QUALITY
        return sprite_symbol, icon_size

    def _convert_line_symbol(
        self,
        symbol: QgsSymbol,
        style_name: str,
        source_layer_name: str,
        source_name: str,
        min_zoom: int = -1,
        max_zoom: int = -1,
    ):
        """Convert every symbol layer of a QGIS line symbol into MapLibre layer(s).

        A QGIS line symbol can stack multiple symbol layers - typically a
        base ``QgsSimpleLineSymbolLayer`` stroke plus one or more
        ``QgsMarkerLineSymbolLayer`` marker decorations. Each sub-layer is
        converted independently and appended to ``self.style["layers"]`` in
        the same bottom-to-top order QGIS uses for its symbol layers, so
        marker decorations correctly render above or below the base stroke
        exactly as they do in QGIS.
        """
        for index in range(symbol.symbolLayerCount()):
            symbol_layer = symbol.symbolLayer(index)
            # The first sub-layer keeps the original style name (so existing
            # look-ups by id keep working); subsequent sub-layers get a
            # unique suffixed id since MapLibre layer ids must be unique.
            layer_id = style_name if index == 0 else f"{style_name}_layer{index}"

            if (LinePropertyExtractor.is_marker_line(symbol_layer)
                    or LinePropertyExtractor.is_hashed_line(symbol_layer)):
                self._convert_marker_line_symbol_layer(
                    symbol_layer, layer_id, source_layer_name, source_name,
                    min_zoom, max_zoom,
                )
            else:
                self._convert_simple_or_pattern_line_symbol_layer(
                    symbol_layer, symbol, layer_id, source_layer_name, source_name,
                    min_zoom, max_zoom,
                )

    def _convert_marker_line_symbol_layer(
        self,
        symbol_layer: QgsSymbolLayer,
        style_name: str,
        source_layer_name: str,
        source_name: str,
        min_zoom: int = -1,
        max_zoom: int = -1,
    ):
        """Convert a QGIS ``QgsMarkerLineSymbolLayer`` into a MapLibre ``symbol`` layer.

        The marker line's sub-symbol is rendered to a sprite (registered via
        ``self.marker_symbols`` exactly like a standalone marker symbol) and
        referenced through ``icon-image``. QGIS's interval/vertex placement
        is mapped to ``symbol-placement``/``symbol-spacing``,
        ``rotateSymbols()`` controls ``icon-rotation-alignment``, and the
        marker line's perpendicular ``offset()`` is reproduced via
        ``icon-offset`` so markers displaced from the line stay displaced
        the same way in MapLibre.
        """
        sub_symbol = None
        try:
            sub_symbol = symbol_layer.subSymbol()
        except (AttributeError, RuntimeError):
            pass
        if sub_symbol is None or sub_symbol.symbolLayerCount() == 0:
            return

        layer_def = self._base_layer_def(
            "symbol", style_name, source_layer_name, source_name, min_zoom, max_zoom
        )

        icon_rotate = 0
        if LinePropertyExtractor.is_hashed_line(symbol_layer):
            # A hashed line: a short stroke of its line sub-symbol, drawn
            # across the line (at the hash angle) at every interval.
            length = PropertyExtractor.convert_length_to_pixels(
                symbol_layer.hashLength(),
                PropertyExtractor.get_attribute(symbol_layer, "hashLengthUnit"),
            )
            marker_name = self._register_sprite(HashMark(sub_symbol.clone(), max(length, 1.0)), "marker")
            icon_rotate = symbol_layer.hashAngle()
            icon_size = 1.0 / _SPRITE_QUALITY
        else:
            marker_name = self._register_sprite(sub_symbol.clone(), "marker")
            icon_size = IconPropertyExtractor.get_icon_size(sub_symbol.symbolLayer(0), 1.0)

        rotate_with_line = LinePropertyExtractor.get_marker_line_rotate_symbols(symbol_layer)
        offset_px = LinePropertyExtractor.get_marker_line_offset(symbol_layer)

        layer_def["layout"].update({
            "icon-image": IconPropertyExtractor.get_icon_image(marker_name),
            "icon-size": icon_size,
            "icon-rotate": icon_rotate,
            "icon-padding": IconPropertyExtractor.get_icon_padding(),
            # Following the line's bearing ("map") reproduces rotateSymbols()
            # == True; "viewport" keeps markers upright regardless of the
            # line's direction, matching rotateSymbols() == False.
            "icon-rotation-alignment": "map" if rotate_with_line else "viewport",
            "icon-pitch-alignment": IconPropertyExtractor.get_icon_pitch_alignment(),
            "icon-anchor": IconPropertyExtractor.get_icon_anchor(),
            "icon-allow-overlap": IconPropertyExtractor.get_icon_allow_overlap(True),
            # QGIS marker lines stamp every marker unconditionally - there is
            # no collision system in QGIS to reproduce. icon-allow-overlap
            # alone only stops a marker from being hidden by collisions;
            # each repeat along the line is still registered in MapLibre's
            # collision index and will suppress its own later siblings (and
            # be suppressed by them) unless icon-ignore-placement is also
            # true. Both flags together are required to fully disable
            # collision-based hiding, matching QGIS's "always show" look.
            "icon-ignore-placement": True,
            "icon-optional": IconPropertyExtractor.get_icon_optional(),
            "icon-keep-upright": False,
            "symbol-placement": LinePropertyExtractor.get_marker_line_symbol_placement(symbol_layer),
            "symbol-spacing": LinePropertyExtractor.get_marker_line_spacing(symbol_layer),
            "symbol-avoid-edges": IconPropertyExtractor.get_symbol_avoid_edges(),
            "symbol-sort-key": IconPropertyExtractor.get_symbol_sort_key(),
            "symbol-z-order": IconPropertyExtractor.get_symbol_z_order(),
            "visibility": "visible",
        })
        if offset_px:
            # icon-offset is applied in the icon's own pixel space and
            # rotates together with icon-rotate/icon-rotation-alignment, so
            # when the icon follows the line bearing this reproduces QGIS's
            # perpendicular marker-to-line offset.
            icon_size_value = layer_def["layout"]["icon-size"]
            if isinstance(icon_size_value, (int, float)) and icon_size_value:
                compensated_offset = offset_px / icon_size_value
            else:
                # icon-size is data-defined (rare: only when the QGIS marker
                # sub-symbol itself has a per-feature size override). There is
                # no way to build a dynamic two-element icon-offset array in
                # the MapLibre expression language, so fall back to the
                # sprite-quality constant used for the non-data-defined case.
                compensated_offset = offset_px * _SPRITE_QUALITY
            layer_def["layout"]["icon-offset"] = [0, compensated_offset]

        layer_def["paint"].update({
            "icon-opacity": IconPropertyExtractor.get_icon_opacity(),
            "icon-halo-color": IconPropertyExtractor.get_icon_halo_color(),
            "icon-halo-width": IconPropertyExtractor.get_icon_halo_width(),
            "icon-halo-blur": IconPropertyExtractor.get_icon_halo_blur(),
            "icon-translate": IconPropertyExtractor.get_icon_translate(),
            "icon-translate-anchor": IconPropertyExtractor.get_icon_translate_anchor(),
        })

        self.style["layers"].append(layer_def)

    def _convert_simple_or_pattern_line_symbol_layer(
        self,
        symbol_layer: QgsSymbolLayer,
        symbol: QgsSymbol,          # add this
        style_name: str,
        source_layer_name: str,
        source_name: str,
        min_zoom: int = -1,
        max_zoom: int = -1,
    ):
        """Convert a plain-stroke or raster-pattern line symbol layer to MapLibre ``line``."""
        layer_def = self._base_layer_def(
            "line", style_name, source_layer_name, source_name, min_zoom, max_zoom
        )

        if isinstance(symbol_layer, QgsSimpleLineSymbolLayer):
            layer_def["paint"].update({
                "line-color": LinePropertyExtractor.get_line_color(symbol_layer),
                "line-width": LinePropertyExtractor.get_line_width(symbol_layer),
                "line-opacity": LinePropertyExtractor.get_line_opacity(symbol_layer, symbol),
                "line-blur": LinePropertyExtractor.get_line_blur(),
                "line-gap-width": LinePropertyExtractor.get_line_gap_width(),
                "line-translate": LinePropertyExtractor.get_line_translate(symbol_layer),
                "line-translate-anchor": LinePropertyExtractor.get_line_translate_anchor(symbol_layer),
            })

            offset = LinePropertyExtractor.get_line_offset(symbol_layer)
            width = layer_def["paint"]["line-width"]
            if (symbol_layer.drawInsidePolygon() and isinstance(width, (int, float))
                    and isinstance(offset, (int, float))):
                # Only the half of a polygon's outline inside it: polygon
                # rings run with the inside on their right (vector tiles),
                # where a positive offset draws.
                layer_def["paint"]["line-width"] = width / 2
                offset += width / 4
            # Emit offset only when non-zero (preserve compact output) or when
            # the offset is data-defined (a list expression, never numerically zero).
            if isinstance(offset, list) or (isinstance(offset, (int, float)) and offset != 0):
                layer_def["paint"]["line-offset"] = offset

            width_value = layer_def["paint"]["line-width"]
            width_px = width_value if isinstance(width_value, (int, float)) else 1.0
            dasharray = LinePropertyExtractor.get_line_dasharray(symbol_layer, width_px)
            if dasharray:
                layer_def["paint"]["line-dasharray"] = dasharray

            layer_def["layout"].update({
                "line-cap": LinePropertyExtractor.get_line_cap(symbol_layer),
                "line-join": LinePropertyExtractor.get_line_join(symbol_layer),
                "line-miter-limit": LinePropertyExtractor.get_line_miter_limit(),
                "line-round-limit": LinePropertyExtractor.get_line_round_limit(),
                "line-sort-key": LinePropertyExtractor.get_line_sort_key(symbol_layer),
                "visibility": "visible",
            })
        elif (LinePropertyExtractor.is_pattern_line(symbol_layer)
              and LinePropertyExtractor.get_line_pattern_path(symbol_layer)):
            # Raster line: its image repeated along the line, at the line's
            # width (MapLibre draws a pattern at the line's width).
            width = LinePropertyExtractor.get_line_width(symbol_layer)
            height = width if isinstance(width, (int, float)) and width > 0 else 4.0
            pattern_name = self._register_sprite(LinePattern(
                LinePropertyExtractor.get_line_pattern_path(symbol_layer), height
            ), "pattern")
            layer_def["paint"].update({
                "line-pattern": pattern_name,
                "line-width": width,
                "line-opacity": LinePropertyExtractor.get_line_opacity(symbol_layer, symbol),
                "line-translate": LinePropertyExtractor.get_line_translate(symbol_layer),
                "line-translate-anchor": LinePropertyExtractor.get_line_translate_anchor(symbol_layer),
            })
            layer_def["layout"].update({
                "line-cap": "round",
                "line-join": "round",
                "line-miter-limit": LinePropertyExtractor.get_line_miter_limit(),
                "line-round-limit": LinePropertyExtractor.get_line_round_limit(),
                "visibility": "visible",
            })
        else:
            # A line MapLibre can't draw (burst, arrow, interpolated, ...):
            # a plain line of its colour and width.
            color, width = LinePropertyExtractor.get_fallback_line(symbol_layer)
            layer_def["paint"].update({
                "line-color": color,
                "line-width": width,
                "line-opacity": LinePropertyExtractor.get_line_opacity(symbol_layer, symbol),
            })
            layer_def["layout"].update({
                "line-cap": "round", "line-join": "round", "visibility": "visible",
            })

        self.style["layers"].append(layer_def)

    def _convert_fill_symbol(
        self,
        symbol_layer: QgsSymbolLayer,
        symbol: QgsSymbol,          # add this
        style_name: str,
        source_layer_name: str,
        source_name: str,
        min_zoom: int = -1,
        max_zoom: int = -1,
    ):
        """Convert a QGIS fill symbol into a MapLibre ``fill`` layer."""
        layer_def = self._base_layer_def(
            "fill", style_name, source_layer_name, source_name, min_zoom, max_zoom
        )

        brush = PropertyExtractor.enum_int(symbol_layer.brushStyle()) if isinstance(
            symbol_layer, QgsSimpleFillSymbolLayer) else None
        if brush in _BRUSH_PATTERNS:
            # Hatching and the like: the brush pattern, repeated. (A flat
            # fill would hide everything below it.)
            pattern_name = self._register_sprite(BrushPattern(
                symbol_layer.color().name(QColor.NameFormat.HexArgb), brush), "pattern")
            layer_def["paint"].update({
                "fill-pattern": pattern_name,
                "fill-opacity": FillPropertyExtractor.get_fill_opacity(symbol_layer, symbol),
                "fill-antialias": FillPropertyExtractor.get_fill_antialias(),
                "fill-translate": FillPropertyExtractor.get_fill_translate(symbol_layer),
                "fill-translate-anchor": FillPropertyExtractor.get_fill_translate_anchor(symbol_layer),
            })
            layer_def["layout"]["visibility"] = "visible"
        elif isinstance(symbol_layer, QgsSimpleFillSymbolLayer):
            if symbol_layer.brushStyle() != Qt.BrushStyle.NoBrush:
                layer_def["paint"].update({
                    "fill-color": FillPropertyExtractor.get_fill_color(symbol_layer),
                    "fill-opacity": FillPropertyExtractor.get_fill_opacity(symbol_layer, symbol),
                    "fill-antialias": FillPropertyExtractor.get_fill_antialias(),
                    "fill-translate": FillPropertyExtractor.get_fill_translate(symbol_layer),
                    "fill-translate-anchor": FillPropertyExtractor.get_fill_translate_anchor(symbol_layer),
                })

            else:
                layer_def["paint"].update({
                    "fill-color": "rgba(0, 0, 0, 0.0)"})
            if symbol_layer.strokeStyle() != Qt.PenStyle.NoPen:
                outline_color = FillPropertyExtractor.get_fill_outline_color(symbol_layer)
                if outline_color:
                    layer_def["paint"]["fill-outline-color"] = outline_color

                layer_def["layout"].update({
                    "fill-sort-key": FillPropertyExtractor.get_fill_sort_key(symbol_layer),
                    "visibility": "visible",
                })
        elif FillPropertyExtractor.is_pattern_fill(symbol_layer):
            # Pattern-based fill: emit fill-pattern; fill-color is ignored by MapLibre.
            pattern_name = self._register_pattern(symbol_layer)
            layer_def["paint"].update({
                "fill-pattern": pattern_name,
                "fill-opacity": FillPropertyExtractor.get_fill_opacity(symbol_layer, symbol),
                "fill-antialias": FillPropertyExtractor.get_fill_antialias(),
                "fill-translate": FillPropertyExtractor.get_fill_translate(symbol_layer),
                "fill-translate-anchor": FillPropertyExtractor.get_fill_translate_anchor(symbol_layer),
            })
            layer_def["layout"].update({
                "fill-sort-key": FillPropertyExtractor.get_fill_sort_key(symbol_layer),
                "visibility": "visible",
            })
        else:
            # A fill MapLibre can't draw (gradient, shapeburst, ...): a flat
            # fill of its mean colour.
            layer_def["paint"].update({
                "fill-color": FillPropertyExtractor.get_fallback_fill_color(symbol_layer),
                "fill-opacity": FillPropertyExtractor.get_fill_opacity(symbol_layer, symbol),
                "fill-antialias": FillPropertyExtractor.get_fill_antialias(),
            })
            layer_def["layout"]["visibility"] = "visible"

        self.style["layers"].append(layer_def)

    def _convert_label(
        self,
        label_settings: QgsPalLayerSettings,
        style_name: str,
        source_layer_name: str,
        source_name: str,
        min_zoom: int = -1,
        max_zoom: int = -1,
        geometry_type=None,
    ):
        """Convert a ``QgsPalLayerSettings`` into a MapLibre ``symbol`` layer."""
        layer_def = self._base_layer_def(
            "symbol", style_name, source_layer_name, source_name, min_zoom, max_zoom
        )

        text_format = label_settings.format()

        text_field = TextPropertyExtractor.get_text_field(label_settings)
        if text_field:
            layer_def["layout"]["text-field"] = text_field
        font = TextPropertyExtractor.get_text_font(text_format)
        text_font, fonts = self._label_fonts(label_settings, text_format, font, source_layer_name)
        # The dataset and field holding this label's text, whose characters
        # the fonts' glyphs are made for.
        dataset = join(self.utils_dir, f'{source_layer_name}.{_TEMP_RULE_FORMAT}')
        for one in fonts:
            self.glyphs.setdefault(one, []).append((dataset, label_settings.fieldName))
        layer_def["layout"].update({
            "text-font": text_font,
            "text-size": TextPropertyExtractor.get_text_size(text_format, label_settings),
            "text-anchor": TextPropertyExtractor.get_text_anchor(label_settings),
            "text-justify": TextPropertyExtractor.get_text_justify(label_settings),
            "text-offset": TextPropertyExtractor.get_text_offset(label_settings),
            "text-allow-overlap": TextPropertyExtractor.get_text_allow_overlap(label_settings),
            "text-ignore-placement": TextPropertyExtractor.get_text_ignore_placement(),
            "text-optional": TextPropertyExtractor.get_text_optional(),
            "text-padding": TextPropertyExtractor.get_text_padding(),
            "text-line-height": TextPropertyExtractor.get_text_line_height(text_format),
            "text-letter-spacing": TextPropertyExtractor.get_text_letter_spacing(text_format),
            "text-transform": TextPropertyExtractor.get_text_transform(label_settings),
            "text-max-width": TextPropertyExtractor.get_text_max_width(label_settings),
            "text-max-angle": TextPropertyExtractor.get_text_max_angle(label_settings),
            "text-keep-upright": TextPropertyExtractor.get_text_keep_upright(),
            "text-rotate": TextPropertyExtractor.get_text_rotate(label_settings),
            "text-rotation-alignment": TextPropertyExtractor.get_text_rotation_alignment(),
            "text-pitch-alignment": TextPropertyExtractor.get_text_pitch_alignment(),
            "symbol-placement": IconPropertyExtractor.get_symbol_placement(label_settings),
            "symbol-spacing": IconPropertyExtractor.get_symbol_spacing(label_settings),
            "symbol-avoid-edges": IconPropertyExtractor.get_symbol_avoid_edges(),
            "symbol-sort-key": TextPropertyExtractor.get_text_sort_key(label_settings),
            "symbol-z-order": IconPropertyExtractor.get_symbol_z_order(),
            "visibility": "visible",
        })

        # Placements that let QGIS choose the side of the point. MapLibre
        # uses a radial offset instead of text-offset with them.
        variable_anchor = TextPropertyExtractor.get_text_variable_anchor(label_settings)
        if variable_anchor:
            radial = TextPropertyExtractor.get_text_radial_offset(label_settings)
            if not TextPropertyExtractor.keeps_callout_side(label_settings):
                layer_def["layout"]["text-variable-anchor"] = variable_anchor
                layer_def["layout"]["text-radial-offset"] = radial
            # Renderers without variable anchors (OpenLayers' ol-mapbox-style,
            # QGIS's style converter) read text-anchor and text-offset
            # instead: the first position, at the same distance. MapLibre
            # ignores both when the variable anchors are set.
            layer_def["layout"]["text-anchor"] = variable_anchor[0]
            layer_def["layout"]["text-offset"] = TextPropertyExtractor.radial_to_offset(
                variable_anchor[0], radial
            )
        # A point displacement renderer's own labels: placed by the tiles,
        # beside each point, all drawn, as QGIS draws them.
        # A callout drawn in the tiles is drawn whatever the labels collide
        # with: its label is shown on it (and still keeps others away).
        if TextPropertyExtractor.has_drawn_callout(label_settings, geometry_type):
            layer_def["layout"]["text-allow-overlap"] = True
        displacement = displacement_label_settings(label_settings)
        if displacement is not None:
            layer_def["metadata"] = {"q2vt:displacement": self._displacement_meta(displacement)}
            layer_def["layout"].update({
                "text-allow-overlap": True, "text-ignore-placement": True,
                "text-anchor": ["coalesce", ["get", DISPLACEMENT_ANCHOR_FIELD], "bottom-left"],
                "text-offset": [0, 0],
            })
        # Labels at their data-defined position: anchored there as QGIS
        # aligns them (variable anchors can't vary by feature: those
        # labels then keep them).
        pinned_anchor = TextPropertyExtractor.get_pinned_anchor(label_settings)
        pinned_def = None
        if pinned_anchor and text_field:
            layout = layer_def["layout"]
            pinned = ["==", ["get", pinned_field(text_field[1])], 1]
            if variable_anchor:
                # Variable anchors can't vary by feature: the labels at their
                # position are a layer of their own, anchored there (above,
                # placed first), the others keep theirs.
                pinned_def = copy.deepcopy(layer_def)
                pinned_def["id"] = f"{style_name}_pinned"
                pinned_def["layout"].pop("text-variable-anchor", None)
                pinned_def["layout"].pop("text-radial-offset", None)
                pinned_def["layout"]["text-anchor"] = pinned_anchor
                pinned_def["layout"]["text-offset"] = [0, 0]
                pinned_def["_only"] = pinned
                layer_def["_only"] = ["!=", ["get", pinned_field(text_field[1])], 1]
            else:
                offset = layout["text-offset"]
                if not (offset and isinstance(offset[0], str)):
                    offset = ["literal", offset]
                layout["text-anchor"] = ["case", pinned, pinned_anchor, layout["text-anchor"]]
                layout["text-offset"] = ["case", pinned, ["literal", [0, 0]], offset]


        #         "text-halo-blur": TextPropertyExtractor.get_text_halo_blur(
        #     text_format, label_settings
        # ),
        layer_def["paint"].update({
            "text-color": TextPropertyExtractor.get_text_color(text_format, label_settings),
            "text-opacity": TextPropertyExtractor.get_text_opacity(text_format, label_settings),
            "text-halo-color": TextPropertyExtractor.get_text_halo_color(
                text_format, label_settings
            ),
            "text-halo-width": TextPropertyExtractor.get_text_halo_width(text_format, label_settings),
            "text-translate": TextPropertyExtractor.get_text_translate(),
            "text-translate-anchor": TextPropertyExtractor.get_text_translate_anchor(),
        })

        # Deep-copy label settings to avoid mutating the caller's object.
        label_settings = QgsPalLayerSettings(label_settings)
        label_format = QgsTextFormat(label_settings.format())
        background = QgsTextBackgroundSettings(label_format.background())
        if background.markerSymbol():
            background.setMarkerSymbol(background.markerSymbol().clone())
        label_format.setBackground(background)
        label_settings.setFormat(label_format)

        shape_type = PropertyExtractor.enum_int(background.type()) if background.enabled() else None
        if shape_type == _BACKGROUND_MARKER and background.markerSymbol():
            self._apply_icon_from_background(layer_def, background, style_name)
        elif shape_type in _BACKGROUND_SHAPES and (shape := self._label_shape(background, shape_type))[0]:
            self._apply_icon_from_background(layer_def, background, style_name, shape)
        elif TextPropertyExtractor.balloon_icon(label_settings, geometry_type):
            self._apply_icon_from_background(layer_def, background, style_name,
                                             self._balloon_shape(label_settings))
        else:
            self._apply_default_icon_props(layer_def)

        self.style["layers"].append(layer_def)
        if pinned_def is not None:
            for section in ("layout", "paint"):
                for key, value in layer_def.get(section, {}).items():
                    if key not in ("text-variable-anchor", "text-radial-offset"):
                        pinned_def[section].setdefault(key, value)
            if "metadata" in layer_def:
                pinned_def["metadata"] = layer_def["metadata"]
            self.style["layers"].append(pinned_def)

    def _apply_icon_from_background(self, layer_def: dict, background, style_name: str,
                                    shape: Optional[tuple] = None):
        """Configure icon layout/paint from a label background: its marker
        symbol, or a shape's (sprite name, icon-text-fit, padding)."""
        if shape is not None:
            layer_def["layout"]["icon-image"], text_fit, text_fit_padding, background_box = shape
            if background_box:
                # The OpenLayers viewer draws it as the text's own background
                # (it can't stretch an icon around a label).
                layer_def["metadata"] = {"q2vt:text-background": background_box}
        else:
            try:
                marker = background.markerSymbol() if hasattr(background, "markerSymbol") else None
                if marker and marker.type() == QgsSymbol.SymbolType.Marker:
                    layer_def["layout"]["icon-image"] = self._register_sprite(marker.clone(), "marker")
                else:
                    layer_def["layout"]["icon-image"] = style_name
            except (RuntimeError, AttributeError):
                layer_def["layout"]["icon-image"] = style_name
            text_fit = IconPropertyExtractor.get_icon_text_fit(background)
            text_fit_padding = IconPropertyExtractor.get_icon_text_fit_padding(background)

        if text_fit:
            layer_def["layout"]["icon-text-fit"] = text_fit
        if text_fit_padding:
            layer_def["layout"]["icon-text-fit-padding"] = text_fit_padding

        layer_def["layout"].update({
            "icon-anchor": IconPropertyExtractor.get_icon_anchor(),
            "icon-rotate": IconPropertyExtractor.get_icon_rotate(background=background),
            "icon-padding": IconPropertyExtractor.get_icon_padding(),
            "icon-rotation-alignment": IconPropertyExtractor.get_icon_rotation_alignment(),
            "icon-pitch-alignment": IconPropertyExtractor.get_icon_pitch_alignment(),
            "icon-allow-overlap": IconPropertyExtractor.get_icon_allow_overlap(),
            "icon-ignore-placement": IconPropertyExtractor.get_icon_ignore_placement(),
            "icon-keep-upright": IconPropertyExtractor.get_icon_keep_upright(),
            "icon-offset": IconPropertyExtractor.get_icon_offset(background),
        })

        try:
            layer_def["paint"].update({
                "icon-opacity": IconPropertyExtractor.get_icon_opacity(background),
                "icon-color": IconPropertyExtractor.get_icon_color(background),
                "icon-halo-color": IconPropertyExtractor.get_icon_halo_color(background),
                "icon-halo-width": IconPropertyExtractor.get_icon_halo_width(background),
                "icon-halo-blur": IconPropertyExtractor.get_icon_halo_blur(),
                "icon-translate": IconPropertyExtractor.get_icon_translate(),
                "icon-translate-anchor": IconPropertyExtractor.get_icon_translate_anchor(),
            })
        except (OSError, RuntimeError):
            pass

    def _label_shape(self, background: QgsTextBackgroundSettings, shape_type: int) -> tuple:
        """(sprite name, icon-text-fit, icon-text-fit-padding, its box for
        the OpenLayers viewer or None) of a label
        background rectangle, square, ellipse, circle or SVG (see LabelShape):
        stretched around the text plus the buffer size (QGIS's x and y
        margins), or drawn at its fixed size. A square or circle is drawn
        as a rectangle or ellipse: MapLibre can't keep a stretched icon's
        proportions."""
        def px(value, unit, fallback=0.0) -> float:
            length = PropertyExtractor.length_to_style_value(value, unit)
            return length if isinstance(length, (int, float)) else fallback  # Ground: a guess.

        stroke_px = px(background.strokeWidth(), background.strokeWidthUnit(), 1.0)
        radii = background.radii()
        radius_px = 0.0
        if radii.width() > 0:
            if PropertyExtractor.enum_int(background.radiiUnit()) == _PERCENTAGE_UNIT:
                radius_px = radii.width() / 100 * 16  # Of the text's height, about.
            else:
                radius_px = px(radii.width(), background.radiiUnit(), 4.0)
        size = background.size()
        fixed = PropertyExtractor.enum_int(background.sizeType()) == _BACKGROUND_SIZE_FIXED
        shape = LabelShape(
            ellipse=shape_type in _BACKGROUND_ELLIPSES,
            fill=background.fillColor().name(QColor.NameFormat.HexArgb),
            stroke=background.strokeColor().name(QColor.NameFormat.HexArgb),
            stroke_px=stroke_px, radius_px=radius_px,
            width=px(size.width(), background.sizeUnit(), 16.0) if fixed else 0.0,
            height=px(size.height(), background.sizeUnit(), 16.0) if fixed else 0.0,
            svg=background.svgFile() if shape_type == _BACKGROUND_SVG else "",
        )
        if shape_type == _BACKGROUND_SVG and not shape.svg:
            return None, None, None, None
        name = self._register_sprite(shape, "label_bg")
        if fixed:
            return name, None, None, None
        margin_x = max(px(size.width(), background.sizeUnit(), 2.0), 0.0)
        margin_y = max(px(size.height(), background.sizeUnit(), 2.0), 0.0)
        padding = [margin_y, margin_x, margin_y, margin_x]
        box = {
            "fill": PropertyExtractor.convert_qcolor_to_maplibre(background.fillColor()),
            "stroke": PropertyExtractor.convert_qcolor_to_maplibre(background.strokeColor()),
            "stroke-width": stroke_px, "padding": padding,
        }
        return name, "both", padding, box

    def _balloon_shape(self, label_settings: QgsPalLayerSettings) -> tuple:
        """(sprite name, icon-text-fit, icon-text-fit-padding, its box for
        the OpenLayers viewer) of a label's balloon callout drawn as its
        icon (see BalloonShape): its wedge pointing at the label's point,
        from where the style places the text."""
        def px(value, unit) -> float:
            length = PropertyExtractor.length_to_style_value(value, unit)
            return length if isinstance(length, (int, float)) else 0.0  # Ground: none.

        callout = label_settings.callout()
        anchor, offset = TextPropertyExtractor.fixed_anchor(label_settings)
        size = TextPropertyExtractor.get_text_size_px(label_settings.format())
        margins = callout.margins()
        unit = callout.marginsUnit()
        shape = BalloonShape(
            symbol=callout.fillSymbol().clone(), anchor=anchor,
            origin=(-offset[0] * size, -offset[1] * size),
            margins=tuple(px(m, unit) for m in (margins.left(), margins.top(), margins.right(), margins.bottom())),
            wedge_width=px(callout.wedgeWidth(), callout.wedgeWidthUnit()),
            corner_radius=px(callout.cornerRadius(), callout.cornerRadiusUnit()),
            from_anchor=(px(callout.offsetFromAnchor(), callout.offsetFromAnchorUnit())
                         if hasattr(callout, "offsetFromAnchor") else 0.0),
        )
        name = self._register_sprite(shape, "balloon")
        # The text box as QGIS has it (the balloon's inside): MapLibre's is
        # a line height a line, its baselines (as the glyphs are made, see
        # GlyphGenerator) the font's ascent and descent centred on each line;
        # QGIS's starts the first line's ascent above the first baseline and
        # ends below the last by the descent.
        padding = None
        text_format = label_settings.format()
        metrics = TextPropertyExtractor.label_box_metrics(text_format)
        if metrics is not None:
            first, _, last = metrics
            font = QFont(text_format.font())
            font.setPixelSize(24)
            glyph_metrics = QFontMetrics(font)
            centre = (glyph_metrics.ascent() - glyph_metrics.descent()) / 2 / 24
            half_line = TextPropertyExtractor.get_text_line_height(text_format) / 2
            padding = [round((first - half_line - centre) * size, 2), 0,
                       round((last - half_line + centre) * size, 2), 0]
        # The OpenLayers viewer draws its first simple fill as the text's background.
        box = None
        for symbol_layer in callout.fillSymbol().symbolLayers():
            if isinstance(symbol_layer, QgsSimpleFillSymbolLayer) and symbol_layer.enabled():
                no_stroke = symbol_layer.strokeStyle() == Qt.PenStyle.NoPen
                box = {
                    "fill": PropertyExtractor.convert_qcolor_to_maplibre(symbol_layer.fillColor()),
                    "stroke": PropertyExtractor.convert_qcolor_to_maplibre(
                        QColor(0, 0, 0, 0) if no_stroke else symbol_layer.strokeColor()),
                    "stroke-width": 0.0 if no_stroke else px(symbol_layer.strokeWidth(),
                                                             symbol_layer.strokeWidthUnit()),
                    "padding": [shape.margins[1] + (padding[0] if padding else 0), shape.margins[2],
                                shape.margins[3] + (padding[2] if padding else 0), shape.margins[0]],
                }
                break
        return name, "both", padding, box

    def _apply_default_icon_props(self, layer_def: dict):
        """Apply default icon layout/paint when no background marker is present."""
        layer_def["layout"].update({
            "icon-allow-overlap": IconPropertyExtractor.get_icon_allow_overlap(),
            "icon-ignore-placement": IconPropertyExtractor.get_icon_ignore_placement(),
            "icon-optional": IconPropertyExtractor.get_icon_optional(),
            "icon-padding": IconPropertyExtractor.get_icon_padding(),
        })
        layer_def["paint"].update({
            "icon-opacity": IconPropertyExtractor.get_icon_opacity(),
            "icon-halo-color": IconPropertyExtractor.get_icon_halo_color(),
            "icon-halo-width": IconPropertyExtractor.get_icon_halo_width(),
            "icon-halo-blur": IconPropertyExtractor.get_icon_halo_blur(),
            "icon-translate": IconPropertyExtractor.get_icon_translate(),
            "icon-translate-anchor": IconPropertyExtractor.get_icon_translate_anchor(),
        })

    def to_json(self, indent: int = 2) -> str:
        """Serialise the in-memory style dict to a JSON string."""
        return json.dumps(self.style, indent=indent)

    def save_to_file(self, filename: str = "style.json", indent: Optional[int] = None) -> str:
        """Write the style JSON and the sprite sheet to the output directory.

        Args:
            filename: Filename for the JSON file (relative to ``style/``).
            indent:   JSON indentation level for human-readable output; by
                      default compact, the smallest for viewers to load.

        Returns:
            The absolute path of the written ``style.json`` file.
        """
        style_dir = os.path.join(self.output_dir, "style")
        os.makedirs(style_dir, exist_ok=True)

        if self.marker_symbols:
            SpriteGenerator(
                self.marker_symbols, style_dir, _SPRITE_QUALITY, False
            ).generate()
        else:
            del self.style["sprite"]
        if self.glyphs:
            glyphs_dir = os.path.join(self.output_dir, "style", "glyphs")
            os.mkdir(glyphs_dir)
            fontstacks = GlyphGenerator(self.glyphs, glyphs_dir).generate()
            self._use_fontstacks(fontstacks)
        else:
             del self.style["glyphs"]
        rounded_style = self.round_numeric_values(self.style)
        rounded_style.pop("metadata", None)
        strip_defaults(rounded_style["layers"])
        filepath = os.path.join(style_dir, filename)
        with open(filepath, "w", encoding="utf8") as f:
            json.dump(rounded_style, f, indent=indent,
                      separators=None if indent else (",", ":"))
        return filepath
    
    def _use_fontstacks(self, fontstacks: Dict[str, str]) -> None:
        """Point each label's text-font at the glyph folder written for it.

        A font without a style name (e.g. "MS Shell Dlg 2 ") would otherwise
        request glyphs from a folder that doesn't exist. In a data-defined
        font's ``match``, a font whose glyphs weren't written gives way to
        the label's default font (its fallback).
        """
        for layer_def in self.style["layers"]:
            layout = layer_def.get("layout", {})
            fonts = layout.get("text-font")
            if not fonts:
                continue
            if fonts[0] != "match":
                layout["text-font"] = [fontstacks.get(font, font.strip()) for font in fonts]
                continue
            default = fonts[-1][1][0]
            default = fontstacks.get(default, default.strip())
            fonts[-1] = ["literal", [default]]
            for i in range(3, len(fonts) - 1, 2):
                fonts[i] = ["literal", [fontstacks.get(fonts[i][1][0], default)]]

    def round_numeric_values(self, obj):
        """
        Recursively walk through dict/list values.
        Convert anything numeric-like to float and round to 2 decimals.
        """
        if isinstance(obj, dict):
            return {k: self.round_numeric_values(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.round_numeric_values(item) for item in obj]
        if isinstance(obj, bool):
            return obj
        try:
            # Try converting strings, ints, floats, etc.
            num = float(obj)

            # Keep integers as int if possible after rounding. Small values
            # (a map-unit width at zoom 0) keep 3 significant digits instead.
            rounded = round(num, 2) if abs(num) >= 0.1 or num == 0 else float(f"{num:.3g}")
            return int(rounded) if rounded.is_integer() else rounded

        except (ValueError, TypeError):
            # Not numeric → keep original
            return obj



if __name__ == "__console__":
    exporter = QgisMapLibreStyleExporter(output_dir=QgsProcessingUtils.tempFolder())
    output_file = exporter.export()