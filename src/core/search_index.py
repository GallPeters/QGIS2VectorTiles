"""
search_index.py

SearchIndexBuilder — writes style/search.json, the index the web viewers
search. It is built from the exported rule datasets, so it holds exactly
what the output package holds: the features the rules show inside the
export extent, with the fields the Included Fields setting exports (all,
visible, or none) and their label text.

    {"layers": [{"name": "...", "fields": ["...", ...]}, ...],
     "features": [[layer index, [value, ...], [west, south, east, north]], ...],
     "truncated": false}

A source feature is in several datasets (fill, outline, label...) and, when
multi-part, in several parts: they are merged by layer and q2vt_orig_id.

Depends on: config, flattened_rule
"""

import json
import math
from os import makedirs
from os.path import dirname
from typing import Any, Dict, List, Optional, Tuple

from qgis.core import (
    QgsFeatureRequest,
    QgsRectangle,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QDate, QDateTime, QTime

from ..utils.config import _FIELD_PREFIX
from ..utils.flattened_rule import FlattenedRule

# The Web Mercator (EPSG:3857) sphere's radius, in metres.
_MERCATOR_RADIUS = 6378137.0
# Features in the index at most; beyond it the index is cut and says so.
_MAX_FEATURES = 200000
# Entries read at most: duplicates (see _without_duplicates) are removed
# before the index is cut to _MAX_FEATURES.
_MAX_READ_FEATURES = 4 * _MAX_FEATURES
# Longest field value kept, in characters.
_MAX_VALUE_LENGTH = 120
# Entries with the same name this close (degrees, about 1 km) are one place.
_DUPLICATE_DISTANCE = 0.01
_ORIG_ID = f"{_FIELD_PREFIX}_orig_id"
_LABEL = f"{_FIELD_PREFIX}_label"


class _IndexedLayer:
    """One source layer's merged features while the index is built."""

    def __init__(self, name: str):
        self.name = name
        self.fields: List[str] = []      # Output field names, in first-seen order.
        # Zoom levels (the style's) at which the layer's rules draw it.
        self.minzoom: Optional[int] = None
        self.maxzoom: Optional[int] = None
        self.features: Dict[Any, Tuple[Dict[str, Any], QgsRectangle]] = {}


class SearchIndexBuilder:
    """Build the viewers' search index from the exported rule datasets."""

    def __init__(
        self,
        datasets: List[QgsVectorLayer],
        rules: List[FlattenedRule],
        output_path: str,
        feedback,
    ):
        self.datasets = datasets
        self.rules = rules
        self.output_path = output_path
        self.feedback = feedback

    def build(self) -> Optional[str]:
        """Write the index; returns its path, or None if nothing was indexed."""
        source_of = {rule.output_dataset: rule.layer for rule in self.rules}
        layers: Dict[str, _IndexedLayer] = {}
        for rule in self.rules:
            indexed = layers.setdefault(rule.layer.id(), _IndexedLayer(rule.layer.name()))
            low, high = rule.get_attr("o"), rule.get_attr("i")
            if low is not None:
                indexed.minzoom = low if indexed.minzoom is None else min(indexed.minzoom, low)
            if high is not None:
                indexed.maxzoom = high if indexed.maxzoom is None else max(indexed.maxzoom, high)
        count = 0
        truncated = False
        for dataset in self.datasets:
            source = source_of.get(dataset.name())
            if source is None:
                continue
            indexed = layers.setdefault(source.id(), _IndexedLayer(source.name()))
            count, cut = self._read_dataset(dataset, indexed, count)
            truncated = truncated or cut
            if self.feedback.isCanceled():
                return None
        if not count:
            return None
        truncated = self._write(list(layers.values()), truncated)
        if truncated:
            self.feedback.pushWarning(
                f"The search index holds the first {_MAX_FEATURES} features only."
            )
        return self.output_path

    def _read_dataset(self, dataset: QgsVectorLayer, indexed: _IndexedLayer, count: int) -> Tuple[int, bool]:
        """Merge one rule dataset into its layer's features; returns the new
        feature count and whether the cap cut it short."""
        fields = dataset.fields()
        orig_idx = fields.lookupField(_ORIG_ID)
        if orig_idx < 0:
            return count, False
        searchable = [
            (i, fields.at(i).name()) for i in range(fields.count())
            if self._searchable(fields.at(i).name())
        ]
        # A dataset's label fields (q2vt_label, or q2vt_label_NNNN for labels
        # drawn from their features' rows) are one "label" column.
        searchable = [(i, _LABEL if name.startswith(_LABEL) else name) for i, name in searchable]
        for _, name in searchable:
            if name not in indexed.fields:
                indexed.fields.append(name)
        request = QgsFeatureRequest()
        request.setInvalidGeometryCheck(QgsFeatureRequest.InvalidGeometryCheck.GeometryNoCheck)
        truncated = False
        for feature in dataset.getFeatures(request):
            geometry = feature.geometry()
            if geometry.isNull():
                continue
            key = feature.attribute(orig_idx)
            entry = indexed.features.get(key)
            if entry is None:
                if count >= _MAX_READ_FEATURES:
                    truncated = True
                    continue
                entry = indexed.features[key] = ({}, QgsRectangle(geometry.boundingBox()))
                count += 1
            else:
                entry[1].combineExtentWith(geometry.boundingBox())
            values = entry[0]
            for i, name in searchable:
                if values.get(name) is None:
                    value = self._value(feature.attribute(i))
                    if value is not None:
                        values[name] = value
        return count, truncated

    @staticmethod
    def _searchable(name: str) -> bool:
        """Output fields worth searching: the source's and the label text,
        not the plugin's other internal fields."""
        if name.startswith(_LABEL):
            return True
        return not name.startswith(f"{_FIELD_PREFIX}_") and name.lower() != "fid"

    @staticmethod
    def _to_wgs84(box: QgsRectangle) -> List[float]:
        """A Web Mercator box as [west, south, east, north] in degrees.

        Web Mercator maps longitude and latitude each on their own axis,
        monotonically, so a box's corners are the corners of its box in
        degrees: two formulas, not QGIS's densified box transform (about
        a hundred projected points per box, seconds for a large index).
        """
        def lon(x: float) -> float:
            return round(math.degrees(x / _MERCATOR_RADIUS), 6)

        def lat(y: float) -> float:
            return round(math.degrees(math.atan(math.sinh(y / _MERCATOR_RADIUS))), 6)

        return [lon(box.xMinimum()), lat(box.yMinimum()), lon(box.xMaximum()), lat(box.yMaximum())]

    def _write(self, layers: List[_IndexedLayer], truncated: bool) -> bool:
        """Write the index; returns whether it was cut short."""
        layers_out: List[Dict[str, Any]] = []
        features_out: List[list] = []
        for layer in layers:
            if not layer.features:
                continue
            index = len(layers_out)
            layers_out.append({
                "name": layer.name,
                "fields": ["label" if f == _LABEL else f for f in layer.fields],
                # The viewers zoom to a found feature within these levels.
                "minzoom": layer.minzoom,
                "maxzoom": layer.maxzoom,
            })
            for values, box in layer.features.values():
                features_out.append([
                    index,
                    [values.get(f) for f in layer.fields],
                    self._to_wgs84(box),
                ])
        features_out = self._without_duplicates(features_out, layers_out)
        if len(features_out) > _MAX_FEATURES:
            features_out = features_out[:_MAX_FEATURES]
            truncated = True
        makedirs(dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"layers": layers_out, "features": features_out, "truncated": truncated},
                f, ensure_ascii=False, separators=(",", ":"),
            )
        return truncated

    @staticmethod
    def _without_duplicates(features: List[list], layers: List[Dict[str, Any]]) -> List[list]:
        """One result per place: entries with the same name whose boxes
        touch or lie within _DUPLICATE_DISTANCE degrees are one feature
        shown by several layers (a layer copied per scale range, a place
        in two themes, a street in segments). They become one entry: the
        one from the layer drawn at the most zoom levels (the viewer zooms
        to it within them), with the boxes' union. Same names elsewhere
        (a town and a street named after it) stay apart.
        """
        def name_of(entry) -> Optional[str]:
            fields = layers[entry[0]]["fields"]
            values = entry[1]
            if "label" in fields and isinstance(values[fields.index("label")], str):
                return values[fields.index("label")].strip().casefold() or None
            return next((v.strip().casefold() for v in values if isinstance(v, str) and v.strip()), None)

        def span(entry) -> int:
            layer = layers[entry[0]]
            return (layer.get("maxzoom") or 22) - (layer.get("minzoom") or 0)

        cell = _DUPLICATE_DISTANCE * 4
        kept: List[Optional[list]] = []
        # Per (name, grid cell): indices into kept of the entries there;
        # per name, the entries too large for the grid (checked one by one).
        grid: Dict[tuple, List[int]] = {}
        large: Dict[str, List[int]] = {}
        by_name: Dict[str, List[int]] = {}

        def cells_of(box) -> Optional[List[tuple]]:
            xs = range(int(math.floor(box[0] / cell)), int(math.floor(box[2] / cell)) + 1)
            ys = range(int(math.floor(box[1] / cell)), int(math.floor(box[3] / cell)) + 1)
            if len(xs) * len(ys) > 64:
                return None
            return [(cx, cy) for cx in xs for cy in ys]

        for entry in features:
            name = name_of(entry)
            if name is None:
                kept.append(entry)
                continue
            west, south, east, north = entry[2]
            near = (west - _DUPLICATE_DISTANCE, south - _DUPLICATE_DISTANCE,
                    east + _DUPLICATE_DISTANCE, north + _DUPLICATE_DISTANCE)
            near_cells = cells_of(near)
            candidates = list(large.get(name, ()))
            if near_cells is None:  # A large box: every entry of the name.
                candidates = list(by_name.get(name, ()))
            else:
                for cx, cy in near_cells:
                    candidates += grid.get((name, cx, cy), ())
            match = None
            for i in candidates:
                box = kept[i][2]
                if box[0] <= near[2] and box[2] >= near[0] and box[1] <= near[3] and box[3] >= near[1]:
                    match = i
                    break
            if match is None:
                kept.append(entry)
                index = len(kept) - 1
                by_name.setdefault(name, []).append(index)
            else:
                index = match
                other = kept[match]
                box = [min(other[2][0], west), min(other[2][1], south),
                       max(other[2][2], east), max(other[2][3], north)]
                best = entry if span(entry) > span(other) else other
                kept[match] = [best[0], best[1], box]
            box_cells = cells_of(kept[index][2])
            if box_cells is None:
                if index not in large.setdefault(name, []):
                    large[name].append(index)
            else:
                for cx, cy in box_cells:
                    slot = grid.setdefault((name, cx, cy), [])
                    if index not in slot:
                        slot.append(index)
        return [entry for entry in kept if entry is not None]

    @staticmethod
    def _value(value) -> Any:
        """A field value as JSON: numbers and booleans as they are, everything
        else as (shortened) text; NULL and empty text as None."""
        if value is None or (hasattr(value, "isNull") and value.isNull()):
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value if value == value else None  # NaN is not JSON.
        if isinstance(value, QDateTime):
            text = value.toString("yyyy-MM-dd HH:mm:ss")
        elif isinstance(value, QDate):
            text = value.toString("yyyy-MM-dd")
        elif isinstance(value, QTime):
            text = value.toString("HH:mm:ss")
        elif isinstance(value, (list, tuple)):
            text = ", ".join(str(v) for v in value)
        else:
            text = str(value)
        text = text.strip()
        return text[:_MAX_VALUE_LENGTH] or None
