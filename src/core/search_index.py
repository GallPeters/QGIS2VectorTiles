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
# Longest field value kept, in characters.
_MAX_VALUE_LENGTH = 120
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
        if truncated:
            self.feedback.pushWarning(
                f"The search index holds the first {_MAX_FEATURES} features only."
            )
        self._write(list(layers.values()), truncated)
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
                if count >= _MAX_FEATURES:
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

    def _write(self, layers: List[_IndexedLayer], truncated: bool) -> None:
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
        makedirs(dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"layers": layers_out, "features": features_out, "truncated": truncated},
                f, ensure_ascii=False, separators=(",", ":"),
            )

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
