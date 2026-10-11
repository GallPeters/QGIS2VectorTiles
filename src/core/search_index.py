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
from os import makedirs
from os.path import dirname
from typing import Any, Dict, List, Optional, Tuple

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsCsException,
    QgsFeatureRequest,
    QgsProject,
    QgsRectangle,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QDate, QDateTime, QTime

from ..utils.config import _EPSG_CRS, _FIELD_PREFIX
from ..utils.flattened_rule import FlattenedRule

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
        if name == _LABEL:
            return True
        return not name.startswith(f"{_FIELD_PREFIX}_") and name.lower() != "fid"

    def _write(self, layers: List[_IndexedLayer], truncated: bool) -> None:
        to_wgs84 = QgsCoordinateTransform(
            QgsCoordinateReferenceSystem(f"EPSG:{_EPSG_CRS}"),
            QgsCoordinateReferenceSystem("EPSG:4326"),
            QgsProject.instance().transformContext(),
        )
        layers_out: List[Dict[str, Any]] = []
        features_out: List[list] = []
        for layer in layers:
            if not layer.features:
                continue
            index = len(layers_out)
            layers_out.append({
                "name": layer.name,
                "fields": ["label" if f == _LABEL else f for f in layer.fields],
            })
            for values, box in layer.features.values():
                try:
                    wgs = to_wgs84.transformBoundingBox(box)
                except QgsCsException:
                    continue
                features_out.append([
                    index,
                    [values.get(f) for f in layer.fields],
                    [round(wgs.xMinimum(), 6), round(wgs.yMinimum(), 6),
                     round(wgs.xMaximum(), 6), round(wgs.yMaximum(), 6)],
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
