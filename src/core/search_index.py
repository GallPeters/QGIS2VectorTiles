"""
search_index.py

SearchIndexBuilder — writes style/search.json, the index the web viewers
search: every feature the exported rules show inside the export extent, with
all its fields and its bounding box (WGS84, clipped to the extent).

    {"layers": [{"name": "...", "fields": ["...", ...]}, ...],
     "features": [[layer index, [value, ...], [west, south, east, north]], ...],
     "truncated": false}

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
    QgsExpression,
    QgsExpressionContext,
    QgsExpressionContextUtils,
    QgsFeatureRequest,
    QgsProject,
    QgsRectangle,
    QgsVectorLayer,
    QgsVectorLayerFeatureSource,
)
from qgis.PyQt.QtCore import QDate, QDateTime, QTime

from ..utils.config import _EPSG_CRS, _FIELD_PREFIX
from ..utils.flattened_rule import FlattenedRule

# Features in the index at most; beyond it the index is cut and says so.
_MAX_FEATURES = 200000
# Longest field value kept, in characters.
_MAX_VALUE_LENGTH = 120
# Providers read from the project layer itself (as in the rules export).
_LIVE_PROVIDERS = frozenset({"memory"})
# Field types that can't be searched as text (binary, geometry, maps).
_SKIPPED_TYPES = frozenset({12, 8})  # QMetaType ByteArray, QVariantMap


class SearchIndexBuilder:
    """Build the viewers' search index from the exported rules' layers."""

    def __init__(self, rules: List[FlattenedRule], extent: QgsRectangle, output_path: str, feedback):
        self.rules = rules
        self.extent = extent
        self.output_path = output_path
        self.feedback = feedback
        self._extent_crs = QgsCoordinateReferenceSystem(f"EPSG:{_EPSG_CRS}")
        self._wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        self._context = QgsProject.instance().transformContext()

    def build(self) -> Optional[str]:
        """Write the index; returns its path, or None if nothing was indexed."""
        layers_out: List[Dict[str, Any]] = []
        features_out: List[list] = []
        truncated = False
        for layer, filter_expression in self._layers_and_filters():
            if len(features_out) >= _MAX_FEATURES:
                truncated = True
                break
            try:
                fields, rows, cut = self._read_layer(
                    layer, filter_expression, _MAX_FEATURES - len(features_out)
                )
            except Exception as e:  # noqa: BLE001  (one layer must not stop the export)
                self.feedback.pushWarning(f'The "{layer.name()}" layer was left out of the search: {e}')
                continue
            truncated = truncated or cut
            if not rows:
                continue
            index = len(layers_out)
            layers_out.append({"name": layer.name(), "fields": fields})
            features_out.extend([index, values, bbox] for values, bbox in rows)
        if not features_out:
            return None
        if truncated:
            self.feedback.pushWarning(
                f"The search index holds the first {_MAX_FEATURES} features only."
            )
        makedirs(dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"layers": layers_out, "features": features_out, "truncated": truncated},
                f, ensure_ascii=False, separators=(",", ":"),
            )
        return self.output_path

    def _layers_and_filters(self) -> List[Tuple[QgsVectorLayer, Optional[str]]]:
        """Each exported layer once, with the filter that selects the
        features its rules show (None: all of them)."""
        layers: Dict[str, QgsVectorLayer] = {}
        filters: Dict[str, Optional[set]] = {}
        for rule in self.rules:
            layer_id = rule.layer.id()
            layers.setdefault(layer_id, rule.layer)
            expression = rule.rule.filterExpression()
            if not expression:
                filters[layer_id] = None  # A rule shows every feature.
            elif filters.get(layer_id, set()) is not None:
                filters.setdefault(layer_id, set()).add(expression)
        result = []
        for layer_id, layer in layers.items():
            expressions = filters.get(layer_id)
            combined = " OR ".join(f"({e})" for e in sorted(expressions)) if expressions else None
            if combined and QgsExpression(combined).hasParserError():
                combined = None
            result.append((layer, combined))
        return result

    def _read_layer(
        self, project_layer: QgsVectorLayer, filter_expression: Optional[str], limit: int
    ) -> Tuple[List[str], List[Tuple[list, list]], bool]:
        """(field names, [(values, bbox)], truncated) for one layer."""
        if project_layer.providerType() in _LIVE_PROVIDERS:
            reader, layer = QgsVectorLayerFeatureSource(project_layer), None
        else:
            layer = QgsVectorLayer(project_layer.source(), project_layer.name(), project_layer.providerType())
            if not layer.isValid():
                raise RuntimeError("the layer could not be opened")
            reader = layer
        fields = project_layer.fields()
        indices = [
            i for i in range(fields.count())
            if self._type_code(fields.at(i).type()) not in _SKIPPED_TYPES
            and not fields.at(i).name().startswith(f"{_FIELD_PREFIX}_")
        ]
        to_extent = QgsCoordinateTransform(project_layer.crs(), self._extent_crs, self._context)
        to_wgs84 = QgsCoordinateTransform(self._extent_crs, self._wgs84, self._context)

        request = QgsFeatureRequest()
        request.setInvalidGeometryCheck(QgsFeatureRequest.InvalidGeometryCheck.GeometryNoCheck)
        try:
            request.setFilterRect(
                QgsCoordinateTransform(self._extent_crs, project_layer.crs(), self._context)
                .transformBoundingBox(self.extent)
            )
        except QgsCsException:
            pass  # Read everything; features are still checked against the extent.
        if filter_expression:
            context = QgsExpressionContext(
                QgsExpressionContextUtils.globalProjectLayerScopes(project_layer)
            )
            request.setFilterExpression(filter_expression)
            request.setExpressionContext(context)

        rows: List[Tuple[list, list]] = []
        truncated = False
        for n, feature in enumerate(reader.getFeatures(request)):
            if n % 1000 == 0 and self.feedback.isCanceled():
                break
            geometry = feature.geometry()
            if geometry.isNull() or geometry.isEmpty():
                continue
            try:
                box = to_extent.transformBoundingBox(geometry.boundingBox())
            except QgsCsException:
                continue
            # Only what lies in the export extent is in the tiles.
            if box.width() == 0 and box.height() == 0:
                if not self.extent.contains(box.center()):
                    continue
            elif self.extent.intersects(box):
                box = box.intersect(self.extent)
            else:
                continue
            try:
                wgs = to_wgs84.transformBoundingBox(box)
            except QgsCsException:
                continue
            if len(rows) >= limit:
                truncated = True
                break
            values = [self._value(feature.attribute(i)) for i in indices]
            bbox = [round(wgs.xMinimum(), 6), round(wgs.yMinimum(), 6),
                    round(wgs.xMaximum(), 6), round(wgs.yMaximum(), 6)]
            rows.append((values, bbox))
        return [fields.at(i).name() for i in indices], rows, truncated

    @staticmethod
    def _type_code(field_type) -> int:
        return int(getattr(field_type, "value", field_type))

    @staticmethod
    def _value(value) -> Any:
        """A field value as JSON: numbers and booleans as they are, everything
        else as (shortened) text; NULL and empty text as null."""
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
