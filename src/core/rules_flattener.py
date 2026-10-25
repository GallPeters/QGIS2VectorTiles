"""
rules_flattener.py

RulesFlattener — walks every visible vector layer in the current QGIS project,
converts any non-rule-based renderer / labeling system to rule-based, then
recursively flattens the rule hierarchy with full property inheritance
(scale range, filter expression, symbol layers).

Depends on: config, zoom_levels, flattened_rule
"""

import json
from typing import List, Optional, Union

from qgis.core import (
    Qgis,
    QgsProject,
    QgsVectorLayer,
    QgsRuleBasedRenderer,
    QgsRuleBasedLabeling,
    QgsPalLayerSettings,
    QgsTextFormat,
    QgsTextBackgroundSettings,
    QgsGraduatedSymbolRenderer,
    QgsCategorizedSymbolRenderer,
    QgsReadWriteContext,
    QgsExpression,
    QgsExpressionNode,
    QgsSimpleLineSymbolLayer,
    QgsFillSymbol,
    QgsSymbolLayer,
    QgsSimpleFillSymbolLayer,
    QgsProperty,
    QgsArrowSymbolLayer,
    QgsEllipseSymbolLayer,
    QgsGeometryGeneratorSymbolLayer,
    QgsLineSymbol,
    QgsMarkerLineSymbolLayer,
    QgsMarkerSymbol,
    QgsSimpleMarkerSymbolLayer,
    QgsGradientColorRamp,
    )
from qgis.PyQt.QtCore import QPointF

from ..utils.config import Qt
from ..utils.config import QDomDocument
from ..utils.flattened_rule import (
    FlattenedRule, MARKER_ALL_VERTICES, MARKER_CENTRAL_POINT, MARKER_FIRST_VERTEX,
    MARKER_INNER_VERTICES, MARKER_LAST_VERTEX, MARKER_SEGMENT_CENTERS, OFFSET_PROPERTY,
    HEATMAP_PROPERTY, CLUSTER_PROPERTY,
)
from ..utils.zoom_levels import ZoomLevels


class RulesFlattener:
    """Flatten QGIS rule-based styling with full property inheritance."""

    RULE_TYPES = {0: "renderer", 1: "labeling"}

    def __init__(self, min_zoom: int, max_zoom: int, utils_dir, feedback):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.utils_dir = utils_dir
        self.layer_tree_root = QgsProject.instance().layerTreeRoot()
        self.flattened_rules: List[FlattenedRule] = []
        self.feedback = feedback
        # Tree-unique counter; reset per (layer, rule_type) pass. Used only to
        # disambiguate output_dataset when sibling subtrees share (l,t,d,r,...).
        self._unique_counter = 0
        # Numbers the rules split into zoom bands (FlattenedRule.band_group).
        self._band_groups = 0

    def flatten_all_rules(self) -> List[FlattenedRule]:
        """Extract and flatten all rules from visible vector layers."""
        for layer_idx, node in enumerate(self.layer_tree_root.findLayers()):
            if self._is_valid_layer(node.layer()):
                self._process_layer_rules(node.layer(), layer_idx)

        if self.flattened_rules:
            self._clone_rule_objects()

        self._remove_unusable_rules()
        return self.flattened_rules

    def _remove_unusable_rules(self):
        """Remove rules without a symbol (renderer) or label expression (labeling)."""
        for flat_rule in list(self.flattened_rules):
            usable = False
            if flat_rule.get_attr("t") == 0 and flat_rule.rule.symbol():
                usable = True
            elif flat_rule.get_attr("t") == 1 and flat_rule.rule.settings():
                if flat_rule.rule.settings().getLabelExpression().expression():
                    usable = True
            if not usable:
                self.flattened_rules.remove(flat_rule)

    def _clone_rule_objects(self):
        """Clone each rule's underlying object to prevent shared-property mutations."""
        for flat_rule in self.flattened_rules:
            rule = flat_rule.rule
            rule_type = flat_rule.get_attr("t")

            if rule_type == 0:
                clone = QgsRuleBasedRenderer.Rule(None)
            else:
                clone = QgsRuleBasedLabeling.Rule(None)

            clone.setDescription(rule.description())
            clone.setFilterExpression(rule.filterExpression())
            clone.setMinimumScale(rule.minimumScale())
            clone.setMaximumScale(rule.maximumScale())

            if rule_type == 0 and rule.symbol():
                clone.setSymbol(rule.symbol().clone())
            elif rule_type == 1 and rule.settings():
                new_settings = QgsPalLayerSettings(rule.settings())
                new_format = QgsTextFormat(new_settings.format())
                new_bg = QgsTextBackgroundSettings(new_format.background())
                if new_bg.markerSymbol():
                    new_bg.setMarkerSymbol(new_bg.markerSymbol().clone())
                new_format.setBackground(new_bg)
                new_settings.setFormat(new_format)
                clone.setSettings(new_settings)

            flat_rule.rule = clone

    def _is_valid_layer(self, layer) -> bool:
        """Return True if the layer is a visible non-geometry-collection vector layer."""
        is_vector = layer.type() == 0 and layer.geometryType() != 4
        node = self.layer_tree_root.findLayer(layer.id())
        is_visible = node.isVisible() if node is not None else False
        return is_vector and is_visible

    def _process_layer_rules(self, layer: QgsVectorLayer, layer_idx: int):
        """Process both renderer and labeling rules for a single layer."""
        for rule_type, type_name in self.RULE_TYPES.items():
            rule_system = self._get_or_convert_rule_system(layer, rule_type)
            if not rule_system:
                continue
            # rule_system is a converted clone; it is NOT set on the layer.
            # Doing so changed the user's project styling and, from the
            # processing thread, flooded the GUI with legend refreshes.
            # Flattened rules are clones, so rule_system only needs to live
            # for this pass.
            root_rule = self._prepare_root_rule(rule_system, layer)
            if root_rule:
                # Reset per (layer, rule_type) pass; values must stay < 100
                # because FlattenedRule.set_attr formats as 2 digits.
                self._unique_counter = 0
                self._flatten_rule(layer, layer_idx, root_rule, rule_type, 0, 0)

    def _get_or_convert_rule_system(self, layer: QgsVectorLayer, rule_type: int):
        """Return the layer's rule system, converting from single/graduated/categorized if needed."""
        if rule_type == 0:
            return self._convert_renderer_to_rules(layer)
        return self._convert_labeling_to_rules(layer)

    def _convert_renderer_to_rules(self, layer: QgsVectorLayer):
        """Convert any renderer to a QgsRuleBasedRenderer, preserving active items only."""
        system = layer.renderer()
        if not system:
            return None

        system = system.clone()
        if isinstance(system, QgsRuleBasedRenderer):
            return system
        if system.type() == "heatmapRenderer":
            return self._heatmap_rules(system, layer)
        if system.type() == "pointCluster":
            return self._cluster_rules(system)

        inactive_indices = self._get_inactive_item_indices(system)
        rule_renderer = QgsRuleBasedRenderer.convertFromRenderer(system)

        if rule_renderer and inactive_indices:
            for idx in sorted(inactive_indices, reverse=True):
                rule_renderer.rootRule().removeChildAt(idx)

        return rule_renderer

    @staticmethod
    def _heatmap_rules(heatmap, layer: QgsVectorLayer):
        """A point layer's heatmap renderer as one rule drawing every
        feature with a placeholder marker that carries the heatmap's
        settings (see HEATMAP_PROPERTY); None for other layers (QGIS draws
        heatmaps of points)."""
        if layer.geometryType() != Qgis.GeometryType.Point:
            return None
        ramp = heatmap.colorRamp()
        stops = []
        if ramp is not None:
            points = [i / 10 for i in range(11)]
            if isinstance(ramp, QgsGradientColorRamp) and not ramp.isDiscrete():
                points = sorted({0.0, 1.0, *(stop.offset for stop in ramp.stops())})
            for t in points:
                color = ramp.color(t)
                stops.append([t, [color.red(), color.green(), color.blue(), color.alphaF()]])
        scale = heatmap.radiusMapUnitScale()
        settings = {
            "radius": heatmap.radius(),
            "unit": int(getattr(heatmap.radiusUnit(), "value", heatmap.radiusUnit())),
            "min_scale": scale.minScale, "max_scale": scale.maxScale,
            "max_value": heatmap.maximumValue(),
            "stops": stops,
        }
        marker = QgsSimpleMarkerSymbolLayer()
        marker.setDataDefinedProperty(HEATMAP_PROPERTY, QgsProperty.fromValue(json.dumps(settings), False))
        weight = heatmap.weightExpression()
        if weight:
            marker.setDataDefinedProperty(QgsSymbolLayer.Property.PropertySize,
                                          QgsProperty.fromExpression(weight))
        root = QgsRuleBasedRenderer.Rule(None)
        root.appendChild(QgsRuleBasedRenderer.Rule(QgsMarkerSymbol([marker])))
        return QgsRuleBasedRenderer(root)

    def _cluster_rules(self, cluster):
        """A point cluster renderer as its own renderer's rules (the points
        drawn alone) and, drawn above them, a rule drawing every point with
        the cluster symbol, carrying the clustering distance (see
        CLUSTER_PROPERTY)."""
        embedded = cluster.embeddedRenderer()
        rules = QgsRuleBasedRenderer.convertFromRenderer(embedded) if embedded else None
        if rules is None:
            return None
        symbol = cluster.clusterSymbol()
        if symbol is None or not symbol.symbolLayerCount():
            return rules
        symbol = symbol.clone()
        scale = cluster.toleranceMapUnitScale()
        settings = {
            "tolerance": cluster.tolerance(),
            "unit": int(getattr(cluster.toleranceUnit(), "value", cluster.toleranceUnit())),
            "min_scale": scale.minScale, "max_scale": scale.maxScale,
        }
        # On every symbol layer: each becomes a rule of its own.
        for index in range(symbol.symbolLayerCount()):
            symbol.symbolLayer(index).setDataDefinedProperty(
                CLUSTER_PROPERTY, QgsProperty.fromValue(json.dumps(settings), False))
        rules.rootRule().appendChild(QgsRuleBasedRenderer.Rule(symbol))
        return rules

    def _get_inactive_item_indices(self, system) -> List[int]:
        """Return indices of inactive items from graduated or categorized renderers."""
        items_method = None
        if isinstance(system, QgsGraduatedSymbolRenderer):
            items_method = "ranges"
        elif isinstance(system, QgsCategorizedSymbolRenderer):
            items_method = "categories"

        if items_method:
            items = getattr(system, items_method)()
            return [i for i, item in enumerate(items) if not item.renderState()]
        return []

    def _convert_labeling_to_rules(self, layer: QgsVectorLayer):
        """Convert any labeling system to QgsRuleBasedLabeling."""
        system = layer.labeling()
        if not system or not layer.labelsEnabled():
            return None

        system = system.clone()
        if isinstance(system, QgsRuleBasedLabeling):
            return system

        rule = QgsRuleBasedLabeling.Rule(system.settings())
        root = QgsRuleBasedLabeling.Rule(QgsPalLayerSettings())
        root.appendChild(rule)
        return QgsRuleBasedLabeling(root)

    def _prepare_root_rule(self, rule_system, layer: QgsVectorLayer):
        """Set layer-level scale visibility on the root rule."""
        root_rule = rule_system.rootRule()
        if layer.hasScaleBasedVisibility():
            root_rule.setMinimumScale(layer.minimumScale())
            root_rule.setMaximumScale(layer.maximumScale())
        return root_rule

    def _set_rule_attributes(
        self,
        flat_rule: FlattenedRule,
        layer_idx: int,
        rule_type: int,
        rule_level: int,
        rule_idx: int,
    ):
        """Stamp identification and zoom attributes onto the flat rule."""
        flat_rule.set_attr("l", layer_idx)
        flat_rule.set_attr("t", rule_type)
        flat_rule.set_attr("d", rule_level)
        flat_rule.set_attr("r", rule_idx)
        flat_rule.set_attr("g", flat_rule.layer.geometryType())
        flat_rule.set_attr("c", flat_rule.layer.geometryType())
        flat_rule.set_attr("o", self._rule_min_zoom(flat_rule.rule))
        flat_rule.set_attr("i", self._rule_max_zoom(flat_rule.rule))
        # "s" = symbol layer index for renderer; "f" = renderer index for labeling
        flat_rule.set_attr("s" if rule_type == 0 else "f", 0)

    def _rule_min_zoom(self, rule) -> int:
        return int(ZoomLevels.scale_to_zoom(rule.minimumScale(), "o"))

    def _rule_max_zoom(self, rule) -> int:
        return int(ZoomLevels.scale_to_zoom(rule.maximumScale(), "i"))

    def _flatten_rule(
        self,
        layer: QgsVectorLayer,
        layer_idx: int,
        rule: Union[QgsRuleBasedLabeling.Rule, QgsRuleBasedRenderer.Rule],
        rule_type: int,
        rule_level: int,
        rule_idx: int,
        inherited_parent=None,
    ):
        """Recursively flatten the rule hierarchy with property inheritance."""
        # Children of an unprocessed node (the root) inherit from the node itself;
        # children of a processed node inherit from its accumulated (flattened) state.
        parent_for_children = rule
        if rule.parent():
            inheritance_source = self._process_rule(
                layer, layer_idx, rule, rule_type, rule_level, rule_idx, inherited_parent
            )
            if inheritance_source is not None:
                parent_for_children = inheritance_source
        for child_idx, child in enumerate(rule.children()):
            if not child.active():
                continue
            if child.filterExpression() == "ELSE":
                self._convert_else_filter(child, rule)
            self._flatten_rule(
                layer, layer_idx, child, rule_type, rule_level + 1, child_idx,
                parent_for_children,
            )

    def _process_rule(
        self,
        layer: QgsVectorLayer,
        layer_idx: int,
        rule,
        rule_type: int,
        rule_level: int,
        rule_idx: int,
        inherited_parent,
    ):
        """Inherit, validate, split, register; return the cumulative source for descendants."""
        if rule_type == 1:
            self._sync_labeling_scale_range(rule)

        inherited_rule = self._inherit_rule_properties(rule, rule_type, inherited_parent)
        if not inherited_rule:
            return None

        # Snapshot cumulative state (AND-chain filter, intersected scales, stacked
        # symbols) BEFORE applying NOT-children — descendants must not see the
        # output-only exclusion (otherwise C inherits "... AND NOT C" from B*).
        inheritance_source = inherited_rule.clone()
        # self._exclude_children_from_filter(inherited_rule, rule)

        flat_rule = FlattenedRule(inherited_rule, layer)
        flat_rule.rule.setDescription("")
        self._set_rule_attributes(flat_rule, layer_idx, rule_type, rule_level, rule_idx)

        if not self._is_within_zoom_range(flat_rule):
            return inheritance_source

        split_rules = self._split_rule(flat_rule, rule_type)
        self._clip_rules_to_zoom_range(split_rules)
        for split_rule in split_rules:
            self.flattened_rules.extend(self._split_by_scale_expressions(split_rule))
        return inheritance_source

    def _sync_labeling_scale_range(self, rule):
        """Copy label settings scale visibility to the rule if the rule has no range set."""
        if rule.minimumScale() != 0 or rule.maximumScale() != 0:
            return
        settings = rule.settings()
        if settings and settings.scaleVisibility:
            rule.setMinimumScale(settings.minimumScale)
            rule.setMaximumScale(settings.maximumScale)
            settings.scaleVisibility = False

    def _inherit_rule_properties(self, rule, rule_type: int, inherited_parent):
        """Clone and inherit scale range, filter expression, and symbol layers from cumulative parent."""
        clone = rule.clone()
        self._inherit_min_scale(clone, rule, inherited_parent)
        self._inherit_max_scale(clone, rule, inherited_parent)
        self._inherit_filter_expression(clone, rule, inherited_parent)
        # if rule_type == 0:
        #     self._inherit_symbol_layers(clone, rule, inherited_parent)
        return clone

    def _inherit_min_scale(self, clone, rule, inherited_parent):
        """Inherit minimum scale; 0 means no restriction → use the largest available scale.

        The 0-substitution is applied symmetrically to both the rule's own scale
        and the parent's scale; otherwise an unrestricted ancestor would zero out
        every descendant when min(child, 0) is computed.
        """
        rule_scale = rule.minimumScale() if rule.minimumScale() != 0 else max(ZoomLevels.SCALES)
        parent_min = inherited_parent.minimumScale()
        parent_scale = parent_min if parent_min != 0 else max(ZoomLevels.SCALES)
        clone.setMinimumScale(min(rule_scale, parent_scale))

    def _inherit_max_scale(self, clone, rule, inherited_parent):
        """Inherit maximum scale; 0 means no restriction → use the smallest available scale."""
        rule_scale = rule.maximumScale() if rule.maximumScale() != 0 else min(ZoomLevels.SCALES)
        parent_max = inherited_parent.maximumScale()
        parent_scale = parent_max if parent_max != 0 else min(ZoomLevels.SCALES)
        clone.setMaximumScale(max(rule_scale, parent_scale))

    def _inherit_filter_expression(self, clone, rule, inherited_parent):
        """Combine cumulative ancestor filter with this rule's filter (AND-chain only).

        NOT-children exclusion is NOT applied here — it would poison the snapshot
        used for descendant inheritance. It is applied to the output clone in
        _exclude_children_from_filter after the snapshot is taken.
        """
        parent_filter = inherited_parent.filterExpression() if inherited_parent is not None else ""
        rule_filter = rule.filterExpression()

        if parent_filter and rule_filter:
            combined = f"({parent_filter}) AND ({rule_filter})"
        else:
            combined = parent_filter or rule_filter or ""

        clone.setFilterExpression(combined)

    def _exclude_children_from_filter(self, clone, rule):
        """Exclude this rule's children's filter conditions (output-only — descendants must
        not see this exclusion or grandchildren would inherit a tautologically-false filter)."""
        child_filters = [
            f"({child.filterExpression()})"
            for child in rule.children()
            if child.filterExpression() and child.filterExpression() != "ELSE"
        ]
        if not child_filters:
            return
        children_expr = " OR ".join(child_filters)
        combined = clone.filterExpression()
        final = f"({combined}) AND NOT ({children_expr})" if combined else f"NOT ({children_expr})"
        clone.setFilterExpression(final)

    def _inherit_symbol_layers(self, clone, rule, inherited_parent):
        """Append the cumulative parent's symbol layers to the cloned rule's symbol."""
        clone_symbol = clone.symbol()
        parent_symbol = inherited_parent.symbol() if inherited_parent is not None else None
        if parent_symbol and clone_symbol:
            for i in range(parent_symbol.symbolLayerCount()):
                clone_symbol.appendSymbolLayer(parent_symbol.symbolLayer(i).clone())

    def _split_rule(self, flat_rule: FlattenedRule, rule_type: int) -> List[FlattenedRule]:
        """Split rule by symbol layers (renderer) or matching renderers (labeling)."""
        if rule_type == 0:
            return self._split_by_symbol_layers(flat_rule)
        return self._split_by_matching_renderers(flat_rule)

    def _convert_else_filter(self, else_rule, parent_rule):
        """Replace ELSE filter with an explicit exclusion of sibling conditions."""
        sibling_filters = [
            sibling.filterExpression()
            for sibling in parent_rule.children()
            if sibling.active() and sibling.filterExpression() not in ("ELSE", "")
        ]
        if sibling_filters:
            else_rule.setFilterExpression(
                f'NOT ({" OR ".join(f"({f})" for f in sibling_filters)}) IS 1'
            )
        else:
            else_rule.setFilterExpression("")

    def _is_within_zoom_range(self, flat_rule: FlattenedRule) -> bool:
        """Return True if the rule's zoom range overlaps with the requested range."""
        return self._ranges_overlap(
            flat_rule.get_attr("o"), flat_rule.get_attr("i"),
            self.min_zoom, self.max_zoom,
        )

    def _clip_rules_to_zoom_range(self, flat_rules: List[FlattenedRule]):
        """Clamp each rule's zoom range to the global min/max zoom."""
        for flat_rule in flat_rules:
            if flat_rule.get_attr("o") < self.min_zoom:
                flat_rule.set_attr("o", self.min_zoom)
            if flat_rule.get_attr("i") > self.max_zoom:
                flat_rule.set_attr("i", self.max_zoom)

    def _split_by_symbol_layers(self, flat_rule: FlattenedRule) -> List[FlattenedRule]:
        """Split a renderer rule into one rule per enabled symbol layer."""
        symbol = flat_rule.rule.symbol()
        if not symbol:
            return [flat_rule]
        self._expand_symbol_layers(symbol, flat_rule.layer)

        layer_count = symbol.symbolLayerCount()
        split_rules = []

        for layer_idx in reversed(range(layer_count)):
            symbol_layer = symbol.symbolLayer(layer_idx)
            if not symbol_layer.enabled():
                continue
            geom_generator = None
            sub_symbol = symbol_layer.subSymbol()
            layer_type = symbol_layer.layerType() if symbol_layer else None
            if layer_type == "GeometryGenerator":
                symbol_type = sub_symbol.type()
                geom_generator = True
            elif layer_type == "CentroidFill":
                symbol_type = 0
            else:
                symbol_type = symbol_layer.type()
                if int(symbol_type) == 1 and flat_rule.get_attr("g") == 2:
                    # A line of a fill symbol (an outline): drawn along the
                    # polygons' own rings, from their dataset (the web map
                    # strokes polygons), not from a copy of their boundaries.
                    symbol_type = 2

            rule_clone = FlattenedRule(flat_rule.rule.clone(), flat_rule.layer)
            rule_clone.set_attr("c", symbol_type)
            rule_clone.set_attr("s", layer_idx, geom_generator)

            clone_symbol = rule_clone.rule.symbol()
            for remove_idx in reversed(range(layer_count)):
                if remove_idx != layer_idx:
                    clone_symbol.deleteSymbolLayer(remove_idx)


            clone_symbol_layer = clone_symbol.symbolLayers()[0]
            if layer_type == "MarkerLine":
                split_rules.extend(self._split_marker_line(rule_clone, flat_rule.layer))
                continue
            if rule_clone and layer_type == "SimpleFill":
                if clone_symbol_layer.strokeStyle() != Qt.PenStyle.NoPen:
                    # Drawn along the polygons' rings, from their dataset.
                    outline_rule = FlattenedRule(rule_clone.rule.clone(), flat_rule.layer)
                    outline_rule.set_attr("c", 2)
                    fill_symbol = outline_rule.rule.symbol()
                    outline_symbol = self._convert_fill_outline_to_line_symbol(fill_symbol)
                    if outline_symbol:
                        outline_rule.rule.setSymbol(outline_symbol)
                        split_rules.append(outline_rule)
                        clone_symbol_layer.setStrokeStyle(Qt.PenStyle.NoPen)
                if clone_symbol_layer.brushStyle() == Qt.BrushStyle.NoBrush:
                    continue
            if rule_clone:
                split_rules.append(rule_clone)

        return split_rules

    # Marker line placements drawn at points (exported as point rules), and
    # the "p" attribute of each.
    _POINT_PLACEMENTS = (
        (Qgis.MarkerLinePlacement.FirstVertex, MARKER_FIRST_VERTEX),
        (Qgis.MarkerLinePlacement.LastVertex, MARKER_LAST_VERTEX),
        (Qgis.MarkerLinePlacement.CentralPoint, MARKER_CENTRAL_POINT),
        (Qgis.MarkerLinePlacement.Vertex, MARKER_ALL_VERTICES),
        (Qgis.MarkerLinePlacement.InnerVertices, MARKER_INNER_VERTICES),
        (Qgis.MarkerLinePlacement.SegmentCenter, MARKER_SEGMENT_CENTERS),
    )

    # --- Symbol layers drawn as others the web map can draw ---

    def _expand_symbol_layers(self, symbol, layer: QgsVectorLayer) -> None:
        """Replace, in place, symbol layers the web map can't draw by ones
        it can, drawing the same: an arrow by its body (a line) and its
        head(s) (markers at its end(s), turned with it); a trimmed line of a
        line layer by a geometry generator cutting the line, then drawing it
        untrimmed."""
        for index in reversed(range(symbol.symbolLayerCount())):
            symbol_layer = symbol.symbolLayer(index)
            if not symbol_layer.enabled():
                continue
            if isinstance(symbol_layer, QgsArrowSymbolLayer):
                replacements = self._arrow_layers(symbol_layer)
            elif (isinstance(symbol_layer, QgsSimpleLineSymbolLayer)
                  and layer.geometryType() == Qgis.GeometryType.Line
                  and (symbol_layer.trimDistanceStart() or symbol_layer.trimDistanceEnd())):
                replacements = [self._trimmed_line(symbol_layer, layer)]
            else:
                continue
            symbol.deleteSymbolLayer(index)
            for offset, replacement in enumerate(replacements):
                symbol.insertSymbolLayer(index + offset, replacement)

    def _arrow_layers(self, arrow: QgsArrowSymbolLayer) -> list:
        """An arrow as a line (its body, at its mean width) and marker lines
        drawing its head at its end, start or both: a triangle of the head's
        length and width (the body's plus its thickness on each side), its
        tip at the line's end, pointing along it. Curved and half arrows
        are drawn straight and whole."""
        fill = arrow.subSymbol()
        color = fill.color() if fill is not None else arrow.color()
        width = arrow.arrowWidth()
        if arrow.arrowStartWidth() > 0:
            width = (width + arrow.arrowStartWidth()) / 2
        body = QgsSimpleLineSymbolLayer(color, width)
        body.setWidthUnit(arrow.arrowWidthUnit())
        body.setWidthMapUnitScale(arrow.arrowWidthUnitScale())
        body.setPenCapStyle(Qt.PenCapStyle.FlatCap)
        layers = [body]
        head_type = arrow.headType()
        ends = {
            QgsArrowSymbolLayer.HeadType.HeadSingle: [(Qgis.MarkerLinePlacement.LastVertex, 90)],
            QgsArrowSymbolLayer.HeadType.HeadReversed: [(Qgis.MarkerLinePlacement.FirstVertex, 270)],
            QgsArrowSymbolLayer.HeadType.HeadDouble: [(Qgis.MarkerLinePlacement.FirstVertex, 270),
                                                      (Qgis.MarkerLinePlacement.LastVertex, 90)],
        }.get(head_type, [])
        if arrow.headLength() <= 0:
            return layers
        same_unit = arrow.headThicknessUnit() == arrow.arrowWidthUnit()
        head_width = (arrow.arrowWidth() if same_unit else 0) + 2 * arrow.headThickness()
        for placement, angle in ends:
            head = QgsEllipseSymbolLayer()
            head.setShape(QgsEllipseSymbolLayer.Shape.Triangle)
            head.setSymbolWidth(head_width)
            head.setSymbolWidthUnit(arrow.headThicknessUnit())
            head.setSymbolHeight(arrow.headLength())
            head.setSymbolHeightUnit(arrow.headLengthUnit())
            head.setColor(color)
            head.setStrokeStyle(Qt.PenStyle.NoPen)
            # The tip (the triangle's top) at the line's end: the marker is
            # drawn turned so that its top points along the line.
            head.setOffset(QPointF(0, arrow.headLength() / 2))
            head.setOffsetUnit(arrow.headLengthUnit())
            head.setAngle(angle)
            marker_line = QgsMarkerLineSymbolLayer(True)
            marker_line.setPlacements(placement)
            marker_line.setSubSymbol(QgsMarkerSymbol([head]))
            layers.append(marker_line)
        return layers

    def _trimmed_line(self, line: QgsSimpleLineSymbolLayer, layer: QgsVectorLayer):
        """A line trimmed at its start and end, as a geometry generator
        cutting the line (lengths in screen units follow the scale:
        @map_scale gives each zoom its own cut) and drawing it untrimmed."""
        start = self._ground_length(line.trimDistanceStart(), line.trimDistanceStartUnit(), layer)
        end = self._ground_length(line.trimDistanceEnd(), line.trimDistanceEndUnit(), layer)
        # Generators are evaluated in the layer's CRS (RulesExporter): the
        # line is cut in EPSG:3857, the lengths' unit, and given back.
        crs = layer.crs().authid()
        expression = (
            f"transform(with_variable('q2vt_g', transform(@geometry, '{crs}', 'EPSG:3857'), "
            "with_variable('q2vt_len', length(@q2vt_g), "
            f"if(({start}) + ({end}) < @q2vt_len, "
            f"line_substring(@q2vt_g, {start}, @q2vt_len - ({end})), NULL))), 'EPSG:3857', '{crs}')"
        )
        untrimmed = line.clone()
        untrimmed.setTrimDistanceStart(0)
        untrimmed.setTrimDistanceEnd(0)
        generator = QgsGeometryGeneratorSymbolLayer.create({"geometryModifier": expression})
        generator.setSymbolType(Qgis.SymbolType.Line)
        generator.setSubSymbol(QgsLineSymbol([untrimmed]))
        return generator

    @staticmethod
    def _ground_length(value: float, unit, layer: QgsVectorLayer) -> str:
        """A QGIS length as an expression of EPSG:3857 metres (the unit of
        the exported base layers), with @map_scale for screen units and
        @q2vt_len (the line's length) for percentages."""
        if not value:
            return "0"
        unit = int(getattr(unit, "value", unit))
        stretch = ZoomLevels.GROUND_STRETCH
        if unit == int(Qgis.RenderUnit.Percentage.value):
            return f"(@q2vt_len * {value / 100!r})"
        if unit == int(Qgis.RenderUnit.MapUnits.value):
            metres = 111319.49079327357 if layer.crs().isGeographic() else stretch
            return repr(value * metres)
        if unit == int(Qgis.RenderUnit.MetersInMapUnits.value):
            return repr(value * stretch)
        mm = value * {
            int(Qgis.RenderUnit.Points.value): 25.4 / 72,
            int(Qgis.RenderUnit.Pixels.value): 25.4 / 96,
            int(Qgis.RenderUnit.Inches.value): 25.4,
        }.get(unit, 1.0)
        # On screen: as long on the ground as the scale makes it, in
        # EPSG:3857 metres (see ZoomLevels.map_scale_factor).
        return f"({mm!r} * @map_scale / {1000 * ZoomLevels.map_scale_factor()!r})"

    def _split_marker_line(self, rule_clone: FlattenedRule, layer) -> List[FlattenedRule]:
        """A marker line rule as one point rule per vertex-like placement
        (first / last vertex, central point, vertices, segment centres),
        whose markers are then drawn exactly at those points, plus a line
        rule for the remaining placements (interval, curve points), which
        follow the line.
        """
        symbol_layer = rule_clone.rule.symbol().symbolLayers()[0]
        try:
            placements = symbol_layer.placements()
        except AttributeError:
            return [rule_clone]
        rules = []
        remaining = placements
        for flag, code in self._POINT_PLACEMENTS:
            if not placements & flag:
                continue
            remaining &= ~flag
            if code == MARKER_INNER_VERTICES and placements & Qgis.MarkerLinePlacement.Vertex:
                continue  # Already drawn on every vertex.
            point_rule = FlattenedRule(rule_clone.rule.clone(), layer)
            point_rule.set_attr("c", 0)
            point_rule.set_attr("p", code)
            point_symbol_layer = point_rule.rule.symbol().symbolLayers()[0]
            point_symbol_layer.setPlacements(flag)
            rules.append(point_rule)
        if remaining:
            symbol_layer.setPlacements(remaining)
            rules.append(rule_clone)
        return rules

    @staticmethod
    def _convert_fill_outline_to_line_symbol(fill_symbol: QgsFillSymbol) -> QgsFillSymbol | None:
        """ Convert a QgsFillSymbol containing a single QgsSimpleFillSymbolLayer into 
        a cloned QgsFillSymbol containing a single QgsSimpleLineSymbolLayer representing
        the original outline.
        """
        if not isinstance(fill_symbol, QgsFillSymbol):
            return None

        layers = fill_symbol.symbolLayers()
        if len(layers) != 1:
            return None

        fill_layer = layers[0]
        if not isinstance(fill_layer, QgsSimpleFillSymbolLayer):
            return None

        new_symbol = fill_symbol.clone()
        line_layer = QgsSimpleLineSymbolLayer()

        line_layer.setColor(fill_layer.strokeColor())
        line_layer.setWidth(fill_layer.strokeWidth())
        line_layer.setWidthUnit(fill_layer.strokeWidthUnit())
        line_layer.setPenStyle(fill_layer.strokeStyle())
        line_layer.setPenJoinStyle(fill_layer.penJoinStyle())
        line_layer.setWidthMapUnitScale(fill_layer.strokeWidthMapUnitScale())
        offset = fill_layer.offset()
        if offset.x() or offset.y():
            # The outline moves with the fill (see OFFSET_PROPERTY).
            scale = fill_layer.offsetMapUnitScale()
            line_layer.setDataDefinedProperty(OFFSET_PROPERTY, QgsProperty.fromValue(json.dumps({
                "x": offset.x(), "y": offset.y(),
                "unit": int(getattr(fill_layer.offsetUnit(), "value", fill_layer.offsetUnit())),
                "min_scale": scale.minScale, "max_scale": scale.maxScale,
                "min_mm_on": scale.minSizeMMEnabled, "min_mm": scale.minSizeMM,
                "max_mm_on": scale.maxSizeMMEnabled, "max_mm": scale.maxSizeMM,
            }), False))

        property_keys = [
            QgsSymbolLayer.Property.PropertyStrokeColor,
            QgsSymbolLayer.Property.PropertyStrokeWidth,
            QgsSymbolLayer.Property.PropertyStrokeStyle,
            QgsSymbolLayer.Property.PropertyJoinStyle
        ]

        for key in property_keys:
            prop = fill_layer.dataDefinedProperties().property(key.value)
            if prop is not None and prop.isActive():
                line_layer.setDataDefinedProperty(key, QgsProperty(prop))

        new_symbol.changeSymbolLayer(0, line_layer)

        return new_symbol


    def _split_by_matching_renderers(self, label_rule: FlattenedRule) -> List[FlattenedRule]:
        """Split a label rule by matching renderer rules with overlapping scale ranges."""
        split_rules = []
        seen_datasets: set = set()
        renderer_idx = 0

        for renderer_rule in self.flattened_rules:
            if label_rule.layer.id() != renderer_rule.layer.id():
                continue
            if renderer_rule.get_attr("t") == 1:
                continue
            filter_id = f'{renderer_rule.get_attr('r')}{renderer_rule.get_attr('d')}'
            if filter_id in seen_datasets:
                continue
            matched = self._match_label_to_renderer(label_rule, renderer_rule, renderer_idx)
            if matched:
                split_rules.append(matched)
                seen_datasets.add(filter_id)
            renderer_idx += 1

        return split_rules if split_rules else [label_rule]

    def _match_label_to_renderer(
        self,
        label_rule: FlattenedRule,
        renderer_rule: FlattenedRule,
        renderer_idx: int,
    ) -> Optional[FlattenedRule]:
        """Return a combined label/renderer rule if their zoom ranges overlap."""
        label_min, label_max = label_rule.get_attr("o"), label_rule.get_attr("i")
        renderer_min, renderer_max = renderer_rule.get_attr("o"), renderer_rule.get_attr("i")

        if not self._ranges_overlap(label_min, label_max, renderer_min, renderer_max):
            return None

        rule_clone = FlattenedRule(label_rule.rule.clone(), label_rule.layer)
        label_filter = rule_clone.rule.filterExpression()
        renderer_filter = renderer_rule.rule.filterExpression()

        if label_filter and renderer_filter:
            combined = f"({renderer_filter}) AND ({label_filter})"
        else:
            combined = renderer_filter or label_filter or ""
        rule_clone.rule.setFilterExpression(combined)

        if label_min < renderer_min:
            rule_clone.set_attr("o", renderer_min)
        if label_max > renderer_max:
            rule_clone.set_attr("i", renderer_max)
        # Shown where both are: the narrower scale range (the style draws
        # the exact zooms of the scales, see TilesStyler).
        rule_clone.rule.setMinimumScale(
            min(rule_clone.rule.minimumScale(), renderer_rule.rule.minimumScale()))
        rule_clone.rule.setMaximumScale(
            max(rule_clone.rule.maximumScale(), renderer_rule.rule.maximumScale()))

        rule_clone.set_attr("f", renderer_idx)
        rule_clone.renderer = renderer_rule
        rule_clone.label_filter = label_filter or ""
        return rule_clone

    @staticmethod
    def _ranges_overlap(r1_start: int, r1_end: int, r2_start: int, r2_end: int) -> bool:
        a_min, a_max = sorted((r1_start, r1_end))
        b_min, b_max = sorted((r2_start, r2_end))
        return a_min <= b_max and b_min <= a_max

    def _split_by_scale_expressions(self, flat_rule: FlattenedRule) -> List[FlattenedRule]:
        """Split a rule by zoom level when it contains @map_scale-dependent
        expressions.

        Consecutive zoom levels at which every such expression gives the
        same result share one rule (and one dataset): e.g. a width of
        "CASE WHEN @map_scale > 50000 THEN 1 ELSE 2 END" needs two rules,
        not one per zoom level.
        """
        if not self._has_scale_dependencies(flat_rule):
            return [flat_rule]

        min_zoom = flat_rule.get_attr("o")
        max_zoom = min(self.max_zoom, flat_rule.get_attr("i"))
        document = self._rule_document(flat_rule).toString()
        # [first zoom, last zoom, signature, rule] per run of equal zooms.
        runs: list = []
        for zoom in range(min_zoom, max_zoom + 1):
            clone = self._create_zoom_specific_rule(flat_rule, zoom)
            if flat_rule.get_attr("t") != 1 and not self._symbol_layer_visible_at_zoom(clone):
                continue
            signature = self._scale_signature(document, ZoomLevels.zoom_to_scale(zoom))
            last = runs[-1] if runs else None
            if last is not None and last[1] == zoom - 1 and last[2] == signature:
                last[1] = zoom
            else:
                runs.append([zoom, zoom, signature, clone])
        split_rules = []
        self._band_groups += 1
        for index, (first, last, signature, clone) in enumerate(runs):
            clone.set_attr("o", first)
            clone.set_attr("i", last)
            if len(runs) > 1:
                clone.band_group = self._band_groups
            previous = runs[index - 1] if index else None
            if previous is not None and previous[1] + 1 == first:
                self._start_band_at_switch(document, previous[3], clone, first, signature)
            split_rules.append(clone)
        return split_rules

    # Bisection steps finding where a rule's @map_scale results change
    # between two zoom levels: 1/4096 of a zoom level.
    _SWITCH_STEPS = 12

    def _start_band_at_switch(self, document: str, before: FlattenedRule, band: FlattenedRule,
                              zoom: int, signature: tuple) -> None:
        """Draw a zoom band from the exact scale its results start at, as
        QGIS switches there, not from the next whole zoom level.

        Bands are found at whole zoom levels; the switch lies between the
        previous one (zoom - 1) and this one: bisection finds it. The band
        then starts there (its data from that zoom level's tiles on) and
        the band before ends there; each keeps the scale its expressions
        were evaluated at (scale_zoom).
        """
        low, high = zoom - 1.0, float(zoom)
        for _ in range(self._SWITCH_STEPS):
            middle = (low + high) / 2
            if self._scale_signature(document, ZoomLevels.SCALES[0] / 2 ** middle) == signature:
                high = middle
            else:
                low = middle
        if high >= zoom:
            return
        switch_scale = ZoomLevels.SCALES[0] / 2 ** high
        band.scale_zoom = zoom
        band.set_attr("o", zoom - 1)
        band.rule.setMinimumScale(min(band.rule.minimumScale() or switch_scale, switch_scale))
        before.rule.setMaximumScale(max(before.rule.maximumScale(), switch_scale))

    def _scale_signature(self, document: str, scale_value: float) -> tuple:
        """What a rule's @map_scale expressions give at a scale.

        Each expression (an attribute or text of the rule's XML) with
        @map_scale replaced by the scale is, if it reads no feature,
        evaluated to its value; otherwise kept as written, which differs
        from every other zoom's.
        """
        scale = str(scale_value)
        doc = QDomDocument()
        doc.setContent(document)
        tokens: list = []

        def visit(node):
            if node.isElement():
                attributes = node.toElement().attributes()
                for i in range(attributes.count()):
                    self._scale_token(attributes.item(i).nodeValue(), scale, tokens)
            elif node.isText():
                self._scale_token(node.nodeValue(), scale, tokens)
            child = node.firstChild()
            while not child.isNull():
                visit(child)
                child = child.nextSibling()

        visit(doc.documentElement())
        return tuple(tokens)

    @classmethod
    def _scale_token(cls, text: str, scale: str, tokens: list) -> None:
        """One expression's part of a zoom's signature: its value if it
        reads no feature, else its form with every part that reads no
        feature folded to its value ("@map_scale < 30000 OR "rank" >= 4"
        is "false OR "rank" >= 4" at every zoom out of 1:30,000), so zooms
        it gives the same results at share one rule."""
        if not text or "@map_scale" not in text:
            return
        replaced = text.replace("@map_scale", scale)
        expr = QgsExpression(replaced)
        if expr.hasParserError() or expr.rootNode() is None:
            tokens.append(("expression", replaced))
            return
        tokens.append(("value", cls._folded(expr.rootNode())))

    _Node = QgsExpressionNode.NodeType

    @classmethod
    def _folded(cls, node) -> str:
        """A canonical form of an expression node, its parts that read no
        feature, field, geometry or variable replaced by their values:
        equal forms give equal results for every feature."""
        part = QgsExpression(node.dump())
        if (not part.hasParserError() and not part.referencedColumns()
                and not part.needsGeometry() and not part.referencedVariables()):
            value = part.evaluate()
            if not part.hasEvalError():
                return f"<{value!r}>"
        node_type = node.nodeType()
        if node_type == cls._Node.ntBinaryOperator:
            return f"({cls._folded(node.opLeft())} {getattr(node.op(), "value", node.op())} {cls._folded(node.opRight())})"
        if node_type == cls._Node.ntUnaryOperator:
            return f"({getattr(node.op(), 'value', node.op())} {cls._folded(node.operand())})"
        if node_type == cls._Node.ntFunction:
            args = node.args().list() if node.args() else []
            return f"f{node.fnIndex()}({', '.join(cls._folded(a) for a in args)})"
        if node_type == cls._Node.ntCondition:
            whens = " ".join(f"when {cls._folded(w.whenExp())} then {cls._folded(w.thenExp())}"
                             for w in node.conditions())
            otherwise = cls._folded(node.elseExp()) if node.elseExp() is not None else ""
            return f"case {whens} else {otherwise}"
        if node_type == cls._Node.ntInOperator:
            items = ", ".join(cls._folded(i) for i in node.list().list())
            return f"({cls._folded(node.node())} {'not in' if node.isNotIn() else 'in'} [{items}])"
        return node.dump()

    def _symbol_layer_visible_at_zoom(self, flat_rule: FlattenedRule) -> bool:
        """Return False if the symbol layer's visibility DDP evaluates to falsy at the zoom."""
        symbol = flat_rule.rule.symbol()
        if not symbol:
            return True
        symbol_layer = symbol.symbolLayers()[0]
        vis_prop = symbol_layer.dataDefinedProperties().property(44)
        if vis_prop and vis_prop.isActive():
            min_scale = str(ZoomLevels.zoom_to_scale(flat_rule.get_attr("o")))
            zoom_expr = QgsExpression(vis_prop.expressionString().replace("@map_scale", min_scale))
            # Only an expression of the scale alone tells (one reading the
            # feature can't be evaluated without one: some functions crash).
            if (zoom_expr.hasParserError() or zoom_expr.referencedColumns()
                    or zoom_expr.needsGeometry() or zoom_expr.referencedVariables()):
                return True
            evaluation = zoom_expr.evaluate()
            if evaluation is not None and not evaluation:
                return False
        return True

    def _has_scale_dependencies(self, flat_rule: FlattenedRule) -> bool:
        """Return True if the rule's serialized XML contains @map_scale."""
        return "@map_scale" in self._rule_document(flat_rule).toString()

    @staticmethod
    def _rule_document(flat_rule: FlattenedRule) -> QDomDocument:
        """The rule, saved as a renderer or labeling XML document."""
        doc = QDomDocument("style")
        context = QgsReadWriteContext()
        rule_clone = flat_rule.rule.clone()

        if flat_rule.get_attr("t") == 1:
            root = QgsRuleBasedLabeling.Rule(None)
            root.appendChild(rule_clone)
            elem = QgsRuleBasedLabeling(root).save(doc, context)
        else:
            root = QgsRuleBasedRenderer.Rule(None)
            root.appendChild(rule_clone)
            elem = QgsRuleBasedRenderer(root).save(doc, context)

        doc.appendChild(elem)
        return doc

    def _create_zoom_specific_rule(self, flat_rule: FlattenedRule, zoom: int) -> FlattenedRule:
        """Clone a rule with @map_scale replaced by the exact scale for the given zoom."""
        rule_clone = FlattenedRule(flat_rule.rule.clone(), flat_rule.layer)
        scale = str(ZoomLevels.zoom_to_scale(zoom))
        rule_clone.renderer = flat_rule.renderer
        if flat_rule.label_filter is not None:
            rule_clone.label_filter = flat_rule.label_filter.replace("@map_scale", scale)

        filter_exp = flat_rule.rule.filterExpression()
        if "@map_scale" in filter_exp:
            rule_clone.rule.setFilterExpression(filter_exp.replace("@map_scale", scale))

        if flat_rule.get_attr("t") == 1 and flat_rule.rule.settings():
            label_exp = flat_rule.rule.settings().getLabelExpression().expression()
            if label_exp and "@map_scale" in label_exp:
                rule_clone.rule.settings().fieldName = label_exp.replace("@map_scale", scale)

        rule_clone.set_attr("o", zoom)
        rule_clone.set_attr("i", zoom)
        return rule_clone