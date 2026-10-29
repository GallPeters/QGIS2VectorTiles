"""
rules_flattener.py

RulesFlattener — walks every visible vector layer in the current QGIS project,
converts any non-rule-based renderer / labeling system to rule-based, then
recursively flattens the rule hierarchy with full property inheritance
(scale range, filter expression, symbol layers).

Depends on: config, zoom_levels, flattened_rule
"""

import json
import math
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
    QgsRenderContext,
    QgsExpression,
    QgsExpressionNode,
    QgsSimpleLineSymbolLayer,
    QgsFillSymbol,
    QgsSymbolLayer,
    QgsSimpleFillSymbolLayer,
    QgsProperty,
    QgsArrowSymbolLayer,
    QgsGeometryGeneratorSymbolLayer,
    QgsLineSymbol,
    QgsMarkerSymbol,
    QgsSimpleMarkerSymbolLayer,
    QgsGradientColorRamp,
    QgsSymbolBufferSettings,
    )
from qgis.PyQt.QtCore import QPointF

from ..utils.config import Qt
from ..utils.config import QDomDocument
from ..utils.flattened_rule import (
    FlattenedRule, MARKER_ALL_VERTICES, MARKER_CENTRAL_POINT, MARKER_FIRST_VERTEX,
    MARKER_INNER_VERTICES, MARKER_LAST_VERTEX, MARKER_SEGMENT_CENTERS, OFFSET_PROPERTY,
    HEATMAP_PROPERTY, CLUSTER_PROPERTY, DISPLACEMENT_LABEL_PROPERTY, displacement_label_settings,
)
from ..utils.zoom_levels import ZoomLevels
# Registers the q2vt_* expression functions the rules' geometry generators call.
from . import symbol_geometry  # noqa: F401  pylint: disable=unused-import
from .symbol_geometry import BALLOON_VARIABLE


# EPSG:3857 metres a screen millimetre covers at web zoom 0 (MapLibre's
# 512-pixel tiles of the world, 96 pixels an inch, as the style sizes them).
_METRES_PER_MM_AT_ZOOM_0 = 96 / 25.4 * 40075016.68557849 / 512


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
                self._distance_from_symbol_point(new_settings, flat_rule.layer)
                clone.setSettings(new_settings)

            flat_rule.rule = clone

    @staticmethod
    def _distance_from_symbol_point(settings: QgsPalLayerSettings, layer) -> None:
        """A "cartographic" label's distance from its symbol's bounds, as
        one from its point (the web map knows no symbol): QGIS adds half the
        symbol's width and height (the layer's largest marker, in pixels)
        along each axis of the label's first position, so a corner adds
        half a size times the square root of two."""
        # pylint: disable=import-outside-toplevel
        from .maplibre_converter import PropertyExtractor, TextPropertyExtractor
        if (PropertyExtractor.enum_int(settings.placement) != TextPropertyExtractor._ORDERED_POSITIONS
                or PropertyExtractor.enum_int(settings.offsetType)
                != PropertyExtractor.enum_int(Qgis.LabelOffsetType.FromSymbolBounds)):
            return
        try:
            context = QgsRenderContext()
            context.setScaleFactor(96 / 25.4)
            sizes = [max(bounds.width(), bounds.height())
                     for symbol in layer.renderer().symbols(context)
                     if symbol.type() == Qgis.SymbolType.Marker
                     for bounds in [symbol.bounds(QPointF(0, 0), context)]]
        except (AttributeError, RuntimeError):
            return
        anchors = TextPropertyExtractor.get_text_variable_anchor(settings) or ["center"]
        half = max(sizes, default=0.0) / 2
        extra = half * (math.sqrt(2) if "-" in anchors[0] else 1)
        distance = PropertyExtractor.convert_length_to_pixels(
            settings.dist, PropertyExtractor.get_attribute(settings, "distUnits", "distUnit")) or 0.0
        settings.dist = distance + extra
        settings.distUnits = Qgis.RenderUnit.Pixels
        settings.offsetType = Qgis.LabelOffsetType.FromPoint

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
        if system.type() == "pointDisplacement":
            return self._displacement_rules(system)

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

    def _displacement_rules(self, displacement):
        """A point displacement renderer as its own renderer's rules (each
        point drawn where QGIS draws it: alone, or around its group's
        centre) and, drawn below them, a rule drawing the groups' circle (a
        placeholder marker) and the centre symbol's rules, all carrying the
        displacement settings (see displacement_settings). The tiles hold,
        per zoom level, the points moved and the groups' centres
        (GDALTilesGenerator)."""
        embedded = displacement.embeddedRenderer()
        rules = QgsRuleBasedRenderer.convertFromRenderer(embedded) if embedded else None
        if rules is None:
            return None
        settings = self._displacement_settings(displacement)
        center = displacement.centerSymbol()
        color = displacement.circleColor()

        def marked(symbol, role: str):
            for index in range(symbol.symbolLayerCount()):
                symbol.symbolLayer(index).setDataDefinedProperty(
                    CLUSTER_PROPERTY, QgsProperty.fromValue(json.dumps({**settings, "role": role}), False))
            return symbol

        # QGIS draws a group's circle, then its centre symbol, then its points.
        if center is not None and center.symbolLayerCount():
            rules.rootRule().insertChild(0, QgsRuleBasedRenderer.Rule(marked(center.clone(), "center")))
        if color.isValid() and color.alpha() > 0:
            circle = QgsSimpleMarkerSymbolLayer()
            rules.rootRule().insertChild(0, QgsRuleBasedRenderer.Rule(marked(QgsMarkerSymbol([circle]), "circle")))
        return rules

    @staticmethod
    def _displacement_settings(displacement) -> dict:
        """A point displacement renderer's settings, as the style converter
        and GDALTilesGenerator read them (see displacement_settings)."""
        # pylint: disable=import-outside-toplevel
        from .maplibre_converter import PropertyExtractor
        embedded = displacement.embeddedRenderer()
        # The largest symbol's diagonal (QGIS spaces the points by the
        # group's largest).
        diagonal = 0.0
        for symbol in (embedded.symbols(QgsRenderContext()) if embedded else []):
            if isinstance(symbol, QgsMarkerSymbol):
                size = PropertyExtractor.convert_length_to_pixels(symbol.size(), symbol.sizeUnit())
                diagonal = max(diagonal, math.sqrt(2) * (size or 0))
        center = displacement.centerSymbol()
        center_diagonal = 0.0
        if center is not None:
            size = PropertyExtractor.convert_length_to_pixels(center.size(), center.sizeUnit())
            center_diagonal = math.sqrt(2) * (size or 0)
        scale = displacement.toleranceMapUnitScale()
        color = displacement.circleColor()
        settings = {
            "tolerance": displacement.tolerance(),
            "unit": int(getattr(displacement.toleranceUnit(), "value", displacement.toleranceUnit())),
            "min_scale": scale.minScale, "max_scale": scale.maxScale,
            "placement": int(getattr(displacement.placement(), "value", displacement.placement())),
            "diagonal_px": diagonal, "center_px": center_diagonal,
            "addition_px": PropertyExtractor.convert_length_to_pixels(
                displacement.circleRadiusAddition(), Qgis.RenderUnit.Millimeters),
            "circle_px": PropertyExtractor.convert_length_to_pixels(
                displacement.circleWidth(), Qgis.RenderUnit.Millimeters),
            "circle_color": [color.red(), color.green(), color.blue(), color.alphaF()],
            "label_factor": displacement.labelDistanceFactor(),
        }
        return settings

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
        """Convert any labeling system to QgsRuleBasedLabeling, with a point
        displacement renderer's own labels (see _displacement_label)."""
        system = layer.labeling()
        renderer_label = self._displacement_label(layer)
        if (not system or not layer.labelsEnabled()) and renderer_label is None:
            return None
        if system and layer.labelsEnabled():
            system = system.clone()
            if not isinstance(system, QgsRuleBasedLabeling):
                root = QgsRuleBasedLabeling.Rule(QgsPalLayerSettings())
                root.appendChild(QgsRuleBasedLabeling.Rule(system.settings()))
                system = QgsRuleBasedLabeling(root)
        else:
            system = QgsRuleBasedLabeling(QgsRuleBasedLabeling.Rule(QgsPalLayerSettings()))
        if renderer_label is not None:
            system.rootRule().appendChild(renderer_label)
        return system

    def _displacement_label(self, layer: QgsVectorLayer):
        """A point displacement renderer's own labels (its label attribute,
        font and colour, drawn by every point, from its label scale in) as
        a label rule carrying the displacement settings (see
        displacement_label_settings): placed in the tiles, per zoom level,
        where QGIS draws them, beside each point (GDALTilesGenerator)."""
        renderer = layer.renderer()
        if renderer is None or renderer.type() != "pointDisplacement" or not renderer.labelAttributeName():
            return None
        settings = QgsPalLayerSettings()
        settings.fieldName = renderer.labelAttributeName()
        settings.isExpression = False
        settings.placement = Qgis.LabelPlacement.OverPoint
        settings.quadOffset = Qgis.LabelQuadrantPosition.AboveRight
        settings.displayAll = True
        text_format = QgsTextFormat()
        font = renderer.labelFont()
        text_format.setFont(font)
        text_format.setSize(font.pointSizeF() if font.pointSizeF() > 0 else 10)
        text_format.setSizeUnit(Qgis.RenderUnit.Points)
        text_format.setColor(renderer.labelColor())
        settings.setFormat(text_format)
        settings.dataDefinedProperties().setProperty(
            DISPLACEMENT_LABEL_PROPERTY,
            QgsProperty.fromValue(json.dumps({**self._displacement_settings(renderer), "role": "label"}), False))
        rule = QgsRuleBasedLabeling.Rule(settings)
        if renderer.minimumLabelScale() > 0:
            rule.setMinimumScale(renderer.minimumLabelScale())
        return rule

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

        callouts = self._callout_rules(flat_rule) if rule_type == 1 else []
        split_rules = self._split_rule(flat_rule, rule_type) + callouts
        self._clip_rules_to_zoom_range(split_rules)
        for split_rule in split_rules:
            self.flattened_rules.extend(self._split_by_scale_expressions(split_rule))
        return inheritance_source

    # The "s" attribute of a label's callout rule (a renderer rule of the
    # label's own numbers, no symbol layer has).
    CALLOUT_SYMBOL_LAYER = 99

    # q2vt_callout's kind of each QGIS callout type.
    _CALLOUT_KINDS = {"simple": 0, "manhattan": 1, "curved": 2, "balloon": 3}

    def _callout_rules(self, label_rule: FlattenedRule) -> List[FlattenedRule]:
        """A label's callout, for labels at a fixed place: at a data-defined
        position (X, Y), or a point's label at its point (a balloon there is
        the label's icon instead: TextPropertyExtractor.balloon_icon). As
        renderer rules drawing what QGIS draws (q2vt_callout): the simple,
        Manhattan or curved line from the label's box (as the web map lays
        it out there) to the feature, with the callout's line symbol, or the
        balloon around the label pointing at the feature, with each layer of
        its fill symbol (fills and their outlines, geometry generators drawn
        from the balloon's shape, such as a marker in its corner)."""
        # pylint: disable=import-outside-toplevel
        from .maplibre_converter import TextPropertyExtractor
        settings = label_rule.rule.settings()
        callout = settings.callout() if settings else None
        kind = self._CALLOUT_KINDS.get(callout.type()) if callout is not None else None
        if callout is None or not callout.enabled() or kind is None:
            return []
        balloon = kind == self._CALLOUT_KINDS["balloon"]
        symbol = callout.fillSymbol() if balloon else callout.lineSymbol()
        if symbol is None:
            return []
        props = settings.dataDefinedProperties()
        x = props.property(QgsPalLayerSettings.Property.PositionX)
        y = props.property(QgsPalLayerSettings.Property.PositionY)
        positioned = bool(x and y and x.isActive() and y.isActive())
        layer = label_rule.layer
        fixed = layer.geometryType() == Qgis.GeometryType.Point
        if not positioned and not fixed:
            return []
        if TextPropertyExtractor.balloon_icon(settings, layer.geometryType()):
            return []  # The label's icon (see BalloonShape).
        label_exp = settings.getLabelExpression().expression()
        if not label_exp:
            return []
        text = label_exp if settings.isExpression else QgsExpression.quotedColumnRef(label_exp)
        if settings.wrapChar:
            text = f"replace(({text}), {QgsExpression.quotedString(settings.wrapChar)}, '\\n')"
        case = TextPropertyExtractor.get_text_transform(settings)
        if case != "none":
            text = f"{'upper' if case == 'uppercase' else 'lower'}({text})"
        text_format = settings.format()
        crs = layer.crs().authid()
        if positioned:
            # The label's box as the web map lays it out at the position
            # (aligned there as QGIS aligns it: the style converter's anchor).
            anchor, offset = TextPropertyExtractor.pinned_anchor_expression(settings), [0, 0]
            point = (f"transform(make_point(to_real({x.asExpression()}), to_real({y.asExpression()})), "
                     f"'{crs}', 'EPSG:3857')")
        else:
            # At the point, with the anchor and offset the style gives it
            # (a label with a callout placed around its point keeps the first
            # side: TextPropertyExtractor.keeps_callout_side).
            anchor_name, offset = TextPropertyExtractor.fixed_anchor(settings)
            if not all(isinstance(v, (int, float)) for v in offset):
                return []  # A data-defined offset: the box isn't known.
            anchor, point = QgsExpression.quotedString(anchor_name), "@geometry"
        ground = lambda value, unit: self._ground_length(value, unit, layer)  # noqa: E731

        def length(name: str) -> str:
            """A callout length (one a balloon has none of: 0)."""
            if not hasattr(callout, name):
                return "0"
            return ground(getattr(callout, name)(), getattr(callout, f"{name}Unit")())
        curvature = callout.curvature() if hasattr(callout, "curvature") else 0
        orientation = int(callout.orientation()) if hasattr(callout, "orientation") else 0
        wedge = ground(callout.wedgeWidth(), callout.wedgeWidthUnit()) if balloon else "0"
        corner = ground(callout.cornerRadius(), callout.cornerRadiusUnit()) if balloon else "0"
        margins = callout.margins() if balloon else None
        margin_values = ([ground(m, callout.marginsUnit()) for m in
                          (margins.left(), margins.top(), margins.right(), margins.bottom())]
                         if balloon else ["0"] * 4)
        arguments = [
            "@geometry", point, f"to_string({text})",
            QgsExpression.quotedString(text_format.font().family()),
            repr(TextPropertyExtractor.get_text_size_px(text_format)),
            repr(TextPropertyExtractor.get_text_line_height(text_format)),
            anchor, repr(offset[0]), repr(offset[1]),
            str(int(callout.anchorPoint())), str(int(callout.labelAnchorPoint())),
            str(kind),
            length("minimumLength"), length("offsetFromAnchor"), length("offsetFromLabel"),
            ground(1, Qgis.RenderUnit.Pixels),
            repr(float(curvature)), str(orientation), wedge, corner, *margin_values,
        ]

        def callout_expression(generator: str = "") -> str:
            return self._in_3857(
                f"q2vt_callout({', '.join(arguments + [QgsExpression.quotedString(generator)])})", layer)
        expression = callout_expression()

        def rule_drawing(sub_symbol, geometry: str = expression, index: int = 0) -> FlattenedRule:
            generator = QgsGeometryGeneratorSymbolLayer.create({"geometryModifier": geometry})
            generator.setSymbolType(sub_symbol.type())
            generator.setSubSymbol(sub_symbol)
            rule_symbol = (QgsFillSymbol if balloon else QgsLineSymbol)([generator])
            rule = QgsRuleBasedRenderer.Rule(rule_symbol)
            rule.setFilterExpression(label_rule.rule.filterExpression())
            rule.setMinimumScale(label_rule.rule.minimumScale())
            rule.setMaximumScale(label_rule.rule.maximumScale())
            # The label's numbers, as a renderer rule's ("f": the label's).
            rule.setDescription(label_rule.rule.description().replace("f00", ""))
            callout_rule = FlattenedRule(rule, layer)
            callout_rule.set_attr("t", 0)
            callout_rule.set_attr("c", {Qgis.SymbolType.Marker: 0, Qgis.SymbolType.Line: 1}.get(
                sub_symbol.type(), 2))
            # A callout symbol layer's own number (its index below 99).
            callout_rule.set_attr("s", self.CALLOUT_SYMBOL_LAYER - index)
            return callout_rule

        if not balloon:
            return [rule_drawing(symbol.clone())]
        # The balloon's fill symbol, layer by layer, the top one first (as
        # _split_by_symbol_layers lists a symbol's layers).
        rules = []
        for index in reversed(range(symbol.symbolLayerCount())):
            symbol_layer = symbol.symbolLayer(index)
            if not symbol_layer.enabled():
                continue
            if symbol_layer.layerType() == "GeometryGenerator" and symbol_layer.subSymbol() is not None:
                # Drawn from the balloon (as QGIS evaluates it: q2vt_callout).
                sub_symbol = symbol_layer.subSymbol().clone()
                sub_symbol.setOpacity(sub_symbol.opacity() * symbol.opacity())
                # Its geometry as a variable (the export rewrites @geometry).
                generator = symbol_layer.geometryExpression()
                for name in ("@geometry", "$geometry"):
                    generator = generator.replace(name, f"@{BALLOON_VARIABLE}")
                rules.append(rule_drawing(sub_symbol, callout_expression(generator), index))
                continue
            fill = QgsFillSymbol([symbol_layer.clone()])
            fill.setOpacity(symbol.opacity())
            if isinstance(symbol_layer, QgsSimpleFillSymbolLayer):
                # Its outline, as a fill's: a rule of its own, along the
                # rings, listed first (drawn above it).
                outline = self._convert_fill_outline_to_line_symbol(fill)
                if outline and symbol_layer.strokeStyle() != Qt.PenStyle.NoPen:
                    rules.append(rule_drawing(outline, expression, index))
                    fill.symbolLayer(0).setStrokeStyle(Qt.PenStyle.NoPen)
                    if fill.symbolLayer(0).brushStyle() == Qt.BrushStyle.NoBrush:
                        continue
            rules.append(rule_drawing(fill, expression, index))
        return rules

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

    @staticmethod
    def _without_buffer(symbol) -> None:
        """The symbol with its buffer (QgsSymbolBufferSettings) turned off."""
        try:
            buffer = symbol.bufferSettings()
            if buffer is None or not buffer.enabled():
                return
            settings = QgsSymbolBufferSettings(buffer)
            settings.setEnabled(False)
            symbol.setBufferSettings(settings)
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _split_by_symbol_layers(self, flat_rule: FlattenedRule) -> List[FlattenedRule]:
        """Split a renderer rule into one rule per enabled symbol layer."""
        symbol = flat_rule.rule.symbol()
        if not symbol:
            return [flat_rule]
        self._expand_symbol_layers(symbol, flat_rule.layer)

        layer_count = symbol.symbolLayerCount()
        split_rules = []
        bottom_idx = next((i for i in range(layer_count) if symbol.symbolLayer(i).enabled()), 0)

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
            if layer_idx != bottom_idx:
                # The symbol's buffer, which QGIS draws once around the whole
                # symbol: kept by its bottom layer alone (most often holding
                # the others), not around each (a halo filling an SVG's holes).
                self._without_buffer(clone_symbol)

            clone_symbol_layer = clone_symbol.symbolLayers()[0]
            if layer_type == "GeometryGenerator" and int(symbol_type) == 2:
                # A generated fill's outline, as a fill's (below): a rule of
                # its own, drawing the generated polygons' rings.
                fill = clone_symbol_layer.subSymbol()
                outline_symbol = self._convert_fill_outline_to_line_symbol(fill)
                if outline_symbol and fill.symbolLayer(0).strokeStyle() != Qt.PenStyle.NoPen:
                    outline_rule = FlattenedRule(rule_clone.rule.clone(), flat_rule.layer)
                    outline_rule.rule.symbol().symbolLayer(0).setSubSymbol(outline_symbol)
                    split_rules.append(outline_rule)
                    fill.symbolLayer(0).setStrokeStyle(Qt.PenStyle.NoPen)
                    if fill.symbolLayer(0).brushStyle() == Qt.BrushStyle.NoBrush:
                        continue
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
        it can, drawing the same: an arrow by a geometry generator making
        the arrows' polygons, filled with its fill symbol; a trimmed line of
        a line layer by one cutting the line, then drawing it untrimmed; a
        fill with a data-defined offset by one moving the polygons, then
        drawing them unmoved."""
        for index in reversed(range(symbol.symbolLayerCount())):
            symbol_layer = symbol.symbolLayer(index)
            if not symbol_layer.enabled():
                continue
            if isinstance(symbol_layer, QgsArrowSymbolLayer):
                replacement = self._arrow_generator(symbol_layer, layer)
            elif (isinstance(symbol_layer, QgsSimpleLineSymbolLayer)
                  and layer.geometryType() == Qgis.GeometryType.Line
                  and (symbol_layer.trimDistanceStart() or symbol_layer.trimDistanceEnd()
                       or self._active(symbol_layer, "TrimStart")
                       or self._active(symbol_layer, "TrimEnd"))):
                replacement = self._trimmed_line(symbol_layer, layer)
            elif (type(symbol_layer).__name__ in self._OFFSET_FILLS
                  and self._active(symbol_layer, "Offset")):
                replacement = self._offset_fill(symbol_layer, layer)
            else:
                continue
            symbol.deleteSymbolLayer(index)
            symbol.insertSymbolLayer(index, replacement)

    # Fill symbol layers whose offset() is an x/y offset of the whole fill.
    _OFFSET_FILLS = frozenset({"QgsSimpleFillSymbolLayer", "QgsGradientFillSymbolLayer",
                               "QgsShapeburstFillSymbolLayer"})

    @staticmethod
    def _property_key(name: str):
        """A data-defined property key by its name ("Offset", "TrimStart"...)."""
        key = getattr(QgsSymbolLayer.Property, name, None)
        return key if key is not None else getattr(QgsSymbolLayer.Property, f"Property{name}", None)

    @classmethod
    def _property(cls, symbol_layer, name: str):
        """A symbol layer's data-defined property, by its key's name."""
        key = cls._property_key(name)
        return None if key is None else symbol_layer.dataDefinedProperties().property(key)

    @classmethod
    def _active(cls, symbol_layer, name: str) -> bool:
        prop = cls._property(symbol_layer, name)
        return bool(prop and prop.isActive())

    @classmethod
    def _deactivate(cls, symbol_layer, name: str) -> None:
        """Turn a symbol layer's data-defined property off (its value is
        then drawn into the geometry)."""
        prop = cls._property(symbol_layer, name)
        if prop is not None and prop.isActive():
            prop.setActive(False)
            symbol_layer.setDataDefinedProperty(cls._property_key(name), prop)

    @classmethod
    def _number(cls, symbol_layer, name: str, static: float):
        """A length as QGIS reads it: its data-defined value (an expression
        string) where a feature has a number, else ``static``."""
        prop = cls._property(symbol_layer, name)
        if not (prop and prop.isActive()):
            return static
        return f"coalesce(to_real({prop.asExpression()}), {static!r})"

    @staticmethod
    def _in_3857(expression: str, layer: QgsVectorLayer) -> str:
        """A geometry expression written for EPSG:3857 (its @geometry and
        its result transformed): generators are evaluated in the layer's
        CRS (RulesExporter)."""
        crs = layer.crs().authid()
        geometry = f"transform(@geometry, '{crs}', 'EPSG:3857')"
        return f"transform({expression.replace('@geometry', geometry)}, 'EPSG:3857', '{crs}')"

    @classmethod
    def _type_value(cls, symbol_layer, name: str) -> str:
        """A type property as an expression: its data-defined value, else
        NULL (the static type, see q2vt_arrow)."""
        prop = cls._property(symbol_layer, name)
        return f"({prop.asExpression()})" if prop and prop.isActive() else "NULL"

    def _arrow_generator(self, arrow: QgsArrowSymbolLayer, layer: QgsVectorLayer):
        """An arrow as a geometry generator making the polygons QGIS draws
        (q2vt_arrow: straight or curved, repeated or not, head(s), half
        arrows, offset), filled with the arrow's own fill symbol."""
        lengths = [
            self._ground_length(self._number(arrow, "ArrowStartWidth", arrow.arrowStartWidth()),
                                arrow.arrowStartWidthUnit(), layer),
            self._ground_length(self._number(arrow, "ArrowWidth", arrow.arrowWidth()),
                                arrow.arrowWidthUnit(), layer),
            self._ground_length(self._number(arrow, "ArrowHeadLength", arrow.headLength()),
                                arrow.headLengthUnit(), layer),
            self._ground_length(self._number(arrow, "ArrowHeadThickness", arrow.headThickness()),
                                arrow.headThicknessUnit(), layer),
            self._ground_length(self._number(arrow, "Offset", arrow.offset()),
                                arrow.offsetUnit(), layer),
        ]
        # A screen pixel, in which QGIS flattens curves; lengths all on the
        # ground draw the same at every zoom (in 1 m pixels then).
        on_screen = any("@map_scale" in length for length in lengths)
        pixel = self._ground_length(1, Qgis.RenderUnit.Pixels, layer) if on_screen else "1"
        expression = self._in_3857(
            f"q2vt_arrow(@geometry, {', '.join(lengths)}, {self._type_value(arrow, 'ArrowHeadType')}, "
            f"{self._type_value(arrow, 'ArrowType')}, {int(arrow.isCurved())}, {int(arrow.isRepeated())}, "
            f"{pixel}, {int(arrow.headType())}, {int(arrow.arrowType())})",
            layer)
        generator = QgsGeometryGeneratorSymbolLayer.create({"geometryModifier": expression})
        generator.setSymbolType(Qgis.SymbolType.Fill)
        fill = arrow.subSymbol()
        generator.setSubSymbol(fill.clone() if fill is not None else QgsFillSymbol.createSimple({}))
        return generator

    def _trimmed_line(self, line: QgsSimpleLineSymbolLayer, layer: QgsVectorLayer):
        """A line trimmed at its start and end (static or data-defined
        lengths), as a geometry generator cutting the line (lengths in
        screen units follow the scale: @map_scale gives each zoom its own
        cut) and drawing it untrimmed."""
        start = self._ground_length(self._number(line, "TrimStart", line.trimDistanceStart()),
                                    line.trimDistanceStartUnit(), layer)
        end = self._ground_length(self._number(line, "TrimEnd", line.trimDistanceEnd()),
                                  line.trimDistanceEndUnit(), layer)
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
        self._deactivate(untrimmed, "TrimStart")
        self._deactivate(untrimmed, "TrimEnd")
        generator = QgsGeometryGeneratorSymbolLayer.create({"geometryModifier": expression})
        generator.setSymbolType(Qgis.SymbolType.Line)
        generator.setSubSymbol(QgsLineSymbol([untrimmed]))
        return generator

    def _offset_fill(self, fill, layer: QgsVectorLayer):
        """A fill with a data-defined x/y offset, as a geometry generator
        moving each polygon by its feature's offset (q2vt_offset; offsets
        on screen follow the scale), drawing it unmoved: the fill and its
        outline move together, as in QGIS."""
        offset = fill.offset()
        expression = self._in_3857(
            f"q2vt_offset(@geometry, {self._property(fill, 'Offset').asExpression()}, "
            f"{offset.x()!r}, {offset.y()!r}, {self._ground_length(1, fill.offsetUnit(), layer)})",
            layer)
        unmoved = fill.clone()
        unmoved.setOffset(QPointF(0, 0))
        self._deactivate(unmoved, "Offset")
        generator = QgsGeometryGeneratorSymbolLayer.create({"geometryModifier": expression})
        generator.setSymbolType(Qgis.SymbolType.Fill)
        generator.setSubSymbol(QgsFillSymbol([unmoved]))
        return generator

    @staticmethod
    def _ground_length(value, unit, layer: QgsVectorLayer) -> str:
        """A QGIS length (a number or an expression) as an expression of
        EPSG:3857 metres (the unit of the exported base layers), with
        @map_scale for screen units and @q2vt_len (the line's length) for
        percentages."""
        if not value:
            return "0"
        unit = int(getattr(unit, "value", unit))
        stretch = ZoomLevels.GROUND_STRETCH
        static = not isinstance(value, str)
        if unit == int(Qgis.RenderUnit.Percentage.value):
            return f"(@q2vt_len * {value / 100!r})" if static else f"(@q2vt_len * ({value}) / 100)"
        if unit in (int(Qgis.RenderUnit.MapUnits.value), int(Qgis.RenderUnit.MetersInMapUnits.value)):
            metres = (111319.49079327357 if unit == int(Qgis.RenderUnit.MapUnits.value)
                      and layer.crs().isGeographic() else stretch)
            return repr(value * metres) if static else f"(({value}) * {metres!r})"
        per_mm = {
            int(Qgis.RenderUnit.Points.value): 25.4 / 72,
            int(Qgis.RenderUnit.Pixels.value): 25.4 / 96,
            int(Qgis.RenderUnit.Inches.value): 25.4,
        }.get(unit, 1.0)
        mm = repr(value * per_mm) if static else f"({value}) * {per_mm!r}"
        # On screen: as long as the web map draws it at the rule's zoom, whose
        # scale @map_scale stands for (ZoomLevels.SCALES), in EPSG:3857 metres.
        return f"({mm} * @map_scale * {_METRES_PER_MM_AT_ZOOM_0 / ZoomLevels.SCALES[0]!r})"

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
            if code in (MARKER_FIRST_VERTEX, MARKER_LAST_VERTEX):
                offset_marker = self._offset_vertex_marker(
                    point_symbol_layer, code == MARKER_LAST_VERTEX, rule_clone.get_attr("g"), layer)
                if offset_marker is not None:
                    point_rule.rule.setSymbol(offset_marker)
            rules.append(point_rule)
        if remaining:
            symbol_layer.setPlacements(remaining)
            rules.append(rule_clone)
        return rules

    def _offset_vertex_marker(self, marker_line, last: bool, source_geom: int,
                              layer: QgsVectorLayer) -> Optional[QgsMarkerSymbol]:
        """A first (or last) vertex marker moved along its line by the marker
        line's offset along line, as QGIS moves it: forward from the first
        vertex (back from the last), not drawn past the line's other end,
        around a ring; turned with the segment it lands on. A geometry
        generator placing the marker there (screen lengths follow the scale);
        None without an offset."""
        static = marker_line.offsetAlongLine()
        if not static and not self._active(marker_line, "OffsetAlongLine"):
            return None
        offset = self._ground_length(self._number(marker_line, "OffsetAlongLine", static),
                                     marker_line.offsetAlongLineUnit(), layer)
        ring = source_geom == 2
        line = "@geometry" if source_geom == 1 else "boundary(@geometry)"
        if ring:
            # Around the ring: any offset, kept within its length.
            along = (f"if(@q2vt_o >= 0, @q2vt_o % @q2vt_len, "
                     f"@q2vt_len - ((-@q2vt_o) % @q2vt_len))")
        else:
            along = "if(@q2vt_o > 0, @q2vt_o, 0)"
        distance = f"(@q2vt_len - ({along}))" if last else f"({along})"

        def at(expression: str) -> str:
            return (f"with_variable('q2vt_l', {line}, with_variable('q2vt_len', length(@q2vt_l), "
                    f"with_variable('q2vt_o', {offset}, {expression})))")

        inside = "(@q2vt_len - ({along}) > 0)" if last else f"(({along}) < @q2vt_len)"
        inside = inside.replace("{along}", along)
        point = at(f"if({inside}, line_interpolate_point(@q2vt_l, {distance}), NULL)")
        generator = QgsGeometryGeneratorSymbolLayer.create({"geometryModifier": self._in_3857(point, layer)})
        generator.setSymbolType(Qgis.SymbolType.Marker)
        marker = marker_line.subSymbol().clone()
        if marker_line.rotateSymbols():
            # The segment's direction where the marker lands (the tiles'
            # EPSG:3857, as the exported features), with the marker's angle;
            # back from the last vertex, QGIS walks the line backwards.
            turn = 90 if last else -90
            for index in range(marker.symbolLayerCount()):
                own = marker.symbolLayer(index).angle()
                marker.symbolLayer(index).setDataDefinedProperty(
                    QgsSymbolLayer.Property.PropertyAngle,
                    QgsProperty.fromExpression(at(f"line_interpolate_angle(@q2vt_l, {distance}) + {turn} + {own!r}")))
        generator.setSubSymbol(marker)
        return QgsMarkerSymbol([generator])

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
        if displacement_label_settings(label_rule.rule.settings()) is not None:
            return [label_rule]  # The renderer's own labels: one per point, as it draws them.
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