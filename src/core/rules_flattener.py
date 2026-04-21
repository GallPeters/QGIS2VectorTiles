"""
rules_flattener.py

RulesFlattener — walks every visible vector layer in the current QGIS project,
converts any non-rule-based renderer / labeling system to rule-based, then
recursively flattens the rule hierarchy with full property inheritance
(scale range, filter expression, symbol layers).

Depends on: config, zoom_levels, flattened_rule
"""

from typing import List, Optional, Union

from qgis.core import (
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
)

from ..utils.config import QDomDocument
from ..utils.flattened_rule import FlattenedRule
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
            getattr(layer, f"set{type_name.capitalize()}")(rule_system)
            root_rule = self._prepare_root_rule(rule_system, layer)
            if root_rule:
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

        inactive_indices = self._get_inactive_item_indices(system)
        rule_renderer = QgsRuleBasedRenderer.convertFromRenderer(system)

        if rule_renderer and inactive_indices:
            for idx in sorted(inactive_indices, reverse=True):
                rule_renderer.rootRule().removeChildAt(idx)

        return rule_renderer

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
    ):
        """Recursively flatten the rule hierarchy with property inheritance."""
        if rule.parent():
            self._process_rule(layer, layer_idx, rule, rule_type, rule_level, rule_idx)
        for child_idx, child in enumerate(rule.children()):
            if not child.active():
                continue
            if child.filterExpression() == "ELSE":
                self._convert_else_filter(child, rule)
            self._flatten_rule(layer, layer_idx, child, rule_type, rule_level + 1, child_idx)

    def _process_rule(
        self,
        layer: QgsVectorLayer,
        layer_idx: int,
        rule,
        rule_type: int,
        rule_level: int,
        rule_idx: int,
    ):
        """Inherit properties, validate zoom range, split, and register the rule."""
        if rule_type == 1:
            self._sync_labeling_scale_range(rule)

        inherited_rule = self._inherit_rule_properties(rule, rule_type)
        if not inherited_rule:
            return

        flat_rule = FlattenedRule(inherited_rule, layer)
        flat_rule.rule.setDescription("")
        self._set_rule_attributes(flat_rule, layer_idx, rule_type, rule_level, rule_idx)

        if not self._is_within_zoom_range(flat_rule):
            return

        split_rules = self._split_rule(flat_rule, rule_type)
        self._clip_rules_to_zoom_range(split_rules)
        for split_rule in split_rules:
            self.flattened_rules.extend(self._split_by_scale_expressions(split_rule))

    def _sync_labeling_scale_range(self, rule):
        """Copy label settings scale visibility to the rule if the rule has no range set."""
        if rule.minimumScale() != 0 or rule.maximumScale() != 0:
            return
        settings = rule.settings()
        if settings and settings.scaleVisibility:
            rule.setMinimumScale(settings.minimumScale())
            rule.setMaximumScale(settings.maximumScale())
            settings.scaleVisibility = False

    def _inherit_rule_properties(self, rule, rule_type: int):
        """Clone and inherit scale range, filter expression, and symbol layers from parents."""
        clone = rule.clone()
        self._inherit_min_scale(clone, rule)
        self._inherit_max_scale(clone, rule)
        self._inherit_filter_expression(clone, rule)
        if rule_type == 0:
            self._inherit_symbol_layers(clone, rule)
        return clone

    def _inherit_min_scale(self, clone, rule):
        """Inherit minimum scale; 0 means no restriction → use the largest available scale."""
        rule_scale = rule.minimumScale() if rule.minimumScale() != 0 else max(ZoomLevels.SCALES)
        clone.setMinimumScale(min(rule_scale, rule.parent().minimumScale()))

    def _inherit_max_scale(self, clone, rule):
        """Inherit maximum scale; 0 means no restriction → use the smallest available scale."""
        rule_scale = rule.maximumScale() if rule.maximumScale() != 0 else min(ZoomLevels.SCALES)
        clone.setMaximumScale(max(rule_scale, rule.parent().maximumScale()))

    def _inherit_filter_expression(self, clone, rule):
        """Combine parent and child filter expressions, excluding children's conditions."""
        parent_filter = rule.parent().filterExpression()
        rule_filter = rule.filterExpression()

        if parent_filter and rule_filter:
            combined = f"({parent_filter}) AND ({rule_filter})"
        else:
            combined = parent_filter or rule_filter or ""

        child_filters = [
            f"({child.filterExpression()})"
            for child in rule.children()
            if child.filterExpression() and child.filterExpression() != "ELSE"
        ]

        if child_filters:
            children_expr = " OR ".join(child_filters)
            final = f"({combined}) AND NOT ({children_expr})" if combined else f"NOT ({children_expr})"
        else:
            final = combined

        clone.setFilterExpression(final)

    def _inherit_symbol_layers(self, clone, rule):
        """Append parent symbol layers to the cloned rule's symbol."""
        clone_symbol = clone.symbol()
        parent_symbol = rule.parent().symbol()
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

        layer_count = symbol.symbolLayerCount()
        split_rules = []

        for layer_idx in reversed(range(layer_count)):
            symbol_layer = symbol.symbolLayer(layer_idx)
            if not symbol_layer.enabled():
                continue

            sub_symbol = symbol_layer.subSymbol()
            if symbol_layer.layerType() == "GeometryGenerator":
                symbol_type = sub_symbol.type()
            elif symbol_layer.layerType() == "CentroidFill":
                symbol_type = 0
            else:
                symbol_type = symbol_layer.type()

            rule_clone = FlattenedRule(flat_rule.rule.clone(), flat_rule.layer)
            rule_clone.set_attr("c", symbol_type)
            rule_clone.set_attr("s", layer_idx)

            clone_symbol = rule_clone.rule.symbol()
            for remove_idx in reversed(range(layer_count)):
                if remove_idx != layer_idx:
                    clone_symbol.deleteSymbolLayer(remove_idx)

            split_rules.append(rule_clone)

        return split_rules

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
            if renderer_rule.output_dataset in seen_datasets:
                continue

            matched = self._match_label_to_renderer(label_rule, renderer_rule, renderer_idx)
            if matched:
                split_rules.append(matched)
                seen_datasets.add(renderer_rule.output_dataset)
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

        rule_clone.set_attr("f", renderer_idx)
        return rule_clone

    @staticmethod
    def _ranges_overlap(r1_start: int, r1_end: int, r2_start: int, r2_end: int) -> bool:
        a_min, a_max = sorted((r1_start, r1_end))
        b_min, b_max = sorted((r2_start, r2_end))
        return a_min <= b_max and b_min <= a_max

    def _split_by_scale_expressions(self, flat_rule: FlattenedRule) -> List[FlattenedRule]:
        """Split a rule per zoom level when it contains @map_scale-dependent expressions."""
        if not self._has_scale_dependencies(flat_rule):
            return [flat_rule]

        min_zoom = flat_rule.get_attr("o")
        max_zoom = min(self.max_zoom, flat_rule.get_attr("i") + 1)
        split_rules = []
        for zoom in range(min_zoom, max_zoom + 1):
            clone = self._create_zoom_specific_rule(flat_rule, zoom)
            if flat_rule.get_attr("t") == 1 or self._symbol_layer_visible_at_zoom(clone):
                split_rules.append(clone)
        return split_rules

    def _symbol_layer_visible_at_zoom(self, flat_rule: FlattenedRule) -> bool:
        """Return False if the symbol layer's visibility DDP evaluates to falsy at the zoom."""
        symbol = flat_rule.rule.symbol()
        if not symbol:
            return True
        symbol_layer = symbol.symbolLayers()[0]
        vis_prop = symbol_layer.dataDefinedProperties().property(44)
        if vis_prop and vis_prop.isActive():
            min_scale = str(ZoomLevels.zoom_to_scale(flat_rule.get_attr("o")))
            zoom_expr = vis_prop.expressionString().replace("@map_scale", min_scale)
            evaluation = QgsExpression(zoom_expr).evaluate()
            if evaluation is not None and not evaluation:
                return False
        return True

    def _has_scale_dependencies(self, flat_rule: FlattenedRule) -> bool:
        """Return True if the rule's serialized XML contains @map_scale."""
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
        return "@map_scale" in doc.toString()

    def _create_zoom_specific_rule(self, flat_rule: FlattenedRule, zoom: int) -> FlattenedRule:
        """Clone a rule with @map_scale replaced by the exact scale for the given zoom."""
        rule_clone = FlattenedRule(flat_rule.rule.clone(), flat_rule.layer)
        scale = str(ZoomLevels.zoom_to_scale(zoom))

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
