"""
rules_flattener.py

RulesFlattener — walks every visible vector layer in the current QGIS project,
converts any non-rule-based renderer / labeling system to rule-based, then
recursively flattens the rule hierarchy with full property inheritance
(scale range, filter expression, symbol layers).

Depends on: config, zoom_levels, f;lattened_rule, data_defined_properties
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
    """Flattens QGIS rule-based styling with property inheritance."""

    RULE_TYPES = {0: "renderer", 1: "labeling"}

    def __init__(self, min_zoom: int, max_zoom: int, utils_dir, feedback):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.utils_dir = utils_dir
        self.layer_tree_root = QgsProject.instance().layerTreeRoot()
        self.flattened_rules = []
        self.feedback = feedback

    def flatten_all_rules(self) -> List[FlattenedRule]:
        """Extract and flatten all rules from visible vector layers."""
        for layer_idx, layer in enumerate(self.layer_tree_root.findLayers()):
            if self._is_valid_layer(layer.layer()):
                self._process_layer_rules(layer.layer(), layer_idx)
        if self.flattened_rules:
            self._split_flattened_rules_layers()
        self._clear_unuseable_rules()
        return self.flattened_rules

    def _clear_unuseable_rules(self):
        """Clear rules without properties - renderer rules without symbols
        or labeling rules without settings"""
        for flat_rule in self.flattened_rules:
            useable = False
            if flat_rule.get_attr("t") == 0 and flat_rule.rule.symbol():
                useable = True
            elif flat_rule.get_attr("t") == 1 and flat_rule.rule.settings():
                if flat_rule.rule.settings().getLabelExpression().expression():
                    useable = True
            if not useable:
                self.flattened_rules.remove(flat_rule)

    def _split_flattened_rules_layers(self):
        """Seperate every rule to diffrent layer and save it on memory
        in order to prevent shared properties overwritten"""
        for flattened_rule in self.flattened_rules:
            rule = flattened_rule.rule
            rule_type = flattened_rule.get_attr("t")
            if rule_type == 0:
                rule_clone = QgsRuleBasedRenderer.Rule(None)
            else:
                rule_clone = QgsRuleBasedLabeling.Rule(None)

            rule_clone.setDescription(rule.description())
            rule_clone.setFilterExpression(rule.filterExpression())
            rule_clone.setMinimumScale(rule.minimumScale())
            rule_clone.setMaximumScale(rule.maximumScale())

            if rule_type == 0 and rule.symbol():
                rule_clone.setSymbol(rule.symbol().clone())
            elif rule_type == 1 and rule.settings():
                new_settings = QgsPalLayerSettings(rule.settings())
                new_format = QgsTextFormat(new_settings.format())
                new_bg = QgsTextBackgroundSettings(new_format.background())
                if new_bg.markerSymbol():
                    new_bg.setMarkerSymbol(new_bg.markerSymbol().clone())
                new_format.setBackground(new_bg)
                new_settings.setFormat(new_format)
                rule_clone.setSettings(new_settings)
            flattened_rule.rule = rule_clone

    def _is_valid_layer(self, layer) -> bool:
        """Check if layer is a visible vector layer."""
        is_vector = layer.type() == 0 and layer.geometryType() != 4
        layer_node = self.layer_tree_root.findLayer(layer.id())
        is_visible = layer_node.isVisible() if layer_node is not None else False
        return is_vector and is_visible

    def _process_layer_rules(self, layer: QgsVectorLayer, layer_idx: int):
        """Process both renderer and labeling rules for a layer."""
        for rule_type, type_name in self.RULE_TYPES.items():
            rule_system = self._get_or_convert_rule_system(layer, rule_type)
            if not rule_system:
                continue
            getattr(layer, f"set{type_name.capitalize()}")(rule_system)
            root_rule = self._prepare_root_rule(rule_system, layer)
            if root_rule:
                self._flat_rule(layer, layer_idx, root_rule, rule_type, 0, 0)

    def _get_or_convert_rule_system(self, layer: QgsVectorLayer, rule_type: int):
        """Get or convert layer styling to rule-based system."""
        if rule_type == 0:
            return self._convert_renderer_to_rules(layer)
        else:
            return self._convert_labeling_to_rules(layer)

    def _convert_renderer_to_rules(self, layer: QgsVectorLayer):
        """Convert renderer to rule-based system."""
        system = layer.renderer()

        if not system:
            return

        system = system.clone()
        if isinstance(system, QgsRuleBasedRenderer):
            return system

        inactive_items = self._get_inactive_items(system)
        rulebased_renderer = QgsRuleBasedRenderer.convertFromRenderer(system) if system else None

        if rulebased_renderer and inactive_items:
            for rule_index in sorted(inactive_items, reverse=True):
                rulebased_renderer.rootRule().removeChildAt(rule_index)

        return rulebased_renderer

    def _get_inactive_items(self, system) -> List[int]:
        """Get indices of inactive items from graduated/categorized renderer."""
        inactive_items = []
        items_method = None

        if isinstance(system, QgsGraduatedSymbolRenderer):
            items_method = "ranges"
        elif isinstance(system, QgsCategorizedSymbolRenderer):
            items_method = "categories"

        if items_method:
            items = getattr(system, items_method)()
            inactive_items = [i for i, item in enumerate(items) if not item.renderState()]

        return inactive_items

    def _convert_labeling_to_rules(self, layer: QgsVectorLayer):
        """Convert labeling to rule-based system."""
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
        """Prepare root rule with layer scale visibility."""
        root_rule = rule_system.rootRule()
        if layer.hasScaleBasedVisibility():
            root_rule.setMinimumScale(layer.minimumScale())
            root_rule.setMaximumScale(layer.maximumScale())
        return root_rule

    def _set_rule_properties(
        self,
        layer: QgsVectorLayer,
        layer_idx: int,
        rule: Union[QgsRuleBasedLabeling.Rule, QgsRuleBasedRenderer.Rule],
        rule_type: int,
        rule_level: int,
        rule_idx: int,
    ):
        """Set rule properties using its and its parent attributes"""
        if rule_type == 1:
            self._fix_labeling_rule_scale_range(rule)
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
            final_rules = self._split_by_scale_expressions(split_rule)
            self.flattened_rules.extend(final_rules)

    def _flat_rule(
        self,
        layer: QgsVectorLayer,
        layer_idx: int,
        rule: Union[QgsRuleBasedLabeling.Rule, QgsRuleBasedRenderer.Rule],
        rule_type: int,
        rule_level: int,
        rule_idx: int,
    ):
        """Recursively flatten rule hierarchy with inheritance."""
        if rule.parent():
            self._set_rule_properties(layer, layer_idx, rule, rule_type, rule_level, rule_idx)
        for child_idx, child in enumerate(rule.children()):
            if not child.active():
                continue
            if child.filterExpression() == "ELSE":
                self._convert_else_filter(child, rule)

            self._flat_rule(layer, layer_idx, child, rule_type, rule_level + 1, child_idx)

    def _fix_labeling_rule_scale_range(self, rule):
        """Copy labeling settings visiblity scale to rule's visblity scales if not activated"""
        if rule.minimumScale() != 0 or rule.maximumScale() != 0:
            return
        settings = rule.settings()
        if settings and settings.scaleVisibility:
            rule.setMinimumScale(settings.minimumScale())
            rule.setMaximumScale(settings.maximumScale())
            settings.scaleVisibility = False

    def _set_rule_attributes(
        self,
        flat_rule: FlattenedRule,
        layer_idx: int,
        rule_type: int,
        rule_level: int,
        rule_idx: int,
    ):
        """Set rule attributes for identification and processing."""
        flat_rule.set_attr("l", layer_idx)
        flat_rule.set_attr("t", rule_type)
        flat_rule.set_attr("d", rule_level)
        flat_rule.set_attr("r", rule_idx)
        flat_rule.set_attr("g", flat_rule.layer.geometryType())
        flat_rule.set_attr("c", flat_rule.layer.geometryType())
        flat_rule.set_attr("o", self._get_rule_zoom(flat_rule, min))
        flat_rule.set_attr("i", self._get_rule_zoom(flat_rule, max))
        flat_rule.set_attr("s" if rule_type == 0 else "f", 0)

    def _get_rule_zoom(self, flat_rule: FlattenedRule, comparator) -> int:
        """Extract rule zoom level from scale."""
        attr_name = f"{comparator.__name__}imumScale"
        rule_scale = getattr(flat_rule.rule, attr_name)()
        edge = "i" if comparator.__name__ == "max" else "o"
        return int(ZoomLevels.scale_to_zoom(rule_scale, edge))

    def _is_within_zoom_range(self, flat_rule: FlattenedRule) -> bool:
        """Check if rule is within requested zoom range."""
        min_zoom = flat_rule.get_attr("o")
        max_zoom = flat_rule.get_attr("i")
        return self._ranges_overlap(min_zoom, max_zoom, self.min_zoom, self.max_zoom)

    def _clip_rules_to_zoom_range(self, flat_rules: List[FlattenedRule]):
        """Clip rule zoom range to general zoom range."""
        for flat_rule in flat_rules:
            if flat_rule.get_attr("o") < self.min_zoom:
                flat_rule.set_attr("o", self.min_zoom)
            if flat_rule.get_attr("i") > self.max_zoom:
                flat_rule.set_attr("i", self.max_zoom)

    def _split_rule(self, flat_rule: FlattenedRule, rule_type: int) -> List[FlattenedRule]:
        """Split rule based on type."""
        if rule_type == 0:  # Renderer
            return self._split_by_symbol_layers(flat_rule)
        else:  # Labeling
            return self._split_by_matching_renderers(flat_rule)

    def _convert_else_filter(self, else_rule, parent_rule):
        """Convert ELSE filter to explicit exclusion of sibling conditions."""
        sibling_filters = []

        for sibling in parent_rule.children():
            if sibling.active() and sibling.filterExpression() not in ("ELSE", ""):
                sibling_filters.append(sibling.filterExpression())

        if sibling_filters:
            else_expression = f'NOT ({" OR ".join(f"({f})" for f in sibling_filters)}) IS 1'
        else:
            else_expression = ""

        else_rule.setFilterExpression(else_expression)

    def _inherit_rule_properties(self, rule, rule_type: int):
        """Inherit all properties from parent hierarchy."""
        clone = rule.clone()

        self._inherit_scale_range(clone, rule, min)
        self._inherit_scale_range(clone, rule, max)
        self._inherit_filter_expression(clone, rule)

        if rule_type == 0:
            self._inherit_symbol_layers(clone, rule)

        return clone

    def _inherit_scale_range(self, clone, rule, comparator):
        """Inherit scale limits using min/max comparator."""
        attr_name = f"{comparator.__name__}imumScale"
        rule_scale = getattr(rule, attr_name)()
        if rule_scale == 0:
            opposite = min if comparator.__name__ == "max" else max
            rule_scale = opposite(ZoomLevels.SCALES)
        parent_scale = getattr(rule.parent(), attr_name)()
        inherited_scale = comparator(rule_scale, parent_scale)
        setter_name = f"set{comparator.__name__.capitalize()}imumScale"
        getattr(clone, setter_name)(inherited_scale)

    def _inherit_filter_expression(self, clone, rule):
        """Inherit and combine filter expressions from parent hierarchy."""
        parent_filter = rule.parent().filterExpression()
        rule_filter = rule.filterExpression()

        if parent_filter and rule_filter:
            combined_filter = f"({parent_filter}) AND ({rule_filter})"
        else:
            combined_filter = parent_filter or rule_filter or ""

        child_filters = []
        for child in rule.children():
            if child.filterExpression() and child.filterExpression() != "ELSE":
                child_filters.append(f"({child.filterExpression()})")

        if child_filters:
            children_expression = " OR ".join(child_filters)
            if combined_filter:
                final_filter = f"({combined_filter}) AND NOT ({children_expression})"
            else:
                final_filter = f"NOT ({children_expression})"
        else:
            final_filter = combined_filter

        clone.setFilterExpression(final_filter)

    def _inherit_symbol_layers(self, clone, rule):
        """Inherit symbol layers from parent."""
        clone_symbol = clone.symbol()
        parent_symbol = rule.parent().symbol()

        if parent_symbol and clone_symbol:
            for i in range(parent_symbol.symbolLayerCount()):
                symbol_layer = parent_symbol.symbolLayer(i).clone()
                clone_symbol.appendSymbolLayer(symbol_layer)

    def _split_by_symbol_layers(self, flat_rule: FlattenedRule) -> List[FlattenedRule]:
        """Split renderer rule by individual symbol layers."""
        symbol = flat_rule.rule.symbol()
        if not symbol:
            return [flat_rule]

        symbol_layer_count = symbol.symbolLayerCount()
        split_rules = []

        for layer_idx in reversed(range(symbol_layer_count)):
            rule_clone = FlattenedRule(flat_rule.rule.clone(), flat_rule.layer)

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

            rule_clone.set_attr("c", symbol_type)
            rule_clone.set_attr("s", layer_idx)

            clone_symbol = rule_clone.rule.symbol()
            for remove_idx in reversed(range(symbol_layer_count)):
                if remove_idx != layer_idx:
                    clone_symbol.deleteSymbolLayer(remove_idx)

            split_rules.append(rule_clone)

        return split_rules

    def _split_by_matching_renderers(self, label_rule: FlattenedRule) -> List[FlattenedRule]:
        """Split label rule by matching renderer rules with overlapping scales."""
        splitted_rules = []
        renderer_datasets = []
        renderer_idx = 0

        for renderer_rule in self.flattened_rules:
            if label_rule.layer.id() != renderer_rule.layer.id():
                continue
            if renderer_rule.get_attr("t") == 1:
                continue
            if renderer_rule.output_dataset in renderer_datasets:
                continue
            splitted_rule = self._match_label_to_renderer(label_rule, renderer_rule, renderer_idx)
            if splitted_rule:
                splitted_rules.append(splitted_rule)
                renderer_datasets.append(renderer_rule.output_dataset)
            renderer_idx += 1
        return splitted_rules if splitted_rules else [label_rule]

    def _match_label_to_renderer(
        self, label_rule: FlattenedRule, renderer_rule: FlattenedRule, renderer_idx: int
    ) -> Optional[FlattenedRule]:
        """Create combined rule matching label to renderer with overlapping scales."""
        label_min = label_rule.get_attr("o")
        label_max = label_rule.get_attr("i")
        renderer_min = renderer_rule.get_attr("o")
        renderer_max = renderer_rule.get_attr("i")

        if not self._ranges_overlap(label_min, label_max, renderer_min, renderer_max):
            return None

        rule_clone = FlattenedRule(label_rule.rule.clone(), label_rule.layer)
        clone_rule = rule_clone.rule

        label_filter = clone_rule.filterExpression()
        renderer_filter = renderer_rule.rule.filterExpression()

        if label_filter and renderer_filter:
            combined_filter = f"({renderer_filter}) AND ({label_filter})"
        else:
            combined_filter = renderer_filter or label_filter or ""

        clone_rule.setFilterExpression(combined_filter)

        if label_min < renderer_min:
            rule_clone.set_attr("o", renderer_min)
        if label_max > renderer_max:
            rule_clone.set_attr("i", renderer_max)

        rule_clone.set_attr("f", renderer_idx)
        return rule_clone

    @staticmethod
    def _ranges_overlap(r1_start: int, r1_end: int, r2_start: int, r2_end: int) -> bool:
        """Check if two ranges overlap."""
        a_min, a_max = sorted((r1_start, r1_end))
        b_min, b_max = sorted((r2_start, r2_end))
        return a_min <= b_max and b_min <= a_max

    def _split_by_scale_expressions(self, flat_rule: FlattenedRule) -> List[FlattenedRule]:
        """Split rule by zoom levels if contains scale-dependent expressions."""
        if not self._has_scale_dependencies(flat_rule):
            return [flat_rule]

        min_zoom = flat_rule.get_attr("o")
        max_zoom = flat_rule.get_attr("i")
        max_zoom = min(self.max_zoom, max_zoom + 1)
        relevant_zooms = list(range(min_zoom, max_zoom + 1))
        split_rules = []
        for zoom in relevant_zooms:
            rule_clone = self._create_scale_specific_rule(flat_rule, zoom)
            if flat_rule.get_attr("t") == 1 or self._symbol_layer_visible_at_zoom(rule_clone):
                split_rules.append(rule_clone)
        return split_rules

    def _symbol_layer_visible_at_zoom(self, flat_rule: FlattenedRule) -> bool:
        """Check if symbol layer is visible at specific zoom level."""
        min_zoom = flat_rule.get_attr("o")
        symbol = flat_rule.rule.symbol()
        if not symbol:
            return
        symbol_layer = symbol.symbolLayers()[0]
        visiblity_prop = symbol_layer.dataDefinedProperties().property(44)
        if visiblity_prop and visiblity_prop.isActive():
            expression = visiblity_prop.expressionString()
            min_scale = str(ZoomLevels.zoom_to_scale(min_zoom))
            zoom_expression = expression.replace("@map_scale", min_scale)
            evaluation = QgsExpression(zoom_expression).evaluate()
            if evaluation is not None and not evaluation:
                return False
        return True

    def _has_scale_dependencies(self, flat_rule: FlattenedRule) -> bool:
        """Check if rule has scale-dependent expressions."""
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

    def _create_scale_specific_rule(self, flat_rule: FlattenedRule, zoom: int) -> FlattenedRule:
        """Create rule with scale-specific values."""
        rule_clone = FlattenedRule(flat_rule.rule.clone(), flat_rule.layer)
        scale = str(ZoomLevels.zoom_to_scale(zoom))

        filter_exp = flat_rule.rule.filterExpression()
        if "@map_scale" in filter_exp:
            scale_specific_filter = filter_exp.replace("@map_scale", scale)
            rule_clone.rule.setFilterExpression(scale_specific_filter)

        rule_type = flat_rule.get_attr("t")
        if rule_type == 1 and flat_rule.rule.settings():
            label_exp = flat_rule.rule.settings().getLabelExpression().expression()
            if label_exp and "@map_scale" in label_exp:
                scale_specific_label = label_exp.replace("@map_scale", scale)
                rule_clone.rule.settings().fieldName = scale_specific_label

        rule_clone.set_attr("o", zoom)
        rule_clone.set_attr("i", zoom)

        return rule_clone
