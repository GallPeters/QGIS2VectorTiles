from qgis.core import *


class RuleExtractor:
    """
    Compact PyQGIS class that extracts and flattens all renderer and labeling rules
    from vector layers in the current QGIS project using unified processing logic.
    """

    def __init__(self):
        self.project = QgsProject.instance()
        self.extracted_rules = []

    def extract_all_rules(self):
        """Extract all rules from all vector layers in the project."""
        self.extracted_rules = []
        vector_layers = [
            layer
            for layer in self.project.mapLayers().values()
            if layer.type() == 0 and layer.geometryType() != 4
        ]

        for layer in vector_layers:
            try:
                # Process both renderer and labeling using unified approach
                for rule_type in ["renderer", "labeling"]:
                    rule_based = self._get_rule_based_system(layer, rule_type)
                    if rule_based and rule_based.rootRule():
                        rules = self._flatten_rules(
                            rule_based.rootRule(), layer, rule_type
                        )
                        self.extracted_rules.extend(rules)
            except Exception as e:
                print(f"Error processing layer {layer.name()}: {str(e)}")

        return self.extracted_rules

    def _get_rule_based_system(self, layer, rule_type):
        """Get or convert to rule-based system for both renderer and labeling."""
        system = layer.renderer() if rule_type == "renderer" else layer.labeling()
        if not system:
            return None

        # Return if already rule-based
        if isinstance(system, (QgsRuleBasedRenderer, QgsRuleBasedLabeling)):
            return system

        # Convert to rule-based
        if rule_type == "renderer":
            return QgsRuleBasedRenderer.convertFromRenderer(system)
        else:
            rule = QgsRuleBasedLabeling.Rule(system.settings())
            root = QgsRuleBasedLabeling.Rule(QgsPalLayerSettings())
            root.appendChild(rule)
            return QgsRuleBasedLabeling(root)

    def _flatten_rules(self, rule, layer, rule_type):
        """Unified rule flattening for both renderer and labeling."""
        flattened = []
        if rule.parent():
            clone = rule.clone()
            self._inherit_rule_name(rule, clone, rule_type, len(flattened))
            self._inherit_rule_filter(rule, clone)
            self._inherit_scale(rule, clone, max)
            self._inherit_scale(rule, clone, min)
            current_data = self._inherit_rule_data(parent_data, rule, rule_type)

            # Append flattened rule
            flattened.append(flattened_rule)

        # Process children recursively or add leaf rule
        if rule.children():
            for child in rule.children():
                flattened.extend(
                    self._flatten_rules(
                        child,
                        layer,
                        rule_type,
                        current_path,
                        current_filter,
                        current_min_scale,
                        current_max_scale,
                        current_data,
                    )
                )
        return flattened


    def _inherit_rule_name(self, rule, clone, rule_type, index):
        """Inherit rule-specific name (label or description)."""
        clone_name = rule.label() if rule_type == "renderer" else rule.description()
        if not clone_name:
            parent_name = rule.label() if rule_type == "renderer" else rule.description()
            clone_name = f"{parent_name} > {index}"
        clone.setDescription(clone_name)

    def _inherit_rule_filter(self, rule, clone):
        """Combine filters using AND logic."""
        parent_filter = rule.parent().filterExpression()
        rule_filter = rule.filterExpression()

        if not parent_filter and not rule_filter:
            clone_filter = ""
        elif not parent_filter:
            clone_filter =  rule_filter
        elif not rule_filter:
            clone_filter = parent_filter
        else:
            clone_filter = f"({parent_filter}) AND ({rule_filter})"
        clone.setFilterExpression(clone_filter)

    def _inherit_scale(self, rule, clone, comparator):
        """Inherit scale using comparator function (min/max)."""
        rule_scale = getattr(rule, f'{comparator.__name__}imumScale')()
        parent_scale = getattr(rule.parent(), f'{comparator.__name__}imumScale')()
        if rule_scale > 0 and parent_scale > 0:
            clone_scale= comparator(rule_scale, parent_scale)
        clone_scale =  rule_scale if rule_scale > 0 else parent_scale
        setattr(clone, f'set{comparator.__name__.capitalize()}imumScale', clone_scale)

    def _inherit_rule_data(self, parent_data, rule, rule_type):
        """Inherit rule-specific data (symbol or settings)."""
        if rule_type == "renderer":
            current_symbol = rule.symbol()
            if parent_data and current_symbol:
                inherited_symbol = current_symbol.clone()
                for i in range(parent_data.symbolLayerCount()):
                    inherited_symbol.appendSymbolLayer(
                        parent_data.symbolLayer(i).clone()
                    )
                return inherited_symbol
            return current_symbol.clone() if current_symbol else parent_data
        else:
            # For labeling, child settings override parent
            return rule.settings() or parent_data



# Console execution
if __name__ == "__console__":
    extractor = RuleExtractor()
    print("Extracting rules from current QGIS project...")
    rules = extractor.extract_all_rules()
    extractor.print_summary()
    print(f"\nRule extractor ready! Use:")
