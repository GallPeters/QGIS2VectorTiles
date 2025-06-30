from qgis.core import *


class RuleExtractor:
    """
    Compact PyQGIS class that extracts and flattens all renderer and labeling rules
    from vector layers in the current QGIS project using unified processing logic.
    """

    def __init__(self):
        self.project = QgsProject.instance()
        self.rules_types = ["Renderer", "Labeling"]
        self.extracted_rules = []

    def extract_all_rules(self):
        """Extract all rules from all vector layers in the project."""
        self.extracted_rules = {}
        vector_layers = [
            layer
            for layer in self.project.mapLayers().values()
            if layer.type() == 0 and layer.geometryType() != 4
        ]
        
        for layer in vector_layers:
            rules_dict = {}
            try:
                # Process both renderer and labeling using unified approach
                for rule_type in self.rules_types:
                    rule_based = self._get_rule_based_system(layer, rule_type)
                    if rule_based and rule_based.rootRule():
                        rules_dict[rule_type] = self._flatten_rules(
                            rule_based.rootRule(), rule_type
                        )
                self.extracted_rules[f"{layer.name()} | {layer.id()}"] = rules_dict
            except Exception as e:
                print(f"Error processing layer {layer.name()}: {str(e)}")

        return self.extracted_rules
    
    def print_rules(self):
        """Print extracted rules with elegant tree structure."""
        print('\n -------- Extracted rules --------\n')
        print('Project: ')
        
        for layer_idx, (layer, rules_dict) in enumerate(self.extracted_rules.items()):
            is_last_layer = layer_idx == len(self.extracted_rules) - 1
            layer_prefix = "    " if is_last_layer else "  │  "
            print(f"{'  └─ ' if is_last_layer else '  ├─ '}Layer {layer_idx + 1}: {layer.split(' | ')[0]}")
            
            for type_idx, (rule_type, rules) in enumerate(rules_dict.items()):
                is_last_type = type_idx == len(rules_dict) - 1
                type_prefix = "        " if is_last_type else "   │    "
                print(f"{layer_prefix}{'   └─ ' if is_last_type else '   ├─ '}{rule_type}:")
                
                for rule_idx, rule in enumerate(rules):
                    is_last_rule = rule_idx == len(rules) - 1
                    rule_prefix = "     " if is_last_rule else " │   "
                    print(f"{layer_prefix}{type_prefix}{' └─ ' if is_last_rule else ' ├─ '}Rule {rule_idx + 1}:")
                    
                    details = [("Name", rule.description()), ("MinScale", rule.minimumScale()), 
                            ("MaxScale", rule.maximumScale()), ("Filter", rule.filterExpression())]
                    for i, (label, value) in enumerate(details):
                        connector = "└─ " if i == 3 else "├─ "
                        print(f"{layer_prefix}{type_prefix}{rule_prefix}{connector}{label}: {value}")
        
        print('\n -------- Extraction has been finished successfully --------')

    def _get_rule_based_system(self, layer, rule_type):
        """Get or convert to rule-based system for both renderer and labeling."""
        system = layer.renderer() if rule_type == "Renderer" else layer.labeling()
        if not system:
            return None

        # Return if already rule-based
        if isinstance(system, (QgsRuleBasedRenderer, QgsRuleBasedLabeling)):
            return system

        # Convert to rule-based
        if rule_type == "Renderer":
            return QgsRuleBasedRenderer.convertFromRenderer(system)
        else:
            rule = QgsRuleBasedLabeling.Rule(system.settings())
            root = QgsRuleBasedLabeling.Rule(QgsPalLayerSettings())
            root.appendChild(rule)
            return QgsRuleBasedLabeling(root)

    def _flatten_rules(self, rule, rule_type):
        """Unified rule flattening for both renderer and labeling."""
        flattened = []
        if rule.parent():
            clone = rule.clone()
            self._inherit_rule_name(rule, clone, rule_type, len(flattened))
            self._inherit_rule_filter(rule, clone)
            self._inherit_scale(rule, clone, max)
            self._inherit_scale(rule, clone, min)
            self._inherit_rule_data(rule, clone, rule_type)

            # Append flattened rule
            flattened.append(clone)
        else:
            clone = rule
        # Process children recursively or add leaf rule
        if clone.children():
            for child in clone.children():
                flattened.extend(self._flatten_rules(child, rule_type))
        return flattened

    def _inherit_rule_name(self, rule, clone, rule_type, index):
        """Inherit rule-specific name (label or description)."""
        clone_name = rule.label() if rule_type == "Renderer" else rule.description()
        if not clone_name:
            parent_name = (
                rule.label() if rule_type == "Renderer" else rule.description()
            )
            clone_name = f"{parent_name} > {index}"
        clone.setDescription(clone_name)

    def _inherit_rule_filter(self, rule, clone):
        """Combine filters using AND logic."""
        parent_filter = rule.parent().filterExpression()
        rule_filter = rule.filterExpression()

        if not parent_filter and not rule_filter:
            clone_filter = ""
        elif not parent_filter:
            clone_filter = rule_filter
        elif not rule_filter:
            clone_filter = parent_filter
        else:
            clone_filter = f"({parent_filter}) AND ({rule_filter})"
        clone.setFilterExpression(clone_filter)

    def _inherit_scale(self, rule, clone, comparator):
        """Inherit scale using comparator function (min/max)."""
        rule_scale = getattr(rule, f"{comparator.__name__}imumScale")()
        parent_scale = getattr(rule.parent(), f"{comparator.__name__}imumScale")()
        if rule_scale > 0 and parent_scale > 0:
            clone_scale = comparator(rule_scale, parent_scale)
        clone_scale = rule_scale if rule_scale > 0 else parent_scale
        setattr(clone, f"set{comparator.__name__.capitalize()}imumScale", clone_scale)

    def _inherit_rule_data(self, rule, clone, rule_type):
        """Inherit rule-specific data (symbol or settings)."""
        if rule_type == "Renderer":
            clone_symbol = clone.symbol()
            parent_symbol = rule.parent().symbol()
            if parent_symbol and clone_symbol:
                for index in range(parent_symbol.symbolLayerCount()):
                    symbol_layer = parent_symbol.symbolLayer(index).clone()
                    clone_symbol.appendSymbolLayer(symbol_layer)


# Console execution
if __name__ == "__console__":
    extractor = RuleExtractor()
    print("Extracting rules from current QGIS project...")
    rules = extractor.extract_all_rules()
    print(f"Printing rules...")
    extractor.print_rules()

