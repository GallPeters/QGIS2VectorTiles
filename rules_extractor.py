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
        vector_layers = [layer for layer in self.project.mapLayers().values() 
                        if isinstance(layer, QgsVectorLayer)]
        
        for layer in vector_layers:
            try:
                # Process both renderer and labeling using unified approach
                for rule_type in ['renderer', 'labeling']:
                    rule_based = self._get_rule_based_system(layer, rule_type)
                    if rule_based and rule_based.rootRule():
                        rules = self._flatten_rules(rule_based.rootRule(), layer.name(), rule_type)
                        self.extracted_rules.extend(rules)
            except Exception as e:
                print(f"Error processing layer {layer.name()}: {str(e)}")
        
        return self.extracted_rules
    
    def _get_rule_based_system(self, layer, rule_type):
        """Get or convert to rule-based system for both renderer and labeling."""
        system = layer.renderer() if rule_type == 'renderer' else layer.labeling()
        if not system:
            return None
        
        # Return if already rule-based
        if isinstance(system, (QgsRuleBasedRenderer, QgsRuleBasedLabeling)):
            return system
        
        # Convert to rule-based
        if rule_type == 'renderer':
            return QgsRuleBasedRenderer.convertFromRenderer(system)
        else:
            rule = QgsRuleBasedLabeling.Rule(system.settings())
            root = QgsRuleBasedLabeling.Rule(QgsPalLayerSettings())
            root.appendChild(rule)
            return QgsRuleBasedLabeling(root)
    
    def _flatten_rules(self, rule, layer_name, rule_type, parent_path="root", parent_filter="", 
                      parent_min_scale=0, parent_max_scale=0, parent_data=None):
        """Unified rule flattening for both renderer and labeling."""
        flattened = []
        if parent_path != "root":
            # Build rule properties with inheritance
            rule_label = self._get_rule_label(rule, rule_type, len(flattened))
            current_path = f"{parent_path} > {rule_label}" if parent_path else rule_label
            current_filter = self._combine_filters(parent_filter, self._get_rule_filter(rule, rule_type))
            current_min_scale = self._inherit_scale(parent_min_scale, self._get_rule_min_scale(rule, rule_type), max)
            current_max_scale = self._inherit_scale(parent_max_scale, self._get_rule_max_scale(rule, rule_type), min)
            current_data = self._inherit_rule_data(parent_data, rule, rule_type)
            
            # Create flattened rule
            flattened_rule = {
                'type': rule_type,
                'layer_name': layer_name,
                'rule_path': current_path,
                'filter_expression': current_filter,
                'min_scale': current_min_scale,
                'max_scale': current_max_scale,
                'active': self._get_rule_active(rule, rule_type),
                'data': current_data,
                'original_rule': rule
            }
            
            flattened.append(flattened_rule)
            
        # Process children recursively or add leaf rule
        if rule.children():
            for child in rule.children():
                flattened.extend(self._flatten_rules(
                    child, layer_name, rule_type, current_path, current_filter,
                    current_min_scale, current_max_scale, current_data))
        return flattened
    
    def _get_rule_label(self, rule, rule_type, index):
        """Get rule label based on type."""
        rule_name == rule.label() if rule_type =='renderer' else rule.description()
        if rule_name:
            return rule_name
        return f"Rule_{index}"
    
    def _inherit_rule_data(self, parent_data, rule, rule_type):
        """Inherit rule-specific data (symbol or settings)."""
        if rule_type == 'renderer':
            current_symbol = rule.symbol()
            if parent_data and current_symbol:
                inherited_symbol = current_symbol.clone()
                for i in range(parent_data.symbolLayerCount()):
                    inherited_symbol.appendSymbolLayer(parent_data.symbolLayer(i).clone())
                return inherited_symbol
            return current_symbol.clone() if current_symbol else parent_data
        else:
            # For labeling, child settings override parent
            return rule.settings() or parent_data
    
    def _combine_filters(self, parent_filter, child_filter):
        """Combine filters using AND logic."""
        parent_filter = parent_filter.strip() if parent_filter else ""
        child_filter = child_filter.strip() if child_filter else ""
        
        if not parent_filter and not child_filter:
            return ""
        elif not parent_filter:
            return child_filter
        elif not child_filter:
            return parent_filter
        else:
            return f"({parent_filter}) AND ({child_filter})"
    
    def _inherit_scale(self, parent_scale, child_scale, comparator):
        """Inherit scale using comparator function (min/max)."""
        if child_scale > 0 and parent_scale > 0:
            return comparator(child_scale, parent_scale)
        return child_scale if child_scale > 0 else parent_scale

# Console execution
if __name__ == '__console__':
    extractor = RuleExtractor()
    print("Extracting rules from current QGIS project...")
    rules = extractor.extract_all_rules()
    extractor.print_summary()
    print(f"\nRule extractor ready! Use:")