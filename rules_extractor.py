from qgis.core import *
from PyQt5.QtCore import *


class RulesExtractor:
    """
    Comprehensive PyQGIS class that extracts and flattens all renderer (symbology) 
    and labeling rules from vector layers in the current QGIS project.
    """
    
    def __init__(self):
        self.project = QgsProject.instance()
        self.extracted_rules = []
        
    def extract_all_rules(self):
        """
        Main method to extract all rules from all vector layers in the project.
        Returns a flattened list of standalone rules with inherited properties.
        """
        self.extracted_rules = []
        
        # Get all vector layers from the project
        vector_layers = [layer for layer in self.project.mapLayers().values() 
                        if isinstance(layer, QgsVectorLayer)]
        
        for layer in vector_layers:
            try:
                # Extract renderer rules
                renderer_rules = self._extract_renderer_rules(layer)
                self.extracted_rules.extend(renderer_rules)
                
                # Extract labeling rules
                labeling_rules = self._extract_labeling_rules(layer)
                self.extracted_rules.extend(labeling_rules)
                
            except Exception as e:
                print(f"Error processing layer {layer.name()}: {str(e)}")
                continue
        
        return self.extracted_rules
    
    def _extract_renderer_rules(self, layer):
        """Extract and flatten renderer rules from a vector layer."""
        renderer = layer.renderer()
        renderer_rules = []
        
        if not renderer:
            return renderer_rules
        
        # Convert to rule-based renderer if necessary
        rule_based_renderer = self._convert_to_rule_based_renderer(renderer)
        
        if rule_based_renderer:
            # Get the root rule and flatten its children
            root_rule = rule_based_renderer.rootRule()
            flattened_rules = self._flatten_renderer_rules(root_rule, layer.name())
            renderer_rules.extend(flattened_rules)
        
        return renderer_rules
    
    def _extract_labeling_rules(self, layer):
        """Extract and flatten labeling rules from a vector layer."""
        labeling = layer.labeling()
        labeling_rules = []
        
        if not labeling:
            return labeling_rules
        
        # Convert to rule-based labeling if necessary
        rule_based_labeling = self._convert_to_rule_based_labeling(labeling)
        
        if rule_based_labeling:
            # Get the root rule and flatten its children
            root_rule = rule_based_labeling.rootRule()
            flattened_rules = self._flatten_labeling_rules(root_rule, layer.name())
            labeling_rules.extend(flattened_rules)
        
        return labeling_rules
    
    def _convert_to_rule_based_renderer(self, renderer):
        """Convert any renderer type to rule-based renderer."""
        if renderer.type() == 'RuleRenderer':
            return renderer
        
        # Create a new rule-based renderer
        rule_based = QgsRuleBasedRenderer(None)
        root_rule = rule_based.rootRule()
        
        if isinstance(renderer, QgsSingleSymbolRenderer):
            # Single symbol - create one rule with the symbol
            symbol = renderer.symbol().clone() if renderer.symbol() else None
            child_rule = root_rule.appendChild(QgsRendererRule(symbol))
            
        elif isinstance(renderer, QgsCategorizedSymbolRenderer):
            # Categorized - create rule for each category
            for category in renderer.categories():
                symbol = category.symbol().clone() if category.symbol() else None
                filter_exp = f'"{renderer.classAttribute()}" = \'{category.value()}\''
                label = category.label() or str(category.value())
                child_rule = root_rule.appendChild(QgsRendererRule(symbol, filter_exp, label))
                
        elif isinstance(renderer, QgsGraduatedSymbolRenderer):
            # Graduated - create rule for each range
            for range_item in renderer.ranges():
                symbol = range_item.symbol().clone() if range_item.symbol() else None
                attr = renderer.classAttribute()
                lower = range_item.lowerValue()
                upper = range_item.upperValue()
                filter_exp = f'"{attr}" >= {lower} AND "{attr}" <= {upper}'
                label = range_item.label() or f"{lower} - {upper}"
                child_rule = root_rule.appendChild(QgsRendererRule(symbol, filter_exp, label))
        
        return rule_based
    
    def _convert_to_rule_based_labeling(self, labeling):
        """Convert any labeling type to rule-based labeling."""
        if isinstance(labeling, QgsRuleBasedLabeling):
            return labeling
        
        # Create a new rule-based labeling
        rule_based = QgsRuleBasedLabeling(None)
        root_rule = rule_based.rootRule()
        
        if isinstance(labeling, QgsSimpleLabeling):
            # Simple labeling - create one rule with the settings
            settings = labeling.settings()
            child_rule = root_rule.appendChild(QgsRuleBasedLabelProvider.Rule(settings))
        
        return rule_based
    
    def _flatten_renderer_rules(self, rule, layer_name, parent_path="", parent_filter="", 
                               parent_min_scale=0, parent_max_scale=0, parent_symbol=None):
        """Recursively flatten renderer rules with inheritance."""
        flattened = []
        
        # Skip root rules (they're containers only)
        if parent_path == "":
            for i, child in enumerate(rule.children()):
                child_rules = self._flatten_renderer_rules(
                    child, layer_name, "", "", 0, 0, None
                )
                flattened.extend(child_rules)
            return flattened
        
        # Build rule path
        rule_label = rule.label() or f"Rule_{len(flattened)}"
        current_path = f"{parent_path} > {rule_label}" if parent_path else rule_label
        
        # Inherit and combine filter expressions
        current_filter = self._combine_filters(parent_filter, rule.filterExpression())
        
        # Inherit scale ranges
        current_min_scale = self._inherit_min_scale(parent_min_scale, rule.minimumScale())
        current_max_scale = self._inherit_max_scale(parent_max_scale, rule.maximumScale())
        
        # Inherit symbol layers
        current_symbol = self._inherit_symbol(parent_symbol, rule.symbol())
        
        # Create flattened rule object
        flattened_rule = {
            'type': 'renderer',
            'layer_name': layer_name,
            'rule_path': current_path,
            'filter_expression': current_filter,
            'min_scale': current_min_scale,
            'max_scale': current_max_scale,
            'symbol': current_symbol.clone() if current_symbol else None,
            'active': rule.active(),
            'original_rule': rule
        }
        
        # If this rule has children, process them recursively
        if rule.children():
            for child in rule.children():
                child_rules = self._flatten_renderer_rules(
                    child, layer_name, current_path, current_filter,
                    current_min_scale, current_max_scale, current_symbol
                )
                flattened.extend(child_rules)
        else:
            # This is a leaf rule, add it to the flattened list
            flattened.append(flattened_rule)
        return flattened
    
    def _flatten_labeling_rules(self, rule, layer_name, parent_path="", parent_filter="",
                               parent_min_scale=0, parent_max_scale=0, parent_settings=None):
        """Recursively flatten labeling rules with inheritance."""
        flattened = []
        
        # Skip root rules (they're containers only)
        if parent_path == "":
            for i, child in enumerate(rule.children()):
                child_rules = self._flatten_labeling_rules(
                    child, layer_name, "", "", 0, 0, None
                )
                flattened.extend(child_rules)
            return flattened
        
        # Build rule path
        rule_description = rule.description() or f"Label_Rule_{len(flattened)}"
        current_path = f"{parent_path} > {rule_description}" if parent_path else rule_description
        
        # Inherit and combine filter expressions
        current_filter = self._combine_filters(parent_filter, rule.filterExpression())
        
        # Inherit scale ranges
        current_min_scale = self._inherit_min_scale(parent_min_scale, rule.minimumScale())
        current_max_scale = self._inherit_max_scale(parent_max_scale, rule.maximumScale())
        
        # Inherit label settings
        current_settings = self._inherit_label_settings(parent_settings, rule.settings())
        
        # Create flattened rule object
        flattened_rule = {
            'type': 'labeling',
            'layer_name': layer_name,
            'rule_path': current_path,
            'filter_expression': current_filter,
            'min_scale': current_min_scale,
            'max_scale': current_max_scale,
            'settings': current_settings,
            'active': rule.active(),
            'original_rule': rule
        }
        
        # If this rule has children, process them recursively
        if rule.children():
            for child in rule.children():
                child_rules = self._flatten_labeling_rules(
                    child, layer_name, current_path, current_filter,
                    current_min_scale, current_max_scale, current_settings
                )
                flattened.extend(child_rules)
        else:
            # This is a leaf rule, add it to the flattened list
            flattened.append(flattened_rule)
        
        return flattened
    
    def _combine_filters(self, parent_filter, child_filter):
        """Combine parent and child filter expressions using AND logic."""
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
    
    def _inherit_min_scale(self, parent_min, child_min):
        """Inherit minimum scale (more restrictive wins)."""
        if child_min > 0 and parent_min > 0:
            return max(child_min, parent_min)  # More restrictive (larger number)
        elif child_min > 0:
            return child_min
        elif parent_min > 0:
            return parent_min
        else:
            return 0
    
    def _inherit_max_scale(self, parent_max, child_max):
        """Inherit maximum scale (more restrictive wins)."""
        if child_max > 0 and parent_max > 0:
            return min(child_max, parent_max)  # More restrictive (smaller number)
        elif child_max > 0:
            return child_max
        elif parent_max > 0:
            return parent_max
        else:
            return 0
    
    def _inherit_symbol(self, parent_symbol, child_symbol):
        """Inherit symbol layers from parent to child."""
        if not child_symbol and not parent_symbol:
            return None
        
        if not parent_symbol:
            return child_symbol.clone() if child_symbol else None
        
        if not child_symbol:
            return parent_symbol.clone()
        
        # Clone child symbol and append parent symbol layers
        inherited_symbol = child_symbol.clone()
        
        # Add parent symbol layers to child
        for i in range(parent_symbol.symbolLayerCount()):
            parent_layer = parent_symbol.symbolLayer(i).clone()
            inherited_symbol.appendSymbolLayer(parent_layer)
        
        return inherited_symbol
    
    def _inherit_label_settings(self, parent_settings, child_settings):
        """Inherit label settings from parent to child."""
        if not child_settings and not parent_settings:
            return None
        
        if not parent_settings:
            return child_settings
        
        if not child_settings:
            return parent_settings
        
        # For labeling, child settings typically override parent settings
        # This is a simplified approach - in practice, you might want more sophisticated inheritance
        return child_settings
    
    def print_summary(self):
        """Print a summary of extracted rules."""
        if not self.extracted_rules:
            print("No rules extracted.")
            return
        
        renderer_count = len([r for r in self.extracted_rules if r['type'] == 'renderer'])
        labeling_count = len([r for r in self.extracted_rules if r['type'] == 'labeling'])
        
        print(f"\n=== QGIS Rule Extraction Summary ===")
        print(f"Total rules extracted: {len(self.extracted_rules)}")
        print(f"Renderer rules: {renderer_count}")
        print(f"Labeling rules: {labeling_count}")
        print()
        
        # Group by layer
        layers = {}
        for rule in self.extracted_rules:
            layer_name = rule['layer_name']
            if layer_name not in layers:
                layers[layer_name] = {'renderer': 0, 'labeling': 0}
            layers[layer_name][rule['type']] += 1
        
        for layer_name, counts in layers.items():
            print(f"Layer: {layer_name}")
            print(f"  - Renderer rules: {counts['renderer']}")
            print(f"  - Labeling rules: {counts['labeling']}")
        
        print("\n=== Sample Rules ===")
        for i, rule in enumerate(self.extracted_rules[:5]):  # Show first 5 rules
            print(f"\nRule {i+1}:")
            print(f"  Type: {rule['type']}")
            print(f"  Layer: {rule['layer_name']}")
            print(f"  Path: {rule['rule_path']}")
            print(f"  Filter: {rule['filter_expression'] or 'None'}")
            print(f"  Scale: {rule['min_scale']} - {rule['max_scale']}")
            print(f"  Active: {rule['active']}")
    
    def get_rules_by_layer(self, layer_name):
        """Get all rules for a specific layer."""
        return [rule for rule in self.extracted_rules if rule['layer_name'] == layer_name]
    
    def get_rules_by_type(self, rule_type):
        """Get all rules of a specific type ('renderer' or 'labeling')."""
        return [rule for rule in self.extracted_rules if rule['type'] == rule_type]
    
    def export_rules_to_dict(self):
        """Export rules to a serializable dictionary format."""
        export_data = []
        
        for rule in self.extracted_rules:
            rule_data = {
                'type': rule['type'],
                'layer_name': rule['layer_name'],
                'rule_path': rule['rule_path'],
                'filter_expression': rule['filter_expression'],
                'min_scale': rule['min_scale'],
                'max_scale': rule['max_scale'],
                'active': rule['active']
            }
            
            # Add type-specific data
            if rule['type'] == 'renderer' and rule['symbol']:
                rule_data['symbol_type'] = rule['symbol'].type()
                rule_data['symbol_layer_count'] = rule['symbol'].symbolLayerCount()
            elif rule['type'] == 'labeling' and rule['settings']:
                rule_data['field_name'] = rule['settings'].fieldName
                rule_data['enabled'] = rule['settings'].enabled
            
            export_data.append(rule_data)
        
        return export_data


# Console execution
if __name__ == '__console__':
    # Create the rule extractor
    extractor = RulesExtractor()
    
    # Extract all rules from the current project
    print("Extracting rules from current QGIS project...")
    rules = extractor.extract_all_rules()
    
    # Print summary
    extractor.print_summary()
    
    # Example usage:
    # Get rules for a specific layer
    # layer_rules = extractor.get_rules_by_layer("Your Layer Name")
    
    # Get only renderer rules
    # renderer_rules = extractor.get_rules_by_type("renderer")
    
    # Export to dictionary format
    # exported_rules = extractor.export_rules_to_dict()
    
    # print(f"\nRule extractor created successfully!")
    # print("Available methods:")
    # print("- extractor.extract_all_rules()")
    # print("- extractor.get_rules_by_layer(layer_name)")
    # print("- extractor.get_rules_by_type('renderer' or 'labeling')")
    # print("- extractor.export_rules_to_dict()")
    # print("- extractor.print_summary()")