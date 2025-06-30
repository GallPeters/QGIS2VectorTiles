from qgis.core import *
import re


class VectorTileCompatibilityProcessor:
    
    def __init__(self, rules_dict):
        self.rules_dict = rules_dict
        self.project = QgsProject.instance()
        self.created_layers = []
        self.warnings = []
    
    def process_all_layers(self):
        for layer, rules in self.rules_dict.items():
            if not isinstance(layer, QgsVectorLayer) or layer.geometryType() != QgsWkbTypes.PolygonGeometry:
                continue
            
            if "Labeling" in rules:
                self._process_labeling_rule(layer, rules["Labeling"])
            
            if "Renderer" in rules:
                self._process_renderer_rule(layer, rules["Renderer"])
        
        return self.created_layers
    
    def _process_labeling_rule(self, layer, labeling_rule):
        try:
            centroid_layer = self._create_centroid_query_layer(layer)
            self._apply_labeling_to_centroid(centroid_layer, layer, labeling_rule)
            
            self.project.addMapLayer(centroid_layer)
            self.created_layers.append(centroid_layer)
            print(f"Created centroid query layer: {centroid_layer.name()}")
            
        except Exception as e:
            print(f"Error processing labeling rule for {layer.name()}: {str(e)}")
    
    def _process_renderer_rule(self, layer, renderer_rule):
        try:
            if self._has_outline_symbol_layers(layer.renderer()):
                line_layer = self._create_boundary_query_layer(layer)
                self._apply_line_symbology(line_layer, layer)
                
                self.project.addMapLayer(line_layer)
                self.created_layers.append(line_layer)
                print(f"Created boundary query layer: {line_layer.name()}")
            
            self._check_polygon_outlines(layer)
            
        except Exception as e:
            print(f"Error processing renderer rule for {layer.name()}: {str(e)}")
    
    def _create_centroid_query_layer(self, layer):
        # Get layer source info
        source_uri = QgsDataSourceUri(layer.source())
        
        # Build query with centroid and area
        table_name = source_uri.table() or source_uri.schema()
        geometry_col = source_uri.geometryColumn() or 'geom'
        
        query = f"""
        SELECT *, 
               ST_Area({geometry_col}) as poly_area,
               ST_Centroid({geometry_col}) as geom
        FROM {table_name}
        """
        
        # Create new URI for query layer
        new_uri = QgsDataSourceUri()
        new_uri.setConnection(source_uri.host(), source_uri.port(), 
                             source_uri.database(), source_uri.username(), 
                             source_uri.password())
        new_uri.setDataSource("", f"({query})", "geom", "", "")
        
        # Create layer
        centroid_layer = QgsVectorLayer(new_uri.uri(), f"{layer.name()}_centroids", layer.providerType())
        return centroid_layer
    
    def _create_boundary_query_layer(self, layer):
        # Get layer source info
        source_uri = QgsDataSourceUri(layer.source())
        
        # Build query with boundary and area
        table_name = source_uri.table() or source_uri.schema()
        geometry_col = source_uri.geometryColumn() or 'geom'
        
        query = f"""
        SELECT *, 
               ST_Area({geometry_col}) as poly_area,
               ST_Boundary({geometry_col}) as geom
        FROM {table_name}
        """
        
        # Create new URI for query layer
        new_uri = QgsDataSourceUri()
        new_uri.setConnection(source_uri.host(), source_uri.port(), 
                             source_uri.database(), source_uri.username(), 
                             source_uri.password())
        new_uri.setDataSource("", f"({query})", "geom", "", "")
        
        # Create layer
        boundary_layer = QgsVectorLayer(new_uri.uri(), f"{layer.name()}_boundaries", layer.providerType())
        return boundary_layer
    
    def _apply_labeling_to_centroid(self, centroid_layer, polygon_layer, labeling_rule):
        original_labeling = polygon_layer.labeling()
        if not original_labeling:
            return
        
        if isinstance(original_labeling, QgsVectorLayerSimpleLabeling):
            settings = QgsPalLayerSettings(original_labeling.settings())
            self._update_area_expressions(settings)
            centroid_layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
            
        elif isinstance(original_labeling, QgsVectorLayerRuleBasedLabeling):
            root_rule = original_labeling.rootRule().clone()
            self._update_rule_labeling_expressions(root_rule)
            centroid_layer.setLabeling(QgsVectorLayerRuleBasedLabeling(root_rule))
        
        centroid_layer.setLabelsEnabled(True)
    
    def _update_area_expressions(self, settings):
        if settings.fieldName:
            settings.fieldName = self._replace_area_expressions(settings.fieldName)
    
    def _update_rule_labeling_expressions(self, rule):
        if rule.settings():
            self._update_area_expressions(rule.settings())
        
        if rule.filterExpression():
            rule.setFilterExpression(self._replace_area_expressions(rule.filterExpression()))
        
        for child in rule.children():
            self._update_rule_labeling_expressions(child)
    
    def _replace_area_expressions(self, expression):
        if not expression:
            return expression
        
        updated = expression.replace('$area', '"poly_area"')
        updated = re.sub(r'area\s*\(\s*\$geometry\s*\)', '"poly_area"', updated, flags=re.IGNORECASE)
        return updated
    
    def _has_outline_symbol_layers(self, renderer):
        def check_symbol(symbol):
            if not symbol or symbol.type() != QgsSymbol.Fill:
                return False
            
            for i in range(symbol.symbolLayerCount()):
                layer_type = symbol.symbolLayer(i).layerType()
                if 'line' in layer_type.lower() or layer_type in ['SimpleLine', 'RasterLine']:
                    return True
            return False
        
        if hasattr(renderer, 'symbol') and renderer.symbol():
            return check_symbol(renderer.symbol())
        elif hasattr(renderer, 'symbols'):
            return any(check_symbol(s) for s in renderer.symbols() if s)
        elif isinstance(renderer, QgsRuleBasedRenderer):
            return self._check_rule_symbols(renderer.rootRule(), check_symbol)
        
        return False
    
    def _check_rule_symbols(self, rule, check_func):
        if rule.symbol() and check_func(rule.symbol()):
            return True
        return any(self._check_rule_symbols(child, check_func) for child in rule.children())
    
    def _apply_line_symbology(self, line_layer, polygon_layer):
        # Extract line properties from first outline symbol layer found
        renderer = polygon_layer.renderer()
        line_symbol = QgsLineSymbol.createSimple({'color': 'black', 'width': '0.5'})
        
        def extract_line_props(symbol):
            if not symbol or symbol.type() != QgsSymbol.Fill:
                return None
            
            for i in range(symbol.symbolLayerCount()):
                symbol_layer = symbol.symbolLayer(i)
                layer_type = symbol_layer.layerType()
                if 'line' in layer_type.lower() or layer_type in ['SimpleLine', 'RasterLine']:
                    if hasattr(symbol_layer, 'color'):
                        line_symbol.setColor(symbol_layer.color())
                    if hasattr(symbol_layer, 'width'):
                        line_symbol.setWidth(symbol_layer.width())
                    return True
            return False
        
        # Try to extract from renderer
        if hasattr(renderer, 'symbol') and renderer.symbol():
            extract_line_props(renderer.symbol())
        elif hasattr(renderer, 'symbols') and renderer.symbols():
            for symbol in renderer.symbols():
                if extract_line_props(symbol):
                    break
        
        line_layer.setRenderer(QgsSingleSymbolRenderer(line_symbol))
    
    def _check_polygon_outlines(self, layer):
        def check_symbol_outlines(symbol, rule_name=""):
            if not symbol or symbol.type() != QgsSymbol.Fill:
                return
            
            for i in range(symbol.symbolLayerCount()):
                symbol_layer = symbol.symbolLayer(i)
                if isinstance(symbol_layer, QgsSimpleFillSymbolLayer):
                    if (hasattr(symbol_layer, 'strokeWidth') and 
                        hasattr(symbol_layer, 'strokeColor') and
                        symbol_layer.strokeWidth() > 0 and
                        symbol_layer.strokeColor().alpha() > 0):
                        
                        warning = f"Warning: {layer.name()} rule '{rule_name}' symbol layer {i} has visible polygon outline"
                        self.warnings.append(warning)
                        print(warning)
        
        renderer = layer.renderer()
        if hasattr(renderer, 'symbol') and renderer.symbol():
            check_symbol_outlines(renderer.symbol(), "default")
        elif hasattr(renderer, 'symbols'):
            for i, symbol in enumerate(renderer.symbols()):
                if symbol:
                    check_symbol_outlines(symbol, f"symbol_{i}")
        elif isinstance(renderer, QgsRuleBasedRenderer):
            self._check_rule_outlines(renderer.rootRule(), check_symbol_outlines)
    
    def _check_rule_outlines(self, rule, check_func):
        if rule.symbol():
            rule_label = rule.label() or "unnamed_rule"
            check_func(rule.symbol(), rule_label)
        
        for child in rule.children():
            self._check_rule_outlines(child, check_func)
    
    def get_warnings(self):
        return self.warnings
    
    def get_created_layers(self):
        return self.created_layers