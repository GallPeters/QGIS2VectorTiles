"""
Q2StyledTiles:

Converts QGIS vector layer styling to vector tiles format by:
1. Flattening nested rule-based renderers/labeling with property inheritance
2. Splitting rules by symbol layers and matching label rules to renderer rules
3. Exporting each rule as a separate dataset with geometry transformations
4. Generating vector tiles using GDAL MVT driver
5. Loading and styling the tiles in QGIS with appropriate symbology
"""

from dataclasses import dataclass
from tomllib import load
from datetime import datetime
from os import makedirs, cpu_count
from os.path import join, basename, exists
from tempfile import gettempdir
from time import perf_counter, sleep
from typing import List, Optional, Tuple, Union
from urllib.request import pathname2url
from shutil import rmtree
from gc import collect

import processing
from osgeo import gdal, ogr, osr
from PyQt5.QtCore import QVariant
from qgis.core import (
    QgsProject, QgsRuleBasedRenderer, QgsRuleBasedLabeling, QgsPalLayerSettings,
    QgsVectorLayer, QgsLayerDefinition, QgsVectorTileLayer, 
    QgsGraduatedSymbolRenderer, QgsCategorizedSymbolRenderer, QgsWkbTypes, QgsVectorTileBasicRenderer,
    QgsProcessingFeatureSourceDefinition, QgsVectorTileBasicRendererStyle,
    QgsVectorTileBasicLabeling, QgsVectorTileBasicLabelingStyle,
    QgsMarkerSymbol, QgsLineSymbol, QgsFillSymbol, QgsTextFormat,
    QgsFeatureRequest, QgsProcessingFeedback, QgsApplication,QgsPropertyDefinition, QgsLabelThinningSettings, Qgis, QgsExpression
)

from .settings import _CONF, _PLUGIN_DIR


_TILES_CONF = load(open(_CONF, "rb"))
_TILING_SCHEME = _TILES_CONF['TILING_SCHEME']


class ZoomLevels:
    """Manages zoom level scales and conversions for web mapping standards."""
    SCALES = [_TILING_SCHEME['SCALE'] / (2 ** zoom) for zoom in range(23)]

    @classmethod
    def scale_to_zoom(cls, scale: float, edge: str) -> str:
        """Convert scale to zero-padded zoom level string."""
        if scale in [0, 0.0]:
            scale = cls.SCALES[0 if edge == "o" else -1]
        if edge == 'o':
            for zoom, zoom_scale in enumerate(cls.SCALES):
                if scale >= zoom_scale:
                    return zoom
        else:
             for zoom, zoom_scale in sorted(enumerate(cls.SCALES), reverse=True):
                if scale <= zoom_scale:
                    return zoom
        return len(cls.SCALES) - 1

    @classmethod
    def zoom_to_scale(cls, zoom: int) -> Optional[float]:
        """Convert zoom level to scale."""
        if 0 <= zoom < len(cls.SCALES):
            return cls.SCALES[zoom]
        return None


@dataclass
class FlattenedRule:
    """A flattened rule with inherited properties from parent hierarchy."""

    rule: Union[QgsRuleBasedLabeling.Rule, QgsRuleBasedRenderer.Rule]
    layer: QgsVectorLayer
    name: Optional[str]= ""
    output_dataset_name: str = ""

    def __post_init__(self):
        """Construct rule's name"""
        if not self.name:
            lyr_name = self.layer.name() or self.layer.id()  
            rule_type = 'symbology' if isinstance(self.rule,  QgsRuleBasedRenderer.Rule) else 'labeling'
            if rule_type == 'symbology':
                rule_name = self.rule.label()
            else:
                rule_name = self.rule.description()
            rule_name = rule_name or self.rule.ruleKey()
            self.name = f'{lyr_name} > {rule_type} > {rule_name}'

    def get_attribute(self, char: str) -> Optional[int]:
        """Extract rule attribute from description by character prefix."""
        desc = self.rule.description()
        start = desc.find(char) + 1
        if start == 0:
            return None
        return int(desc[start:start + 2])

    def set_attribute(self, char: str, value: int) -> None:
        """Set rule attribute in description."""
        value = int(value)
        new_attr = f"{char}{value:02d}"
        current = self.get_attribute(char)

        desc = self.rule.description()
        if current is not None:
            old_attr = f"{char}{current:02d}"
            desc = desc.replace(old_attr, new_attr)
        else:
            desc = f"{desc}{new_attr}"

        self.rule.setDescription(desc)
        self.output_dataset_name = desc
        
        # Normalize scale attribute for dataset naming
        i = desc.find("s")
        if i >= 0:
            self.output_dataset_name = self.output_dataset_name.replace(desc[i:i + 3], "s00")


class TilesStyler:
    """Applies styling to vector tile layers from flattened rules."""

    def __init__(self, flattened_rules: List[FlattenedRule], output_dir: str, tiles_path: str):
        self.flattened_rules = flattened_rules
        self.output_dir = output_dir
        self.tiles_layer = self._create_tiles_layer(tiles_path)
        self.renderer_styles = []
        self.labeling_styles = []

    def apply_styling(self) -> QgsVectorTileLayer:
        """Apply styles to vector tiles layer and add to project."""
        for rule in self.flattened_rules[::-1]:
            self._create_style_from_rule(rule)
        
        self._apply_styles_to_layer()
        self._save_style()
        return self.tiles_layer

    def _create_tiles_layer(self, tiles_path: Optional[str]) -> QgsVectorTileLayer:
        """Create and add vector tiles layer to project."""
        suffix = "&http-header:referer=" if tiles_path and tiles_path.split(".")[-1] != "mbtiles" else ""
        layer = QgsVectorTileLayer(f"{tiles_path}{suffix}", "Vector Tiles")
        layer = QgsProject.instance().addMapLayer(layer, False)
        QgsProject.instance().layerTreeRoot().insertLayer(0, layer)
        return layer

    def _create_style_from_rule(self, flat_rule: FlattenedRule) -> None:
        """Create appropriate style from flattened rule."""
        rule_type = flat_rule.get_attribute("t")

        if rule_type == 0:  # Renderer
            style = QgsVectorTileBasicRendererStyle()
            self._setup_renderer_style(style, flat_rule)
            self.renderer_styles.append(style)
        else:  # Labeling
            style = QgsVectorTileBasicLabelingStyle()
            self._setup_labeling_style(style, flat_rule)
            self.labeling_styles.append(style)

    def _setup_renderer_style(self, style: QgsVectorTileBasicRendererStyle, 
                              flat_rule: FlattenedRule) -> None:
        """Configure renderer style properties."""
        self._setup_base_style_properties(style, flat_rule)

        symbol = flat_rule.rule.symbol()
        symbol_layer = symbol.symbolLayers()[-1]
        sub_symbol = symbol_layer.subSymbol()
        source_geom = int(flat_rule.get_attribute("g"))
        target_geom = int(flat_rule.get_attribute("c"))

        # Handle geometry transformations
        if source_geom != target_geom:
            if sub_symbol and symbol_layer.layerType() == "GeometryGenerator":
                self._copy_data_driven_properties(symbol, sub_symbol)
                self._copy_data_driven_properties(
                    symbol.symbolLayers()[-1], 
                    sub_symbol.symbolLayers()[-1]
                )
                symbol = sub_symbol
            else:
                symbol = self._create_transformed_symbol(target_geom, symbol_layer)

        style.setSymbol(symbol.clone())

    def _create_transformed_symbol(self, target_geom: int, symbol_layer):
        """Create symbol with transformed geometry type."""
        symbol_map = {
            0: QgsMarkerSymbol,
            1: QgsLineSymbol,
            2: QgsFillSymbol,
        }
        symbol = symbol_map.get(target_geom, QgsMarkerSymbol)()
        symbol.appendSymbolLayer(symbol_layer.clone())
        symbol.deleteSymbolLayer(0)
        return symbol

    def _setup_labeling_style(self, style: QgsVectorTileBasicLabelingStyle, 
                              flat_rule: FlattenedRule) -> None:
        """Configure labeling style properties."""
        self._setup_base_style_properties(style, flat_rule)
        settings = QgsPalLayerSettings(flat_rule.rule.settings())
        if Qgis.versionInt() >= 34400:
            self._remove_duplicates_labels(settings)
        style.setLabelSettings(settings)

    def _remove_duplicates_labels(self, settings):
        """Remove duplicate labels to avoid labels appears in each tile"""
        thin = QgsLabelThinningSettings(settings.thinningSettings())
        thin.setAllowDuplicateRemoval(True)
        remove_distance = _TILES_CONF['GENERAL_CONF']['REMOVE_DUPLICATES_DISTANCE']
        thin.setMinimumDistanceToDuplicate(remove_distance)
        thin.setMinimumDistanceToDuplicateUnit(Qgis.RenderUnit.Points)
        settings.setThinningSettings(thin)

    def _setup_base_style_properties(self, style, flat_rule: FlattenedRule) -> None:
        """Setup common style properties."""
        style.setEnabled(True)
        style.setLayerName(flat_rule.output_dataset_name)
        style.setStyleName(flat_rule.rule.description())
        style.setMinZoomLevel(flat_rule.get_attribute('o'))
        style.setMaxZoomLevel(flat_rule.get_attribute('i'))
        geom_types = {
            0: QgsWkbTypes.PointGeometry,
            1: QgsWkbTypes.LineGeometry,
            2: QgsWkbTypes.PolygonGeometry,
        }
        geom_code = flat_rule.get_attribute("c")
        style.setGeometryType(geom_types.get(geom_code, QgsWkbTypes.PointGeometry))

    def _copy_data_driven_properties(self, source_obj, target_obj) -> None:
        """Copy data-driven properties between objects."""
        source_props = source_obj.dataDefinedProperties()
        target_props = target_obj.dataDefinedProperties()   

        for prop_key in source_obj.propertyDefinitions():
            prop = source_props.property(prop_key)
            target_props.setProperty(prop_key, prop)
            target_props.property(prop_key).setActive(True)
        
    def _apply_styles_to_layer(self) -> None:
        """Apply collected styles to the tiles layer."""
        renderer = QgsVectorTileBasicRenderer()
        renderer.setStyles(self.renderer_styles)

        labeling = QgsVectorTileBasicLabeling()
        labeling.setStyles(self.labeling_styles)

        self.tiles_layer.setRenderer(renderer)
        self.tiles_layer.setLabeling(labeling)

    def _save_style(self) -> None:
        """Save layer style to QLR file."""
        makedirs(self.output_dir, exist_ok=True)
        qlr_path = join(self.output_dir, "tiles.qlr")
        layer = QgsProject.instance().layerTreeRoot().findLayer(self.tiles_layer.id())
        QgsLayerDefinition().exportLayerDefinition(qlr_path, [layer])


class GDALTilesGenerator:
    """Generate mbtiles/XYZ tiles from GeoJSON layers using GDAL MVT driver."""

    def __init__(self, layers: List[QgsVectorLayer], output_dir: str, output_type: str,
                 extent, cpu_percent: int, feedback: QgsProcessingFeedback):
        self.layers = layers
        self.output_dir = output_dir
        self.output_type = output_type
        self.extent = extent
        self.cpu_percent = cpu_percent
        self.feedback = feedback

    def generate(self) -> str:
        """Generate tiles file from configured layers."""
        self._configure_gdal_threading()

        spatial_ref = osr.SpatialReference()
        crs_id = _TILING_SCHEME['EPSG_CRS']
        spatial_ref.ImportFromEPSG(crs_id)

        output, uri = self._prepare_output_paths()
        creation_options = self._get_creation_options()

        driver = gdal.GetDriverByName("MVT")
        dataset = driver.Create(output, 0, 0, 0, gdal.GDT_Unknown, options=creation_options)

        for layer in self.layers:
            self._process_layer(dataset, layer, spatial_ref)

        dataset.FlushCache()
        dataset = None

        return uri

    def _configure_gdal_threading(self) -> None:
        """Configure GDAL threading based on CPU percentage."""
        cpu_num = str(max(1, int(cpu_count() * self.cpu_percent / 100)))
        gdal.SetConfigOption("GDAL_NUM_THREADS", cpu_num)

    def _prepare_output_paths(self) -> Tuple[str, str]:
        """Prepare output paths based on output type."""
        if self.output_type == "xyz":
            template = pathname2url(r"/{z}/{x}/{y}.pbf")
            output = join(self.output_dir, "tiles")
            uri = f"type=xyz&zmin=0&zmax=22&url=file:///{output}{template}"
            return output, uri
        else:
            output = join(self.output_dir, "tiles.mbtiles")
            uri = f"type=mbtiles&url={output}"
            return output, uri

    def _get_creation_options(self) -> List[str]:
        """Generate GDAL creation options based on output type."""
        range_options = [f"MINZOOM={self._get_global_min_zoom()}", f"MAXZOOM={self._get_global_max_zoom()}"]
        if self.output_type == "mbtiles":
            return range_options
        scheme_conf = list(_TILING_SCHEME.values())[:-1]
        scheme_options = [f"TILING_SCHEME=EPSG{','.join([str(val) for val in scheme_conf])}"]
        additional_options = [f'{key}={value}' for key, value in _TILES_CONF['GDAL_OPTIONS'].items()]
        return range_options + scheme_options + additional_options

    def _process_layer(self, dataset, layer: QgsVectorLayer, spatial_ref) -> None:
        """Process a single layer and add it to the dataset."""
        layer_name = basename(layer.source()).split(".")[0]
        min_zoom = int(layer_name.split("o")[1][:2])
        max_zoom = int(layer_name.split("i")[1][:2])

        source = layer.source().split("|layername=")[0]
        src_dataset = ogr.Open(source)
        src_layer = src_dataset.GetLayer(0)

        layer_options = [f"MINZOOM={min_zoom}", f"MAXZOOM={max_zoom}"]
        out_layer = dataset.CreateLayer(
            layer_name, 
            srs=spatial_ref, 
            geom_type=src_layer.GetGeomType(), 
            options=layer_options
        )

        # Copy field definitions
        src_defn = src_layer.GetLayerDefn()
        for i in range(src_defn.GetFieldCount()):
            field_defn = src_defn.GetFieldDefn(i)
            out_layer.CreateField(field_defn)

        # Copy features
        out_defn = out_layer.GetLayerDefn()
        src_layer.ResetReading()

        for src_feat in src_layer:
            out_feat = ogr.Feature(out_defn)
            out_feat.SetGeometry(src_feat.GetGeometryRef())

            for i in range(out_defn.GetFieldCount()):
                out_feat.SetField(i, src_feat.GetField(i))

            out_layer.CreateFeature(out_feat)
            out_feat = None

        src_dataset = None

    def _get_global_min_zoom(self) -> int:
        """Get minimum zoom level across all layers."""
        min_zoom = float('inf')
        for layer in self.layers:
            layer_name = basename(layer.source()).split(".")[0]
            zoom = int(layer_name.split("o")[1][:2])
            min_zoom = min(min_zoom, zoom)
        return int(min_zoom) if min_zoom != float('inf') else 0

    def _get_global_max_zoom(self) -> int:
        """Get maximum zoom level across all layers."""
        max_zoom = 0
        for layer in self.layers:
            layer_name = basename(layer.source()).split(".")[0]
            zoom = int(layer_name.split("i")[1][:2])
            max_zoom = max(max_zoom, zoom)
        return max_zoom if max_zoom > 0 else 14


class RulesExporter:
    """Export all rules to datasets with geometry transformations."""

    FIELD_PREFIX = "q2vt"

    def __init__(self, flattened_rules: List[FlattenedRule],
                 extent, include_required_fields_only, max_zoom, cent_source, feedback: QgsProcessingFeedback):
        self.flattened_rules = flattened_rules
        self.extent = extent
        self.include_required_fields_only = include_required_fields_only
        self.max_zoom = max_zoom
        self.cent_source = cent_source
        self.temp_dir = self._create_temp_dir()
        self.processed_layers = []
        self.feedback = feedback

    def export(self) -> List[QgsVectorLayer]:
        """Export all rules to datasets."""
        self._export_base_layers()
        
        # Group rules by dataset name
        rules_by_dataset = {}
        for flat_rule in self.flattened_rules:
            rules_by_dataset.setdefault(flat_rule.output_dataset_name, []).append(flat_rule)

        for rules in rules_by_dataset.values():
            if not self._export_rule_group(rules):
                for rule in rules:
                    self.flattened_rules.remove(rule)
        return self.processed_layers, self.flattened_rules

    def _match_zoom_levels_to_qgis_tiling_scheme(self, rules: list[FlattenedRule]):
        """The tiling scheme zoom levels need to be updated because QGIS treats zoom level 0 differently than GDAL.
        For some reason, this only affects the rendering rules of the vector tiling layer while the labeling rules
        are displayed at the correct zoom level."""
        for rule in rules:
            for attr in ['i', 'o']:
                attr_val = rule.get_attribute(attr)
                fitted = max(attr_val - 1, 0)
                rule.set_attribute(attr, fitted)
        # else:
        #     real_maxzoom = self.max_zoom - 1
        #     for rule in rules:
        #         for attr in ['i', 'o']:
        #             attr_val = rule.get_attribute(attr)
        #             if attr == 'o' and attr_val > real_maxzoom:
        #                 return
        #             if attr == 'i' and attr_val > real_maxzoom:
        #                 rule.set_attribute(attr, real_maxzoom)
        # return rules
    
    def _create_temp_dir(self) -> Tuple[str, str]:
        """Create temporary and datasets directories."""
        temp_dir = join(_PLUGIN_DIR, '_temp')
        if exists(temp_dir):
            try:
                collect()
                rmtree(temp_dir)
                makedirs(temp_dir, exist_ok=True)
            except PermissionError:
                pass
        else:
            makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def _export_base_layers(self) -> None:
        """Export base vector layers to FlatGeobuf format."""
        layers_fields = {}
        for flat_rule in self.flattened_rules:
            required_fields = self._get_required_fields(flat_rule)
            calculated_fields = self._get_calculated_fields(flat_rule)
            all_fields = required_fields.union(calculated_fields)
            
            if flat_rule.layer not in layers_fields:
                layers_fields[flat_rule.layer] = set()
            layers_fields[flat_rule.layer] = layers_fields[flat_rule.layer].union(all_fields)

        for layer, fields in layers_fields.items():
            self._export_single_base_layer(layer, fields)

    def _get_required_fields(self, flat_rule: FlattenedRule) -> set:
        """Get fields required by rule for rendering/labeling."""
        return set(f.name() for f in flat_rule.layer.fields())
        # if self.include_required_fields_only != 0:
        #     return set(f.name() for f in flat_rule.layer.fields())
        # rule_type = flat_rule.get_attribute("t")
        # if rule_type == 0:  # Renderer
        #     return set(flat_rule.rule.symbol().usedAttributes(QgsRenderContext()))    
        # return set(flat_rule.rule.settings().referencedFields(QgsRenderContext()))

    def _get_calculated_fields(self, flat_rule: FlattenedRule) -> set:
        """Get calculated fields from filter expression."""
        return set([
            f.name() for f in flat_rule.layer.fields() 
            if f.name() in flat_rule.rule.filterExpression()
        ])

    def _export_single_base_layer(self, layer: QgsVectorLayer, fields: set) -> None:
        """Export single base layer with field selection and transformations."""
        layer_id = layer.id()
        fields_str = ','.join([f for f in fields if f not in ('#!allattributes!#', 'fid')])
        selection = f' -select {fields_str}' if fields_str else ''
        
        extent_wkt = self.extent.asWktCoordinates().replace(",",'')
        subset_string = layer.subsetString()
        where = f' -where "{subset_string}"' if subset_string else ''
        
        output_path = join(self.temp_dir, f'map_layer_{layer_id}.fgb')
        
        # Build processing pipeline
        options = f'-spat {extent_wkt} -spat_srs EPSG:{_TILING_SCHEME['EPSG_CRS']} -t_srs EPSG:{_TILING_SCHEME['EPSG_CRS']} -dim XY{where} -nlt PROMOTE_TO_MULTI{selection}'
        layer = self._run_processing("buildvirtualvector", "gdal", INPUT=layer)
        layer = self._run_processing("convertformat", "gdal", INPUT=layer, OPTIONS=options)

        if layer.featureCount() > 0:
            layer = self._run_processing("multiparttosingleparts", INPUT=layer)
            layer = self._run_processing("simplifygeometries", INPUT=layer, 
                                        METHOD=0, TOLERANCE=_TILES_CONF['GENERAL_CONF']['DATA_SIMPLIFICATION_TOLERANCE'])
            layer = self._run_processing("fixgeometries", INPUT=layer, MTHOD=1)
            layer = self._run_processing("extractbyexpression", INPUT=layer,
                                        EXPRESSION="is_valid($geometry) AND NOT is_empty_or_null($geometry)",
                                        OUTPUT=output_path)

    def _export_rule_group(self, rules: List[FlattenedRule]) -> Optional[QgsVectorLayer]:
        """Export group of rules sharing the same dataset."""
        flat_rule = rules[0]
        layer = QgsVectorLayer(join(self.temp_dir, f"map_layer_{flat_rule.layer.id()}.fgb"))
        if not layer.featureCount() > 0:
            return

        fields = self._create_expression_fields(rules)
        
        if flat_rule.get_attribute("t") == 1:  # Labeling
            fields = self._add_label_expression_field(flat_rule, fields)
        
        if layer.featureCount() <= 0:
            return 
        selected_ids = self._select_features_by_expression(layer, flat_rule)
        if selected_ids is None:
            return
        transformation = self._get_geometry_transformation(rules[0])
        layer = self._apply_field_mapping(layer, fields, selected_ids, transformation, rules)
        if not layer or not layer.featureCount() > 0:
            return
        layer.setName(flat_rule.output_dataset_name)
        self.processed_layers.append(layer)
        
        return layer
    
    def _select_features_by_expression(self, layer: QgsVectorLayer, 
                                      flat_rule: FlattenedRule) -> Optional[List]:
        """Select features matching rule's filter expression."""
        expression = flat_rule.rule.filterExpression()
        if not expression:
            return []

        if QgsExpression(expression).isValid():
            layer.selectByExpression(expression)
            if layer.selectedFeatureCount() > 0:
                return layer.selectedFeatureIds()
            
        return None

    def _apply_field_mapping(self, layer: QgsVectorLayer, fields: list, 
                            selected_ids: Optional[List], transformation, rules: list[FlattenedRule]) -> QgsVectorLayer:
        """Apply field mapping and geometry transformation."""
        single_rule = rules[0]
        field_mapping = [{'type': 4, 'expression': '"fid"', 'name': f"{self.FIELD_PREFIX}_fid"}]
        field_mapping.extend(fields)
        field_mapping.append({'type': 10, 'expression': f"'{single_rule.name}'", 'name':  f"{self.FIELD_PREFIX}_description"})
        if self.include_required_fields_only != 0:
            field_mapping.extend([{'type': field.type(), 'expression': f'"{field.name()}"', 'name': f'{field.name()}'} 
            for field in layer.fields()])

        layer = QgsProject.instance().addMapLayer(layer, False)
        layer_id = layer.id()
        feature_source = QgsProcessingFeatureSourceDefinition(
            layer.source(), 
            selectedFeaturesOnly=False if not selected_ids else True, 
            featureLimit=-1,
            geometryCheck=QgsFeatureRequest.GeometryAbortOnInvalid
        )
        output_dataset = single_rule.output_dataset_name
        output_dataset = join(self.temp_dir, f"{output_dataset}.fgb")
        sleep(1) # Avoid project crashing
        if not exists(output_dataset):
            layer = self._run_processing("refactorfields", INPUT=feature_source,
                                        FIELDS_MAPPING=field_mapping)
            layer = self._apply_transformation(layer, transformation, output_dataset)
        else:
            layer = None
        QgsProject.instance().removeMapLayer(layer_id)

        return layer

    def _apply_transformation(self, layer: QgsVectorLayer, transformation, 
                             output_dataset: str) -> QgsVectorLayer:
        """Apply geometry transformation to layer."""
        output_geometry = abs(transformation[0] - 2)
        layer = self._run_processing(
            "geometrybyexpression",
            INPUT=layer,
            OUTPUT_GEOMETRY=output_geometry,
            EXPRESSION=transformation[1]
        )
        layer = self._run_processing(
            "removenullgeometries",
            INPUT=layer,
            REMOVE_EMPTY=True,
            
        )
        return self._run_processing(
            "multiparttosingleparts", 
            INPUT=layer, 
            OUTPUT=output_dataset
        )
            
    def _get_polygon_centroids_expression(self):
        """Get polygon centroids expression based on user perference - visible polygon/whole polygon"""
        source_polygons = f"intersection(@geometry, geom_from_wkt('{self.extent.asWktPolygon()}'))" if self.cent_source == 1 else '@geometry'
        centroids = f"with_variable('source', {source_polygons}, if(intersects(centroid(@source), @source), centroid(@source),  point_on_surface(@source)))"
        return centroids
    
    def _add_label_expression_field(self, flat_rule: FlattenedRule, 
                                    fields: dict) -> dict:
        """Add label expression as a calculated field."""
        field_name = f"{self.FIELD_PREFIX}_label"
        expression = flat_rule.rule.settings().getLabelExpression().expression()
        
        exp =  f'"{expression}"' if not flat_rule.rule.settings().isExpression else expression
        fields.append({'type': 10, 'expression': exp, 'name': field_name})

        flat_rule.rule.settings().isExpression = False
        flat_rule.rule.settings().fieldName = field_name
        return fields

    def _get_geometry_transformation(self, flat_rule: FlattenedRule) -> Union[str, Tuple, None]:
        """Determine geometry transformation needed for rule."""
        rule_type = flat_rule.get_attribute("t")
        if rule_type == 1:  # Labeling
            transformation = self._get_labeling_transformation(flat_rule)
        else:  # Renderer
            transformation = self._get_renderer_transformation(flat_rule)
        # Clip output geometry to extent
        transformation[1] = f"with_variable('clip',intersection({transformation[1]}, geom_from_wkt('{self.extent.asWktPolygon()}')), if(not is_empty_or_null(@clip), @clip, NULL))"
        return tuple(transformation)
    
    def _get_labeling_transformation(self, flat_rule: FlattenedRule) -> Union[Tuple, str, None]:
        """Get geometry transformation for labeling rules."""
        settings = flat_rule.rule.settings()
        target_geom = flat_rule.get_attribute("g")
        transform_expr = '@geometry'

        if settings.geometryGeneratorEnabled:
            target_geom = settings.geometryGeneratorType
            transform_expr = settings.geometryGenerator
            settings.geometryGeneratorEnabled = False
            flat_rule.set_attribute("c", target_geom)

        elif target_geom == 2:  # Polygon to centroid
            flat_rule.set_attribute("c", 0)
            target_geom = 0
            transform_expr = self._get_polygon_centroids_expression()
        return [target_geom, transform_expr]

    def _get_renderer_transformation(self, flat_rule: FlattenedRule) -> Union[Tuple, str, None]:
        """Get geometry transformation for renderer rules."""
        symbol_layer = flat_rule.rule.symbol().symbolLayers()[0]
        target_geom = flat_rule.get_attribute("g")
        transform_expr = '@geometry'

        if symbol_layer.layerType() == "GeometryGenerator":
            target_geom = symbol_layer.subSymbol().type()
            transform_expr = symbol_layer.geometryExpression()
        else:
            target_geom = flat_rule.get_attribute("c")
            source_geom = flat_rule.get_attribute("g")

            if source_geom != target_geom:
                if target_geom == 0:
                    transform_expr = self._get_polygon_centroids_expression()
                elif target_geom == 1:
                    transform_expr = "boundary(@geometry)"
        return [target_geom, transform_expr]


    def _get_ddp(self, current_object, parent_obj, ddp, depth = 0, index=0):
        """Get data defined properties objects"""
        if hasattr(current_object, 'dataDefinedProperties'):
            ddp.append((current_object, parent_obj, depth, index))
        for attr in dir(current_object):
            try:
                if any(char in attr.lower() for char in ['_',  'node', 'class', 'clone', 'create', 'copy', 'paste', 'clear', 'remove']):
                    continue
                obj = getattr(current_object, attr)
                if attr != 'subSymbol' and len(attr) > 4 and attr[3].isupper():
                    continue
                if any(word in type(obj).__name__.lower() for word in ['qgis', 'enum']):
                    continue
                call_obj = obj()
                if isinstance(call_obj, type(current_object)):
                    continue
                if attr != 'symbolLayers' and type(call_obj).__module__  != 'qgis._core':
                    continue
                iterable_object = [call_obj] if not hasattr(call_obj, '__iter__') else call_obj
                for index, subobj in enumerate(iterable_object):
                    try:
                        if not isinstance(subobj, type(current_object)):
                            subdepth = depth + 1
                            subindex = index + 1
                            parent_obj = current_object
                            self._get_ddp(subobj, parent_obj, ddp, subdepth, subindex)
                    except (NameError, ValueError, AttributeError, TypeError):
                        break
            except (NameError, ValueError, AttributeError, TypeError):
                continue

    def _create_expression_fields(self, flat_rules: List[FlattenedRule]) -> dict:
        """Create calculated fields from data-driven properties."""
        fields = []
        
        for flat_rule in flat_rules:
            # Duplicate object to avoid script modify original instance
            rule_type = flat_rule.get_attribute("t")
            if rule_type == 0:
                clone = flat_rule.rule.symbol().clone()
                flat_rule.rule.setSymbol(clone)
                root_object = flat_rule.rule.symbol()  
            else:
                
                format_clone = QgsTextFormat(flat_rule.rule.settings().format())
                if flat_rule.rule.settings().format().background().markerSymbol():
                    symbol_clone = flat_rule.rule.settings().format().background().markerSymbol().clone()
                    format_clone.background().setMarkerSymbol(symbol_clone)
                flat_rule.rule.settings().setFormat(format_clone)
                root_object = flat_rule.rule.settings()

            # Extract rules objects which may contain data defined properties
            ddp_objects = []
            self._get_ddp(root_object, root_object, ddp_objects)
            for obj, parent, depth, index in ddp_objects:
                dd_props = obj.dataDefinedProperties()
                for prop_key in dd_props.propertyKeys():
                    prop = dd_props.property(prop_key)
                    if prop and prop.isActive():
                        attr = 's' if flat_rule.get_attribute("t") == 0 else 'f'
                        extra_val = f'{attr}{int(flat_rule.get_attribute(attr)):02d}'
                        prop_id = f'property_{prop_key}d{int(depth):02d}i{int(index):02d}{extra_val}'
                        field_name, expression = self._create_field_from_property(
                            prop_id, prop, flat_rule
                        )
                        
                        type_map = {
                            QgsPropertyDefinition.DataTypeString: QVariant.String,
                            QgsPropertyDefinition.DataTypeNumeric: QVariant.Double,
                            QgsPropertyDefinition.DataTypeBoolean: QVariant.Bool
                            }
                        def_obj = getattr(obj, 'propertyDefinitions') or getattr(parent, 'propertyDefinitions')
                        if def_obj:
                            prop_def = def_obj().get(prop_key)
                            if prop_def:
                                field_type = type_map.get(prop_def.dataType()) or 10
                                field_props = {'type':field_type, 'expression': expression, 'name': field_name} 
                                fields.append(field_props)
                                prop.setExpressionString(f'"{field_name}"')
        return fields

    def _create_field_from_property(self, prop_id: str, prop, 
                                    flat_rule: FlattenedRule) -> Tuple[str, str]:
        """Create field name and expression from data-driven property."""

        expression = prop.expressionString().replace(
            "@map_scale",
            str(ZoomLevels.zoom_to_scale(flat_rule.get_attribute("o")))
        )

        field_name = f"{self.FIELD_PREFIX}_{prop_id}"
        # expression = f"array_to_string(array({expression}))"
        return field_name, expression

    def _run_processing(self, algorithm: str, algorithm_type: str = "native", **params):
        """Execute QGIS processing algorithm."""
        if not params.get("OUTPUT"):
            params["OUTPUT"] = "TEMPORARY_OUTPUT"
        output = processing.run(f"{algorithm_type}:{algorithm}", params)["OUTPUT"]
        
        if isinstance(output, str):
            output = QgsVectorLayer(output)
        
        return output


class RuleFlattener:
    """Flattens QGIS rule-based styling with property inheritance."""

    RULE_TYPES = {0: "renderer", 1: "labeling"}

    def __init__(self, min_zoom: int, max_zoom: int):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.layer_tree_root = QgsProject.instance().layerTreeRoot()
        self.flattened_rules = []

    def flatten_all_rules(self) -> List[FlattenedRule]:
        """Extract and flatten all rules from visible vector layers."""
        layers = [layer.layer() for layer in self.layer_tree_root.findLayers() if layer.layer() and layer.layer().isValid()]

        for layer_idx, layer in enumerate(layers):
            if self._is_valid_layer(layer):
                self._process_layer_rules(layer.clone(), layer_idx)
        
        return self.flattened_rules

    def _is_valid_layer(self, layer) -> bool:
        """Check if layer is a visible vector layer."""
        is_vector = layer.type() == 0 and layer.geometryType() != 4
        layer_node = self.layer_tree_root.findLayer(layer.id())
        is_visible = layer_node.isVisible() if layer_node else False
        return is_vector and is_visible
    
    def _process_layer_rules(self, layer: QgsVectorLayer, layer_idx: int) -> None:
        """Process both renderer and labeling rules for a layer."""
        for rule_type in self.RULE_TYPES:
            rule_system = self._get_or_convert_rule_system(layer, rule_type)
            if rule_system:
                root_rule = self._prepare_root_rule(rule_system, layer)
                if root_rule:
                    self._flatten_rule_hierarchy(layer, layer_idx, root_rule, rule_type, 0, 0)

    def _get_or_convert_rule_system(self, layer: QgsVectorLayer, rule_type: int):
        """Get or convert layer styling to rule-based system."""
        if rule_type == 0:  # Renderer
            return self._convert_renderer_to_rules(layer)
        else:  # Labeling
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

    def _flatten_rule_hierarchy(self, layer: QgsVectorLayer, layer_idx: int, rule,
                                rule_type: int, rule_level: int, rule_idx: int) -> None:
        """Recursively flatten rule hierarchy with inheritance."""
        if rule.parent():
            if rule_type == 1:
                self._fix_labeling_rule_scale_range(rule)
            inherited_rule = self._inherit_rule_properties(rule, rule_type)
            if inherited_rule:
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

        for child_idx, child in enumerate(rule.children()):
            if child.active() or rule_type == 1:
                if child.filterExpression() == "ELSE":
                    self._convert_else_filter(child, rule)

                self._flatten_rule_hierarchy(layer, layer_idx, child, rule_type, 
                                            rule_level + 1, child_idx)
    def _fix_labeling_rule_scale_range(self, rule):
        """Copy labeling rule's settings visiblity scale to rule's visblity scales if they are not activated"""
        if rule.minimumScale() == 0 and rule.maximumScale() == 0:
            settings = rule.settings()
            if settings.scaleVisibility:
                rule.setMinimumScale(settings.minimumScale()) 
                rule.setMaximumScale(settings.maximumScale())
                settings.scaleVisibility = False

    def _set_rule_attributes(self, flat_rule: FlattenedRule, layer_idx: int, 
                            rule_type: int, rule_level: int, rule_idx: int) -> None:
        """Set rule attributes for identification and processing."""
        flat_rule.set_attribute("l", layer_idx)
        flat_rule.set_attribute("t", rule_type)
        flat_rule.set_attribute("d", rule_level)
        flat_rule.set_attribute("r", rule_idx)
        flat_rule.set_attribute("g", flat_rule.layer.geometryType())
        flat_rule.set_attribute("c", flat_rule.layer.geometryType())
        flat_rule.set_attribute("o", self._get_rule_zoom(flat_rule, min))
        flat_rule.set_attribute("i", self._get_rule_zoom(flat_rule, max))
        flat_rule.set_attribute("s" if rule_type == 0 else "f", 0)

    def _get_rule_zoom(self, flat_rule: FlattenedRule, comparator) -> int:
        """Extract rule zoom level from scale."""
        attr_name = f"{comparator.__name__}imumScale"
        rule_scale = getattr(flat_rule.rule, attr_name)()
        edge = "i" if comparator == max else "o"
        return int(ZoomLevels.scale_to_zoom(rule_scale, edge))

    def _is_within_zoom_range(self, flat_rule: FlattenedRule) -> bool:
        """Check if rule is within requested zoom range."""
        min_zoom = flat_rule.get_attribute("o")
        max_zoom = flat_rule.get_attribute("i")
        return self._ranges_overlap(min_zoom, max_zoom, self.min_zoom, self.max_zoom)

    def _clip_rules_to_zoom_range(self, flat_rules: List[FlattenedRule]) -> None:
        """Clip rule zoom range to general zoom range."""
        for flat_rule in flat_rules:
            if flat_rule.get_attribute("o") < self.min_zoom:
                flat_rule.set_attribute("o", self.min_zoom)
            if flat_rule.get_attribute("i") > self.max_zoom:
                flat_rule.set_attribute("i", self.max_zoom)

    def _split_rule(self, flat_rule: FlattenedRule, rule_type: int) -> List[FlattenedRule]:
        """Split rule based on type."""
        if rule_type == 0:  # Renderer
            return self._split_by_symbol_layers(flat_rule)
        else:  # Labeling
            return self._split_by_matching_renderers(flat_rule)

    def _convert_else_filter(self, else_rule, parent_rule) -> None:
        """Convert ELSE filter to explicit exclusion of sibling conditions."""
        sibling_filters = [
            sibling.filterExpression() 
            for sibling in parent_rule.children()
            if sibling.active() and sibling.filterExpression() not in ("ELSE", "")
        ]

        if sibling_filters:
            else_expression = f'NOT ({" OR ".join(f"({f})" for f in sibling_filters)})'
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

    def _inherit_scale_range(self, clone, rule, comparator) -> None:
        """Inherit scale limits using min/max comparator."""        
        attr_name = f"{comparator.__name__}imumScale"
        rule_scale = getattr(rule, attr_name)()
        if rule_scale == 0:
            opposite = min if comparator == max else max
            rule_scale = opposite(ZoomLevels.SCALES)
        parent_scale = getattr(rule.parent(), attr_name)()
        inherited_scale = comparator(rule_scale, parent_scale)
        setter_name = f"set{comparator.__name__.capitalize()}imumScale"
        getattr(clone, setter_name)(inherited_scale)

    def _inherit_filter_expression(self, clone, rule) -> None:
        """Inherit and combine filter expressions from parent hierarchy."""
        parent_filter = rule.parent().filterExpression()
        rule_filter = rule.filterExpression()

        if parent_filter and rule_filter:
            combined_filter = f"({parent_filter}) AND ({rule_filter})"
        else:
            combined_filter = parent_filter or rule_filter or ""

        child_filters = [
            f"({child.filterExpression()})"
            for child in rule.children()
            if child.filterExpression() and child.filterExpression() != "ELSE"
        ]

        if child_filters:
            children_expression = " OR ".join(child_filters)
            if combined_filter:
                final_filter = f"({combined_filter}) AND NOT ({children_expression})"
            else:
                final_filter = f"NOT ({children_expression})"
        else:
            final_filter = combined_filter

        clone.setFilterExpression(final_filter)

    def _inherit_symbol_layers(self, clone, rule) -> None:
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
            rule_clone = FlattenedRule(flat_rule.rule.clone(), flat_rule.layer, flat_rule.name)

            symbol_layer = symbol.symbolLayer(layer_idx)
            sub_symbol = symbol_layer.subSymbol()
            symbol_type = (sub_symbol.type() if symbol_layer.layerType() == "GeometryGenerator"
                          else symbol_layer.type())
            
            rule_clone.set_attribute("c", symbol_type)
            rule_clone.set_attribute("s", layer_idx)

            clone_symbol = rule_clone.rule.symbol()
            for remove_idx in reversed(range(symbol_layer_count)):
                if remove_idx != layer_idx:
                    clone_symbol.deleteSymbolLayer(remove_idx)

            split_rules.append(rule_clone)

        return split_rules

    def _split_by_matching_renderers(self, label_rule: FlattenedRule) -> List[FlattenedRule]:
        """Split label rule by matching renderer rules with overlapping scales."""
        matching_rules = []
        renderer_idx = 0

        for renderer_rule in self.flattened_rules:
            if (label_rule.layer.id() != renderer_rule.layer.id() or
                renderer_rule.get_attribute("t") != 0):
                continue
            
            if renderer_rule.output_dataset_name in [r.output_dataset_name for r in matching_rules]:
                continue

            matched_rule = self._match_label_to_renderer(label_rule, renderer_rule, renderer_idx)
            if matched_rule:
                matching_rules.append(matched_rule)
            renderer_idx += 1

        return matching_rules if matching_rules else [label_rule]

    def _match_label_to_renderer(self, label_rule: FlattenedRule, 
                                 renderer_rule: FlattenedRule, 
                                 renderer_idx: int) -> Optional[FlattenedRule]:
        """Create combined rule matching label to renderer with overlapping scales."""
        label_min = label_rule.get_attribute("o")
        label_max = label_rule.get_attribute("i")
        renderer_min = renderer_rule.get_attribute("o")
        renderer_max = renderer_rule.get_attribute("i")

        if not self._ranges_overlap(label_min, label_max, renderer_min, renderer_max):
            return None

        rule_clone = FlattenedRule(label_rule.rule.clone(), label_rule.layer, label_rule.name)
        clone_rule = rule_clone.rule

        label_filter = clone_rule.filterExpression()
        renderer_filter = renderer_rule.rule.filterExpression()

        if label_filter and renderer_filter:
            combined_filter = f"({renderer_filter}) AND ({label_filter})"
        else:
            combined_filter = renderer_filter or label_filter or ""

        clone_rule.setFilterExpression(combined_filter)

        if label_min < renderer_min:
            rule_clone.set_attribute("o", renderer_min)
        if label_max > renderer_max:
            rule_clone.set_attribute("i", renderer_max)

        rule_clone.set_attribute("f", renderer_idx)
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

        min_zoom = flat_rule.get_attribute("o")
        max_zoom = flat_rule.get_attribute("i")
        # min_zoom = max(self.min_zoom, min_zoom - 1)
        max_zoom = min(self.max_zoom, max_zoom + 1)
        relevant_zooms = list(range(min_zoom, max_zoom + 1))
        split_rules = []
        for zoom in relevant_zooms:
            rule_clone = self._create_scale_specific_rule(flat_rule, zoom)
            split_rules.append(rule_clone)

        return split_rules

    def _has_scale_dependencies(self, flat_rule: FlattenedRule) -> bool:
        """Check if rule has scale-dependent expressions."""
        expressions = []
        rule_type = flat_rule.get_attribute("t")

        if rule_type == 0:  # Renderer
            objects = [flat_rule.rule.symbol()] + flat_rule.rule.symbol().symbolLayers()
        else:  # Labeling
            objects = [flat_rule.rule.settings()]

        for obj in objects:
            dd_props = obj.dataDefinedProperties()
            for prop_key in dd_props.propertyKeys():
                prop = dd_props.property(prop_key)
                if prop:
                    expressions.append(prop.expressionString())

        expressions.append(flat_rule.rule.filterExpression())
        
        if rule_type == 1:
            label_exp = flat_rule.rule.settings().getLabelExpression().expression()
            expressions.append(label_exp)

        return "@map_scale" in ', '.join(expressions)

    def _create_scale_specific_rule(self, flat_rule: FlattenedRule, 
                                    zoom: int) -> FlattenedRule:
        """Create rule with scale-specific values."""
        rule_clone = FlattenedRule(flat_rule.rule.clone(), flat_rule.layer, flat_rule.name)
        scale = str(ZoomLevels.zoom_to_scale(zoom))

        filter_exp = flat_rule.rule.filterExpression()
        if "@map_scale" in filter_exp:
            scale_specific_filter = filter_exp.replace("@map_scale", scale)
            rule_clone.rule.setFilterExpression(scale_specific_filter)

        rule_type = flat_rule.get_attribute("t")
        if rule_type == 1:
            label_exp = flat_rule.rule.settings().getLabelExpression().expression()
            if "@map_scale" in label_exp:
                scale_specific_label = label_exp.replace("@map_scale", scale)
                rule_clone.rule.settings().fieldName = scale_specific_label

        rule_clone.set_attribute("o", zoom)
        rule_clone.set_attribute("i", zoom)

        return rule_clone


class QGIS2StyledTiles:
    """
    Main adapter class that orchestrates the conversion process from QGIS
    vector layer styling to vector tiles format.
    """

    def __init__(self, min_zoom: int = 0, max_zoom: int = 10, extent=None,
                 output_dir: str = None, include_required_fields_only=0, output_type: str = "xyz", cpu_percent: int = 100, output_content: int = 0,
                 cent_source: int = 0, feedback: QgsProcessingFeedback = None):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.extent = extent or iface.mapCanvas().extent()
        self.output_dir = output_dir or gettempdir()
        self.include_required_fields_only = include_required_fields_only
        self.output_type = output_type.lower()
        self.cpu_percent = cpu_percent
        self.output_content = output_content
        self.cent_source = cent_source
        self.feedback = feedback or QgsProcessingFeedback()

    def convert_project_to_vector_tiles(self) -> Optional[QgsVectorTileLayer]:
        """
        Convert current QGIS project to vector tiles format.

        Returns:
            QgsVectorTileLayer: The created vector tiles layer, or None if failed
        """
        try:
            temp_dir = self._create_temp_directory()
            self._log(". Starting conversion process...")
            start_time = perf_counter()

            # Step 1: Flatten all rules
            self._log(". Flattening rules...")
            rules = self._flatten_rules()
            if not rules:
                self._log(". No visible vector layers found in project.")
                return None
            
            flatten_time = perf_counter()
            self._log(f". Successfully extracted {len(rules)} rules "
                     f"({self._elapsed_minutes(start_time, flatten_time)} minutes).")
            output = tiles_uri = layers = None
            if self.output_content == 0:
                # Step 2: Export rules to datasets
                self._log(". Exporting rules to layers...")
                layers, rules = self._export_rules(rules)
                export_time = perf_counter()
                self._log(f". Successfully exported {len(layers)} layers "
                        f"({self._elapsed_minutes(flatten_time, export_time)} minutes).")
                
                # Step 3: Generate and style tiles
                if self._has_features(layers):
                    self._log(". Generating tiles...")
                    tiles_uri = self._generate_tiles(layers, temp_dir)
                    tiles_time = perf_counter()
                    self._log(f". Successfully generated tiles "
                    f"({self._elapsed_minutes(export_time, tiles_time)} minutes).")
            
            self._log(". Loading and styling tiles...")
            self._sytle_tiles(rules, temp_dir, tiles_uri)
            
            total_time = perf_counter()
            self._log(f". Process completed successfully "
                     f"({self._elapsed_minutes(start_time, total_time)} minutes).")
            
            return output

        except Exception as e:
            self._log(f". Error during conversion: {e}")

    def _create_temp_directory(self) -> str:
        """Create temporary directory for processing."""
        temp_dir = join(self.output_dir, datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f"))
        makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def _flatten_rules(self) -> List[FlattenedRule]:
        """Flatten all rules from project layers."""
        flattener = RuleFlattener(self.min_zoom, self.max_zoom)
        return flattener.flatten_all_rules()

    def _export_rules(self, rules: List[FlattenedRule]) -> List[QgsVectorLayer]:
        """Export rules to vector layers."""
        exporter = RulesExporter(
            rules, self.extent, self.include_required_fields_only, self.max_zoom, self.cent_source, self.feedback)
        return exporter.export()

    def _has_features(self, layers: List[QgsVectorLayer]) -> bool:
        """Check if any layer has features."""
        return any(layer.featureCount() > 0 for layer in layers)

    def _generate_tiles(self, layers: List[QgsVectorLayer], temp_dir: str) -> str:
        """Generate vector tiles."""
        generator = GDALTilesGenerator(
            layers, temp_dir, self.output_type, self.extent, self.cpu_percent, self.feedback
        )
        tiles_uri = generator.generate()
        return tiles_uri

    def _sytle_tiles(self, rules, temp_dir, tiles_uri):
        """Style tiles."""
        styler = TilesStyler(rules, temp_dir, tiles_uri)
        styler.apply_styling()

    def _log(self, message: str) -> None:
        """Log message to feedback or console."""
        if __name__ != "__console__":
            self.feedback.pushInfo(message)
        else:
            print(message)

    @staticmethod
    def _elapsed_minutes(start: float, end: float) -> str:
        """Calculate elapsed time in minutes."""
        return f"{round((end - start) / 60, 2)}"


# Main execution for QGIS console
if __name__ == "__console__":
    from qgis.utils import iface
    adapter = QGIS2StyledTiles()
    adapter.convert_project_to_vector_tiles()