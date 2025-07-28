"""
QGIS Vector Tiles Adapter (QVTA)

Converts QGIS vector layer styling to vector tiles format by:
1. Flattening nested rule-based renderers/labeling with property inheritance
2. Splitting rules by symbol layers and matching label rules to renderer rules
3. Exporting each rule as a separate dataset with geometry transformations
4. Generating MBTiles using GDAL MVT driver with multi-threading support
5. Loading and styling the tiles in QGIS with appropriate symbology

The adapter handles complex styling scenarios including geometry generators,
scale-dependent styling, nested filters, and ensures proper tile-based rendering.
"""

from os.path import join
from os import cpu_count
from tempfile import mkdtemp, TemporaryDirectory
from osgeo import gdal, ogr, osr
from dataclasses import dataclass
from typing import Union, List, Optional, Tuple
import processing
from qgis.core import (
    QgsProject,
    QgsRuleBasedRenderer,
    QgsRuleBasedLabeling,
    QgsPalLayerSettings,
    QgsVectorLayer,
    QgsSymbol,
    QgsRenderContext,
    QgsVectorTileLayer,
    QgsWkbTypes,
    QgsVectorTileBasicRenderer,
    QgsVectorTileBasicRendererStyle,
    QgsVectorTileBasicLabeling,
    QgsVectorTileBasicLabelingStyle,
)


class ZoomLevels:
    """Manages zoom level scales and conversions for web mapping standards."""

    SCALES = [
        591657528,
        295828764,
        147914382,
        73957191,
        36978595,
        18489298,
        9244649,
        4622324,
        2311162,
        1155581,
        577791,
        288895,
        144448,
        72224,
        36112,
        18056,
        9028,
        4514,
        2257,
        1128,
        0.433333333333333,
        0.2375,
        0.139583333333333,
        0.090972222222222,
    ]

    @classmethod
    def snap_scale(cls, scale: float, snap_up: bool = True) -> float:
        """Snap scale to nearest zoom level."""
        if scale <= 0:
            return cls.SCALES[0] if snap_up else cls.SCALES[-1]

        for i, level in enumerate(cls.SCALES):
            if scale >= level:
                if i == 0 or not snap_up:
                    return level
                return level if not snap_up else cls.SCALES[i - 1]
        return cls.SCALES[-1]

    @classmethod
    def scale_to_zoom(cls, scale: float) -> str:
        """Convert scale to zero-padded zoom level string."""
        zoom = cls.SCALES.index(scale)
        return f"{zoom:02d}"

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

    def get_attribute(self, char: str) -> Optional[int]:
        """Extract rule attribute from description by character prefix."""
        desc = self.rule.description()
        start = desc.find(char) + 1
        if start == 0:
            return None
        return int(desc[start : start + 2])

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


class VectorTileStyler:
    """Applies styling to vector tile layers from flattened rules."""

    def __init__(self, flattened_rules: List[FlattenedRule], tiles_path: str):
        self.flattened_rules = flattened_rules
        self.tiles_layer = self._create_tiles_layer(tiles_path)
        self.renderer_styles = []
        self.labeling_styles = []

    def apply_styling(self) -> QgsVectorTileLayer:
        """Apply styles to vector tiles layer and add to project."""
        for rule in self.flattened_rules:
            self._create_style_from_rule(rule)
        self._apply_styles_to_layer()
        return self.tiles_layer

    def _create_tiles_layer(self, tiles_path: str) -> QgsVectorTileLayer:
        """Create and add vector tiles layer to project."""
        layer = QgsVectorTileLayer(f"type=mbtiles&url={tiles_path}", "Vector Tiles")
        return QgsProject.instance().addMapLayer(layer)

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

    def _setup_renderer_style(
        self, style: QgsVectorTileBasicRendererStyle, flat_rule: FlattenedRule
    ) -> None:
        """Configure renderer style properties."""
        self._setup_base_style_properties(style, flat_rule)

        symbol = flat_rule.rule.symbol()
        # Handle subsymbol for geometry changes
        sub_symbol = symbol.symbolLayers()[-1].subSymbol()
        if sub_symbol:
            self._copy_data_driven_properties(symbol, sub_symbol)
            symbol = sub_symbol
        style.setSymbol(symbol.clone())

    def _setup_labeling_style(
        self, style: QgsVectorTileBasicLabelingStyle, flat_rule: FlattenedRule
    ) -> None:
        """Configure labeling style properties."""
        self._setup_base_style_properties(style, flat_rule)
        settings = QgsPalLayerSettings(flat_rule.rule.settings())
        style.setLabelSettings(settings)

    def _setup_base_style_properties(self, style, flat_rule: FlattenedRule) -> None:
        """Setup common style properties."""
        style.setEnabled(True)
        style.setLayerName(flat_rule.rule.description())
        style.setStyleName(flat_rule.rule.description())
        style.setMinZoomLevel(flat_rule.get_attribute("o"))
        style.setMaxZoomLevel(flat_rule.get_attribute("i"))

        # Set geometry type
        geom_code = flat_rule.get_attribute("c")
        geom_types = {
            0: QgsWkbTypes.PointGeometry,
            1: QgsWkbTypes.LineGeometry,
            2: QgsWkbTypes.PolygonGeometry,
        }
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


class VectorTileGenerator:
    """Generates MBTiles from GeoPackage using GDAL MVT driver."""

    def __init__(self, output_dir: str, cpu_percent: int, gpkg_path: str):
        self.output_dir = output_dir
        self.cpu_percent = cpu_percent
        self.gpkg_path = gpkg_path

    def generate(self) -> str:
        """Generate MBTiles file with multi-threading support."""
        self._configure_gdal_threading()

        output_path = join(self.output_dir, "tiles.mbtiles")
        web_mercator = self._create_web_mercator_srs()

        # Create MVT dataset
        driver = gdal.GetDriverByName("MVT")
        creation_options = ["EXTENT=16384", "MINZOOM=0", "MAXZOOM=15","SIMPLIFICATION=100", "SIMPLIFICATION_MAX_ZOOM=0"]
        dataset = driver.Create(
            output_path, 0, 0, 0, gdal.GDT_Unknown, options=creation_options
        )

        self._process_gpkg_layers(dataset, web_mercator)
        self._set_dataset_metadata(dataset)

        dataset = None  # Close dataset
        return output_path

    def _configure_gdal_threading(self) -> None:
        """Configure GDAL threading based on CPU percentage."""
        cpu_threads = str(max(1, int(cpu_count() * self.cpu_percent / 100)))
        gdal.SetConfigOption("GDAL_NUM_THREADS", cpu_threads)

    def _create_web_mercator_srs(self) -> osr.SpatialReference:
        """Create Web Mercator spatial reference system."""
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        return srs

    def _process_gpkg_layers(self, dataset, web_mercator: osr.SpatialReference) -> None:
        """Process all layers from GeoPackage into MVT dataset."""
        gpkg = ogr.Open(self.gpkg_path)

        for i in range(gpkg.GetLayerCount()):
            source_layer = gpkg.GetLayer(i)
            layer_name = source_layer.GetName()

            # Extract zoom levels from layer name
            min_zoom = int(layer_name.split("o")[1][:2])
            max_zoom = int(layer_name.split("i")[1][:2])

            self._create_mvt_layer(
                dataset, source_layer, layer_name, web_mercator, min_zoom, max_zoom
            )

    def _create_mvt_layer(
        self,
        dataset,
        source_layer,
        layer_name: str,
        web_mercator: osr.SpatialReference,
        min_zoom: int,
        max_zoom: int,
    ) -> None:
        """Create MVT layer from source layer."""
        # Create layer in MVT dataset
        mvt_layer = dataset.CreateLayer(
            layer_name, srs=web_mercator, geom_type=ogr.wkbUnknown, options=["MINZOOM=0", "MAXZOOM=7"]
        )

        # Set layer metadata
        mvt_layer.SetMetadataItem("MINZOOM", str(min_zoom))
        mvt_layer.SetMetadataItem("MAXZOOM", str(max_zoom))
        mvt_layer.SetMetadataItem("SIMPLIFICATION", "1.0")
        mvt_layer.SetMetadataItem("SIMPLIFICATION_MAX_ZOOM", "23")

        # Setup coordinate transformation
        transform = self._create_coordinate_transform(source_layer, web_mercator)

        # Copy field definitions
        self._copy_field_definitions(source_layer, mvt_layer)

        # Copy features with transformation
        self._copy_features(source_layer, mvt_layer, transform)

    def _create_coordinate_transform(
        self, source_layer, web_mercator: osr.SpatialReference
    ) -> Optional[osr.CoordinateTransformation]:
        """Create coordinate transformation if needed."""
        source_srs = source_layer.GetSpatialRef()
        if source_srs and not source_srs.IsSame(web_mercator):
            return osr.CoordinateTransformation(source_srs, web_mercator)
        return None

    def _copy_field_definitions(self, source_layer, target_layer) -> None:
        """Copy field definitions from source to target layer."""
        source_defn = source_layer.GetLayerDefn()
        for i in range(source_defn.GetFieldCount()):
            field_defn = source_defn.GetFieldDefn(i)
            target_layer.CreateField(field_defn)

    def _copy_features(self, source_layer, target_layer, transform) -> None:
        """Copy features from source to target with coordinate transformation."""
        source_defn = source_layer.GetLayerDefn()

        for source_feature in source_layer:
            target_feature = ogr.Feature(target_layer.GetLayerDefn())

            # Transform and set geometry
            geom = source_feature.GetGeometryRef()
            if geom:
                geom_copy = geom.Clone()
                if transform:
                    geom_copy.Transform(transform)
                target_feature.SetGeometry(geom_copy)

            # Copy attributes
            for i in range(source_defn.GetFieldCount()):
                field_name = source_defn.GetFieldDefn(i).GetName()
                if target_feature.GetFieldIndex(field_name) >= 0:
                    target_feature.SetField(field_name, source_feature.GetField(i))

            target_layer.CreateFeature(target_feature)

    def _set_dataset_metadata(self, dataset) -> None:
        """Set dataset metadata for MBTiles."""
        metadata = {
            "name": "Generated Tiles",
            "description": "MBTiles",
            "version": "1.0.0",
            "format": "pbf",
            "type": "overlay",
        }
        for key, value in metadata.items():
            dataset.SetMetadataItem(key, value)


class RuleDatasetExporter:
    """Exports flattened rules as datasets with geometry transformations."""

    FIELD_PREFIX = "qvta"
    GEOMETRY_ATTRIBUTES = {
        "$area": "area_meters",
        "area(@geometry)": "area_degrees",
        "$length": "length_meters",
        "length(@geometry)": "length_degrees",
    }

    def __init__(
        self,
        flattened_rules: List[FlattenedRule],
        extent,
        output_dir: str,
        include_all_fields: bool,
    ):
        self.flattened_rules = flattened_rules
        self.extent = extent
        self.output_dir = output_dir
        self.include_all_fields = include_all_fields
        self.processed_layers = []

    def export_all_rules(self) -> str:
        """Export all rules and package into single GeoPackage."""
        for rule in self.flattened_rules:
            self._export_single_rule(rule)
        return self._package_layers()

    def _export_single_rule(self, flat_rule: FlattenedRule) -> None:
        """Export single rule as a layer with transformations."""
        layer = self._prepare_base_layer(flat_rule)

        # Apply rule filter
        filter_expr = flat_rule.rule.filterExpression()
        if filter_expr:
            layer = self._run_processing(
                "extractbyexpression", INPUT=layer, EXPRESSION=filter_expr
            )

        # Add geometry attributes for data-driven properties
        layer = self._add_geometry_attributes(layer, flat_rule)

        # Keep only required fields
        if not self.include_all_fields:
            required_fields = list(self._get_required_fields(flat_rule))
            required_fields.append(f"{self.FIELD_PREFIX}_fid")
            layer = self._run_processing(
                "retainfields", INPUT=layer, FIELDS=required_fields
            )

        # Transform geometry if needed
        layer = self._transform_geometry_if_needed(layer, flat_rule)

        layer.setName(flat_rule.rule.description())
        self.processed_layers.append(layer)

    def _prepare_base_layer(self, flat_rule: FlattenedRule) -> QgsVectorLayer:
        """Prepare base layer with extent clipping and geometry fixes."""
        layer = flat_rule.layer
        extent = self.extent or layer.extent()

        # Add unique ID field
        with_id = self._run_processing(
            "fieldcalculator",
            INPUT=layer,
            FIELD_NAME=f"{self.FIELD_PREFIX}_fid",
            FORMULA=f"'{layer.name()}_' || @id",
            FIELD_TYPE=2,
        )

        # Extract by extent and fix geometries
        extracted = self._run_processing(
            "extractbyextent", INPUT=with_id, EXTENT=extent
        )
        fixed_network = self._run_processing("fixgeometries", INPUT=extracted, METHOD=0)
        fixed_structure = self._run_processing(
            "fixgeometries", INPUT=fixed_network, METHOD=1
        )

        return fixed_structure

    def _add_geometry_attributes(
        self, layer: QgsVectorLayer, flat_rule: FlattenedRule
    ) -> QgsVectorLayer:
        """Add geometry attributes needed for data-driven properties."""
        for expression, field_suffix in self.GEOMETRY_ATTRIBUTES.items():
            field_name = f"{self.FIELD_PREFIX}_{field_suffix}"

            # Check if this attribute is used in data-driven properties
            if self._get_data_driven_properties(
                flat_rule, expression, f'"{field_name}"'
            ):
                layer = self._run_processing(
                    "fieldcalculator",
                    INPUT=layer,
                    FIELD_NAME=field_name,
                    FORMULA=expression,
                    FIELD_TYPE=0,
                )
        return layer

    def _transform_geometry_if_needed(
        self, layer: QgsVectorLayer, flat_rule: FlattenedRule
    ) -> QgsVectorLayer:
        """Transform geometry based on rule requirements."""
        target_geom, transform_expr = self._get_geometry_transformation(flat_rule)

        if transform_expr:
            # Convert geometry type code (2 -> 0 for polygon to point)
            geom_type = abs(target_geom - 2)
            layer = self._run_processing(
                "geometrybyexpression",
                INPUT=layer,
                OUTPUT_GEOMETRY=geom_type,
                EXPRESSION=transform_expr,
            )

        return layer

    def _get_geometry_transformation(
        self, flat_rule: FlattenedRule
    ) -> Tuple[Optional[int], Optional[str]]:
        """Determine geometry transformation needed for rule."""
        rule_type = flat_rule.get_attribute("t")

        if rule_type == 1:  # Labeling
            return self._get_labeling_geometry_transform(flat_rule)
        else:  # Renderer
            return self._get_renderer_geometry_transform(flat_rule)

    def _get_labeling_geometry_transform(
        self, flat_rule: FlattenedRule
    ) -> Tuple[Optional[int], Optional[str]]:
        """Get geometry transformation for labeling rules."""
        settings = flat_rule.rule.settings()

        # Geometry generator labeling
        if settings.geometryGeneratorEnabled:
            target_geom = settings.geometryGeneratorType
            transform_expr = settings.geometryGenerator
            settings.geometryGeneratorEnabled = False
            flat_rule.set_attribute("c", target_geom)
            return target_geom, transform_expr

        # Polygon labels need centroid for tile-based rendering
        if flat_rule.get_attribute("g") == 2:  # Polygon source
            flat_rule.set_attribute("c", 0)  # Point target
            return 0, "centroid(@geometry)"

        return None, None

    def _get_renderer_geometry_transform(
        self, flat_rule: FlattenedRule
    ) -> Tuple[Optional[int], Optional[str]]:
        """Get geometry transformation for renderer rules."""
        symbol_layer = flat_rule.rule.symbol().symbolLayers()[0]

        # Geometry generator symbols
        if symbol_layer.layerType() == "GeometryGenerator":
            target_geom = symbol_layer.subSymbol().type()
            transform_expr = symbol_layer.geometryExpression()
            return target_geom, transform_expr

        # Different geometry types (e.g., polygon outline -> line)
        source_geom = flat_rule.get_attribute("g")
        target_geom = flat_rule.get_attribute("c")

        if source_geom != target_geom:
            if target_geom == 0:  # Point
                return target_geom, "centroid(@geometry)"
            elif target_geom == 1:  # Line
                return target_geom, "boundary(@geometry)"

        return None, None

    def _get_required_fields(self, flat_rule: FlattenedRule) -> set:
        """Get fields required by rule for rendering/labeling."""
        rule_type = flat_rule.get_attribute("t")

        if rule_type == 0:  # Renderer
            return flat_rule.rule.symbol().usedAttributes(QgsRenderContext())
        else:  # Labeling
            return flat_rule.rule.settings().referencedFields(QgsRenderContext())

    def _get_data_driven_properties(
        self, flat_rule: FlattenedRule, old_attr: str, new_attr: str = None
    ) -> List:
        """Get and optionally update data-driven properties containing attribute."""
        rule_type = flat_rule.get_attribute("t")

        if rule_type == 0:  # Renderer
            objects = [flat_rule.rule.symbol()] + flat_rule.rule.symbol().symbolLayers()
        else:  # Labeling
            objects = [flat_rule.rule.settings()]

        found_properties = []
        for obj in objects:
            dd_props = obj.dataDefinedProperties()
            for prop_key in obj.propertyDefinitions():
                prop = dd_props.property(prop_key)
                if prop and old_attr in prop.expressionString():
                    found_properties.append(prop)
                    if new_attr:
                        new_expr = prop.expressionString().replace(old_attr, new_attr)
                        prop.setExpressionString(new_expr)

        return found_properties

    def _package_layers(self) -> str:
        """Package all processed layers into single GeoPackage."""
        gpkg_path = join(self.output_dir, "rules.gpkg")
        self._run_processing(
            "package", LAYERS=self.processed_layers, SAVE_STYLES=False, OUTPUT=gpkg_path
        )
        return gpkg_path

    def _run_processing(self, algorithm: str, algorithm_type: str = "native", **params):
        """Execute QGIS processing algorithm."""
        if not params.get("OUTPUT"):
            params["OUTPUT"] = "TEMPORARY_OUTPUT"
        return processing.run(f"{algorithm_type}:{algorithm}", params)["OUTPUT"]


class RuleFlattener:
    """Flattens QGIS rule-based styling with property inheritance."""

    RULE_TYPES = {0: "renderer", 1: "labeling"}

    def __init__(self, min_zoom: int, max_zoom: int):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.project = QgsProject.instance()
        self.layer_tree_root = self.project.layerTreeRoot()
        self.flattened_rules = []

    def flatten_all_rules(self) -> List[FlattenedRule]:
        """Extract and flatten all rules from visible vector layers."""
        layers = self.project.mapLayers().values()

        for layer_idx, layer in enumerate(layers):
            if self._is_valid_layer(layer):
                self._process_layer_rules(layer.clone(), layer_idx)

        return self.flattened_rules

    def _is_valid_layer(self, layer) -> bool:
        """Check if layer is a visible vector layer."""
        is_vector = layer.type() == 0 and layer.geometryType() != 4
        is_visible = self.layer_tree_root.findLayer(layer.id()).isVisible()
        return is_vector and is_visible

    def _process_layer_rules(self, layer: QgsVectorLayer, layer_idx: int) -> None:
        """Process both renderer and labeling rules for a layer."""
        for rule_type in self.RULE_TYPES:
            rule_system = self._get_or_convert_rule_system(layer, rule_type)
            if rule_system:
                root_rule = self._prepare_root_rule(rule_system, layer)
                if root_rule:
                    self._flatten_rule_hierarchy(
                        layer, layer_idx, root_rule, rule_type, 0, 0
                    )

    def _get_or_convert_rule_system(self, layer: QgsVectorLayer, rule_type: int):
        """Get or convert layer styling to rule-based system."""
        if rule_type == 0:  # Renderer
            system = layer.renderer()
            if isinstance(system, QgsRuleBasedRenderer):
                return system
            return QgsRuleBasedRenderer.convertFromRenderer(system) if system else None
        else:  # Labeling
            system = layer.labeling()
            if not system or not layer.labelsEnabled():
                return None

            if isinstance(system, QgsRuleBasedLabeling):
                return system

            # Convert to rule-based labeling
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

    def _flatten_rule_hierarchy(
        self,
        layer: QgsVectorLayer,
        layer_idx: int,
        rule,
        rule_type: int,
        rule_level: int,
        rule_idx: int,
    ) -> None:
        """Recursively flatten rule hierarchy with inheritance."""
        # Process current rule (skip root)
        if rule.parent():
            inherited_rule = self._inherit_rule_properties(rule, rule_type)
            if inherited_rule:
                flat_rule = FlattenedRule(inherited_rule, layer)
                self._set_rule_attributes(
                    flat_rule, layer_idx, rule_type, rule_level, rule_idx
                )

                # Split rule by different criteria
                if rule_type == 0:  # Renderer
                    split_rules = self._split_by_symbol_layers(flat_rule)
                else:  # Labeling
                    split_rules = self._split_by_matching_renderers(flat_rule)

                # Further split by scale dependencies
                for split_rule in split_rules:
                    self.flattened_rules.extend(
                        self._split_by_scale_expressions(split_rule)
                    )

        # Process children recursively
        for child_idx, child in enumerate(rule.children()):
            if child.active():
                # Convert ELSE filters to absolute expressions
                if child.filterExpression() == "ELSE":
                    self._convert_else_filter(child, rule)

                self._flatten_rule_hierarchy(
                    layer, layer_idx, child, rule_type, rule_level + 1, child_idx
                )

    def _set_rule_attributes(
        self,
        flat_rule: FlattenedRule,
        layer_idx: int,
        rule_type: int,
        rule_level: int,
        rule_idx: int,
    ) -> None:
        """Set rule attributes for identification and processing."""
        flat_rule.set_attribute("l", layer_idx)  # Layer index
        flat_rule.set_attribute("t", rule_type)  # Type (0=renderer, 1=labeling)
        flat_rule.set_attribute("d", rule_level)  # Depth in hierarchy
        flat_rule.set_attribute("r", rule_idx)  # Rule index at level
        flat_rule.set_attribute("g", flat_rule.layer.geometryType())  # Source geometry
        flat_rule.set_attribute("c", flat_rule.layer.geometryType())  # Target geometry
        flat_rule.set_attribute(
            "o", ZoomLevels.scale_to_zoom(flat_rule.rule.minimumScale())
        )
        flat_rule.set_attribute(
            "i", ZoomLevels.scale_to_zoom(flat_rule.rule.maximumScale())
        )
        flat_rule.set_attribute(
            "s" if rule_type == 0 else "f", 0
        )  # Symbol/feature index

    def _convert_else_filter(self, else_rule, parent_rule) -> None:
        """Convert ELSE filter to explicit exclusion of sibling conditions."""
        sibling_filters = []
        for sibling in parent_rule.children():
            if sibling.active() and sibling.filterExpression() not in ("ELSE", ""):
                sibling_filters.append(sibling.filterExpression())

        if sibling_filters:
            else_expression = f'NOT ({" OR ".join(f"({f})" for f in sibling_filters)})'
            else_rule.setFilterExpression(else_expression)

    def _inherit_rule_properties(self, rule, rule_type: int):
        """Inherit all properties from parent hierarchy."""
        clone = rule.clone()

        # Inherit scale ranges
        self._inherit_scale_range(clone, rule, min)
        self._inherit_scale_range(clone, rule, max)

        # Skip if outside zoom range
        if self._is_outside_zoom_range(clone):
            return None

        self._inherit_filter_expression(clone, rule)

        if rule_type == 0:  # Renderer
            self._inherit_symbol_layers(clone, rule)

        return clone

    def _inherit_scale_range(self, clone, rule, comparator) -> None:
        """Inherit scale limits using min/max comparator."""
        attr_name = f"{comparator.__name__}imumScale"
        snap_up = comparator.__name__ == "min"

        # Get scales with snapping
        rule_scale = ZoomLevels.snap_scale(getattr(rule, attr_name)(), snap_up)
        parent_scale = ZoomLevels.snap_scale(
            getattr(rule.parent(), attr_name)(), snap_up
        )
        inherited_scale = comparator(rule_scale, parent_scale)

        # Set inherited scale
        setter_name = f"set{comparator.__name__.capitalize()}imumScale"
        getattr(clone, setter_name)(inherited_scale)

    def _is_outside_zoom_range(self, clone) -> bool:
        """Check if rule is outside the specified zoom range."""
        return int(clone.minimumScale()) == int(clone.maximumScale()) == 0

    def _inherit_filter_expression(self, clone, rule) -> None:
        """Inherit and combine filter expressions from parent hierarchy."""
        parent_filter = rule.parent().filterExpression()
        rule_filter = rule.filterExpression()

        # Combine parent and current filters
        if parent_filter and rule_filter:
            combined_filter = f"({parent_filter}) AND ({rule_filter})"
        else:
            combined_filter = parent_filter or rule_filter or ""

        # Exclude children filters to avoid double-filtering
        child_filters = []
        for child in rule.children():
            child_filter = child.filterExpression()
            if child_filter and child_filter != "ELSE":
                child_filters.append(f"({child_filter})")

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

        for layer_idx in range(symbol_layer_count):
            rule_clone = FlattenedRule(flat_rule.rule.clone(), flat_rule.layer)

            # Determine target geometry type
            sub_symbol = symbol.symbolLayer(layer_idx).subSymbol()
            symbol_type = sub_symbol.type() if sub_symbol else symbol.type()
            rule_clone.set_attribute("c", symbol_type)
            rule_clone.set_attribute("s", layer_idx)

            # Keep only the current symbol layer
            clone_symbol = rule_clone.rule.symbol()
            for remove_idx in reversed(range(symbol_layer_count)):
                if remove_idx != layer_idx:
                    clone_symbol.deleteSymbolLayer(remove_idx)

            split_rules.append(rule_clone)

        return split_rules

    def _split_by_matching_renderers(
        self, label_rule: FlattenedRule
    ) -> List[FlattenedRule]:
        """Split label rule by matching renderer rules with overlapping scales."""
        matching_rules = []
        renderer_idx = 0

        for renderer_rule in self.flattened_rules:
            # Skip if different layer or not a renderer rule
            if (
                label_rule.layer.id() != renderer_rule.layer.id()
                or renderer_rule.get_attribute("t") != 0
            ):
                continue

            matched_rule = self._match_label_to_renderer(
                label_rule, renderer_rule, renderer_idx
            )
            if matched_rule:
                matching_rules.append(matched_rule)
            renderer_idx += 1

        return matching_rules if matching_rules else [label_rule]

    def _match_label_to_renderer(
        self, label_rule: FlattenedRule, renderer_rule: FlattenedRule, renderer_idx: int
    ) -> Optional[FlattenedRule]:
        """Create combined rule matching label to renderer with overlapping scales."""
        label_min = label_rule.rule.minimumScale()
        label_max = label_rule.rule.maximumScale()
        renderer_min = renderer_rule.rule.minimumScale()
        renderer_max = renderer_rule.rule.maximumScale()

        # Check for scale overlap
        if label_min <= renderer_min or label_max >= renderer_max:
            rule_clone = FlattenedRule(label_rule.rule.clone(), label_rule.layer)
            clone_rule = rule_clone.rule

            # Combine filters
            label_filter = clone_rule.filterExpression()
            renderer_filter = renderer_rule.rule.filterExpression()

            if label_filter and renderer_filter:
                combined_filter = f"({renderer_filter}) AND ({label_filter})"
            else:
                combined_filter = renderer_filter or label_filter or ""

            clone_rule.setFilterExpression(combined_filter)

            # Adjust scale range to renderer's range
            if label_min > renderer_min:
                clone_rule.setMinimumScale(renderer_min)
                rule_clone.set_attribute("o", ZoomLevels.scale_to_zoom(renderer_min))
            if label_max < renderer_max:
                clone_rule.setMaximumScale(renderer_max)
                rule_clone.set_attribute("i", ZoomLevels.scale_to_zoom(renderer_max))

            rule_clone.set_attribute("f", renderer_idx)
            return rule_clone

        return None

    def _split_by_scale_expressions(
        self, flat_rule: FlattenedRule
    ) -> List[FlattenedRule]:
        """Split rule by zoom levels if contains scale-dependent expressions."""
        filter_expr = flat_rule.rule.filterExpression()
        if "@map_scale" not in filter_expr:
            return [flat_rule]

        # Get scale range and relevant zoom levels
        min_scale = flat_rule.rule.minimumScale()
        max_scale = flat_rule.rule.maximumScale()
        relevant_scales = [
            scale for scale in ZoomLevels.SCALES if max_scale <= scale <= min_scale
        ]

        split_rules = []
        for i, scale in enumerate(relevant_scales):
            rule_clone = FlattenedRule(flat_rule.rule.clone(), flat_rule.layer)

            # Set scale range for this zoom level
            rule_clone.rule.setMinimumScale(scale)
            next_scale = (
                relevant_scales[i + 1] if i + 1 < len(relevant_scales) else max_scale
            )
            rule_clone.rule.setMaximumScale(next_scale)

            # Replace @map_scale with actual scale value
            scale_specific_filter = filter_expr.replace("@map_scale", str(scale))
            rule_clone.rule.setFilterExpression(scale_specific_filter)

            # Update zoom attributes
            rule_clone.set_attribute("o", ZoomLevels.scale_to_zoom(scale))
            rule_clone.set_attribute("i", ZoomLevels.scale_to_zoom(next_scale))

            split_rules.append(rule_clone)

        return split_rules


class QGISVectorTilesAdapter:
    """
    Main adapter class that orchestrates the conversion process from QGIS
    vector layer styling to vector tiles format.
    """

    def __init__(
        self,
        min_zoom: int = 0,
        max_zoom: int = 23,
        extent=None,
        output_dir: TemporaryDirectory = None,
        cpu_percent: int = 70,
        include_all_fields: bool = False,
    ):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.extent = extent or iface.mapCanvas().extent()
        self.output_dir = output_dir or TemporaryDirectory()
        self.cpu_percent = cpu_percent
        self.include_all_fields = include_all_fields

    def convert_project_to_vector_tiles(self) -> Optional[QgsVectorTileLayer]:
        """
        Convert current QGIS project to vector tiles format.

        Returns:
            QgsVectorTileLayer: The created vector tiles layer, or None if failed
        """
        try:
            temp_dir = mkdtemp(dir=self.output_dir.name)
            print("Starting conversion process...")

            # Step 1: Flatten all rules
            print("Extracting and flattening rules...")
            flattener = RuleFlattener(self.min_zoom, self.max_zoom)
            flattened_rules = flattener.flatten_all_rules()

            if not flattened_rules:
                print("No visible vector layers found in project")
                return None

            print(f"Successfully extracted {len(flattened_rules)} rules")

            # Step 2: Export rules as datasets
            print("Exporting rules to datasets...")
            exporter = RuleDatasetExporter(
                flattened_rules, self.extent, temp_dir, self.include_all_fields
            )
            gpkg_path = exporter.export_all_rules()
            print("Successfully exported rule datasets")

            # Step 3: Generate vector tiles
            print("Generating vector tiles...")
            generator = VectorTileGenerator(temp_dir, self.cpu_percent, gpkg_path)
            tiles_path = generator.generate()
            print("Successfully generated vector tiles")

            # Step 4: Load and style tiles
            print("Loading and styling tiles...")
            styler = VectorTileStyler(flattened_rules, tiles_path)
            tiles_layer = styler.apply_styling()
            print("Process completed successfully")

            return tiles_layer

        except Exception as e:
            print(f"Error during conversion: {e}")
            raise e


# Main execution for QGIS console
if __name__ == "__console__":
    adapter = QGISVectorTilesAdapter()
    adapter.convert_project_to_vector_tiles()
