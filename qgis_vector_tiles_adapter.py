"""
QGIS Vector Tiles Adapter (QVTA)

Converts QGIS vector layer styling to vector tiles format by:
1. Flattening nested rule-based renderers/labeling with property inheritance
2. Splitting rules by symbol layers and matching label rules to renderer rules
3. Exporting each rule as a separate dataset with geometry transformations
4. Generating Tiles (pbf) directory using QGIS XYZ vector tiles processing
5. Loading and styling the tiles in QGIS with appropriate symbology

The adapter handles complex styling scenarios including geometry generators,
scale-dependent styling, nested filters, and ensures proper tile-based rendering.
"""

from os import mkdir
from os.path import join, basename
from tempfile import gettempdir
from datetime import datetime
from dataclasses import dataclass
from typing import Union, List, Optional, Tuple
from urllib.request import pathname2url
from time import perf_counter as prf
from osgeo import gdal, ogr, osr
from os import cpu_count
from re import sub

import processing
from PyQt5.QtCore import QVariant
from qgis.core import (
    QgsProject,
    QgsRuleBasedRenderer,
    QgsRuleBasedLabeling,
    QgsPalLayerSettings,
    QgsVectorLayer,
    QgsRenderContext,
    QgsVectorTileLayer,
    QgsCoordinateReferenceSystem,
    QgsGraduatedSymbolRenderer,
    QgsCategorizedSymbolRenderer,
    QgsWkbTypes,
    QgsVectorTileBasicRenderer,
    QgsVectorTileBasicRendererStyle,
    QgsVectorTileBasicLabeling,
    QgsVectorTileBasicLabelingStyle,
    QgsTileMatrix,
    QgsPointXY,
    QgsVectorTileWriter,
    QgsMarkerSymbol,
    QgsLineSymbol,
    QgsFillSymbol,
    QgsGeometry,
    QgsTileXYZ,
    QgsFields,
    QgsField,
    QgsVectorFileWriter,
    QgsFeature,
    QgsFeatureRequest,
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

    def get_rule_index(self):
        """Gets rule number"""
        rule_num = sub(r"[^\d]", "", self.rule.description())
        return float(f"0.{rule_num}")


class TilesStyler:
    """Applies styling to vector tile layers from flattened rules."""

    def __init__(self, flattened_rules: List[FlattenedRule], tiles_path: str):
        self.flattened_rules = flattened_rules
        self.tiles_layer = self._create_tiles_layer(tiles_path)
        self.renderer_styles = []
        self.labeling_styles = []

    def apply_styling(self) -> QgsVectorTileLayer:
        """Apply styles to vector tiles layer and add to project."""

        for rule in self.flattened_rules[::-1]:
            self._create_style_from_rule(rule)
        self._apply_styles_to_layer()
        return self.tiles_layer

    def _create_tiles_layer(self, tiles_path: str) -> QgsVectorTileLayer:
        """Create and add vector tiles layer to project."""
        suffix = (
            "&http-header:referer=" if tiles_path.split(".")[-1] != "mbtiles" else ""
        )
        layer = QgsVectorTileLayer(f"{tiles_path}{suffix}", "Vector Tiles")
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
        symbol_layer = symbol.symbolLayers()[-1]
        sub_symbol = symbol_layer.subSymbol()
        source_geom = int(flat_rule.get_attribute("g"))
        target_geom = int(flat_rule.get_attribute("c"))
        if source_geom != target_geom:
            if sub_symbol and symbol_layer.layerType() == "GeometryGenerator":
                self._copy_data_driven_properties(symbol, sub_symbol)
                self._copy_data_driven_properties(
                    symbol.symbolLayers()[-1], sub_symbol.symbolLayers()[-1]
                )
                symbol = sub_symbol
            else:
                symbol = None
                if target_geom == 0:
                    symbol = QgsMarkerSymbol()
                elif target_geom == 1:
                    symbol = QgsLineSymbol()
                else:
                    symbol = QgsFillSymbol()
                symbol.appendSymbolLayer(symbol_layer.clone())
                symbol.deleteSymbolLayer(0)
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


class GDALTilesGenerator:
    """Generate mbtiles from GeoJSON layers using GDAL MVT driver."""

    def __init__(self, layers, output_dir, output_type, tiles_conf, cpu_percent=75):
        self.layers = layers
        self.output_dir = output_dir
        self.output_type = output_type
        self.tiles_conf = tiles_conf
        self.cpu_percent = cpu_percent

    def generate(self):
        """Generate mbtiles file from configured layers."""
        # Set GDAL threading options

        cpu_num = str(max(1, int(cpu_count() * self.cpu_percent / 100)))
        gdal.SetConfigOption("GDAL_NUM_THREADS", cpu_num)

        # Unpack tile configuration
        crs_id, top_left_x, top_left_y, root_dimension, ratio_width, ratio_height = (
            self.tiles_conf
        )

        # Create spatial reference object
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(crs_id)

        # Determine output path and format

        if self.output_type == "xyz":
            template = pathname2url(r"/{z}/{x}/{y}.pbf")
            tiles_url = join(self.output_dir, "tiles")
            uri = f"{tiles_url}{template}"
            uri = f"type=xyz&url=file:///{uri}"
            output = tiles_url
            # output = join(self.output_dir, 'tiles')
            # template = pathname2url(r"/{z}/{x}/{y}.pbf")
            # tiles_url = output
            # template_path = f"{tiles_url}{template}"
            # uri= f"type=xyz&url=file:///{template_path}"
        else:
            subdir = self._generate_subdir()
            output = join(subdir, "tiles.mbtiles")
            uri = f"type=mbtiles&url={output}"

        # Initialize MVT driver
        driver = gdal.GetDriverByName("MVT")
        if self.output_type == "mbtiles":
            creation_options = []
        else:
            creation_options = [
                f"TILING_SCHEME=EPSG:{crs_id},{top_left_x},{top_left_y},{root_dimension},{ratio_width},{ratio_height}"
            ]
            creation_options = [
                "COMPRESS=YES"
                "TILE_EXTENSION=.mvt",
                "EPSG:3857,-20037508.3427892,20037508.3427892,40075016.6855784",
            ]

        ds = driver.Create(output, 0, 0, 0, gdal.GDT_Unknown, options=creation_options)

        # Process each layer
        for lyr in self.layers:
            self._process_layer(ds, lyr, sr)

        # Flush and close dataset
        ds.FlushCache()
        ds = None

        return uri

    def _generate_subdir(self):
        """Geneare temporary directory which will contains rule's datasets"""
        subdir = join(self.output_dir, "tiles")
        mkdir(subdir)
        return subdir

    def _process_layer(self, ds, lyr, sr):
        """Process a single layer and add it to the dataset."""
        lyr_name = basename(lyr.source()).split(".")[0]

        # Extract zoom levels from layer name (format: ...oXXiYY...)
        min_zoom = int(lyr_name.split("o")[1][:2])
        max_zoom = int(lyr_name.split("i")[1][:2])

        # Open source layer
        src_ds = ogr.Open(lyr.source())

        src_layer = src_ds.GetLayer(0)

        # Create output layer with zoom level options
        layer_options = [f"MINZOOM={min_zoom}", f"MAXZOOM={max_zoom}"]

        out_layer = ds.CreateLayer(
            lyr_name, srs=sr, geom_type=src_layer.GetGeomType(), options=layer_options
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

        # Cleanup
        src_ds = None

    def _get_global_min_zoom(self):
        """Get minimum zoom level across all layers."""
        min_zoom = float("inf")
        for lyr in self.layers:
            lyr_name = basename(lyr.source()).split(".")[0]
            zoom = int(lyr_name.split("o")[1][:2])
            min_zoom = min(min_zoom, zoom)
        return int(min_zoom) if min_zoom != float("inf") else 0

    def _get_global_max_zoom(self):
        """Get maximum zoom level across all layers."""
        max_zoom = 0
        for lyr in self.layers:
            lyr_name = basename(lyr.source()).split(".")[0]
            zoom = int(lyr_name.split("i")[1][:2])
            max_zoom = max(max_zoom, zoom)
        return max_zoom if max_zoom > 0 else 14


class QGISTilesGenerator:
    """Generate vector tiles from datasets list according given configuration"""

    def __init__(self, layers, output_dir, output_type, tiles_conf):
        self.layers: list = layers
        self.output_dir = output_dir
        self.output_type = output_type
        self.tiles_conf = tiles_conf

    def generate(self) -> tuple:
        """Generate vector tiles from rules layers."""
        layers = self._get_layers()
        minzoom, maxzoom = self._get_zoom_levels(layers)
        matrix = self._get_matrix()
        uri = self._get_uri()
        writer = self._set_writer(layers, minzoom, maxzoom, uri, matrix)
        # self._get_tiles(layers, minzoom, maxzoom, matrix)
        # self._set_directories_tree(tiles, minzoom, maxzoom)
        result = self._write_tiles(writer)
        errors = writer.errorMessage()
        if errors:
            print(f"Error during tiles generation: {errors}")
        return uri

    def _get_layers(self) -> list:
        """Extract vector tiles layers"""
        layers = []
        for layer in self.layers:
            tiles_layer = QgsVectorTileWriter.Layer(layer)
            tiles_layer.setLayerName(layer.name())
            tiles_layer.setMinZoom(int(layer.name().split("o")[1][:2]))
            tiles_layer.setMaxZoom(int(layer.name().split("i")[1][:2]))
            layers.append(tiles_layer)
        return layers

    def _get_zoom_levels(self, layers):
        """Get maximum and minimum zoom levels from layers list"""
        min_zoom = min(lyr.minZoom() for lyr in layers)
        max_zoom = max(lyr.maxZoom() for lyr in layers)
        return min_zoom, max_zoom

    def _get_uri(self) -> str:
        """Get output uri string (XYZ directory or mbtiles file, depend on the user preference)"""
        if self.output_type == "xyz":
            template = pathname2url(r"/{z}/{x}/{y}.pbf")
            tiles_url = self.output_dir
            uri = f"{tiles_url}{template}"
            return f"type=xyz&url=file:///{uri}"
        return join(self.output_dir, "tiles.mbtiles")

    def _get_matrix(self):
        """Generate tiles matrix according to user prefernces (Default: Web Mercator)"""
        matrix = QgsTileMatrix()
        crs_id, top_left_x, top_left_y, root_dimention, ratio_width, ratio_height = (
            self.tiles_conf
        )
        crs = QgsCoordinateReferenceSystem(crs_id)
        top_left_pnt = QgsPointXY(top_left_x, top_left_y)
        matrix = matrix.fromCustomDef(
            0, crs, top_left_pnt, root_dimention, ratio_width, ratio_height
        )
        return matrix

    def _get_tiles(self, layers, minzoom, maxzoom, matrix):
        """Get XYZ tiles list"""
        fetcher = TilesFetcher(layers, minzoom, maxzoom, matrix, self.output_dir)
        return fetcher.fetch()

    def _set_directories_tree(self, tiles, minzoom, maxzoom):
        """Set directory tree according to {Z}{X}{Y} template"""
        if self.output_type == "xyz":
            zs = list(range(minzoom, maxzoom + 1))
            zxs = set([(tile.zoomLevel(), tile.column()) for tile in tiles])
            tree = {z: [] for z in zs}
            for z, x in zxs:
                tree[z].append(x)
            for z in tree:
                zp = join(self.output_dir, str(z))
                mkdir(zp)
                for x in tree[z]:
                    xp = join(zp, str(x))
                    mkdir(xp)

    def _set_writer(self, layers, minzoom, maxzoom, uri, matrix):
        """Set vector tiles writer object and its configurations."""
        writer = QgsVectorTileWriter()
        writer.setTransformContext(QgsProject.instance().transformContext())
        writer.setLayers(layers)
        writer.setMinZoom(minzoom)
        writer.setMaxZoom(maxzoom)
        writer.setDestinationUri(uri)
        writer.setRootTileMatrix(matrix)
        writer.setExtent(matrix.extent())
        return writer

    def _write_tiles(self, writer, tiles=None):
        """Write the tiles which covers rules layers extent."""
        writer.writeTiles()
        # for tile in tiles:
        #     with open(
        #         f"{self.output_dir}\\{tile.zoomLevel()}\\{tile.column()}\\{tile.row()}.pbf",
        #         "wb",
        #     ) as f:
        #         f.write(writer.writeSingleTile(tile))


class TilesFetcher:
    """
    High-performance tile index generator for QGIS layers.
    Creates a spatial index of tiles that intersect with input geometries.
    """

    def __init__(self, layers, minzoom, maxzoom, matrix, output_dir):
        self.layers = layers
        self.minzoom = minzoom
        self.maxzoom = maxzoom
        self.matrix = matrix
        self.output_dir = self._generate_subdir(output_dir)
        self.tiles = []

    def fetch(self) -> QgsVectorLayer:
        """Generate tile index for given layers and extent."""

        # Collect layer geometries and extent
        layers_extent = self._get_layers_extent()
        layers_geometries = self._get_layers_geometries()

        # Process zoom levels
        current_extents = [layers_extent]
        for zoom in range(self.minzoom, self.maxzoom + 1):
            current_extents = self._process_zoom_level(
                zoom, current_extents, layers_geometries
            )
        self._export_index()
        return self.tiles

    def _generate_subdir(self, output_dir):
        """Geneare temporary directory which will contains rule's datasets"""
        subdir = join(output_dir, "index")
        mkdir(subdir)
        return subdir

    def _get_layers_extent(self):
        """Get extent of all layers"""
        layers_extents = []

        for layer in self.layers:
            bbox_geom = QgsGeometry.fromWkt(layer.layer().sourceExtent().asWktPolygon())
            layers_extents.append(bbox_geom)

        layers_union = QgsGeometry.unaryUnion(layers_extents)
        return layers_union.boundingBox()

    def _get_layers_geometries(self):
        """Efficiently collect and cache layer geometries."""
        layers_geometries = []

        for layer in self.layers:
            min_zoom = layer.minZoom()
            max_zoom = layer.maxZoom()
            layer = layer.layer()

            # Cache geometry collections to avoid recomputation
            geometries = [f.geometry() for f in list(layer.getFeatures())]

            # Use unaryUnion for badd .etter performance with many geometries
            if len(geometries) == 1:
                combined_geom = geometries[0]
            else:
                combined_geom = QgsGeometry.unaryUnion(geometries)
            layers_geometries.append((min_zoom, max_zoom, combined_geom))

        return layers_geometries

    def _export_index(self):
        """Export index to a given gpk file."""
        # Create Index dataset
        output_index = join(self.output_dir, f"index.parquet")
        fields = QgsFields(
            [
                QgsField("fid", QVariant.Int),
                QgsField("X", QVariant.Int),
                QgsField("Y", QVariant.Int),
                QgsField("Z", QVariant.Int),
            ]
        )

        crs = self.matrix.crs()
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "Parquet"

        QgsVectorFileWriter.create(
            fileName=output_index,
            fields=fields,
            geometryType=QgsWkbTypes.Polygon,
            srs=crs,
            transformContext=QgsProject.instance().transformContext(),
            options=options,
        )
        index_layer = QgsVectorLayer(output_index, "Tiles Index")

        # Export tiles geometries into index dataset
        tiles_features = []
        current_id = 1
        sorted_tiles = sorted(self.tiles, key=lambda x: x.zoomLevel(), reverse=True)
        for tile in sorted_tiles:
            matrix = QgsTileMatrix.fromTileMatrix(tile.zoomLevel(), self.matrix)
            feat = QgsFeature(fields)
            feat.setAttributes(
                [current_id, tile.column(), tile.row(), tile.zoomLevel()]
            )
            feat.setGeometry(
                QgsGeometry.fromWkt(matrix.tileExtent(tile).asWktPolygon())
            )
            tiles_features.append(feat)
            current_id += 1
        index_layer.dataProvider().addFeatures(tiles_features)

        # Add index layer to current project.
        # Layer renderer order determined by tile Z values.
        renderer = index_layer.renderer()
        renderer.setOrderByEnabled(True)
        request = QgsFeatureRequest()
        request.addOrderBy('"Z"', True)
        renderer.setOrderBy(request.orderBy())
        renderer.symbol().setOpacity(0.2)
        QgsProject.instance().addMapLayer(index_layer)
        return index_layer

    def _process_zoom_level(self, zoom: int, extents, layer_geometries):
        """Process a single zoom level and return new extents for next level."""
        # Filter geometries valid for this zoom level
        zoom_geoms = [
            geom for minz, maxz, geom in layer_geometries if minz <= zoom <= maxz
        ]

        if not zoom_geoms:
            return []

        inner_extents = []
        matrix = QgsTileMatrix.fromTileMatrix(zoom, self.matrix)
        for extent in extents:
            tile_range = matrix.tileRangeFromExtent(extent)
            for row in range(tile_range.startRow(), tile_range.endRow() + 1):
                for col in range(tile_range.startColumn(), tile_range.endColumn() + 1):
                    tile = QgsTileXYZ(col, row, zoom)
                    tile_extent = matrix.tileExtent(tile)
                    tile_geom = QgsGeometry.fromWkt(tile_extent.asWktPolygon())

                    # Check intersection with any geometry
                    if any(geom.intersects(tile_geom) for geom in zoom_geoms):
                        inner_extents.append(tile_extent)
                        self.tiles.append(tile)

        return inner_extents


class RulesExporter:
    """Export all rules to memory datasets."""

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
        output_dir,
        extent,
        include_all_fields: bool,
        crs_id,
    ):
        self.flattened_rules = flattened_rules
        self.extent = extent
        self.include_all_fields = include_all_fields
        self.crs_id = crs_id
        self.temp_dir = self._generate_subdir(output_dir)
        self.processed_layers = []

    def export(self) -> list:
        """Export all rules to memory datasets."""
        for rule in self.flattened_rules:
            self._export_single_rule(rule)
        return self.processed_layers

    def _generate_subdir(self, output_dir):
        """Geneare temporary directory which will contains rule's datasets"""
        subdir = join(output_dir, "datasets")
        mkdir(subdir)
        return subdir

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

        # Add labeling expression as field if required
        if flat_rule.get_attribute("t") == 1:
           layer = self._calculate_label_expression(layer, flat_rule)

        # Keep only required fields
        if not self.include_all_fields:
            required_fields = list(self._get_required_fields(flat_rule))
            required_fields.append(f"{self.FIELD_PREFIX}_fid")
            layer = self._run_processing(
                "retainfields", INPUT=layer, FIELDS=required_fields
            )

        # # Extract features by extent
        # layer = self._run_processing(
        #     "extractbyextent", INPUT=layer, CLIP=True, EXTENT=self.extent
        # )

        # Transform geometry if needed
        layer = self._transform_geometry_if_needed(layer, flat_rule)

        # Reproject to destination EPSG
        rule_desc = flat_rule.rule.description()
        output_dataset = join(self.temp_dir, f"{rule_desc}.parquet")
        layer = self._run_processing("savefeatures", INPUT=layer, OUTPUT=output_dataset)

        layer.setName(rule_desc)
        self.processed_layers.append(layer)

    def _prepare_base_layer(self, flat_rule: FlattenedRule) -> QgsVectorLayer:
        """Prepare base layer with extent clipping and geometry fixes."""
        layer = flat_rule.layer

        # Reproject to destination EPSG
        layer = self._run_processing(
            "reprojectlayer",
            INPUT=layer,
            TARGET_CRS=QgsCoordinateReferenceSystem(self.crs_id),
            CONVERT_CURVED_GEOMETRIES=False,
        )

        extent = self.extent or layer.extent()

        # Add unique ID field
        with_id = self._run_processing(
            "fieldcalculator",
            INPUT=layer,
            FIELD_NAME=f"{self.FIELD_PREFIX}_fid",
            FORMULA=f"@id",
            FIELD_TYPE=2,
        )

        # Extract by extent and fix geometries
        extracted = self._run_processing(
            "extractbyextent", INPUT=with_id,CLIP=True, EXTENT=extent
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
    
    def _calculate_label_expression(
        self, layer: QgsVectorLayer, flat_rule: FlattenedRule
    ) -> QgsVectorLayer:
        """Add geometry attributes needed for data-driven properties."""
        field_name = f'{self.FIELD_PREFIX}_label_expression'
        expression = flat_rule.rule.settings().getLabelExpression().expression()
        layer = self._run_processing(
            "fieldcalculator",
            INPUT=layer,
            FIELD_NAME=field_name,
            FORMULA=expression,
            FIELD_TYPE=2
        )
        flat_rule.rule.settings().isExpression = False
        flat_rule.rule.settings().fieldName = field_name
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
                return target_geom, " boundary(@geometry)"

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

    def _run_processing(self, algorithm: str, algorithm_type: str = "native", **params):
        """Execute QGIS processing algorithm."""
        if not params.get("OUTPUT"):
            params["OUTPUT"] = "TEMPORARY_OUTPUT"
            # params["OUTPUT"] = join(r'C:\Users\P0026701\AppData\Local\Temp\06_08_2025_17_24_07_824669',  datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f.shp"))
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
        self.project = QgsProject.instance()
        self.layer_tree_root = self.project.layerTreeRoot()
        self.flattened_rules = []

    def flatten_all_rules(self) -> List[FlattenedRule]:
        """Extract and flatten all rules from visible vector layers."""
        layers = [layer.layer() for layer in self.layer_tree_root.findLayers()]

        for layer_idx, layer in enumerate(layers):
            if self._is_valid_layer(layer):
                self._process_layer_rules(layer.clone(), layer_idx)

        return self.flattened_rules

    def _is_valid_layer(self, layer) -> bool:
        """Check if layer is a visible vector layer."""
        is_vector = layer.type() == 0 and layer.geometryType() != 4
        lyr = self.layer_tree_root.findLayer(layer.id())
        is_visible = lyr.isVisible() if lyr else None
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
            rules_method = None
            inactive_rules = []
            if isinstance(system, QgsGraduatedSymbolRenderer):
                rules_method = 'ranges'
            elif isinstance(system, QgsCategorizedSymbolRenderer):
                rules_method = 'categories'
            if rules_method:
                for index, rule in enumerate(getattr(system, rules_method)()):
                    if not rule.renderState():
                        inactive_rules.append(index)
            rulebased_renderer = QgsRuleBasedRenderer.convertFromRenderer(system) if system else None
            if rulebased_renderer and inactive_rules:
                for rule_index in sorted(inactive_rules, reverse=True):
                    rule = rulebased_renderer.rootRule().children()[rule_index]
                    rulebased_renderer.rootRule().removeChildAt(rule_index)
            return rulebased_renderer
        
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
        if rule.parent() :
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
        flat_rule.set_attribute("o", self._get_rule_zoom_range(flat_rule, min))
        flat_rule.set_attribute("i", self._get_rule_zoom_range(flat_rule, max))
        flat_rule.set_attribute(
            "s" if rule_type == 0 else "f", 0
        )  # Symbol/feature index

    def _get_rule_zoom_range(self, flat_rule, comparator):
        """Extract rule maximum and minimum range using general range."""
        attr_name = f"{comparator.__name__}imumScale"
        rule_scale = getattr(flat_rule.rule, attr_name)()
        rule_zoom = int(ZoomLevels.scale_to_zoom(rule_scale))
        tiles_zoom = getattr(self, f"{comparator.__name__}_zoom")
        opposite = min if comparator == max else max
        return opposite(rule_zoom, tiles_zoom)

    def _convert_else_filter(self, else_rule, parent_rule) -> None:
        """Convert ELSE filter to explicit exclusion of sibling conditions."""
        sibling_filters = []
        for sibling in parent_rule.children():
            if sibling.active() and sibling.filterExpression() not in ("ELSE", ""):
                sibling_filters.append(sibling.filterExpression())

        if sibling_filters:
            else_expression = f'NOT ({" OR ".join(f"({f})" for f in sibling_filters)})'
        else:
            else_expression = ''
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
            symbol_layer = symbol.symbolLayer(layer_idx)
            sub_symbol = symbol_layer.subSymbol()
            symbol_type = (
                sub_symbol.type()
                if symbol_layer.layerType() == "GeometryGenerator"
                else symbol_layer.type()
            )
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
        max_zoom: int = 10,
        extent=None,
        output_dir: str = None,
        include_all_fields: bool = False,
        output_type="xyz",
        tiles_conf=[3857, -20037508.3427892, 20037508.3427892, 40075016.6855784, 1, 1],
    ):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.extent = extent or iface.mapCanvas().extent()
        self.output_dir = output_dir or gettempdir()
        self.include_all_fields = include_all_fields
        self.output_type = output_type.lower()
        self.tiles_conf = tiles_conf

    def convert_project_to_vector_tiles(self) -> Optional[QgsVectorTileLayer]:
        """
        Convert current QGIS project to vector tiles format.

        Returns:
            QgsVectorTileLayer: The created vector tiles layer, or None if failed
        """
        try:
            temp_dir = join(
                self.output_dir, datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f")
            )
            mkdir(temp_dir)
            print("Starting conversion process...")
            start_time = prf()

            # Step 1: Flatten all rules
            print("Flattening rules...")
            flattener = RuleFlattener(self.min_zoom, self.max_zoom)
            rules = flattener.flatten_all_rules()

            if not rules:
                print("No visible vector layers found in project.")
                return None

            print(f"Successfully extracted {len(rules)} rules.")

            # Step 2: Export rules to datasets
            print("Exporting rules to layers...")
            exporter = RulesExporter(
                rules,
                temp_dir,
                self.extent,
                self.include_all_fields,
                self.tiles_conf[0],
            )
            layers = exporter.export()
            print("Successfully exported rules.")

            # Step 3: Generate tiles from datasets
            print("Generating tiles...")
            # generator = GDALTilesGenerator(
            #     layers, temp_dir, self.output_type, self.tiles_conf
            # )
            generator = QGISTilesGenerator(
                layers, temp_dir, self.output_type, self.tiles_conf
            )
            tiles = generator.generate()
            print("Successfully generated tiles.")

            # Step 4: Load and style tiles
            print("Loading and styling tiles...")
            styler = TilesStyler(rules, tiles)
            tiles_layer = styler.apply_styling()
            process_time = prf() - start_time
            print(f"Process completed successfully ({round(process_time,2)} seconds).")

            return tiles_layer

        except Exception as e:
            print(f"Error during conversion: {e}")
            raise e


# Main execution for QGIS console
if __name__ == "__console__":
    adapter = QGISVectorTilesAdapter()
    adapter.convert_project_to_vector_tiles()
