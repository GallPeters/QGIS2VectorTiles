"""
Q2StyledTiles:

Converts QGIS vector layer styling to vector tiles format by:
1. Flattening nested rule-based renderers/labeling with property inheritance
2. Splitting rules by symbol layers and matching label rules to renderer rules
3. Exporting each rule as a separate dataset with geometry transformations
4. Generating vector tiles using GDAL MVT driver
5. Loading and styling the tiles in QGIS with appropriate symbology
"""
import subprocess
import platform
from sys import prefix
from threading import Lock
from dataclasses import dataclass
from tomllib import load
from datetime import datetime
from os import makedirs, cpu_count, listdir
from os.path import join, basename, exists
from time import perf_counter, sleep
from typing import List, Optional, Tuple, Union, Dict
from urllib.request import pathname2url
from shutil import rmtree, copy2
from uuid import uuid4


from processing import run
from osgeo import gdal, ogr, osr
from qgis.PyQt.QtCore import QTimer, QEventLoop
from qgis.PyQt.QtCore import qVersion
from qgis.utils import iface
from tempfile import gettempdir
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProject,
    QgsTask,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsApplication,
    QgsVectorLayer,
    QgsExpression,
    QgsRuleBasedRenderer,
    QgsRuleBasedLabeling,
    QgsReadWriteContext,
    QgsProcessingContext,
    QgsRuleBasedRenderer,
    QgsRuleBasedLabeling,
    QgsPalLayerSettings,
    QgsProcessingException,
    QgsVectorLayer,
    QgsLayerDefinition,
    QgsVectorTileLayer,
    QgsProperty,
    QgsGraduatedSymbolRenderer,
    QgsCategorizedSymbolRenderer,
    QgsWkbTypes,
    QgsReadWriteContext,
    QgsMapLayer,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsVectorTileBasicRenderer,
    QgsProcessingFeatureSourceDefinition,
    QgsVectorTileBasicRendererStyle,
    QgsVectorTileBasicLabeling,
    QgsVectorTileBasicLabelingStyle,
    QgsTextBackgroundSettings,
    QgsMarkerSymbol,
    QgsLineSymbol,
    QgsFillSymbol,
    QgsTextFormat,
    QgsProcessingUtils,
    QgsProcessingFeedback,
    QgsApplication,
    QgsPropertyDefinition,
    QgsLabelThinningSettings,
    Qgis,
    QgsExpression,
)


import os
os.environ["PYTHONIOENCODING"] = "utf-8:replace"

try:
    from .qgis2maplibre import QgisMapLibreStyleExporter
except ImportError:
    pass

# Import by QT version
if int(qVersion()[0]) == 5:
    from PyQt5.QtXml import QDomDocument
    from PyQt5.QtCore import QVariant
    from PyQt5 import sip
else:
    from PyQt6.QtXml import QDomDocument
    from PyQt6.QtCore import QVariant
    from PyQt6 import sip


_PLUGIN_DIR = join(QgsApplication.qgisSettingsDirPath(), "python", "plugins", "QGIS2VectorTiles")
_CONF = join(_PLUGIN_DIR, "resources", "conf.toml")
_VIEWER = join(_PLUGIN_DIR, "resources", "maplibre_viewer.html")
_MAPLIBRE = join(_PLUGIN_DIR, "resources", "maplibre-gl")
_PORT = 9000
_TILES_CONF = load(open(_CONF, "rb"))
_TILING_SCHEME = _TILES_CONF["TILING_SCHEME"]


class DataDefinedPropertiesFetcher:
    """Fetch recursively all data defined properties in a given object"""

    CRASHING_ATTRS = [
        "_",
        "value",
        "index",
        "available",
        "config",
        "next",
        "attr",
        "clone",
        "function",
        "flag",
        "capabil",
        "remove",
        "symbols",
        "clear",
        "prepare",
        "dump",
        "copy",
        "create",
        "update",
        "replace",
    ]
    TYPES_MAP = {
        QgsPropertyDefinition.DataTypeString: QVariant.String,
        QgsPropertyDefinition.DataTypeNumeric: QVariant.Double,
        QgsPropertyDefinition.DataTypeBoolean: QVariant.Bool,
    }
    FIELD_PREFIX = "q2vt"

    def __init__(self, qgis_object, min_zoom):
        self.qgis_object = qgis_object
        self.min_zoom = min_zoom
        self.dd_properties = []

    def fetch(self):
        """Fetch data defined properties from main instance object"""
        self._fetch_ddp(self.qgis_object)
        return self.dd_properties

    def _fetch_ddp(self, qgis_object):
        """Get data defined properties from current object's subobjects"""
        for attr in dir(qgis_object):
            try:
                if any(word.lower() in attr.lower() for word in self.CRASHING_ATTRS):
                    continue
                if attr.startswith("set") and attr != attr.lower():
                    continue
                if attr[0].isupper():
                    continue
                getter = getattr(qgis_object, attr)
                if not callable(getter):
                    continue
                qgis_subobjects = None
                qgis_subobjects = [getter()] if not isinstance(getter(), list) else getter()
                if not qgis_subobjects:
                    continue
                first_subobject = qgis_subobjects[0]
                if isinstance(first_subobject, type(qgis_object)):
                    continue
                if "qgis." not in str(type(first_subobject)):
                    continue
                if first_subobject in self.dd_properties:
                    continue
                if hasattr(first_subobject, "propertyDefinitions"):
                    props_defintions = getattr(first_subobject, "propertyDefinitions")()
                elif hasattr(qgis_object, "propertyDefinitions"):
                    props_defintions = getattr(qgis_object, "propertyDefinitions")()
                else:
                    props_defintions = None
                if not props_defintions:
                    continue
                self._get_properties(qgis_subobjects, props_defintions)
            except (NameError, ValueError, AttributeError, TypeError):
                continue

    def _get_properties(self, qgis_subobjects, props_defintions):
        """Get data defined property properties from qgis object"""
        for qgis_subobject in qgis_subobjects:
            if hasattr(qgis_subobject, "dataDefinedProperties"):
                self._get_propertys_from_subobjects(qgis_subobject, props_defintions)
            self._fetch_ddp(qgis_subobject)

    def _get_propertys_from_subobjects(self, qgis_subobject, props_defintions):
        """Get data defined properties from` subobjects of qgis object"""
        props_collection = qgis_subobject.dataDefinedProperties()
        for key in props_collection.propertyKeys():
            prop = props_collection.property(key)
            if not prop or not prop.isActive():
                continue
            prop_type = prop.propertyType()
            if prop_type not in [2, 3]:
                continue
            prop_def = props_defintions.get(key)
            prop_type = prop_def.dataType() if props_defintions else None
            field_type = self.TYPES_MAP.get(prop_type)
            field_name = f"{self.FIELD_PREFIX}_{uuid4().hex[:8]}"

            if prop_type == 2:
                exp_prop = QgsProperty()
                exp_prop.setExpressionString(prop.asExpression())
                expression = exp_prop.expressionString()
                exp_prop.setExpressionString(f'"{field_name}"')
                props_collection.setProperty(key, exp_prop)
                prop = props_collection.property(key)
            else:
                expression = prop.expressionString().replace("@map_scale", self.min_zoom)
                if "color" in prop_def.name().lower() and field_type == 10:
                    # Convert color to hex string in order to be used in MapLibre style
                    expression = f"'#' || with_variable('hex', array_cat(generate_series(0,9),array('A','B','C','D','E','F')), array_to_string (array_foreach (array ('red','green','blue'),with_variable('colo',color_part ({expression}, @element),@hex[floor(@colo/16)] || @hex[@colo%16] )),''))" # pylint: disable=C0301
                evaluation = QgsExpression(expression).evaluate()
                if evaluation is not None:
                    prop.setExpressionString(str(evaluation))
                    continue
                if "array" in expression:
                    expression = f"try(array_to_string({expression}), {expression})"
                prop.setExpressionString(f'"{field_name}"')

            field_map = [field_type, expression, field_name]

            self.dd_properties.append(field_map)


class ZoomLevels:
    """Manages zoom level scales and conversions for web mapping standards."""

    SCALES = [_TILING_SCHEME["SCALE"] / (2**zoom) for zoom in range(23)]

    @classmethod
    def scale_to_zoom(cls, scale: float, edge: str) -> str:
        """Convert scale to zero-padded zoom level string."""
        if scale in [0, 0.0]:
            scale = cls.SCALES[0 if edge == "o" else -1]
        for zoom, zoom_scale in enumerate(cls.SCALES):
            if scale >= zoom_scale and edge == "o":
                return zoom
        if scale > cls.SCALES[0]:
            return 0
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
    output_dataset: Optional[str] = ""

    def get_attr(self, char: str) -> Optional[int]:
        """Extract rule attribute from description by character prefix."""
        desc = self.rule.description()
        start = desc.find(char) + 1
        if start == 0:
            return None
        return int(desc[start : start + 2])

    def set_attr(self, char: str, value: int):
        """Set rule attribute in description."""
        value = int(value)
        new_attr = f"{char}{value:02d}"
        current = self.get_attr(char)

        desc = self.rule.description()
        if current is not None:
            old_attr = f"{char}{current:02d}"
            desc = desc.replace(old_attr, new_attr)
        else:
            desc = f"{desc}{new_attr}"

        self.rule.setDescription(desc)
        self.output_dataset = desc

        i = desc.find("s")
        if i >= 0:
            self.output_dataset = self.output_dataset.replace(desc[i : i + 3], "s00")

    def get_description(self):
        """Construct rule description for labeling or renderer rule."""
        lyr_name = self.layer.name() or self.layer.id()
        rule_type = "renderer" if self.get_attr("t") == 0 else "labeling"
        rule_num = self.get_attr("r")
        rule_subnum = self.get_attr("s") if rule_type == "renderer" else self.get_attr("f")
        return f"{lyr_name} > {rule_num} > {rule_type} > {rule_subnum}"


class TilesStyler:
    """Applies styling to vector tile layers from flattened rules."""

    def __init__(self, flattened_rules: List[FlattenedRule], output_dir: str, tiles_path: str):
        self.flattened_rules = flattened_rules
        self.output_dir = output_dir or QgsProcessingUtils.tempFolder()
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
        header = "&http-header:referer="
        layer = QgsVectorTileLayer(f"{tiles_path}{header}", "Vector Tiles")
        layer = QgsProject.instance().addMapLayer(layer, False)
        QgsProject.instance().layerTreeRoot().insertLayer(0, layer)
        return layer

    def _create_style_from_rule(self, flat_rule: FlattenedRule):
        """Create appropriate style from flattened rule."""
        rule_type = flat_rule.get_attr("t")

        if rule_type == 0:  # Renderer
            style = QgsVectorTileBasicRendererStyle()
            self._setup_renderer_style(style, flat_rule)
            self.renderer_styles.append(style)
        else:  # Labeling
            style = QgsVectorTileBasicLabelingStyle()
            self._setup_labeling_style(style, flat_rule)
            self.labeling_styles.append(style)

    def _setup_renderer_style(self, style, flat_rule: FlattenedRule):
        """Configure renderer style properties."""
        self._setup_base_style_properties(style, flat_rule)

        symbol = flat_rule.rule.symbol()
        if not symbol:
            return
        symbol_layer = symbol.symbolLayers()[-1]
        sub_symbol = symbol_layer.subSymbol()
        if sub_symbol:
            sub_symbol_layer = sub_symbol.symbolLayers()[-1]
        else:
            sub_symbol_layer = None
        source_geom = int(flat_rule.get_attr("g"))
        target_geom = int(flat_rule.get_attr("c"))

        if source_geom != target_geom:
            if sub_symbol and symbol_layer.layerType() in ["GeometryGenerator", "CentroidFill"]:
                self._copy_data_driven_properties(symbol, sub_symbol)
                self._copy_data_driven_properties(symbol_layer, sub_symbol_layer)
                symbol = sub_symbol
            else:
                symbol = self._create_transformed_symbol(target_geom, symbol_layer)

        style.setSymbol(symbol.clone())

    def _create_transformed_symbol(self, target_geom: int, symbol_layer):
        """Create symbol with transformed geometry type."""
        symbol_map = {0: QgsMarkerSymbol, 1: QgsLineSymbol, 2: QgsFillSymbol}
        symbol = symbol_map.get(target_geom, QgsMarkerSymbol)()
        symbol.appendSymbolLayer(symbol_layer.clone())
        symbol.deleteSymbolLayer(0)
        return symbol

    def _setup_labeling_style(
        self, style: QgsVectorTileBasicLabelingStyle, flat_rule: FlattenedRule
    ):
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
        remove_distance = _TILES_CONF["GENERAL_CONF"]["REMOVE_DUPLICATES_DISTANCE"]
        thin.setMinimumDistanceToDuplicate(remove_distance)
        thin.setMinimumDistanceToDuplicateUnit(Qgis.RenderUnit.Points)
        settings.setThinningSettings(thin)

    def _setup_base_style_properties(self, style, flat_rule: FlattenedRule):
        """Setup common style properties."""
        style.setEnabled(True)
        style.setLayerName(flat_rule.output_dataset)
        style.setStyleName(flat_rule.rule.description())
        style.setMinZoomLevel(flat_rule.get_attr("o"))
        style.setMaxZoomLevel(flat_rule.get_attr("i"))
        geom_types = {
            0: QgsWkbTypes.PointGeometry,
            1: QgsWkbTypes.LineGeometry,
            2: QgsWkbTypes.PolygonGeometry,
        }
        geom_code = flat_rule.get_attr("c")
        style.setGeometryType(geom_types.get(geom_code, QgsWkbTypes.PointGeometry))

    def _copy_data_driven_properties(self, source_obj, target_obj):
        """Copy data-driven properties between objects."""
        source_props = source_obj.dataDefinedProperties()
        target_props = target_obj.dataDefinedProperties()

        for prop_key in source_obj.propertyDefinitions():
            prop = source_props.property(prop_key)
            target_props.setProperty(prop_key, prop)
            target_props.property(prop_key).setActive(True)

    def _apply_styles_to_layer(self):
        """Apply collected styles to the tiles layer."""
        renderer = QgsVectorTileBasicRenderer()
        renderer.setStyles(self.renderer_styles)

        labeling = QgsVectorTileBasicLabeling()
        labeling.setStyles(self.labeling_styles)

        self.tiles_layer.setRenderer(renderer)
        self.tiles_layer.setLabeling(labeling)

    def _save_style(self):
        """Save layer style to QLR file."""
        style_dir = join(self.output_dir, "style")
        makedirs(style_dir, exist_ok=True)
        qlr_path = join(style_dir, "tiles.qlr")
        layer = QgsProject.instance().layerTreeRoot().findLayer(self.tiles_layer.id())
        QgsLayerDefinition().exportLayerDefinition(qlr_path, [layer])



class GDALTilesGenerator:
    """Generate XYZ tiles using GDAL CLI + VRT (fast, multi-layer, per-zoom)."""

    def __init__(
        self,
        layers: List[QgsVectorLayer],
        output_dir: str,
        output_type: str,
        extent,
        cpu_percent: int,
        feedback: QgsProcessingFeedback,
    ):
        self.layers = layers
        self.output_dir = output_dir
        self.output_type = output_type
        self.extent = extent
        self.cpu_percent = cpu_percent
        self.feedback = feedback

    def generate(self) -> Tuple[str, int]:
        """Main entry point."""
        output, uri = self._prepare_output_paths()
        vrt_path = join(self.output_dir, "layers.vrt")

        min_zoom = self._get_global_min_zoom()
        max_zoom = self._get_global_max_zoom()

        self._build_vrt(vrt_path)
        self._run_ogr2ogr(vrt_path, output, min_zoom, max_zoom)

        return uri, min_zoom

    # ---------------------------------------------------------------------
    # VRT CREATION
    # ---------------------------------------------------------------------

    def _build_vrt(self, vrt_path: str):
        """Build VRT with per-layer zoom configuration."""

        with open(vrt_path, "w", encoding="utf-8") as f:
            f.write("<OGRVRTDataSource>\n")

            for layer in self.layers:
                layer_name = basename(layer.source()).split(".")[0]

                # Extract zooms from your naming convention
                min_zoom = int(layer_name.split("o")[1][:2])
                max_zoom = int(layer_name.split("i")[1][:2])

                source = layer.source().split("|layername=")[0]

                f.write(f"""
    <OGRVRTLayer name="{layer_name}">
        <SrcDataSource>{source}</SrcDataSource>
        <LayerSRS>EPSG:3857</LayerSRS>
        <GeometryType>wkbUnknown</GeometryType>
        <LayerCreationOption name="MINZOOM" value="{min_zoom}"/>
        <LayerCreationOption name="MAXZOOM" value="{max_zoom}"/>
    </OGRVRTLayer>
""")

            f.write("</OGRVRTDataSource>\n")

    # ---------------------------------------------------------------------
    # OGR2OGR EXECUTION
    # ---------------------------------------------------------------------

    def _run_ogr2ogr(self, vrt_path: str, output: str, min_zoom: int, max_zoom: int):
        """Run single ogr2ogr command."""

        cpu_num = str(max(1, int(cpu_count() * self.cpu_percent / 100)))
        env = os.environ.copy()
        env["GDAL_NUM_THREADS"] = cpu_num

        cmd = [
            "ogr2ogr",
            "-f", "MVT",
            output,
            vrt_path,
            "-dsco", f"MINZOOM={min_zoom}",
            "-dsco", f"MAXZOOM={max_zoom}",
            "-dsco", "EXTENT=4096",
            "-t_srs", "EPSG:3857",
        ]

        # --- Windows: hide console ---
        startupinfo = None
        creationflags = 0

        if os.name == "nt":
            creationflags = 0x08000000  # CREATE_NO_WINDOW
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0

        try:
            subprocess.run(
                cmd,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                startupinfo=startupinfo,
                creationflags=creationflags,
            )
        except subprocess.CalledProcessError as e:
            error_msg = f"ogr2ogr failed.\nError: {e.stderr}"
            if self.feedback:
                self.feedback.reportError(error_msg)
            raise RuntimeError(error_msg)

    # ---------------------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------------------

    def _prepare_output_paths(self) -> Tuple[str, str]:
        template = pathname2url(r"/{z}/{x}/{y}.pbf")
        output = join(self.output_dir, "tiles")
        uri = f"type=xyz&zmin=0&zmax=22&url=file:///{output}{template}"
        return output, uri

    def _get_global_min_zoom(self) -> int:
        min_zoom = float("inf")
        for layer in self.layers:
            name = basename(layer.source()).split(".")[0]
            zoom = int(name.split("o")[1][:2])
            min_zoom = min(min_zoom, zoom)
        return int(min_zoom) if min_zoom != float("inf") else 0

    def _get_global_max_zoom(self) -> int:
        max_zoom = 0
        for layer in self.layers:
            name = basename(layer.source()).split(".")[0]
            zoom = int(name.split("i")[1][:2])
            max_zoom = max(max_zoom, zoom)
        return max_zoom if max_zoom > 0 else 14


class _ParallelExportTask(QgsTask):
    """Internal task for executing export processing in the background safely."""
    def __init__(self, description, func, *args, **kwargs):
        super().__init__(description, QgsTask.CanCancel)
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.exception = None

    def run(self):
        try:
            # Minor jitter to prevent concurrent initialization race conditions in GDAL
            import random
            sleep(0.5)
            self.result = self.func(*self.args, **self.kwargs)
            return True
        except Exception as e:
            self.exception = e
            return False


class RulesExporter:
    """Export all rules to datasets with geometry transformations using parallel processing."""

    FIELD_PREFIX = "q2vt"

    def __init__(
        self,
        flattened_rules: List['FlattenedRule'], 
        extent,
        include_required_fields_only,
        max_zoom,
        utils_dir,
        cent_source,
        feedback: QgsProcessingFeedback,
        cpu_percent: int = 100
    ):
        self.flattened_rules = flattened_rules
        self.extent = extent
        self.include_required_fields_only = include_required_fields_only
        self.max_zoom = max_zoom
        self.cent_source = cent_source
        self.utils_dir = utils_dir
        self.feedback = feedback
        self.cpu_percent = cpu_percent
        self.processed_layers = []

    def export(self) -> Tuple[List[QgsVectorLayer], List['FlattenedRule']]:
        """Export all rules to datasets using QgsTask parallel workers."""
        num_workers = max(1, int(os.cpu_count() * self.cpu_percent / 100))
        
        # 1. Export base layers parallelly
        output_datasets = self._export_base_layers(num_workers)

        # 2. Export rules parallelly
        total_datasets = len(output_datasets)
        rule_tasks = []
        
        for index, flat_rules in enumerate(output_datasets.values()):
            task_desc = f"Exporting rule {index + 1}/{total_datasets}"
            task = _ParallelExportTask(task_desc, self._export_rule_thread_safe, flat_rules)
            rule_tasks.append((task, flat_rules))

        self._run_tasks(rule_tasks, num_workers)

        successful_flat_rules = []
        for task, flat_rules in rule_tasks:
            rules_dataset = join(self.utils_dir, f"{flat_rules[0].output_dataset}.parquet")
            if exists(rules_dataset):
                layer_path = task.result
                layer = QgsVectorLayer(layer_path, flat_rules[0].output_dataset, "ogr")
                if layer.isValid() and layer.featureCount() > 0:
                    self.processed_layers.append(layer)
                    successful_flat_rules.extend(flat_rules)
            else:
                for rule in flat_rules:
                    if rule in self.flattened_rules:
                        self.flattened_rules.remove(rule)
                        
        return self.processed_layers, successful_flat_rules

    def _run_tasks(self, tasks_with_meta: list, num_workers: int):
        """Throttle and process QgsTasks safely waiting for their execution."""
        manager = QgsApplication.taskManager()
        active_tasks = []
        task_queue = list(tasks_with_meta)
        
        while task_queue or active_tasks:
            if self.feedback.isCanceled():
                for t, _ in active_tasks:
                    t.cancel()
                break

            while len(active_tasks) < num_workers and task_queue:
                t_tuple = task_queue.pop(0)
                t = t_tuple[0]
                manager.addTask(t)
                active_tasks.append(t_tuple)
            
            for t_tuple in active_tasks[:]:
                t = t_tuple[0]
                if not t or sip.isdeleted(t):
                    active_tasks.remove(t_tuple)
                    continue

                if t.status() in (QgsTask.Complete, QgsTask.Terminated):
                    active_tasks.remove(t_tuple)
                    
            QCoreApplication.processEvents()
            sleep(0.5)

    def _export_base_layers(self, num_workers: int):
        """Export base vector layers to FlatGeobuf format utilizing Tasks."""
        output_datasets = {flat_rule.output_dataset: [] for flat_rule in self.flattened_rules}
        
        tasks_meta = []
        unique_layer_ids = set()
        
        for flat_rule in self.flattened_rules:
            output_datasets[flat_rule.output_dataset].append(flat_rule)
            layer_id = flat_rule.layer.id()
            
            if layer_id not in unique_layer_ids:
                output_path = join(self.utils_dir, f"map_layer_{layer_id}.parquet")
                if not exists(output_path):
                    input_source = flat_rule.layer.source()
                    # extent_wkt = self.extent.asWktCoordinates().replace(",", "")
                    extent = self.extent
                    output_crs = f'EPSG:{_TILING_SCHEME["EPSG_CRS"]}'
                    
                    task = _ParallelExportTask(
                        f"Exporting layer {flat_rule.layer.name()}",
                        self._process_base_layer_thread,
                        input_source, output_path, extent, output_crs
                    )
                    tasks_meta.append((task, None))
                unique_layer_ids.add(layer_id)
                
        if tasks_meta:
            self._run_tasks(tasks_meta, num_workers)
            
        return output_datasets

    def _process_base_layer_thread(self, input_source, output_path, extent, output_crs) -> str:
        """Run inside QgsTask: Thread-safe base layer creation."""
        # base_options = f"-dim XY -explodecollections -t_srs {output_crs} -skipinvalid -skipfailures"
        # spat_options = f"-spat {extent} -spat_srs {output_crs}"
        # options = f"{base_options} {spat_options}"
        # params = {"INPUT": input_source, "OPTIONS": options, "OUTPUT": output_path}
        # self._run_alg("convertformat", "gdal", **params)

        fix_geom_linework_params = {"INPUT": input_source, 'METHOD':0}
        fix_geom_linework = self._run_alg("fixgeometries", "native", **fix_geom_linework_params)

        fix_geom_structure_params = {"INPUT": fix_geom_linework, 'METHOD':1}
        fix_geom_structure = self._run_alg("fixgeometries", "native", **fix_geom_structure_params)

        reproject_params = {"INPUT": fix_geom_structure, "TARGET_CRS": QgsCoordinateReferenceSystem(output_crs)}
        reprojected = self._run_alg("reprojectlayer", "native", **reproject_params)
        
        clip_params = {"INPUT": reprojected, "EXTENT": extent, "CLIP": False}
        clipped = self._run_alg("extractbyextent", "native", **clip_params)
        
        singleparts_params = {"INPUT": clipped}
        singleparted = self._run_alg("multiparttosingleparts", "native", **singleparts_params)

        tolerance = _TILES_CONF["GENERAL_CONF"]["DATA_SIMPLIFICATION_TOLERANCE"]
        simplify_params = {"INPUT": singleparted, 'METHOD':0, 'TOLERANCE': tolerance,  "OUTPUT": output_path}
        simplified = self._run_alg("simplifygeometries", "native", **simplify_params)


        return simplified

    def _export_rule_thread_safe(self, flat_rules) -> Optional[str]:
        """Export group of rules sharing the same dataset inside a thread-safe task."""
        flat_rule = flat_rules[0]
        source_path = join(self.utils_dir, f"map_layer_{flat_rule.layer.id()}.parquet")
        
        if not exists(source_path):
            return None
            
        temp_layer = QgsVectorLayer(source_path, "temp", "ogr")
        if not temp_layer.isValid() or temp_layer.featureCount() <= 0:
            return None
            
        self._updated_map_scale_variable(flat_rules)
        fields = self._create_expression_fields(flat_rules)
        
        if flat_rule.get_attr("t") == 1:
            fields = self._add_label_expression_field(flat_rule, fields)
            
        transformation = self._get_geometry_transformation(flat_rule)
        if transformation:
            return self._apply_field_mapping_thread_safe(source_path, fields, transformation, flat_rule)
            
        return None

    def _apply_field_mapping_thread_safe(
        self, source_path: str, fields: Optional[list], transformation, flat_rule
    ) -> Optional[str]:
        """Apply field mapping and geometry transformation without touching UI memory layers."""
        output_dataset = join(self.utils_dir, f"{flat_rule.output_dataset}.parquet")
        if exists(output_dataset):
            return output_dataset

        field_mapping = [(4, '"fid"', f"{self.FIELD_PREFIX}_fid")]
        if fields:
            field_mapping.extend(fields)
        rule_description = f"'{flat_rule.get_description()}'"
        field_mapping.append((10, rule_description, f"{self.FIELD_PREFIX}_description"))
        
        temp_layer = QgsVectorLayer(source_path, "temp", "ogr")
        if self.include_required_fields_only != 0:
            all_fields = [(f.type(), f'"{f.name()}"', f"{f.name()}") for f in temp_layer.fields()]
            field_mapping.extend(all_fields)

        field_mapping = [{"type": f[0], "expression": f[1], "name": f[2]} for f in field_mapping]
        current_input = source_path

        if flat_rule.rule.filterExpression():
            filtered_output = join(self.utils_dir, f"filt_{uuid4().hex[:8]}.parquet")
            params = {
                "INPUT": current_input,
                "EXPRESSION": flat_rule.rule.filterExpression(),
                "OUTPUT": filtered_output
            }
            extracted_path = self._run_alg("extractbyexpression", "native", **params)
            
            check_layer = QgsVectorLayer(extracted_path, "check", "ogr")
            if not check_layer.isValid() or check_layer.featureCount() <= 0:
                return None
            current_input = extracted_path

        refactored = self._run_alg("refactorfields", INPUT=current_input, FIELDS_MAPPING=field_mapping)
        return self._apply_transformation_thread_safe(refactored, transformation, output_dataset)

    def _apply_transformation_thread_safe(
        self, current_input: str, transformation, output_dataset: str
    ) -> Optional[str]:
        """Apply geometry transformation purely resolving file paths inside tasks."""
        geom_type, expression = abs(transformation[0] - 2), transformation[1]
        params = {"INPUT": current_input, "OUTPUT_GEOMETRY": geom_type, "EXPRESSION": expression}
        
        geom_transformed = self._run_alg("geometrybyexpression", **params)
        
        check_layer = QgsVectorLayer(geom_transformed, "check", "ogr")
        if not check_layer.isValid() or check_layer.featureCount() <= 0:
            return None

        # self._run_alg("convertformat", "gdal", **params)

        removenull_parameters = {"INPUT": check_layer, "REMOVE_EMPTY": True}
        removed_nulls = self._run_alg("removenullgeometries", "native", **removenull_parameters)

        singleparts_params = {"INPUT": removed_nulls, "OUTPUT": output_dataset}
        singleparted = self._run_alg("multiparttosingleparts", "native", **singleparts_params)

        return singleparted

    def _updated_map_scale_variable(self, flat_rules):
        """update expressions included @map_scale and replace it with the curren_scale"""
        for flat_rule in flat_rules:
            rule_type = flat_rule.get_attr("t")
            zoom_scale = str(ZoomLevels.zoom_to_scale(flat_rule.get_attr("o")))
            if rule_type == 1 and flat_rule.rule.settings():
                settings = flat_rule.rule.settings()
                label_exp = settings.getLabelExpression().expression()
                if label_exp:
                    updated_exp = label_exp.replace("@map_scale", zoom_scale)
                    settings.fieldName = updated_exp
                if settings.geometryGeneratorEnabled:
                    updated_exp = settings.geometryGenerator.replace("@map_scale", zoom_scale)
                    settings.geometryGenerator = updated_exp
            else:
                symbol = flat_rule.rule.symbol()
                if not symbol:
                    continue
                symbol_layers = flat_rule.rule.symbol().symbolLayers()
                for layer in filter(lambda x: x.layerType() == "GeometryGenerator", symbol_layers):
                    generator_exp = layer.geometryExpression()
                    updated_exp = generator_exp.replace("@map_scale", zoom_scale)
                    layer.setGeometryExpression(updated_exp)

    def _get_polygon_centroids_expression(self):
        """Get polygon centroids expression based on
        user perference - visible polygon/whole polygon"""
        if self.cent_source == 1:
            extent_wkt = self.extent.asWktPolygon()
            polygons = f"intersection(@geometry, geom_from_wkt('{extent_wkt}'))"
        else:
            polygons = "@geometry"
        centroids = f"with_variable('source', {polygons}, if(intersects(centroid(@source), @source), centroid(@source),  point_on_surface(@source)))"
        return centroids

    def _add_label_expression_field(self, flat_rule: 'FlattenedRule', fields: list) -> list:
        """Add label expression as a calculated field."""
        field_name = f"{self.FIELD_PREFIX}_label"
        if not flat_rule.rule.settings():
            return fields
        label_exp = flat_rule.rule.settings().getLabelExpression().expression()
        if not label_exp:
            return fields
        filter_exp = f'"{label_exp}"' if not flat_rule.rule.settings().isExpression else label_exp
        fields.append([10, filter_exp, field_name])
        flat_rule.rule.settings().isExpression = False
        flat_rule.rule.settings().fieldName = field_name
        return fields

    def _get_geometry_transformation(self, flat_rule: 'FlattenedRule') -> Union[str, Tuple, None]:
        """Determine geometry transformation needed for rule."""
        transformation = None
        rule_type = flat_rule.get_attr("t")
        if rule_type == 0 and flat_rule.rule.symbol():
            transformation = self._get_renderer_transformation(flat_rule)
        elif rule_type == 1:
            transformation = self._get_labeling_transformation(flat_rule)
        if not transformation:
            return None
        extent_wkt = self.extent.asWktPolygon()
        clipped_geom = f"with_variable('clip',intersection({transformation[1]}, geom_from_wkt('{extent_wkt}')), if(not is_empty_or_null(@clip), @clip, NULL))"
        transformation[1] = clipped_geom
        return tuple(transformation)

    def _get_labeling_transformation(self, flat_rule: 'FlattenedRule') -> Union[Tuple, str, None]:
        """Get geometry transformation for labeling rules."""
        settings = flat_rule.rule.settings()
        target_geom = flat_rule.get_attr("g")
        transform_expr = "@geometry"

        if settings and settings.geometryGeneratorEnabled:
            target_geom = settings.geometryGeneratorType
            transform_expr = settings.geometryGenerator
            settings.geometryGeneratorEnabled = False
            flat_rule.set_attr("c", target_geom)
        elif target_geom == 2:  # Polygon to centroid
            flat_rule.set_attr("c", 0)
            target_geom = 0
            transform_expr = self._get_polygon_centroids_expression()
        return [target_geom, transform_expr]

    def _get_renderer_transformation(self, flat_rule: 'FlattenedRule') -> Union[Tuple, str, None]:
        """Get geometry transformation for renderer rules."""
        symbol = flat_rule.rule.symbol()
        if not symbol:
            return None
        symbol_layer = flat_rule.rule.symbol().symbolLayers()[0]
        target_geom = flat_rule.get_attr("g")
        transform_expr = "@geometry"

        if symbol_layer.layerType() == "GeometryGenerator":
            target_geom = symbol_layer.subSymbol().type()
            transform_expr = symbol_layer.geometryExpression()
        else:
            target_geom = flat_rule.get_attr("c")
            source_geom = flat_rule.get_attr("g")
            if source_geom != target_geom:
                if target_geom == 0:
                    transform_expr = self._get_polygon_centroids_expression()
                elif target_geom == 1:
                    transform_expr = "boundary(@geometry)"
        return [target_geom, transform_expr]

    def _create_expression_fields(self, flat_rules) -> list:
        """Create calculated fields from data-driven properties."""
        fields = []
        for flat_rule in flat_rules:
            min_zoom = str(ZoomLevels.zoom_to_scale(flat_rule.get_attr("o")))
            fields_list = DataDefinedPropertiesFetcher(flat_rule.rule, min_zoom).fetch()
            if not fields_list:
                continue
            fields.extend(fields_list)
        return fields

    def _run_alg(self, algorithm: str, algorithm_type: str = "native", **params):
        """Runs the processing algorithms securely inside isolated thread contexts."""
        # CRITICAL FIX: Instantiate an isolated Context and Feedback for the current thread.
        # This prevents `processing.run()` from attempting to default to the main thread's QgsProject 
        # map layer registry (which is precisely what triggers the fatal access violation).
        context = QgsProcessingContext()
        context.setExpressionContext(QgsProject.instance().createExpressionContext())
        
        task_feedback = QgsProcessingFeedback()
        
        if not params.get("OUTPUT") or params.get("OUTPUT") == "TEMPORARY_OUTPUT":
            ext = ".parquet" if algorithm == "convertformat" else ".gpkg"
            params["OUTPUT"] = join(self.utils_dir, f"temp_{uuid4().hex[:8]}{ext}")
            
        output = run(
            f"{algorithm_type}:{algorithm}", 
            params,
            context=context,
            feedback=task_feedback
        )["OUTPUT"] 
        
        if isinstance(output, QgsVectorLayer):
            return output.source()
            
        return output


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
            useable= False
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


class QGIS2VectorTiles:
    """
    Main adapter class that orchestrates the conversion process from QGIS
    vector layer styling to vector tiles format.
    """

    def __init__(
        self,
        min_zoom: int = 0,
        max_zoom: int = 5,
        extent=None,
        output_dir: str = None,
        include_required_fields_only=0,
        output_type: str = "xyz",
        cpu_percent: int = 100,
        cent_source: int = 0,
        background_type:int = 0,
        feedback: QgsProcessingFeedback = None,
    ):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.extent = extent or iface.mapCanvas().extent()
        self.utils_dir = self._get_utils_dir()
        self.output_dir = output_dir or self.utils_dir
        self.include_required_fields_only = include_required_fields_only
        self.output_type = output_type.lower()
        self.cpu_percent = min(cpu_percent, 90)
        self.cent_source = cent_source
        self.background_type = background_type
        self.feedback = feedback or QgsProcessingFeedback()

    def convert_project_to_vector_tiles(self) -> Optional[QgsVectorTileLayer]:
        """
        Convert current QGIS project to vector tiles format.

        Returns:
            QgsVectorTileLayer: The created vector tiles layer, or None if failed
        """
        try:
            self._clear_project()
            temp_dir = self._create_temp_directory()
            self._log(". Starting conversion process...")
            start_time = perf_counter()

            # Step 1: Flatten all rules
            self._log(". Flattening rules...")
            rules = self._flatten_rules()
            if not rules:
                self._log(". No visible vector layers found in project.")
                return

            flatten_finish_time = perf_counter()
            flatten_time = self._elapsed_minutes(start_time, flatten_finish_time)
            self._log(f". Successfully extracted {len(rules)} rules " f"({flatten_time} minutes).")
            tiles_uri = layers = None
            # Step 2: Export rules to datasets
            self._log(". Exporting rules to datasets...")
            layers, rules = self._export_rules(rules)
            export_finish_time = perf_counter()
            export_time = self._elapsed_minutes(flatten_finish_time, export_finish_time)
            self._log(f". Successfully exported {len(layers)} layers ({export_time} minutes).")

            # Step 3: Generate and style tiles
            if self._has_features(layers):
                self._log(". Generating tiles...")
                tiles_uri = self._generate_tiles(layers, temp_dir)
                tiles_finish_time = perf_counter()
                tiles_time = self._elapsed_minutes(export_finish_time, tiles_finish_time)
                self._log(f". Successfully generated tiles ({tiles_time} minutes).")

            self._log(". Loading and styling tiles...")
            styled_layer = self._sytle_tiles(rules, temp_dir, tiles_uri)
            self._log(". Successfully loaded and styled tiles.")

            self._log(". Exporting tiles style to MapLibre style package...")
            self._export_maplibre_style(temp_dir, styled_layer)
            self._log(". Successfully exported MapLibre style package.")

            total_time = self._elapsed_minutes(start_time, perf_counter())
            self._log(f". Process completed successfully ({total_time} minutes).")
            self._clear_project()
            self.serve_tiles(temp_dir)
        except QgsProcessingException as e:
            self._log(f". Processing failed: {str(e)}")
            self._clear_project()
            return None

    def _clear_project(self):
        """Clear temp layers which are not visible in the project legend"""
        legend_layers = QgsProject.instance().layerTreeRoot().findLayers()
        legend_layers_ids = [layer.layer().id() for layer in legend_layers]
        for layer in list(QgsProject.instance().mapLayers().values()):
            if layer.id() not in legend_layers_ids:
                QgsProject.instance().removeMapLayer(layer)

    def _get_utils_dir(self) -> str:
        """Clear utils dir"""
        for subfile in listdir(QgsProcessingUtils.tempFolder()):
            try:
                path = join(QgsProcessingUtils.tempFolder(), subfile)
                rmtree(path)
            except (PermissionError, NotADirectoryError):
                continue
        utils_dir = join(QgsProcessingUtils.tempFolder(), f"q2styledtiles_{uuid4().hex}")
        makedirs(utils_dir, exist_ok=True)
        return utils_dir

    def _create_temp_directory(self) -> str:
        """Create temporary directory for processing."""
        temp_dir = join(self.output_dir, datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f"))
        makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def _flatten_rules(self) -> List[FlattenedRule]:
        """Flatten all rules from project layers."""
        flattener = RulesFlattener(self.min_zoom, self.max_zoom, self.utils_dir, self.feedback)
        return flattener.flatten_all_rules()

    def _export_rules(self, rules: List[FlattenedRule]) -> List[QgsVectorLayer]:
        """Export rules to vector layers."""
        exporter = RulesExporter(
            rules,
            self.extent,
            self.include_required_fields_only,
            self.max_zoom,
            self.utils_dir,
            self.cent_source,
            self.feedback,
        )
        return exporter.export()

    def _has_features(self, layers: List[QgsVectorLayer]) -> bool:
        """Check if any layer has features."""
        return any(layer.featureCount() > 0 for layer in layers)

    def _generate_tiles(self, layers: List[QgsVectorLayer], temp_dir: str) -> str:
        """Generate vector tiles."""
        generator = GDALTilesGenerator(
            layers, temp_dir, self.output_type, self.extent, self.cpu_percent, self.feedback
        )
        tiles_uri, min_zoom = generator.generate()
        self.min_zoom = min_zoom
        return tiles_uri

    def _sytle_tiles(self, rules, temp_dir, tiles_uri) -> Optional[QgsVectorTileLayer]:
        """Style tiles."""
        styler = TilesStyler(rules, temp_dir, tiles_uri)
        styled_layer = styler.apply_styling()
        return styled_layer

    def _export_maplibre_style(self, temp_dir, styled_layer):
        """Export MapLibre style."""
        exporter = QgisMapLibreStyleExporter(temp_dir, styled_layer, self.background_type)
        exporter.export()

    def _log(self, message: str):
        """Log message to feedback or console."""
        if __name__ != "__console__":
            self.feedback.pushInfo(message)
        else:
            print(message)

    @staticmethod
    def _elapsed_minutes(start: float, end: float) -> str:
        """Calculate elapsed time in minutes."""
        return f"{round((end - start) / 60, 2)}"

    def serve_tiles(self, output_folder):
        """Serve generated tiles using a simple HTTP server (cross-platform)."""
        # Create utilities direcotry
        utilitie_dir_name = 'utilities'
        utilities = join(output_folder, utilitie_dir_name)
        makedirs(utilities, exist_ok=True)
        
        # Get the extent of the tiles in EPSG:4326 for the MapLibre viewer
        src_crs = QgsCoordinateReferenceSystem("EPSG:3857")
        dest_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        coord_transform = QgsCoordinateTransform(
            src_crs, dest_crs, QgsProject.instance().transformContext()
        )
        center_3857 = self.extent.center()
        center_4326 = coord_transform.transform(center_3857)
        center_corrd = f"[{center_4326.x()}, {center_4326.y()}]"

        # Write the MapLibre viewer HTML with the correct bounds and serve the tiles
        with open(_VIEWER, "r", encoding="utf-8") as html_source:
            html_content = html_source.read()
            html_content = html_content.replace("map.setZoom(2)", f"map.setZoom({self.min_zoom})")
            html_content = html_content.replace(
                "map.setCenter([32, 32])", f"map.setCenter({center_corrd})"
            )
            html_content = html_content.replace('_PORT/utilities_dir_name', f'{_PORT}/{utilitie_dir_name}')

        html_source.close()
        with open(join(utilities, "maplibre_viewer.html"), "w", encoding="utf-8") as html_copy:
            html_copy.write(html_content)
        html_copy.close()

        # Copy maplibre-gl js and css files
        copy2(f'{_MAPLIBRE}.js', utilities)
        copy2(f'{_MAPLIBRE}.css', utilities)

        # Create platform-specific script to serve the tiles and open the viewer
        system = platform.system()

        if system == "Windows":
            self._create_windows_script(output_folder, utilities)
        else:  # Linux and macOS
            self._create_unix_script(output_folder, utilities)

    def _create_windows_script(self, output_folder, utilities_folder):
        """Create and execute Windows batch script."""
        python_exe = join(prefix, "python3.exe")
        
        utilities_dir_name = basename(utilities_folder)
        html_path = f'http://localhost:{_PORT}/{utilities_dir_name}/maplibre_viewer.html'
        activator = join(utilities_folder, "activate_server.bat")
        with open(activator, "w", encoding="utf-8") as bat:
            bat.write(
                "@echo off\n"
                f'for /f "tokens=5" %%a in (\'netstat -aon ^| find ":{_PORT}" ^| find "LISTENING"\') do (\n'  # pylint: disable=C0301
                "  taskkill /F /PID %%a >nul 2>&1\n"
                ")\n"
                f'start /B "" "{python_exe}" -m http.server {_PORT} -d "{output_folder}"'
                "\ntimeout /t 2 /nobreak >nul\n"
            f'start "" "{html_path}"\n'
            )
        
        launcher = join(output_folder, "launch_viewer.vbs")
        
        with open(launcher, "w", encoding="utf-8") as bat:
            bat.write(
            'Set WshShell = CreateObject("WScript.Shell")\n'
            f'WshShell.Run  "cmd /c ""{activator}""" , 0'
            '\nSet WshShell = Nothing'
            )
        command = ["wscript.exe", launcher]
        subprocess.Popen(command)

    def _create_unix_script(self, output_folder, utilities_folder):
        """Create and execute Unix/Linux/macOS shell script."""
        html_path = f'http://localhost:{_PORT}/{basename(utilities_folder)}/maplibre_viewer.html'
        if platform.system() == "Linux":
            python_exe = join(prefix, "bin", "python3")
        else:
            python_exe = join(prefix, "python3")

        launcher = join(output_folder, "launch_viewer.sh")
        with open(launcher, "w", encoding="utf-8") as sh:
            sh.write(
                "#!/bin/bash\n"
                f"PID=$(lsof -ti:{_PORT})\n"
                'if [ ! -z "$PID" ]; then\n'
                "  kill -9 $PID\n"
                "fi\n"
                # start server
                f'"{python_exe}" -m http.server {_PORT} &\n'
                # wait
                "sleep 2\n"
                # open browser
                "if command -v xdg-open >/dev/null 2>&1; then\n"
                f"    xdg-open {html_path}\n"
                "elif command -v open >/dev/null 2>&1; then\n"
                f"    open {html_path}\n"
                "fi\n"
            )
        subprocess.Popen(["bash", launcher], cwd=output_folder)
        


# Main execution for QGIS console
if __name__ == "__console__":
    adapter = QGIS2VectorTiles(output_dir=QgsProcessingUtils.tempFolder())
    adapter.convert_project_to_vector_tiles()