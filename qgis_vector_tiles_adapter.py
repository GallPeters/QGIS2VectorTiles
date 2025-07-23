"""
Extracts and flattens renderer and labeling rules from vector layers in the
current QGIS project with unified processing logic.
"""

from os.path import join
from os import cpu_count
from tempfile import mkdtemp, TemporaryDirectory as tempd
from osgeo import gdal, ogr, osr
from dataclasses import dataclass
from typing import Union
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


class Zooms:
    """Manages predefined zoom levels and scale snapping."""

    LEVELS = [
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
    def _snap_scale(cls, scale, snap_up=True) -> float:
        """Snap scale to the nearest zoom level."""
        if scale <= 0:
            return cls.LEVELS[0] if snap_up else cls.LEVELS[-1]

        for i, level in enumerate(cls.LEVELS):
            if scale >= level:
                if i == 0 or not snap_up:
                    return level
                return level if not snap_up else cls.LEVELS[i - 1]

        return cls.LEVELS[-1]

    @classmethod
    def _zoom(cls, scale):
        """Get scale's corresponding zoom"""
        zoom = cls.LEVELS.index(scale)
        return f"{'0' if zoom < 10 else ''}{zoom}"

    @classmethod
    def _scale(cls, zoom):
        """Get scale's corresponding zoom"""
        if zoom < 0 or zoom > len(cls.LEVELS):
            return
        scale = cls.LEVELS[int(zoom)]
        return scale


@dataclass
class FlatRule:
    """Represents a flattened rule with all inherited properties."""

    rule: Union[QgsRuleBasedLabeling.Rule, QgsRuleBasedRenderer.Rule]
    lyr: QgsVectorLayer

    def _get(self, char):
        """Extract rule attribute value from description by given"""
        init = self.rule.description().find(char) + 1
        if init == 0:
            return
        return int(self.rule.description()[init : init + 2])

    def _set(self, char, num):
        """Set rule attribute"""
        num = int(num)
        new = f"{char}{'0' if num < 10 else ''}{num}"
        current = self._get(char)
        if current != None:
            current = f"{char}{current if current > 10 else f'0{current}'}"
            desc = self.rule.description().replace(current, new)
        else:
            desc = f"{self.rule.description()}{new}"
        self.rule.setDescription(desc)


class TilesLoader:
    def __init__(self, flatrules, tiles):
        self.flatrules = flatrules
        self.tiles_lyr = self._get_tiles_lyr(tiles)
        self.renderer_styles = []
        self.labeling_styles = []

    def load_tiles(self):
        """Load tiles to canvas"""
        for flatrule in self.flatrules:
            self._style_flatrule(flatrule)
        self._apply_styles()
        return self.tiles_lyr

    def _get_tiles_lyr(self, tiles):
        """Get project layer"""
        tiles = QgsVectorTileLayer(f"type=mbtiles&url={tiles}", "MBTiles")
        return QgsProject.instance().addMapLayer(tiles)

    def _style_flatrule(self, flatrule):
        """Style mbtiles labelibg"""
        if flatrule._get("t") == 0:
            style = QgsVectorTileBasicRendererStyle()
        else:
            style = QgsVectorTileBasicLabelingStyle()
        style.setEnabled(True)
        rule_geom = flatrule._get("c")
        if rule_geom == 0:
            geom = QgsWkbTypes.PointGeometry
        elif rule_geom == 1:
            geom = QgsWkbTypes.LineGeometry
        else:
            geom = QgsWkbTypes.PolygonGeometry
        style.setGeometryType(geom)
        style.setLayerName(flatrule.rule.description())
        style.setMinZoomLevel(flatrule._get("o"))
        style.setMaxZoomLevel(flatrule._get("i"))
        style.setStyleName(flatrule.rule.description())
        if flatrule._get("t") == 0:
            sym = flatrule.rule.symbol()
            subsym = sym.symbolLayers()[-1].subSymbol()
            if subsym:
                self._copy_dd_properties(sym, subsym)
                self._copy_dd_properties(sym.symbolLayers()[-1], subsym.symbolLayers()[-1])
                sym = subsym
            style.setSymbol(sym.clone())
            target = self.renderer_styles
        else:
            settings_clone = QgsPalLayerSettings(flatrule.rule.settings())
            style.setLabelSettings(settings_clone)
            target = self.labeling_styles
        target.append(style)

    def _copy_dd_properties(self, obj1, obj2):
        """Copy Data Driven Properties between one object to another"""
        ddp1 = obj1.dataDefinedProperties()
        ddp2 = obj2.dataDefinedProperties()
        for k in obj1.propertyDefinitions():
            v = ddp1.property(k)
            ddp2.setProperty(k, v)
            ddp2.property(k).setActive(True)

    def _apply_styles(self):
        """Apply styles on lyr"""
        renderer = QgsVectorTileBasicRenderer()
        renderer.setStyles(self.renderer_styles)
        labeling = QgsVectorTileBasicLabeling()
        labeling.setStyles(self.labeling_styles)
        self.tiles_lyr.setRenderer(renderer)
        self.tiles_lyr.setLabeling(labeling)


class TilesGenerator:
    """Generate MBTiles from GeoJSON layers using GDAL MVT driver."""

    def __init__(self, output_dir, cpu_percent, gpkg):
        self.cpu_percent = cpu_percent
        self.gpkg = gpkg
        self.output_dir = output_dir

    def generate_tiles(self):
        """Generate MBTiles file from configured layers."""
        # Set GDAL threading options
        cpu_num = str(max(1, int(cpu_count() * self.cpu_percent / 100)))
        gdal.SetConfigOption("GDAL_NUM_THREADS", cpu_num)

        # Create Web Mercator spatial reference (EPSG:3857)
        web_mercator = osr.SpatialReference()
        web_mercator.ImportFromEPSG(3857)

        # Create the MVT dataset
        output_mbtiles = join(self.output_dir, f"tiles.mbtiles")

        driver = gdal.GetDriverByName("MVT")
        creation_options = [
                "TILE_FORMAT=MVT",
                "SIMPLIFICATION=0",
                "EXTENT=16384"
            ]
        ds = driver.Create(output_mbtiles, 0, 0, 0, gdal.GDT_Unknown, options=creation_options)

        # Process each layer
        gpkg = ogr.Open(self.gpkg)
        for index in range(gpkg.GetLayerCount()):
            src_lyr = gpkg.GetLayer(index)
            lyr_name = src_lyr.GetName()
            min_zoom = int(lyr_name.split("o")[1][:2])
            max_zoom = int(lyr_name.split("i")[1][:2])

            # Create layer in MVT dataset with Web Mercator projection
            lyr = ds.CreateLayer(lyr_name, srs=web_mercator, geom_type=ogr.wkbUnknown)

            # Set layer options
            lyr.SetMetadataItem("MINZOOM", str(min_zoom))
            lyr.SetMetadataItem("MAXZOOM", str(max_zoom))
            lyr.SetMetadataItem("SIMPLIFICATION", '1.0')
            lyr.SetMetadataItem("SIMPLIFICATION_MAX_ZOOM", '4')
            
            # Create coordinate transformation to Web Mercator
            src_srs = src_lyr.GetSpatialRef()
            transform = None
            if not src_srs.IsSame(web_mercator):
                transform = osr.CoordinateTransformation(src_srs, web_mercator)

            # Copy field definitions from source
            src_defn = src_lyr.GetLayerDefn()
            for i in range(src_defn.GetFieldCount()):
                field_defn = src_defn.GetFieldDefn(i)
                lyr.CreateField(field_defn)

            # Copy features with coordinate transformation
            feature_count = 0
            for src_feature in src_lyr:
                # Create new feature
                dst_feature = ogr.Feature(lyr.GetLayerDefn())

                # Copy and transform geometry
                geom = src_feature.GetGeometryRef()
                if geom:
                    geom_copy = geom.Clone()
                    if transform:
                        geom_copy.Transform(transform)
                    dst_feature.SetGeometry(geom_copy)

                # Copy attributes
                for i in range(src_defn.GetFieldCount()):
                    field_name = src_defn.GetFieldDefn(i).GetName()
                    if dst_feature.GetFieldIndex(field_name) >= 0:
                        dst_feature.SetField(field_name, src_feature.GetField(i))

                # Add feature to layer
                lyr.CreateFeature(dst_feature)
                feature_count += 1

        # Set dataset metadata
        ds.SetMetadataItem("name", "Generated Tiles")
        ds.SetMetadataItem("description", "MBTiles")
        ds.SetMetadataItem("version", "1.0.0")
        ds.SetMetadataItem("format", "pbf")
        ds.SetMetadataItem("type", "overlay")

        # Close dataset to finalize
        ds = None
        return output_mbtiles


class FlatRulesPackager:
    """Export rules as geoparquets to a destination folder."""

    _FIELDS_PREFIX = "qvta"
    _GEOM_ATTRS = {
        "$area": "area_meters",
        "area(@geometry)": "area_degrees",
        "$length": "length_meters",
        "length(@geometry)": "length_degrees",
    }

    def __init__(self, flatrules, extent, output_dir, all_fields):
        self.flatrules = flatrules
        self.extent = extent
        self.output_dir = output_dir
        self.all_fields = all_fields
        self.flatrule_lyrs = []

    def package_flatrules(self):
        """Export all rules using the matching processing"""
        for flatrule in self.flatrules:
            self._flatrule_to_lyr(flatrule)
        return self._package_lyrs()

    def _dd_properties_fetcher(self, flatrule, attr, new_attr=None):
        """Get data driven properties frm object"""
        rule_type = flatrule._get("t")
        if rule_type == 0:
            objects = [flatrule.rule.symbol()] + flatrule.rule.symbol().symbolLayers()
        else:
            objects = [flatrule.rule.settings()]
        properties = []
        for obj in objects:
            ddp = obj.dataDefinedProperties()
            for idx in obj.propertyDefinitions():
                prop = ddp.property(idx)
                if prop:
                    prop_exp = prop.expressionString()
                    if attr in prop_exp:
                        properties.append(prop)
                        if new_attr:
                            prop.setExpressionString(prop_exp.replace(attr, new_attr))
        return properties

    def _flatrule_to_lyr(self, flatrule):
        """Export dataset using the relevant file keeping only required fields"""
        lyr = self._clean_flatrule_lyr(flatrule)
        rule = flatrule.rule

        # Remove unneccessary features
        expression = rule.filterExpression()
        if expression:
            lyr = self._run("extractbyexpression", INPUT=lyr, EXPRESSION=expression)

        # Get geometry attributes
        for exp, new_field in self._GEOM_ATTRS.items():
            field_name = f"{self._FIELDS_PREFIX}_{new_field}"
            if self._dd_properties_fetcher(flatrule, exp, f'"{field_name}"'):
                lyr = self._run(
                    "fieldcalculator", INPUT=lyr, FIELD_NAME=field_name, FORMULA=exp, FIELD_TYPE=0
                )

        # Remove unneccessary fields
        if not self.all_fields:
            required_fields = list(self._get_required_fields(flatrule)) + [f"{self._FIELDS_PREFIX}_fid"]
            lyr = self._run("retainfields", INPUT=lyr, FIELDS=required_fields)

        # Replace geometry If required
        geom, exp = self._get_geometry_convertion(flatrule)
        if exp:
            geom = abs(geom - 2)
            lyr = self._run(
                "geometrybyexpression", INPUT=lyr, OUTPUT_GEOMETRY=geom, EXPRESSION=exp
            )

        # Insert rule to output dict
        lyr.setName(flatrule.rule.description())
        self.flatrule_lyrs.append(lyr)

    def _get_geometry_convertion(self, flatrule):
        """Get target geometry type and convertion expression"""
        target_geom = geom_exp = None
        if flatrule._get("t") == 1:
            settings = flatrule.rule.settings()
            if settings.geometryGeneratorEnabled:
                target_geom = settings.geometryGeneratorType
                geom_exp = settings.geometryGenerator
                settings.geometryGeneratorEnabled = False
            elif flatrule._get("g") == 2:
                target_geom = 0
                geom_exp = "centroid(@geometry)"
            if target_geom != None:
                flatrule._set("c", target_geom)
        else:
            symlyr = flatrule.rule.symbol().symbolLayers()[0]
            if symlyr.layerType() == "GeometryGenerator":
                target_geom = symlyr.subSymbol().type()
                geom_exp = symlyr.geometryExpression()
            elif flatrule._get("g") != flatrule._get("c"):
                target_geom = flatrule._get("c")
                geom_exp = f'{"centroid" if target_geom ==0 else "boundary"}(@geometry)'
        return target_geom, geom_exp

    def _clean_flatrule_lyr(self, flatrule):
        """Extract and repair rules source layers"""
        lyr = flatrule.lyr
        extent = self.extent or lyr.extent()
        unique_id = self._run(
            "fieldcalculator",
            INPUT=lyr,
            FIELD_NAME=f"{self._FIELDS_PREFIX}_fid",
            FORMULA=f"'{lyr.name}_' || @id",
            FIELD_TYPE=2
        )
        extracted = self._run("extractbyextent", INPUT=unique_id, EXTENT=extent)
        fix_network = self._run("fixgeometries", INPUT=extracted, METHOD=0)
        fix_struct = self._run("fixgeometries", INPUT=fix_network, METHOD=1)
        return fix_struct

    def _get_required_fields(self, flatrule):
        """Get list of target datasets and its required fields"""
        rule = flatrule.rule
        if flatrule._get("t") == 0:
            return rule.symbol().usedAttributes(QgsRenderContext())
        return rule.settings().referencedFields(QgsRenderContext())

    def _add_additional_fields(self):
        """Adds geometry attributes if geometry type is being changed"""
        pass

    def _package_lyrs(self):
        """Package all rule temporary layers into a single gpkg file."""
        gpkg = join(f"{self.output_dir}", "rules.gpkg")
        self._run("package", LAYERS=self.flatrule_lyrs, SAVE_STYLES=False, OUTPUT=gpkg)
        return gpkg

    def _run(self, alg_id, alg_type="native", **params):
        """Run processing tools"""
        if not params.get("OUTPUT"):
            params["OUTPUT"] = "TEMPORARY_OUTPUT"
        return processing.run(f"{alg_type}:{alg_id}", params)["OUTPUT"]


class RulesFlattener:
    """Extracts and flattens QGIS vector layer rules."""

    RULE_TYPES = {0: "renderer", 1: "labeling"}

    def __init__(self, min_zoom, max_zoom):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.project = QgsProject.instance()
        self.lyrs_root = self.project.layerTreeRoot()
        self.flatrules = []

    def flat_rules(self):
        """Extract all rules from visible vector layers."""
        lyrs = self.project.mapLayers().values()

        for idx, lyr in enumerate(lyrs):
            if not self._is_relevant_lyr(lyr):
                continue

            self._process_lyr_rules(lyr.clone(), idx)
        return self.flatrules

    def _is_relevant_lyr(self, lyr) -> bool:
        """Check if layer is a visible vector layer."""
        is_vector = lyr.type() == 0 and lyr.geometryType() != 4
        is_visible = self.lyrs_root.findLayer(lyr.id()).isVisible()
        return is_vector and is_visible

    def _process_lyr_rules(self, lyr, lyr_idx):
        """Process both renderer and labeling rules for a layer."""
        for rule_type in self.RULE_TYPES:
            rule_system = self._get_rule_system(lyr, rule_type)
            if not rule_system:
                continue
            root_rule = self._get_root(rule_system, lyr, lyr_idx, rule_type)
            if root_rule:
                self._flat(lyr, lyr_idx, root_rule, rule_type, 0, 0)

    def _get_rule_system(self, lyr, rule_type):
        """Get or convert layer system to rule-based."""
        system = lyr.renderer() if rule_type == 0 else lyr.labeling()

        if not system or rule_type == 1 and not lyr.labelsEnabled():
            return None

        # Return if already rule-based
        if isinstance(system, (QgsRuleBasedRenderer, QgsRuleBasedLabeling)):
            return system

        # Convert to rule-based
        if rule_type == 0:
            return QgsRuleBasedRenderer.convertFromRenderer(system)
        else:
            rule = QgsRuleBasedLabeling.Rule(system.settings())
            root = QgsRuleBasedLabeling.Rule(QgsPalLayerSettings())
            root.appendChild(rule)
            return QgsRuleBasedLabeling(root)

    def _get_root(self, rule_system, lyr, lyr_idx, rule_type):
        """Prepare root rule with descriptive information."""
        # Get root rule and inherite it scale range from the layer itself
        root_rule = rule_system.rootRule()
        if lyr.hasScaleBasedVisibility():
            root_rule.setMinimumScale(lyr.minimumScale())
            root_rule.setMaximumScale(lyr.maximumScale())
        return root_rule

    def _flat(self, lyr, lyr_idx, rule, rule_type, rule_lvl, rule_idx):
        """Recursively flatten rules with inheritance."""
        # Exclude root rule
        if rule.parent():

            # Set clone attributes
            clone = self._inherit_properties(rule, rule_type)
            if not clone:
                return
            flatrule = FlatRule(clone, lyr)
            self._set_flatrule_attrs(flatrule, lyr_idx, rule_type, rule_lvl, rule_idx)

            # Split by symbol layers (renderer rules), matching renderer rules (labels only)
            # and diffrent scales (filters containing @map_scale only)
            if rule_type == 0:
                splitted_flatrules = self._split_renderer_by_symbol_lyrs(flatrule)
            else:
                splitted_flatrules = self._split_labeling_by_renderer(flatrule)
            for flatrule in splitted_flatrules:
                self.flatrules.extend(self._split_by_scales(flatrule))

        # Process children recursively
        for child_idx, child in enumerate(rule.children()):
            if not child.active():
                continue

            # If child filter is ELSE (all other values) convert it to an absolute expresison
            if child.filterExpression() == "ELSE":
                self._update_else_rule(child, rule)
            self._flat(lyr, lyr_idx, child, rule_type, rule_lvl + 1, child_idx)

    def _set_flatrule_attrs(self, flatrule, lyr_idx, rule_type, rule_lvl, rule_idx):
        """Inherit and combine rule names."""
        flatrule._set("l", lyr_idx)
        flatrule._set("t", rule_type)
        flatrule._set("d", rule_lvl)
        flatrule._set("r", rule_idx)
        flatrule._set("g", flatrule.lyr.geometryType())
        flatrule._set("c", flatrule.lyr.geometryType())
        flatrule._set("o", Zooms._zoom(flatrule.rule.minimumScale()))
        flatrule._set("i", Zooms._zoom(flatrule.rule.maximumScale()))
        flatrule._set("s" if rule_type == 0 else "f", 0)

    def _update_else_rule(self, else_rule, parent):
        """Generate ELSE rule expression by excluding all its brothers"""
        else_filter = []
        for child_rule in parent.children():
            if child_rule.active() and child_rule.filterExpression() != "ELSE":
                else_filter.append(child_rule.filterExpression())
        if else_filter:
            else_rule.setFilterExpression(f'NOT {" AND NOT ".join(else_filter)}')

    def _inherit_properties(self, rule, rule_type):
        """Inherit all properties from parent rule."""
        clone = rule.clone()
        self._inherit_scale(clone, rule, min)
        self._inherit_scale(clone, rule, max)
        if self._outside_range(clone):
            return
        self._inherit_filter(clone, rule)
        if rule_type == 0:
            self._inherit_symbol(clone, rule)
        return clone

    def _inherit_filter(self, clone, rule):
        """Combine parent and rule filters with AND logic."""
        # Combine rule and parent rule filters
        parent_filter = rule.parent().filterExpression()
        rule_filter = rule.filterExpression()
        if parent_filter and rule_filter:
            combined_filter = f"({parent_filter}) AND ({rule_filter})"
        else:
            combined_filter = parent_filter or rule_filter or ""

        # Exclude children filters from rule filter
        children_filters = []
        for child in rule.children():
            child_filter = child.filterExpression()
            if child_filter and child_filter != "ELSE":
                children_filters.append(f"({child_filter})")
        children_filters = " OR ".join(children_filters)
        if children_filters and combined_filter:
            excluded_filter = f"({combined_filter}) AND NOT ({children_filters})"
        else:
            excluded_filter = combined_filter or children_filters or ""
        clone.setFilterExpression(excluded_filter)

    def _inherit_scale(self, clone, rule, comparator):
        """Inherit scale limits using min/max comparator."""
        # Set functions
        comp_name = comparator.__name__
        attr = f"{comp_name}imumScale"
        get_scale = lambda x: Zooms._snap_scale(getattr(x, attr)(), comp_name == "min")

        # Get Scales
        rule_scale = get_scale(rule)
        parent_scale = get_scale(rule.parent())
        inherited_scale = comparator(rule_scale, parent_scale)

        # Inherite scale
        setter_name = f"set{comp_name.capitalize()}imumScale"
        getattr(clone, setter_name)(inherited_scale)

    def _outside_range(self, clone):
        """Return if rule is out of user input range"""
        return int(clone.minimumScale()) == int(clone.maximumScale()) == 0

    def _inherit_symbol(self, clone, rule):
        """Inherit symbol layers from parent."""
        clone_symbol = clone.symbol()
        parent_symbol = rule.parent().symbol()
        if parent_symbol and clone_symbol:
            for i in range(parent_symbol.symbolLayerCount()):
                symbol_lyr = parent_symbol.symbolLayer(i).clone()
                clone_symbol.appendSymbolLayer(symbol_lyr)

    def _split_renderer_by_symbol_lyrs(self, flatrule):
        """Split rule by individual symbol layers."""
        # Split only polygon renderer symbol contains outline symbollayer.
        rule = flatrule.rule
        # split_required = True
        # if not sym or flatrule.lyr.geometryType() != 2:
        #     split_required = False
        # if not any(l.type() == 1 for l in sym.symbolLayers()):
        #     split_required = False
        # if not split_required:
        #     return [flatrule]

        # Clone symbol and keep only the relevant symbol layer
        sym = rule.symbol()
        sym_lyr_count = sym.symbolLayerCount()
        splitted_flatrules = []
        for keep_idx in range(sym_lyr_count):
            flatclone = FlatRule(rule.clone(), flatrule.lyr)
            subsym = sym.symbolLayer(keep_idx).subSymbol()
            sym_type = subsym.type() if subsym else sym.type()
            flatclone._set("c", sym_type)
            flatclone._set("s", keep_idx)

            # Remove all layers except the one to keep
            for remove_idx in reversed(range(sym_lyr_count)):
                if remove_idx != keep_idx:
                    flatclone.rule.symbol().deleteSymbolLayer(remove_idx)

            splitted_flatrules.append(flatclone)

        return splitted_flatrules

    def _split_labeling_by_renderer(self, lbl_flatrule):
        """Split label rule by matching renderer rules with overlapping scales."""
        # Get relevant symbol rules
        splitted_rules = []
        sym_idx = 0
        for sym_flatrule in self.flatrules:
            if lbl_flatrule.lyr.id() != sym_flatrule.lyr.id():
                continue
            if lbl_flatrule._get("t") != 0:
                continue
            flatclone = self._inherite_sym_flatrule(lbl_flatrule, sym_flatrule, sym_idx)
            if flatclone:
                splitted_rules.append(flatclone)
            sym_idx += 1

        if not splitted_rules:
            splitted_rules = [lbl_flatrule]
        return splitted_rules

    def _inherite_sym_flatrule(self, lbl_flatrule, sym_flatrule, sym_idx):
        """Create combined rule with merged filters and constrained scales."""
        lbl_rule = lbl_flatrule.rule
        sym_rule = sym_flatrule.rule
        lbl_min, lbl_max = lbl_rule.minimumScale(), lbl_rule.maximumScale()
        sym_min, sym_max = sym_rule.minimumScale(), sym_rule.maximumScale()
        if lbl_min <= sym_min or lbl_max >= sym_max:
            # Combine filter
            flatclone = FlatRule(lbl_rule.clone(), lbl_flatrule.lyr)
            clone = flatclone.rule
            clone_filter = clone.filterExpression()
            sym_filter = sym_rule.filterExpression()
            if clone_filter and sym_filter:
                combined_filter = f"({sym_filter}) AND ({clone_filter})"
            else:
                combined_filter = sym_filter or clone_filter or ""
            clone.setFilterExpression(combined_filter)

            # Combine scale
            if lbl_min > sym_min:
                clone.setMinimumScale(sym_min)
                flatclone._set("o", Zooms._zoom(sym_min))
            if lbl_max < sym_max:
                clone.setMaximumScale(sym_max)
                flatclone._set("i", Zooms._zoom(sym_max))

            # Combine name
            flatclone._set("f", sym_idx)
            return flatclone
        return

    def _split_by_scales(self, flatrule):
        """Split rule by zoom levels if scale-dependent."""
        # Handle scale-independent rules
        rule = flatrule.rule
        filter_expr = rule.filterExpression()
        if "@map_scale" not in filter_expr:
            return [flatrule]

        # Split scale-dependent rules by zoom levels
        min_scale = rule.minimumScale()
        max_scale = rule.maximumScale()
        rule_lvls = [lvl for lvl in Zooms.LEVELS if max_scale <= lvl <= min_scale]
        rules = []

        # Split by scales
        for idx, level in enumerate(rule_lvls):
            flatclone = FlatRule(rule.clone(), flatrule.lyr)
            flatclone.rule.setMinimumScale(level)
            next_max = rule_lvls[idx] if idx < len(rule_lvls) else max_scale
            flatclone.rule.setMaximumScale(next_max)
            new_filter = filter_expr.replace("@map_scale", str(level))
            flatclone.rule.setFilterExpression(new_filter)
            flatclone._set("o", Zooms._zoom(level))
            flatclone._set("i", Zooms._zoom(next_max))
            rules.append(flatclone)

        return rules


class QVTA:
    def __init__(
        self,
        min_zoom=0,
        max_zoom=23,
        extent=iface.mapCanvas().extent(),
        output_dir=tempd(),
        cpu_percent=70,
        all_fields=False,
    ):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.extent = extent
        self.output_dir = output_dir
        self.cpu_percent = cpu_percent
        self.all_fields = all_fields

    def adapt(self):
        """Adapt project to vector tiles"""
        # Exceution
        try:
            output_dir = mkdtemp(dir=self.output_dir.name)
            print("Starting process...")
            flattener = RulesFlattener(self.min_zoom, self.max_zoom)
            print("Extracting rules...")
            flatrules = flattener.flat_rules()
            if not flatrules:
                print("Project does not contain visible valid vector layers")
                return
            print(f"Successfully extracted {len(flatrules)} rules.")
            packager = FlatRulesPackager(
                flatrules, self.extent, output_dir, self.all_fields
            )
            print(f"Packaging rules...")
            gpkg = packager.package_flatrules()
            print(f"Successfully packed rules.")
            generator = TilesGenerator(output_dir, self.cpu_percent, gpkg)
            print(f"Generating tiles...")
            tiles = generator.generate_tiles()
            print(f"Successfully generated tiles.")
            loader = TilesLoader(flatrules, tiles)
            print(f"Load tiles...")
            lyr = loader.load_tiles()
            print("Process have been finished successfully.")
        except Exception as e:
            # print(f"Error processing layer '{lyr.name()}': {e}")
            raise (e)


if __name__ == "__console__":
    qvta = QVTA()
    qvta.adapt()
