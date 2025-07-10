"""
Extracts and flattens renderer and labeling rules from vector layers in the
current QGIS project with unified processing logic.
"""

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
    def snap_to_level(cls, scale, snap_up=True) -> float:
        """Snap scale to the nearest zoom level."""
        if scale <= 0:
            return cls.LEVELS[-1] if snap_up else cls.LEVELS[0]

        for i, level in enumerate(cls.LEVELS):
            if scale >= level:
                if i == 0 or not snap_up:
                    return level
                return level if not snap_up else cls.LEVELS[i - 1]

        return cls.LEVELS[-1]

    @classmethod
    def get_zoom(cls, scale):
        zoom = cls.LEVELS.index(scale)
        return f"{'0' if zoom < 10 else ''}{zoom}"


@dataclass
class FlatRule:
    """Represents a flattened rule with all inherited properties."""

    rule: Union[QgsRuleBasedLabeling.Rule, QgsRuleBasedRenderer.Rule]
    lyr: QgsVectorLayer
    type: str
    data: Union[QgsSymbol, QgsPalLayerSettings]
    fields: list[str]
    target: str

    def get_attr(self, char):
        """Extract rule attribute value from description by given"""
        init = self.rule.description().find(char) + 1
        return int(self.rule.description()[init : init + 2])

    def set_attr(self, char, value):
        """Extract rule attribute value from description by given"""
        init = self.rule.description().find(char) + 1
        nums = self.rule.description()[init : init + 2]
        current = f"{char}{nums}"
        new = f"{char}{value}"
        self.target = self.target.replace(current, new)


class RuleExporter:
    """Export rules as geoparquets to a destination folder."""

    def __init__(
        self,
        flatrules,
        min_zoom=0,
        max_zoom=len(Zooms.LEVELS),
        extent=None,
        required_field_only=True,
    ):
        self.extent = extent
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.flatrules = self._get_rules(flatrules)
        self.source_lyrs = self._get_source_lyrs()
        self.required_fields = self._get_required_fields(required_field_only)
        self.exported_rules = {}

    def export_rules(self):
        """Export all rules using the matching processing"""
        for flatrule in self.flatrules:
            if flatrule.target not in self.exported_rules:
                self._export_rule(flatrule)
        return self.exported_rules

    def _get_rules(self, flatrules):
        """Filter and modify rules which displayed out of the input scale range"""
        inside_range = []
        for flatrule in flatrules:
            rule_min, rule_max = flatrule.get_attr("o"), flatrule.get_attr("i")
            if rule_max < self.min_zoom or rule_min > self.max_zoom:
                continue
            flatrule.set_attr("o", min(rule_min, self.min_zoom))
            flatrule.set_attr("i", max(rule_max, self.max_zoom))
            inside_range.append(flatrule)
        return inside_range

    def _get_source_lyrs(self):
        """Extract and repair rules source layers"""
        source_lyrs = {}
        flatrules_lyrs = set(flatrule.lyr for flatrule in self.flatrules)
        for lyr in flatrules_lyrs:
            extent = self.extent or lyr.extent()
            extracted = self._run_alg("extractbyextent", INPUT=lyr, EXTENT=extent)
            fix_linetwork = self._run_alg("fixgeometries", INPUT=extracted, METHOD=0)
            fix_structure = self._run_alg(
                "fixgeometries", INPUT=fix_linetwork, METHOD=1
            )
            source_lyrs[lyr.id()] = fix_structure
        return source_lyrs

    def _get_required_fields(self, required_fields_only):
        """Get list of target datasets and its required fields"""
        if not required_fields_only:
            return
        required_fields = dict.fromkeys(
            [flatrule.target for flatrule in self.flatrules], set()
        )
        for flatrule in self.flatrules:
            required_fields.get(flatrule.target).union(set(flatrule.fields))
        return required_fields

    def _add_additional_fields(self):
        """Adds geometry attributes if geometry type is being changed"""
        pass

    def _export_rule(self, rule):
        """Export dataset using the relevant file keeping only required fields"""
        lyr = self.source_lyrs[rule.lyr.id()]

        # Remove unneccessary features
        expression = rule.rule.filterExpression()
        if expression:
            print(expression)
            lyr = self._run_alg("extractbyexpression", INPUT=lyr, EXPRESSION=expression)

        # Remove unneccessary fields
        if self.required_fields:
            required_fields = self.required_fields.get(rule.target) or ["fid"]
            lyr = self._run_alg("retainfields", INPUT=lyr, FIELDS=required_fields)

        # Replace geometry
        target_geom = rule.get_attr("g")
        if target_geom != rule.lyr.geometryType():
            if (
                target_geom == 1
            ):  # Processing refer to linestring as 2 in contrast to layer's gemetry type (1).
                target_geom = 2
            lyr = self._run_alg(
                "convertgeometrytype", "qgis", INPUT=lyr, TYPE=target_geom
            )

        # Insert rule to output dict
        QgsProject.instance().addMapLayer(lyr)
        self.exported_rules[rule.target] = lyr

    @staticmethod
    def _run_alg(alg_id, alg_type="native", **params):
        params["OUTPUT"] = "TEMPORARY_OUTPUT"
        return processing.run(f"{alg_type}:{alg_id}", params)["OUTPUT"]


class RuleExtractor:
    """Extracts and flattens QGIS vector layer rules."""

    RULE_TYPES = {0: "renderer", 1: "labeling"}

    def __init__(self):
        self.project = QgsProject.instance()
        self.lyrs_root = self.project.layerTreeRoot()
        self.extracted_rules = []

    def extract_rules(self):
        """Extract all rules from visible vector layers."""
        lyrs = self.project.mapLayers().values()

        for idx, lyr in enumerate(lyrs):
            if not self._is_relevant_lyr(lyr):
                continue

            try:
                self._process_lyr_rules(lyr.clone(), idx)
            except Exception as e:
                raise (e)
                print(f"Error processing layer '{lyr.name()}': {e}")

        return self.extracted_rules

    def print_rules(self):
        """Print extracted flat rules objects properties"""
        p1 = ".   "
        p2 = f".      -"
        for idx, flat in enumerate(self.extracted_rules):
            rule = flat.rule
            print(".")
            print(f"{p1}# {idx + 1} {rule.description()}")
            print(f"{p2}type: {self.RULE_TYPES[flat.type]}")
            print(f"{p2}lyr: {flat.lyr.name() or f'{flat.lyr.id()} (unnamed)'}")
            print(f"{p2}filter: {rule.filterExpression()}")
            print(f"{p2}range: {int(rule.maximumScale())} - {int(rule.minimumScale())}")

    def find_lyr(self, lyr_id):
        """Find layer by it's ID"""
        return self.lyrs_root.findLayer(lyr_id)

    def _is_relevant_lyr(self, lyr) -> bool:
        """Check if layer is a visible vector layer."""
        is_vector = lyr.type() == 0 and lyr.geometryType() != 4
        is_visible = self.find_lyr(lyr.id()).isVisible()
        return is_vector and is_visible

    def _process_lyr_rules(self, lyr, lyr_idx):
        """Process both renderer and labeling rules for a layer."""
        for rule_type in self.RULE_TYPES:
            rule_system = self._get_rule_system(lyr, rule_type)
            if not rule_system:
                continue
            root_rule = self._get_root(rule_system, lyr, lyr_idx, rule_type)
            if root_rule:
                self._flat_rules(lyr, lyr_idx, root_rule, rule_type, 0, 0)

    def _get_rule_system(self, lyr, rule_type):
        """Get or convert layer system to rule-based."""
        system = lyr.renderer() if rule_type == 0 else lyr.labeling()

        if not system:
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
        root_rule.setMinimumScale(lyr.minimumScale())
        root_rule.setMaximumScale(lyr.maximumScale())
        return root_rule

    def _flat_rules(self, lyr, lyr_idx, rule, rule_type, rule_lvl, rule_idx):
        """Recursively flatten rules with inheritance."""
        flatrules = []

        if rule.parent():
            clone = self._inherit_properties(rule, rule_type)
            self._set_rule_name(lyr_idx, clone, rule_type, rule_lvl, rule_idx)

            # Split by symbol layers (renderer only) then by scales then by symbol filters
            if rule_type == 0:
                splitted_rules = self._split_by_symbol_lyrs(clone, lyr)
            else:
                splitted_rules = self._split_label_by_symbol_filter(clone, lyr)

            for split_rule in splitted_rules:
                flatrules.extend(self._split_by_scales(split_rule))

        # Generate flat rules
        for flatrule in flatrules:
            self._create_flat_rule(flatrule, rule_type, lyr)

        # Process children recursively
        for child_idx, child in enumerate(rule.children()):
            if not child.active():
                continue
            # If child filter is ELSE (all other values) convert it to an absolute expresison
            if child.filterExpression() == "ELSE":
                else_exp = "NOT " + "AND NOT ".join(
                    f"({else_child.filterExpression()})"
                    for else_child in rule.children()
                    if else_child.active()
                    and else_child.filterExpression()
                    and else_child.filterExpression() != "ELSE"
                )
                child.setFilterExpression(else_exp)
            self._flat_rules(lyr, lyr_idx, child, rule_type, rule_lvl + 1, child_idx)
        return flatrules

    def _set_rule_name(self, lyr_idx, rule, rule_type, rule_lvl, rule_idx):
        """Inherit and combine rule names."""
        lyr_desc = f"l{self.get_num(lyr_idx)}"
        type_desc = f"t{self.get_num(rule_type)}"
        lvl_desc = f"d{self.get_num(rule_lvl)}"
        rule_desc = f"r{self.get_num(rule_idx)}"
        rule.setDescription(f"{lyr_desc}{type_desc}{lvl_desc}{rule_desc}")

    def _inherit_properties(self, rule, rule_type):
        """Inherit all properties from parent rule."""
        clone = rule.clone()
        self._inherit_filter(clone, rule)
        self._inherit_scale(clone, rule, min)
        self._inherit_scale(clone, rule, max)
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
        scale_attr = f"{comparator.__name__}imumScale"
        rule_scale = getattr(rule, scale_attr)()
        parent_scale = getattr(rule.parent(), scale_attr)()

        if rule_scale > 0 and parent_scale > 0:
            inherited_scale = comparator(rule_scale, parent_scale)
        else:
            inherited_scale = rule_scale if rule_scale > 0 else parent_scale

        setter_name = f"set{comparator.__name__.capitalize()}imumScale"
        getattr(clone, setter_name)(inherited_scale)

    def _inherit_symbol(self, clone, rule):
        """Inherit symbol layers from parent."""
        clone_symbol = clone.symbol()
        parent_symbol = rule.parent().symbol()

        if parent_symbol and clone_symbol:
            for i in range(parent_symbol.symbolLayerCount()):
                symbol_lyr = parent_symbol.symbolLayer(i).clone()
                clone_symbol.appendSymbolLayer(symbol_lyr)

    def _split_by_symbol_lyrs(self, rule, lyr):
        """Split rule by individual symbol layers."""
        # Split only polygon renderer symbol contains outline symbollayer.
        split_required = True
        sym = rule.symbol()
        if not sym or lyr.geometryType() != 2:
            split_required = False
        if not any(l.type() == 1 for l in sym.symbolLayers()):
            split_required = False
        if not split_required:
            desc = f"{rule.description()}s00g0{lyr.geometryType()}"
            rule.setDescription(desc)
            return [rule]

        # Clone symbol and keep only the relevant symbol layer
        sym_lyr_count = sym.symbolLayerCount()
        split_rules = []
        for keep_idx in range(sym_lyr_count):
            clone = rule.clone()
            line_sym_lyr = sym.symbolLayer(keep_idx).type() == 1
            target_geom = 1 if line_sym_lyr else lyr.geometryType()
            desc = f"{clone.description()}s{self.get_num(keep_idx)}g0{target_geom}"
            clone.setDescription(desc)

            # Remove all layers except the one to keep
            for remove_idx in reversed(range(sym_lyr_count)):
                if remove_idx != keep_idx:
                    clone.symbol().deleteSymbolLayer(remove_idx)

            split_rules.append(clone)

        return split_rules

    def _split_by_scales(self, rule):
        """Split rule by zoom levels if scale-dependent."""
        filter_expr = rule.filterExpression()
        min_scale = Zooms.snap_to_level(rule.minimumScale(), True)
        max_scale = Zooms.snap_to_level(rule.maximumScale(), False)

        # Handle scale-independent rules
        if "@map_scale" not in filter_expr:
            rule.setMinimumScale(min_scale)
            rule.setMaximumScale(max_scale)

            suffix = f"o{Zooms.get_zoom(min_scale)}i{Zooms.get_zoom(max_scale)}"
            rule.setDescription(f"{rule.description()}{suffix}")
            return [rule]

        # Split scale-dependent rules by zoom levels
        rule_lvls = [lvl for lvl in Zooms.LEVELS if max_scale <= lvl <= min_scale]
        rules = []

        for i, level in enumerate(rule_lvls):
            clone = rule.clone()
            clone.setMinimumScale(level)

            next_max = rule_lvls[i + 1] if i + 1 < len(rule_lvls) else max_scale
            clone.setMaximumScale(next_max)

            # Replace scale variable in filter
            new_filter = filter_expr.replace("@map_scale", str(level))
            clone.setFilterExpression(new_filter)

            suffix = f"o{Zooms.get_zoom(level)}i{Zooms.get_zoom(next_max)}"
            clone.setDescription(f"{rule.description()}{suffix}")
            rules.append(clone)

        return rules

    def _split_label_by_symbol_filter(self, rule, lyr):
        """Split label rule by matching renderer rules with overlapping scales."""
        # Get relevant symbol rules
        splitted_rules = []
        sym_flats = {}
        for flatrule in self.extracted_rules:
            sym_name = flatrule.rule.description()
            if sym_name in sym_flats:
                continue
            if flatrule.lyr.id() == lyr.id() and flatrule.type == 0:
                sym_flats[sym_name] = flatrule

        # Split label rule by symbol rules
        for sym_idx, sym_rule in enumerate(sym_flats.values()):
            combined_rule = self._create_combined_rule(rule, sym_idx, sym_rule.rule)
            if combined_rule:
                splitted_rules.append(combined_rule)
        if not splitted_rules:
            rule.setDescription(f"{rule.description()}f00")
            splitted_rules = [rule]

        # add character indicates destination geometry
        for rule in splitted_rules:
            target_geom = 0 if lyr.geometryType() == 2 else lyr.geometryType()
            rule.setDescription(f"{rule.description()}g0{target_geom}")

        return splitted_rules

    def _create_combined_rule(self, lbl_rule, sym_idx, sym_rule):
        """Create combined rule with merged filters and constrained scales."""
        lbl_min, lbl_max = lbl_rule.minimumScale(), lbl_rule.maximumScale()
        sym_min, sym_max = sym_rule.minimumScale(), sym_rule.maximumScale()
        if lbl_min <= sym_min or lbl_max >= sym_max:
            # Combine filter
            clone = lbl_rule.clone()
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
            if lbl_max < sym_max:
                clone.setMaximumScale(sym_max)

            # Combine name
            desc = f"{clone.description()}f{self.get_num(sym_idx)}"
            clone.setDescription(desc)
            return clone
        return

    def _create_flat_rule(self, rule, rule_type, lyr):
        """Create and store a FlatRule instance."""
        # Set the target dataset name by removing the symbol layer affix
        if rule_type == 0:
            target = f"{rule.description()[:12]}00{rule.description()[14:]}"
            data = rule.symbol()
            fields = data.usedAttributes(QgsRenderContext())
        else:
            target = rule.description()
            data = rule.settings()
            fields = data.referencedFields(QgsRenderContext())

        # Generate flat rule object
        flat = FlatRule(rule, lyr, rule_type, data, fields, target)
        self.extracted_rules.append(flat)

    @staticmethod
    def get_num(num):
        """Add zero as prefix to rule index properties if required"""
        return f"{'0' if num < 9 else ''}{num + 1}"


if __name__ == "__console__":
    """Console execution entry point."""
    extractor = RuleExtractor()
    print(".\n. Extracting rules from current QGIS project...")
    flat_rules = extractor.extract_rules()
    print(f".\n. Successfully extracted {len(flat_rules)} rules.")
    exporter = RuleExporter(flat_rules)
    print(f".\n. Exporting rules...")
    exported_rules = exporter.export_rules()
    print(f".\n. Successfully exported {len(exported_rules)} rules.")
    # extractor.print_rules()
