"""
Extracts and flattens renderer and labeling rules from vector layers in the
current QGIS project with unified processing logic.
"""

import re

from dataclasses import dataclass, fields
from typing import List, Union
from qgis.core import (
    QgsProject,
    QgsRuleBasedRenderer,
    QgsRuleBasedLabeling,
    QgsPalLayerSettings,
    QgsVectorLayer,
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


class RuleExtractor:
    """Extracts and flattens QGIS vector layer rules."""

    RULE_TYPES = {0: "renderer", 1: "labeling"}

    def __init__(self):
        self.project = QgsProject.instance()
        self.lyrs_root = self.project.layerTreeRoot()
        self.extracted_rules = []

    def extract_all_rules(self) -> List[FlatRule]:
        """Extract all rules from visible vector layers."""
        lyrs = self.project.mapLayers().values()

        for idx, lyr in enumerate(lyrs):
            if not self._is_relevant_lyr(lyr):
                continue

            try:
                self._process_lyr_rules(lyr, idx)
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
        flattened = []

        if rule.parent():
            clone = self._inherit_properties(rule, rule_type)
            self._set_rule_name(lyr_idx, clone, rule_type, rule_lvl, rule_idx)

            # Split by symbol layers (renderer only) then by scales then by symbol filters
            if rule_type == 0:
                splitted_rules = self._split_by_symbol_lyrs(clone, lyr)
            else:
                splitted_rules = self._split_label_by_symbol_filter(clone, lyr)

            for split_rule in splitted_rules:
                flattened.extend(self._split_by_scales(split_rule))

        # Generate flat rules
        for flat in flattened:
            self._create_flat_rule(flat, rule_type, lyr)

        # Process children recursively
        for child_idx, child in enumerate(rule.children()):
            if not child.active():
                continue
            self._flat_rules(lyr, lyr_idx, child, rule_type, rule_lvl + 1, child_idx)
        return flattened

    def _set_rule_name(self, lyr_idx, rule, rule_type, rule_lvl, rule_idx):
        """Inherit and combine rule names."""
        lyr_desc = f"l{self.get_num(lyr_idx)}t0{rule_type}"
        rule_desc = f"r{self.get_num(rule_lvl, rule_idx)}"
        rule.setDescription(f"{lyr_desc}{rule_desc}")

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
            if child_filter:
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

    def _split_by_symbol_lyrs(self, rule, lyr) -> List:
        """Split rule by individual symbol layers."""
        # Split only polygon renderer symbol contains outline symbollayer.
        split_required = True
        sym = rule.symbol()
        if not sym or lyr.geometryType() != 2:
            split_required = False
        if not any(l.type() == 1 for l in sym.symbolLayers()):
            split_required = False
        if not split_required:
            desc = f"{rule.description()}s00g{self.get_num(lyr.geometryType())}"
            rule.setDescription(desc)
            return [rule]

        # Clone symbol and keep only the relevant symbol layer
        sym_lyr_count = sym.symbolLayerCount()
        split_rules = []
        for keep_idx in range(sym_lyr_count):
            clone = rule.clone()
            line_sym_lyr = sym.symbolLayer(keep_idx).type() == 1
            target_geom = 1 if line_sym_lyr else lyr.geometryType()
            desc = f"{clone.description()}s{self.get_num(keep_idx)}g{self.get_num(target_geom)}"
            clone.setDescription(desc)

            # Remove all layers except the one to keep
            for remove_idx in reversed(range(sym_lyr_count)):
                if remove_idx != keep_idx:
                    clone.symbol().deleteSymbolLayer(remove_idx)

            split_rules.append(clone)

        return split_rules

    def _split_by_scales(self, rule) -> List:
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
        for flat_rule in self.extracted_rules:
            sym_name = flat_rule.rule.description()
            if sym_name in sym_flats:
                continue
            if flat_rule.lyr.id() == lyr.id() and flat_rule.type == 0:
                sym_flats[sym_name] = flat_rule

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
            rule.setDescription(f"{rule.description()}g{self.get_num(target_geom)}")

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

    def _create_flat_rule(self, flat_rule, rule_type, lyr):
        """Create and store a FlatRule instance."""
        flat = FlatRule(rule=flat_rule, lyr=lyr, type=rule_type)
        self.extracted_rules.append(flat)

    @staticmethod
    def get_num(*nums):
        """Add zero as prefix to rule index properties if required"""
        return "".join(f"{'0' if num < 9 else ''}{num + 1}" for num in nums)


if __name__ == "__console__":
    """Console execution entry point."""
    extractor = RuleExtractor()
    print(".\n. Extracting rules from current QGIS project...")
    rules = extractor.extract_all_rules()
    extractor.print_rules()
    print(f".\n. Successfully extracted {len(rules)} rules\n.")
