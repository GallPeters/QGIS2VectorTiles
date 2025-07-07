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
    QgsSymbol,
)


class ZoomLevels:
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


@dataclass
class FlatRule:
    """Represents a flattened rule with all inherited properties."""

    rule: Union[QgsRuleBasedLabeling.Rule, QgsRuleBasedRenderer.Rule]
    name: str
    rule_type: str
    lyr_id: str
    target_name: str
    filter: str
    min_scale: float
    max_scale: float
    data: Union[QgsSymbol, QgsPalLayerSettings]


class RuleExtractor:
    """Extracts and flattens QGIS vector layer rules."""

    RULE_TYPES = ["renderer", "labeling"]

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
        for idx, rule in enumerate(self.extracted_rules):
            print(f".\n.   # {idx + 1} {rule.name}")
            for field in fields(rule):
                if field.name not in ['rule','data', 'name', 'lyr_id']:
                    value = getattr(rule, field.name)
                    print(f".      - {field.name}: {value}")

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
                self._flatten_rules(root_rule, rule_type, lyr.id())

    def _get_rule_system(self, lyr, rule_type):
        """Get or convert layer system to rule-based."""
        system = lyr.renderer() if rule_type == "renderer" else lyr.labeling()

        if not system:
            return None

        # Return if already rule-based
        if isinstance(system, (QgsRuleBasedRenderer, QgsRuleBasedLabeling)):
            return system

        # Convert to rule-based
        if rule_type == "renderer":
            return QgsRuleBasedRenderer.convertFromRenderer(system)
        else:
            rule = QgsRuleBasedLabeling.Rule(system.settings())
            root = QgsRuleBasedLabeling.Rule(QgsPalLayerSettings())
            root.appendChild(rule)
            return QgsRuleBasedLabeling(root)

    def _get_root(self, rule_system, lyr, lyr_idx, rule_type):
        """Prepare root rule with descriptive information."""
        # Get root rule
        root_rule = rule_system.rootRule()

        # Inherit root rule properties from the layer itself
        lyr_name = lyr.name() or "unnamed"
        description = f"layer: {lyr_idx + 1} ({lyr_name}) > type: {rule_type}"
        root_rule.setDescription(description)
        root_rule.setFilterExpression(lyr.subsetString())
        root_rule.setMinimumScale(lyr.minimumScale())
        root_rule.setMaximumScale(lyr.maximumScale())
        return root_rule

    def _flatten_rules(self, rule, rule_type, lyr_id, idx=None) -> List:
        """Recursively flatten rules with inheritance."""
        flattened = []

        if rule.parent():
            clone = self._inherit_properties(rule, rule_type, idx)

            # Split by symbol layers (renderer only) then by scales then by symbol filters
            if rule_type == "renderer":
                splitted_rules = self._split_by_symbol_lyrs(clone)
            else:
                splitted_rules = self._split_label_by_symbol(clone, lyr_id)

            for split_rule in splitted_rules:
                flattened.extend(self._split_by_scales(split_rule))

        # Generate flat rules
        for flat in flattened:
            self._create_flat_rule(flat, rule_type, lyr_id)

        # Process children recursively
        for idx, child in enumerate(rule.children()):
            if child.active():
                self._flatten_rules(child, rule_type, lyr_id, idx)

        return flattened

    def _inherit_properties(self, rule, rule_type, rule_idx):
        """Inherit all properties from parent rule."""
        clone = rule.clone()
        self._inherit_name(clone, rule, rule_type, rule_idx)
        self._inherit_filter(clone, rule)
        self._inherit_scale(clone, rule, min)
        self._inherit_scale(clone, rule, max)
        if rule_type == "renderer":
            self._inherit_symbol(clone, rule)
        return clone

    def _inherit_name(self, clone, rule, rule_type, rule_idx):
        """Inherit and combine rule names."""
        parent = rule.parent()
        is_root_child = not parent.parent()

        attr = "description" if is_root_child or rule_type == "labeling" else "label"

        rule_name = getattr(rule, attr)() or "unnamed"
        parent_name = getattr(parent, attr)() or ""

        prefix = f"{parent_name} > " if parent_name else ""
        clone.setDescription(f"{prefix}rule: {rule_idx + 1} ({rule_name})")

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
            excluded_filter = f"({parent_filter}) AND NOT ({rule_filter})"
        else:
            excluded_filter = children_filters or combined_filter or ""
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

    def _split_by_symbol_lyrs(self, rule) -> List:
        """Split rule by individual symbol layers."""
        if not rule.symbol() or rule.symbol().symbolLayerCount() <= 1:
            return [rule]

        lyr_count = rule.symbol().symbolLayerCount()
        split_rules = []

        for keep_idx in range(lyr_count):
            clone = rule.clone()
            desc = f"{clone.description()} > symbol_lyr: {keep_idx + 1}"
            clone.setDescription(desc)

            # Remove all layers except the one to keep
            for remove_idx in reversed(range(lyr_count)):
                if remove_idx != keep_idx:
                    clone.symbol().deleteSymbolLayer(remove_idx)

            split_rules.append(clone)

        return split_rules

    def _split_by_scales(self, rule) -> List:
        """Split rule by zoom levels if scale-dependent."""
        filter_expr = rule.filterExpression()
        min_scale = rule.minimumScale()
        max_scale = rule.maximumScale()

        # Handle scale-independent rules
        if "@map_scale" not in filter_expr:
            rule.setMinimumScale(ZoomLevels.snap_to_level(min_scale, True))
            rule.setMaximumScale(ZoomLevels.snap_to_level(max_scale, False))
            return [rule]

        # Split scale-dependent rules by zoom levels
        rule_lvls = [lvl for lvl in ZoomLevels.LEVELS if max_scale <= lvl <= min_scale]
        rules = []

        for i, level in enumerate(rule_lvls):
            clone = rule.clone()
            clone.setMinimumScale(level)

            next_max = rule_lvls[i + 1] if i + 1 < len(rule_lvls) else max_scale
            clone.setMaximumScale(next_max)

            # Replace scale variable in filter
            new_filter = filter_expr.replace("@map_scale", str(level))
            clone.setFilterExpression(new_filter)

            desc = f"{rule.description()} > subscale: {round(level, 2)}"
            clone.setDescription(desc)
            rules.append(clone)

        return rules

    def _split_label_by_symbol(self, rule, lyr_id):
        """Split label rule by matching renderer rules with overlapping scales."""
        # Get relevant symbol rules
        splitted_rules = []
        sym_flats = {}
        for flat_rule in self.extracted_rules:
            if flat_rule.target_name in sym_flats:
                continue
            if flat_rule.lyr_id == lyr_id and flat_rule.rule_type == "renderer":
                sym_flats[flat_rule.target_name] = flat_rule

        # Split label rule by symbol rules
        for sym_flat in list(sym_flats.values()):
            combined_rule = self._create_combined_rule(rule, sym_flat)
            if combined_rule:
                splitted_rules.append(combined_rule)
        return splitted_rules

    def _create_combined_rule(self, lbl_rule, sym_flat):
        """Create combined rule with merged filters and constrained scales."""
        lbl_min, lbl_max = lbl_rule.minimumScale(), lbl_rule.maximumScale()
        sym_min, sym_max = sym_flat.min_scale, sym_flat.max_scale
        if lbl_min <= sym_min or lbl_max >= sym_max:
            # Combine filter
            clone = lbl_rule.clone()
            clone_filter = clone.filterExpression()
            sym_filter = sym_flat.filter
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
            desc = f"{clone.description()} > limiter: {sym_flat.target_name}"
            clone.setDescription(desc)
            return clone
        return

    def _create_flat_rule(self, flat_rule, rule_type, lyr_id):
        """Create and store a FlatRule instance."""
        # Get Inner properties
        target_name = self._clean_rule_desc(flat_rule.description())
        data = flat_rule.symbol() if rule_type == "renderer" else flat_rule.settings()

        # Constract FlatRule object
        flat = FlatRule(
            rule=flat_rule,
            name=flat_rule.description(),
            rule_type=rule_type,
            lyr_id=lyr_id,
            target_name=target_name,
            filter=flat_rule.filterExpression(),
            min_scale=flat_rule.minimumScale(),
            max_scale=flat_rule.maximumScale(),
            data=data,
        )
        self.extracted_rules.append(flat)

    @staticmethod
    def _clean_rule_desc(text):
        """Extract layer, type, all rule values, and limiter from formatted string."""
        lyr_match = re.search(r"layer:\s*(\d+)", text)
        type_match = re.search(r"type:\s*(\w+)", text)
        rule_matches = re.findall(r"rule:\s*(\d+)", text)
        limit_match = re.search(r"limiter:\s*(.+?)(?:\s*>|$)", text)

        if lyr_match and type_match and rule_matches:
            lyr = lyr_match.group(1)
            type_val = type_match.group(1)
            rules = "_".join(f"rule_{rule}" for rule in rule_matches)
            result = f"layer_{lyr}_{type_val}_{rules}"

            if limit_match:
                limit_val = limit_match.group(1).strip()
                result += f"_limiter_{limit_val}"

            return result
        return ""




if __name__ == "__console__":
    """Console execution entry point."""
    extractor = RuleExtractor()
    print(".\n. Extracting rules from current QGIS project...")
    rules = extractor.extract_all_rules()
    extractor.print_rules()
    print(f".\n. Successfully extracted {len(rules)} rules")

