"""
flattened_rule.py

FlattenedRule — shared data-transfer object (dataclass) that carries a
flattened QGIS rule together with its source layer and export dataset path.
Imported by RulesFlattener, RulesExporter, and TilesStyler.
"""

from dataclasses import dataclass
from typing import Optional, Union

from qgis.core import QgsRuleBasedRenderer, QgsRuleBasedLabeling, QgsVectorLayer


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
        geom_desc = {0:'Polygon', 1:'Line', 2:'Point', 3:'Unknown'}
        lyr_name = self.layer.name() or self.layer.id()
        rule_type = "renderer" if self.get_attr("t") == 0 else "labeling"
        rule_num = self.get_attr("r")
        rule_depth = self.get_attr("d")
        source_geom = geom_desc.get(self.get_attr("g"))
        target_geom = geom_desc.get(self.get_attr("c"))
        rule_subnum, subnum_desc = (self.get_attr("s"), 'SymboLayer') if rule_type == "renderer" else (self.get_attr("f"), 'Renderer')
        return '{' + f"Layer: {lyr_name}, Type: {rule_type}, Depth: {rule_depth}, Number: {rule_num}, SourceGeom: {source_geom}, TargetGeom: {target_geom}, {subnum_desc}: {rule_subnum}" + '}'
    