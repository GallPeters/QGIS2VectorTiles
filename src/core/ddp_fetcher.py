"""
data_defined_properties.py

DataDefinedPropertiesFetcher — recursively walks a QGIS symbol/label object
tree and collects all active data-defined properties, converting them into
(field_type, expression, field_name) triples that RulesExporter can materialise
as calculated fields.
"""

from uuid import uuid4

from qgis.core import QgsProperty, QgsPropertyDefinition, QgsExpression

from ..utils.config import QVariant


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
                    expression = f"'#' || with_variable('hex', array_cat(generate_series(0,9),array('A','B','C','D','E','F')), array_to_string (array_foreach (array ('red','green','blue'),with_variable('colo',color_part ({expression}, @element),@hex[floor(@colo/16)] || @hex[@colo%16] )),''))"  # pylint: disable=C0301
                evaluation = QgsExpression(expression).evaluate()
                if evaluation is not None:
                    prop.setExpressionString(str(evaluation))
                    continue
                if "array" in expression:
                    expression = f"try(array_to_string({expression}), {expression})"
                prop.setExpressionString(f'"{field_name}"')

            field_map = [field_type, expression, field_name]

            self.dd_properties.append(field_map)
