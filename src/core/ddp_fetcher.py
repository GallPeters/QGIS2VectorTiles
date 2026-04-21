"""
ddp_fetcher.py

DataDefinedPropertiesFetcher — recursively walks a QGIS symbol/label object
tree and collects all active data-defined properties, returning
(field_type, expression, field_name) triples for use as calculated fields.
"""

from uuid import uuid4

from qgis.core import QgsProperty, QgsPropertyDefinition, QgsExpression

from ..utils.config import QVariant


def _to_color_hex_expr(inner_expr: str) -> str:
    """Wrap a color expression to produce an 8-character hex RRGGBBAA string."""
    return (
        f"'#' || with_variable('hex', array_cat(generate_series(0,9),"
        f"array('A','B','C','D','E','F')), array_to_string("
        f"array_foreach(array('red','green','blue','alpha'),"
        f"with_variable('colo', color_part({inner_expr}, @element),"
        f"@hex[floor(@colo/16)] || @hex[@colo%16])), ''))"
    )


class DataDefinedPropertiesFetcher:
    """Recursively fetch active data-defined properties from a QGIS symbol/label object."""

    # Attribute name fragments whose getters are unsafe to call reflectively.
    _SKIP_FRAGMENTS: frozenset = frozenset({
        "_", "value", "index", "available", "config", "next", "attr",
        "clone", "function", "flag", "capabil", "remove", "symbols",
        "clear", "prepare", "dump", "copy", "create", "update", "replace",
    })

    _DATA_TYPE_MAP: dict = {
        QgsPropertyDefinition.DataTypeString: QVariant.String,
        QgsPropertyDefinition.DataTypeNumeric: QVariant.Double,
        QgsPropertyDefinition.DataTypeBoolean: QVariant.Bool,
    }

    FIELD_PREFIX = "q2vt"

    def __init__(self, qgis_object, min_scale):
        self._root = qgis_object
        self._min_scale = str(min_scale)
        self._results: list = []

    def fetch(self) -> list:
        """Return [[field_type, expression, field_name], ...] for all active DDPs."""
        self._walk(self._root)
        return self._results

    def _is_safe_attr(self, attr: str) -> bool:
        lower = attr.lower()
        return not (
            any(frag in lower for frag in self._SKIP_FRAGMENTS)
            or (attr.startswith("set") and attr != lower)
            or attr[0].isupper()
        )

    def _walk(self, obj):
        """Recursively introspect obj's QGIS sub-objects for data-defined properties."""
        for attr in dir(obj):
            if not self._is_safe_attr(attr):
                continue
            try:
                getter = getattr(obj, attr)
                if not callable(getter):
                    continue

                result = getter()
                children = result if isinstance(result, list) else [result]
                if not children:
                    continue

                first = children[0]
                if (
                    isinstance(first, type(obj))
                    or "qgis." not in str(type(first))
                    or first in self._results
                ):
                    continue

                prop_defs = self._resolve_prop_definitions(first, obj)
                if not prop_defs:
                    continue

                for child in children:
                    if hasattr(child, "dataDefinedProperties"):
                        self._collect_from(child, prop_defs)
                    self._walk(child)

            except (NameError, ValueError, AttributeError, TypeError):
                continue

    def _resolve_prop_definitions(self, child, parent):
        if hasattr(child, "propertyDefinitions"):
            return child.propertyDefinitions()
        if hasattr(parent, "propertyDefinitions"):
            return parent.propertyDefinitions()
        return None

    def _collect_from(self, obj, prop_defs):
        """Extract active DDP entries from obj and append to results."""
        props = obj.dataDefinedProperties()
        for key in props.propertyKeys():
            prop = props.property(key)
            if not prop or not prop.isActive():
                continue

            property_kind = prop.propertyType()
            if property_kind not in (2, 3):
                continue

            prop_def = prop_defs.get(key)
            data_type = prop_def.dataType() if prop_defs else None
            field_type = self._DATA_TYPE_MAP.get(data_type)
            field_name = f"{self.FIELD_PREFIX}_{uuid4().hex[:8]}"

            if data_type == QgsPropertyDefinition.DataTypeBoolean:
                expression = self._process_boolean_prop(prop, props, key, field_name)
            else:
                expression = self._process_expression_prop(
                    prop, props, key, prop_def, field_type, field_name
                )
                if expression is None:
                    continue

            self._results.append([field_type, expression, field_name])

    def _process_boolean_prop(self, prop, props_collection, key: int, field_name: str) -> str:
        """Replace boolean property with a field reference; return original expression."""
        exp_prop = QgsProperty()
        exp_prop.setExpressionString(prop.asExpression())
        expression = exp_prop.expressionString()
        exp_prop.setExpressionString(f'"{field_name}"')
        props_collection.setProperty(key, exp_prop)
        return expression

    def _process_expression_prop(
        self, prop, props_collection, key, prop_def, field_type, field_name
    ):
        """
        Build the calculated-field expression for string/numeric DDPs.
        Returns None if the expression evaluates to a static value (no field needed).
        """
        raw = prop.expressionString().replace("@map_scale", self._min_scale)
        is_color = prop_def and "color" in prop_def.name().lower() and field_type == 10

        expression = _to_color_hex_expr(raw) if is_color else raw

        qexpr = QgsExpression(expression)
        static_value = qexpr.evaluate()
        if static_value is not None and not qexpr.needsGeometry():
            prop.setExpressionString(str(static_value))
            return None

        field_ref = f'"{field_name}"'
        if is_color:
            field_ref = (
                f"with_variable('color', \"{field_name}\", "
                f"'#' || substr(@color,8,2) || substr(@color,2,6))"
            )
        if "array" in expression:
            expression = f"try(array_to_string({expression}), {expression})"

        prop.setExpressionString(field_ref)
        return expression
