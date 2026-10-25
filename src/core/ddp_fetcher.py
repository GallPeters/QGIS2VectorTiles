"""
ddp_fetcher.py

DataDefinedPropertiesFetcher — recursively walks a QGIS symbol/label object
tree and collects all active data-defined properties, returning
(field_type, expression, field_name) triples for use as calculated fields.
"""

from qgis.core import QgsProperty, QgsPropertyDefinition, QgsExpression

from ..utils.config import QVariant

# Decimals kept of data-defined numbers in paper or screen units (see
# DataDefinedPropertiesFetcher._collect_from).
_NUMERIC_DECIMALS = 3


# Per colour field expression (_to_color_hex_expr's result): the raw colour
# expression it converts. The export's GDAL path writes the raw value when
# SQLite computes it, and the style maps each value to its colour
# (RulesExporter, QgisMapLibreStyleExporter).
COLOR_RAW_EXPRESSIONS: dict = {}


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

    # The accessors that lead from a rule to every object with data-defined
    # properties: rule -> symbol / label settings -> symbol layers ->
    # sub-symbols, text format, callouts and their symbols. Only these are
    # followed (in alphabetical order, as the walk always did): calling
    # other methods reflectively ran things like symbol preview rendering,
    # which can crash QGIS outside its interface.
    _CHILD_ACCESSORS: tuple = (
        "callout", "fillSymbol", "format", "lineSymbol", "settings",
        "subSymbol", "symbol", "symbolLayers", "textFormat",
    )

    _DATA_TYPE_MAP: dict = {
        QgsPropertyDefinition.DataType.DataTypeString: QVariant.String,
        QgsPropertyDefinition.DataType.DataTypeNumeric: QVariant.Double,
        QgsPropertyDefinition.DataType.DataTypeBoolean: QVariant.Bool,
    }

    FIELD_PREFIX = "q2vt"

    def __init__(self, qgis_object, min_scale, suffix=0):
        self._root = qgis_object
        self._min_scale = str(min_scale)
        self._suffix = f"_{suffix:02d}"
        self._results: list = []

    def fetch(self) -> list:
        """Return [[field_type, expression, field_name], ...] for all active DDPs."""
        self._walk(self._root)
        return self._results

    @classmethod
    def rename_fields(cls, qgis_object, renames: dict) -> None:
        """Point the data-defined properties fetch() rewrote at other field
        names ({old: new}), e.g. a zoom band's own fields."""
        if not renames:
            return

        def visit(obj, depth: int):
            if depth > 8:
                return
            for attr in cls._CHILD_ACCESSORS:
                getter = getattr(obj, attr, None)
                if getter is None or not callable(getter):
                    continue
                try:
                    result = getter()
                except (TypeError, RuntimeError, AttributeError):
                    continue
                for child in result if isinstance(result, list) else [result]:
                    # As _walk: QGIS objects only, never one like its parent.
                    if (child is None or "qgis." not in str(type(child))
                            or isinstance(child, type(obj))):
                        continue
                    if hasattr(child, "dataDefinedProperties"):
                        props = child.dataDefinedProperties()
                        for key in props.propertyKeys():
                            prop = props.property(key)
                            if not prop or not prop.isActive():
                                continue
                            text = prop.expressionString()
                            new_text = text
                            for old, new in renames.items():
                                new_text = new_text.replace(f'"{old}"', f'"{new}"')
                            if new_text != text:
                                prop.setExpressionString(new_text)
                                props.setProperty(key, prop)
                    visit(child, depth + 1)

        visit(qgis_object, 0)

    def _walk(self, obj):
        """Recursively visit obj's QGIS sub-objects for data-defined properties."""
        for attr in self._CHILD_ACCESSORS:
            try:
                getter = getattr(obj, attr, None)
                if getter is None or not callable(getter):
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
            prop_name = '_' + prop_def.description().lower().replace(" ", "_") if prop_def.description() else ''
            field_name = f"{self.FIELD_PREFIX}_property_{prop_name}{self._suffix}"

            if data_type == QgsPropertyDefinition.DataType.DataTypeBoolean:
                expression = self._process_boolean_prop(prop, props, key, field_name)
            else:
                expression = self._process_expression_prop(
                    prop, props, key, prop_def, field_type, field_name
                )
                if expression is None:
                    continue
                if data_type == QgsPropertyDefinition.DataType.DataTypeNumeric and not self._uses_map_units(obj):
                    # Sizes, widths, angles and opacities in paper or
                    # screen units: 3 decimals (0.001 mm is 0.004 px) are
                    # far below anything drawn, and equal values are stored
                    # once per tile. Map-unit lengths (a degree, maybe)
                    # keep every decimal.
                    expression = f"round(({expression}), {_NUMERIC_DECIMALS})"

            self._results.append([field_type, expression, field_name])

    # Unit accessors of symbol layers, text formats and label settings.
    _UNIT_ACCESSORS = (
        "sizeUnit", "widthUnit", "strokeWidthUnit", "offsetUnit", "outputUnit", "intervalUnit",
        "distanceUnit", "patternWidthUnit", "customDashPatternUnit", "hashLengthUnit",
        "averageAngleUnit", "offsetAlongLineUnit", "distanceXUnit", "distanceYUnit",
        "displacementXUnit", "displacementYUnit", "offsetXUnit", "offsetYUnit",
    )
    _UNIT_ATTRIBUTES = ("distUnits", "offsetUnits", "repeatDistanceUnit")
    # Qgis.RenderUnit codes of lengths on the ground: map units, meters.
    _MAP_UNITS = (1, 7)

    @classmethod
    def _uses_map_units(cls, obj) -> bool:
        """Whether any length of ``obj`` (a symbol layer, a label's settings
        and text format) is in map units or meters on the ground."""
        units = []
        owners = [obj]
        if hasattr(obj, "format") and callable(obj.format):
            try:
                text_format = obj.format()
                owners += [text_format, text_format.buffer(), text_format.background()]
            except (AttributeError, RuntimeError, TypeError):
                pass
        for owner in owners:
            for name in cls._UNIT_ACCESSORS:
                getter = getattr(owner, name, None)
                if callable(getter):
                    try:
                        units.append(getter())
                    except (TypeError, RuntimeError):
                        continue
            for name in cls._UNIT_ATTRIBUTES:
                value = getattr(owner, name, None)
                if value is not None and not callable(value):
                    units.append(value)
        for unit in units:
            try:
                if int(getattr(unit, "value", unit)) in cls._MAP_UNITS:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    @staticmethod
    def _static_value(qexpr: QgsExpression):
        """The expression's value if it reads no feature, field, geometry or
        variable (a constant); None otherwise. Evaluating one that does
        without a feature gives a meaningless value (coalesce("x", 5) is
        not 5 for every feature), and some functions crash QGIS then."""
        if (qexpr.hasParserError() or qexpr.referencedColumns() or qexpr.needsGeometry()
                or qexpr.referencedVariables()):
            return None
        return qexpr.evaluate()

    def _process_boolean_prop(self, prop, props_collection, key: int, field_name: str) -> str:
        """Replace boolean property with a field reference; return original expression."""
        exp_prop = QgsProperty()
        # @map_scale is the rule's zoom band's scale, as for other properties
        # (left as is, it has no value during the export: "show label"
        # depending on the scale showed at every scale).
        exp_prop.setExpressionString(prop.asExpression().replace("@map_scale", self._min_scale))
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
        # asExpression() also covers field-based properties, whose
        # expressionString() is empty.
        raw = prop.asExpression().replace("@map_scale", self._min_scale)
        is_color = prop_def and "color" in prop_def.name().lower() and field_type == 10

        expression = _to_color_hex_expr(raw) if is_color else raw

        # Constant if the property's own expression reads nothing (the hex
        # conversion's internal variables don't count).
        static_value = None
        if self._static_value(QgsExpression(raw)) is not None:
            static_value = QgsExpression(expression).evaluate()
        if static_value:
            prop.setExpressionString(f"'{str(static_value)}'")
            return None

        field_ref = f'"{field_name}"'
        if is_color:
            field_ref = (
                f"with_variable('color', \"{field_name}\", "
                f"'#' || substr(@color,8,2) || substr(@color,2,6))"
            )
        # An expression giving an array (the hex conversion only uses them).
        if "array" in raw:
            expression = f"try(array_to_string({expression}), {expression})"
        if is_color:
            COLOR_RAW_EXPRESSIONS[expression] = raw

        prop.setExpressionString(field_ref)
        return expression
