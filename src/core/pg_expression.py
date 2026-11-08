"""
pg_expression.py

PgCompiler — translates QGIS expressions into PostgreSQL / PostGIS SQL that
computes what QGIS computes, for the live tiles (live_sql.py).

QGIS and PostgreSQL differ in many small ways: QGIS writes numbers as text
in their shortest form (1e+06), rounds halves away from zero, divides
integers into reals, compares a number with a text as numbers, takes 'abc'
as true... The SQL follows QGIS, through typed translation (every value
carries its kind) and the q2vt.* helper functions (live_sql_library.py),
which also never raise where QGIS gives NULL: an error would fail a whole
tile, where QGIS loses one value.

Anything without a faithful translation raises NotTranslatable, naming
what; the layer then stays in the static tiles.

Depends on: nothing of the plugin's (qgis.core only)
"""

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from qgis.core import (
    QgsExpression,
    QgsExpressionNode,
    QgsExpressionNodeBinaryOperator,
    QgsExpressionNodeUnaryOperator,
    QgsGeometry,
    QgsStyle,
)
from qgis.PyQt.QtCore import QDate, QDateTime, QTime
from qgis.PyQt.QtGui import QColor


class NotTranslatable(Exception):
    """An expression (or part of one) PostgreSQL can't compute as QGIS does."""


# Kinds of values. Comparisons give "tvl" (QGIS's 1 / 0 / NULL: as text "1"),
# boolean fields and literals "bool" (as text "true"). Arrays are "array:<kind>".
INT, REAL, TEXT, BOOL, TVL, GEOM, NULL = "int", "real", "text", "bool", "tvl", "geom", "null"
DATE, DATETIME, TIME, COLOR = "date", "datetime", "time", "color"
_NUMERIC = (INT, REAL)
_PG_TYPE = {INT: "bigint", REAL: "double precision", TEXT: "text", BOOL: "boolean", TVL: "boolean",
            GEOM: "geometry", DATE: "date", DATETIME: "timestamp", TIME: "time", COLOR: "text"}


@dataclass(frozen=True)
class Sql:
    """A translated value: its SQL, its kind, and for geometries their SRID
    (None: unknown or none, as geometries QGIS makes from coordinates)."""
    sql: str
    kind: str
    srid: Optional[int] = None


def array_of(kind: str) -> str:
    return f"array:{kind}"


def element_kind(kind: str) -> Optional[str]:
    return kind[6:] if kind.startswith("array:") else None


def quote_literal(text: str) -> str:
    """A PostgreSQL string literal (standard_conforming_strings: only ' doubles)."""
    return "'" + str(text).replace("'", "''") + "'"


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _float(value: float) -> str:
    if value != value or value in (float("inf"), float("-inf")):
        return "NULL::double precision"
    return f"{value!r}::double precision"


class PgCompiler:
    """Translate QGIS expressions over one row source.

    ``fields``: QGIS field name -> (SQL column reference, kind).
    ``geometry``: the row's geometry (EPSG:``srid``).
    ``variables``: variable name -> Sql (@geometry, @id, the export's own);
    ``constants``: the global and project variables' values, as the export
    evaluates them; any other variable is NULL there, and here.
    ``value_maps``: field name -> {stored value: shown value}, for
    represent_value().
    """

    _Node = QgsExpressionNode.NodeType
    _Bin = QgsExpressionNodeBinaryOperator.BinaryOperator
    _Un = QgsExpressionNodeUnaryOperator.UnaryOperator

    def __init__(self, fields: Dict[str, Tuple[str, str]], geometry: str, srid: int,
                 variables: Optional[Dict[str, Sql]] = None,
                 value_maps: Optional[Dict[str, Dict[str, str]]] = None,
                 constants: Optional[Dict[str, object]] = None):
        self.fields = {name.lower(): (name, ref, kind) for name, (ref, kind) in fields.items()}
        self.geometry = Sql(geometry, GEOM, srid)
        self.srid = srid
        self.variables: Dict[str, Sql] = dict(variables or {})
        self.variables.setdefault("geometry", self.geometry)
        self.constants = dict(constants or {})
        for name, value in self.constants.items():
            if name not in self.variables:
                try:
                    self.variables[name] = self._literal(value)
                except NotTranslatable:
                    pass
        self.value_maps = value_maps or {}
        self._expressions: List[QgsExpression] = []  # Keeps parsed trees alive.
        self._functions: Dict[str, Callable] = {}
        for name in dir(self):
            if name.startswith("_fn_"):
                self._functions[name[4:]] = getattr(self, name)
        self._functions.update({
            "$geometry": self._fn_geometry_var, "$area": self._fn_dollar_area,
            "$length": self._fn_dollar_length, "$perimeter": self._fn_dollar_perimeter,
            "$x": self._fn_dollar_x, "$y": self._fn_dollar_y, "$id": self._fn_dollar_id,
            "$currentfeature": self._fn_dollar_feature, "$x_at": self._fn_x_at, "$y_at": self._fn_y_at,
        })

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------
    def compile(self, expression: str) -> Sql:
        """Any expression's value."""
        return self._node(self._root(expression))

    def condition(self, expression: str) -> str:
        """A filter: true exactly where QGIS selects a row."""
        return f"COALESCE({self.truth(self.compile(expression))}, false)"

    def value(self, expression: str, field_type: int) -> str:
        """An output field's value, converted as QGIS converts it into a
        field of that type (QMetaType code: 1 bool, 2 int, 4 long long,
        6 double, 10 text; others text)."""
        return self.convert(self.compile(expression), field_type)

    def convert(self, value: Sql, field_type: int) -> str:
        if field_type == 1:
            return self.truth(value)
        if field_type in (2, 3, 4, 5):
            return self.to_qint(value)
        if field_type == 6:
            return self.to_num(value)
        return self.to_text(value)

    def _root(self, expression: str):
        expr = QgsExpression(expression)
        if expr.hasParserError():
            raise NotTranslatable(f"expression not valid: {expr.parserErrorString()}")
        if expr.rootNode() is None:
            raise NotTranslatable("empty expression")
        self._expressions.append(expr)
        return expr.rootNode()

    # ------------------------------------------------------------------
    # Conversions between kinds (as QGIS converts values)
    # ------------------------------------------------------------------
    def to_text(self, v: Sql) -> str:
        k = v.kind
        if k == TEXT or k == COLOR:
            return v.sql
        if k == INT:
            return f"({v.sql})::text"
        if k == REAL:
            return f"q2vt.qstr({v.sql})"
        if k == BOOL:
            return f"(CASE WHEN {v.sql} THEN 'true' WHEN NOT {v.sql} THEN 'false' END)"
        if k == TVL:
            return f"(CASE WHEN {v.sql} THEN '1' WHEN NOT {v.sql} THEN '0' END)"
        if k == NULL:
            return "NULL::text"
        if k == DATE:
            return f"to_char({v.sql}, 'YYYY-MM-DD')"
        if k == DATETIME:
            return f"to_char({v.sql}, 'YYYY-MM-DD\"T\"HH24:MI:SS.MS')"
        if k == TIME:
            return f"to_char({v.sql}, 'HH24:MI:SS.MS')"
        if k.startswith("array:"):
            # QGIS writes an array as its values joined by commas.
            return f"array_to_string({self._text_array(v)}, ',')"
        raise NotTranslatable(f"a {k} value used as text")

    def to_num(self, v: Sql) -> str:
        k = v.kind
        if k == REAL:
            return v.sql
        if k == INT:
            return f"({v.sql})::double precision"
        if k in (TEXT, COLOR):
            return f"q2vt.num({v.sql})"
        if k in (BOOL, TVL):
            return f"(CASE WHEN {v.sql} THEN 1.0 WHEN NOT {v.sql} THEN 0.0 END)"
        if k == NULL:
            return "NULL::double precision"
        raise NotTranslatable(f"a {k} value used as a number")

    def to_int(self, v: Sql) -> str:
        """A whole number where QGIS takes one (e.g. an array index):
        reals rounded, as QVariant does."""
        k = v.kind
        if k == INT:
            return v.sql
        if k == REAL:
            return f"q2vt.qround({v.sql})"
        if k in (TEXT, COLOR):
            return f"q2vt.qround(q2vt.num({v.sql}))"
        if k in (BOOL, TVL):
            return f"(CASE WHEN {v.sql} THEN 1 WHEN NOT {v.sql} THEN 0 END)::bigint"
        if k == NULL:
            return "NULL::bigint"
        raise NotTranslatable(f"a {k} value used as a whole number")

    def to_qint(self, v: Sql) -> str:
        """A value written into an integer field (QgsField::convertCompatible)."""
        return self.to_int(v)

    def truth(self, v: Sql) -> str:
        """QGIS's truth of a value: NULL unknown, numbers non-zero, text
        neither empty, 'false' nor a zero."""
        k = v.kind
        if k in (BOOL, TVL):
            return v.sql
        if k in _NUMERIC:
            return f"(({v.sql}) <> 0)"
        if k in (TEXT, COLOR):
            return f"q2vt.truth({v.sql})"
        if k == NULL:
            return "NULL::boolean"
        if k == GEOM:
            return f"(({v.sql}) IS NOT NULL AND NOT ST_IsEmpty({v.sql}))"
        if k.startswith("array:"):
            return f"(cardinality({v.sql}) > 0)"
        if k in (DATE, DATETIME, TIME):
            return f"(({v.sql}) IS NOT NULL)"
        raise NotTranslatable(f"a {k} value used as a condition")

    def to_geom(self, v: Sql, srid: Optional[int] = None) -> Sql:
        if v.kind == NULL:
            return Sql("NULL::geometry", GEOM, srid)
        if v.kind != GEOM:
            raise NotTranslatable(f"a {v.kind} value used as a geometry")
        return v

    def as_kind(self, v: Sql, kind: str) -> str:
        if kind == v.kind:
            return v.sql
        if kind == TEXT:
            return self.to_text(v)
        if kind == REAL:
            return self.to_num(v)
        if kind == INT:
            return self.to_int(v)
        if kind in (BOOL, TVL):
            return self.truth(v)
        if v.kind == NULL:
            pg = _PG_TYPE.get(kind)
            if pg:
                return f"NULL::{pg}"
            if kind.startswith("array:"):
                return f"NULL::{self._pg_array_type(kind)}"
        if kind == GEOM:
            return self.to_geom(v).sql
        raise NotTranslatable(f"a {v.kind} value where a {kind} is needed")

    def common_kind(self, values: Sequence[Sql]) -> str:
        """The kind several values (if / CASE / coalesce branches) share."""
        kinds = {v.kind for v in values if v.kind != NULL}
        if not kinds:
            return NULL
        if len(kinds) == 1:
            return kinds.pop()
        if kinds <= {INT, REAL}:
            return REAL
        if kinds <= {BOOL, TVL}:
            return TVL
        if kinds <= {TEXT, COLOR}:
            return TEXT
        if any(k == GEOM or k.startswith("array:") for k in kinds):
            raise NotTranslatable("branches of different kinds of value")
        return TEXT

    def _pg_array_type(self, kind: str) -> str:
        element = element_kind(kind)
        if element is None or element.startswith("array:"):
            raise NotTranslatable("arrays of arrays")
        return f"{_PG_TYPE.get(element, 'text')}[]"

    def _text_array(self, v: Sql) -> str:
        element = element_kind(v.kind)
        if element in (TEXT, COLOR):
            return v.sql
        if element == INT:
            return f"({v.sql})::text[]"
        return (f"ARRAY(SELECT {self.to_text(Sql('e', element))} "
                f"FROM unnest({v.sql}) WITH ORDINALITY AS u(e, o) ORDER BY o)")

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    def _node(self, node) -> Sql:
        t = node.nodeType()
        if t == self._Node.ntLiteral:
            return self._literal(node.value())
        if t == self._Node.ntColumnRef:
            return self._column(node.name())
        if t == self._Node.ntUnaryOperator:
            return self._unary(node)
        if t == self._Node.ntBinaryOperator:
            return self._binary(node)
        if t == self._Node.ntInOperator:
            return self._in(node)
        if t == self._Node.ntFunction:
            return self._function(node)
        if t == self._Node.ntCondition:
            return self._case(node)
        if t == self._Node.ntIndexOperator:
            return self._index(node)
        if t == getattr(self._Node, "ntBetweenOperator", None):
            return self._between(node)
        raise NotTranslatable(f"expression part {node.dump()}")

    def _literal(self, value) -> Sql:
        if value is None:
            return Sql("NULL", NULL)
        if isinstance(value, bool):
            return Sql("true" if value else "false", BOOL)
        if isinstance(value, int):
            return Sql(f"{value}::bigint", INT)
        if isinstance(value, float):
            return Sql(_float(value), REAL)
        if isinstance(value, str):
            return Sql(quote_literal(value), TEXT)
        if isinstance(value, QDateTime):
            return Sql(f"timestamp {quote_literal(value.toString('yyyy-MM-dd HH:mm:ss.zzz'))}", DATETIME)
        if isinstance(value, QDate):
            return Sql(f"date {quote_literal(value.toString('yyyy-MM-dd'))}", DATE)
        if isinstance(value, QTime):
            return Sql(f"time {quote_literal(value.toString('HH:mm:ss.zzz'))}", TIME)
        if isinstance(value, QColor):
            return Sql(quote_literal(f"{value.red()},{value.green()},{value.blue()},{value.alpha()}"), COLOR)
        if isinstance(value, QgsGeometry):
            return Sql(f"ST_GeomFromText({quote_literal(value.asWkt())})", GEOM)
        if isinstance(value, (list, tuple)):
            return self._array([self._literal(v) for v in value])
        if hasattr(value, "isNull") and value.isNull():  # A NULL QVariant.
            return Sql("NULL", NULL)
        raise NotTranslatable(f"a value of type {type(value).__name__}")

    def _column(self, name: str) -> Sql:
        entry = self.fields.get(name.lower())
        if entry is None:
            # QGIS evaluates a field the layer lacks as NULL.
            return Sql("NULL", NULL)
        return Sql(entry[1], entry[2])

    def _unary(self, node) -> Sql:
        operand = self._node(node.operand())
        if node.op() == self._Un.uoNot:
            return Sql(f"(NOT {self.truth(operand)})", TVL)
        if operand.kind == INT:
            return Sql(f"(-({operand.sql}))", INT)
        return Sql(f"(-({self.to_num(operand)}))", REAL)

    def _binary(self, node) -> Sql:
        op = node.op()
        b = self._Bin
        left, right = self._node(node.opLeft()), self._node(node.opRight())
        if op == b.boAnd:
            return Sql(f"({self.truth(left)} AND {self.truth(right)})", TVL)
        if op == b.boOr:
            return Sql(f"({self.truth(left)} OR {self.truth(right)})", TVL)
        if op in (b.boEQ, b.boNE, b.boLT, b.boLE, b.boGT, b.boGE, b.boIs, b.boIsNot):
            return self._compare(op, left, right)
        if op in (b.boLike, b.boNotLike, b.boILike, b.boNotILike):
            keyword = {b.boLike: "LIKE", b.boNotLike: "NOT LIKE",
                       b.boILike: "ILIKE", b.boNotILike: "NOT ILIKE"}[op]
            return Sql(f"({self.to_text(left)} {keyword} {self.to_text(right)})", TVL)
        if op == b.boRegexp:
            return Sql(f"({self.to_text(left)} ~ {self.to_text(right)})", TVL)
        if op == b.boConcat:
            return Sql(f"({self.to_text(left)} || {self.to_text(right)})", TEXT)
        if op == b.boPlus and left.kind in (TEXT, COLOR) and right.kind in (TEXT, COLOR):
            return Sql(f"({left.sql} || {right.sql})", TEXT)  # QGIS joins two texts.
        if op in (b.boPlus, b.boMinus, b.boMul):
            sign = {b.boPlus: "+", b.boMinus: "-", b.boMul: "*"}[op]
            if (left.kind in (DATE, DATETIME) or right.kind in (DATE, DATETIME)):
                raise NotTranslatable("arithmetic on dates")
            if left.kind == INT and right.kind == INT:
                return Sql(f"({left.sql} {sign} {right.sql})", INT)
            return Sql(f"({self.to_num(left)} {sign} {self.to_num(right)})", REAL)
        if op == b.boDiv:
            return Sql(f"q2vt.div({self.to_num(left)}, {self.to_num(right)})", REAL)
        if op == b.boIntDiv:
            return Sql(f"q2vt.intdiv({self.to_num(left)}, {self.to_num(right)})", INT)
        if op == b.boMod:
            if left.kind == INT and right.kind == INT:
                return Sql(f"q2vt.imod({left.sql}, {right.sql})", INT)
            return Sql(f"q2vt.fmod({self.to_num(left)}, {self.to_num(right)})", REAL)
        if op == b.boPow:
            return Sql(f"q2vt.pow({self.to_num(left)}, {self.to_num(right)})", REAL)
        raise NotTranslatable(f"operator {node.text()}")

    def _compare(self, op, left: Sql, right: Sql) -> Sql:
        b = self._Bin
        sign = {b.boEQ: "=", b.boNE: "<>", b.boLT: "<", b.boLE: "<=", b.boGT: ">", b.boGE: ">=",
                b.boIs: "IS NOT DISTINCT FROM", b.boIsNot: "IS DISTINCT FROM"}[op]
        texts = (TEXT, COLOR)
        if {left.kind, right.kind} & set(texts) and {left.kind, right.kind} & set(_NUMERIC):
            # A number and a text: as numbers if the text is one, else as texts.
            text = left if left.kind in texts else right
            number = right if text is left else left
            n = f"q2vt.num({text.sql})"
            as_numbers = (f"({n} {sign} {self.to_num(number)})" if text is left
                          else f"({self.to_num(number)} {sign} {n})")
            as_texts = (f'({text.sql} COLLATE "C" {sign} {self.to_text(number)})' if text is left
                        else f'({self.to_text(number)} COLLATE "C" {sign} {text.sql})')
            return Sql(f"(CASE WHEN {n} IS NOT NULL OR ({text.sql}) IS NULL THEN {as_numbers} "
                       f"ELSE {as_texts} END)", TVL)
        a, c = self._comparable(left, right)
        if sign in ("<", "<=", ">", ">=") and left.kind in texts and right.kind in texts:
            a = f'{a} COLLATE "C"'
        return Sql(f"({a} {sign} {c})", TVL)

    def _comparable(self, left: Sql, right: Sql) -> Tuple[str, str]:
        """Both sides as QGIS compares them: two texts as text, numbers (and
        a number with a text) as numbers, else as text."""
        kinds = {left.kind, right.kind} - {NULL}
        if not kinds:
            return "NULL::text", "NULL::text"
        if kinds <= {INT}:
            return self.as_kind(left, INT), self.as_kind(right, INT)
        if kinds <= {TEXT, COLOR}:
            return self.as_kind(left, TEXT), self.as_kind(right, TEXT)
        if kinds & set(_NUMERIC) and kinds <= {INT, REAL, TEXT, COLOR, BOOL, TVL}:
            return self.to_num(left), self.to_num(right)
        if kinds <= {BOOL, TVL}:
            return self.truth(left), self.truth(right)
        if kinds <= {DATE, DATETIME}:
            return f"({self.as_kind(left, DATETIME) if left.kind != NULL else 'NULL'})::timestamp", \
                   f"({self.as_kind(right, DATETIME) if right.kind != NULL else 'NULL'})::timestamp"
        if kinds == {GEOM}:
            return f"ST_AsBinary({left.sql})", f"ST_AsBinary({right.sql})"
        return self.to_text(left), self.to_text(right)

    def _in(self, node) -> Sql:
        value = self._node(node.node())
        items = [self._node(n) for n in node.list().list()]
        kind = INT if value.kind == INT and all(i.kind in (INT, NULL) for i in items) else (
            REAL if value.kind in _NUMERIC or all(i.kind in _NUMERIC + (NULL,) for i in items) else TEXT)
        left = self.as_kind(value, kind) if value.kind != NULL else "NULL"
        listed = ", ".join(self.as_kind(i, kind) if i.kind != NULL else "NULL" for i in items)
        keyword = "NOT IN" if node.isNotIn() else "IN"
        return Sql(f"({left} {keyword} ({listed}))", TVL)

    def _between(self, node) -> Sql:
        value = self._node(node.node())
        low, high = self._node(node.lowerBound()), self._node(node.higherBound())
        a, lo = self._comparable(value, low)
        _, hi = self._comparable(value, high)
        keyword = "NOT BETWEEN" if node.negate() else "BETWEEN"
        return Sql(f"({a} {keyword} {lo} AND {hi})", TVL)

    def _case(self, node) -> Sql:
        branches = [(self._node(w.whenExp()), self._node(w.thenExp())) for w in node.conditions()]
        otherwise = self._node(node.elseExp()) if node.elseExp() is not None else Sql("NULL", NULL)
        values = [then for _, then in branches] + [otherwise]
        kind = self.common_kind(values)
        if kind == NULL:
            return Sql("NULL", NULL)
        parts = [f"WHEN {self.truth(when)} THEN {self.as_kind(then, kind)}" for when, then in branches]
        return Sql(f"(CASE {' '.join(parts)} ELSE {self.as_kind(otherwise, kind)} END)", kind,
                   self._srid_of(values))

    def _index(self, node) -> Sql:
        # PyQGIS has no class for this node: its parts are read from its text.
        text = node.dump()
        if not text.endswith("]"):
            raise NotTranslatable(f"expression part {text}")
        depth, cut = 0, -1
        for i in range(len(text) - 1, -1, -1):
            if text[i] == "]":
                depth += 1
            elif text[i] == "[":
                depth -= 1
                if depth == 0:
                    cut = i
                    break
        if cut <= 0:
            raise NotTranslatable(f"expression part {text}")
        container = self._node(self._root(text[:cut]))
        index = self._node(self._root(text[cut + 1:-1]))
        element = element_kind(container.kind)
        if element is None:
            raise NotTranslatable("map values (by key)")
        return Sql(f"q2vt.array_get({container.sql}, {self.to_int(index)})", element)

    @staticmethod
    def _srid_of(values: Sequence[Sql]) -> Optional[int]:
        srids = {v.srid for v in values if v.kind == GEOM}
        return srids.pop() if len(srids) == 1 else None

    # ------------------------------------------------------------------
    # Functions
    # ------------------------------------------------------------------
    def _function(self, node) -> Sql:
        name = QgsExpression.Functions()[node.fnIndex()].name().lower()
        args = node.args().list() if node.args() is not None else []
        handler = self._functions.get(name)
        if handler is None:
            raise NotTranslatable(f"function {name}()")
        return handler(args)

    def _args(self, args, count: Optional[int] = None) -> List[Sql]:
        values = [self._node(a) for a in args]
        if count is not None and len(values) < count:
            raise NotTranslatable("missing arguments")
        return values

    @staticmethod
    def _static(node):
        """A node's value if it is a literal (QGIS folds constant parts)."""
        if node.nodeType() == QgsExpressionNode.NodeType.ntLiteral:
            return node.value()
        if node.hasCachedStaticValue():
            return node.cachedStaticValue()
        return None

    def _bind(self, name: str, value: Sql, body):
        """``body`` compiled with the variable ``name`` set to ``value``."""
        previous = self.variables.get(name)
        self.variables[name] = value
        try:
            return self._node(body)
        finally:
            if previous is None:
                self.variables.pop(name, None)
            else:
                self.variables[name] = previous

    # --- variables and the feature ---

    def _fn_var(self, args):
        name = self._static(args[0])
        if not isinstance(name, str):
            raise NotTranslatable("a variable named by an expression")
        value = self.variables.get(name)
        if value is None:
            if name in ("feature", "cluster_size"):
                return Sql("NULL", NULL)
            # Not set where the export evaluates expressions: NULL there too.
            return Sql("NULL", NULL)
        return value

    def _fn_with_variable(self, args):
        name = self._static(args[0])
        if not isinstance(name, str):
            raise NotTranslatable("with_variable() of a computed name")
        value = self._node(args[1])
        return self._bind(name, value, args[2])

    def _fn_geometry_var(self, args):
        return self.variables["geometry"]

    def _fn_dollar_id(self, args):
        value = self.variables.get("id")
        if value is None:
            raise NotTranslatable("$id")
        return value

    def _fn_dollar_feature(self, args):
        raise NotTranslatable("the whole feature as a value")

    def _fn_attribute(self, args):
        name_node = args[-1]
        name = self._static(name_node)
        if not isinstance(name, str):
            raise NotTranslatable("attribute() of a computed name")
        if len(args) == 2:
            feature = args[0]
            if not (feature.nodeType() == self._Node.ntFunction and QgsExpression.Functions()[
                    feature.fnIndex()].name().lower() in ("$currentfeature", "var")):
                raise NotTranslatable("attribute() of another feature")
        return self._column(name)

    def _fn_represent_value(self, args):
        node = args[0]
        if node.nodeType() != self._Node.ntColumnRef:
            raise NotTranslatable("represent_value() of an expression")
        value = self._column(node.name())
        mapping = self.value_maps.get(node.name())
        if not mapping:
            return Sql(self.to_text(value), TEXT)
        text = self.to_text(value)
        cases = " ".join(f"WHEN {quote_literal(k)} THEN {quote_literal(v)}" for k, v in mapping.items())
        return Sql(f"(CASE {text} {cases} ELSE {text} END)", TEXT)

    def _fn_eval(self, args):
        source = self._static(args[0])
        node = args[0]
        if source is None and node.nodeType() == self._Node.ntFunction and QgsExpression.Functions()[
                node.fnIndex()].name().lower() == "var":
            name = self._static(node.args().list()[0])
            source = self.constants.get(name) if isinstance(name, str) else None
            if source is None and isinstance(name, str) and name not in self.variables:
                return Sql("NULL", NULL)  # eval(NULL)
        if not isinstance(source, str):
            raise NotTranslatable("eval() of a computed expression")
        return self.compile(source)

    def _fn_try(self, args):
        # q2vt.* helpers return NULL where QGIS fails, so the first value
        # stands; the fallback is only for errors.
        return self._node(args[0])

    # --- conditionals ---

    def _fn_if(self, args):
        cond, yes = self._node(args[0]), self._node(args[1])
        no = self._node(args[2]) if len(args) > 2 else Sql("NULL", NULL)
        kind = self.common_kind([yes, no])
        if kind == NULL:
            return Sql("NULL", NULL)
        return Sql(f"(CASE WHEN {self.truth(cond)} THEN {self.as_kind(yes, kind)} "
                   f"ELSE {self.as_kind(no, kind)} END)", kind, self._srid_of([yes, no]))

    def _fn_coalesce(self, args):
        values = self._args(args)
        kind = self.common_kind(values)
        if kind == NULL:
            return Sql("NULL", NULL)
        return Sql(f"COALESCE({', '.join(self.as_kind(v, kind) for v in values)})", kind,
                   self._srid_of(values))

    def _fn_nullif(self, args):
        value, other = self._args(args, 2)
        a, b = self._comparable(value, other)
        return Sql(f"(CASE WHEN {a} = {b} THEN NULL ELSE {value.sql} END)", value.kind, value.srid)

    # --- conversions ---

    def _fn_to_string(self, args):
        return Sql(self.to_text(self._node(args[0])), TEXT)

    def _fn_to_int(self, args):
        value = self._node(args[0])
        if value.kind in (TEXT, COLOR):
            return Sql(f"q2vt.text_int({value.sql})", INT)  # '3.7' is not a whole number.
        return Sql(self.to_int(value), INT)

    def _fn_to_real(self, args):
        return Sql(self.to_num(self._node(args[0])), REAL)

    def _fn_to_bool(self, args):
        return Sql(self.truth(self._node(args[0])), BOOL)

    def _fn_to_date(self, args):
        value = self._node(args[0])
        if value.kind == DATE:
            return value
        if value.kind == DATETIME:
            return Sql(f"({value.sql})::date", DATE)
        if len(args) > 1 and self._static(args[1]) not in (None, ""):
            raise NotTranslatable("to_date() with a format")
        return Sql(f"q2vt.to_date({self.to_text(value)})", DATE)

    def _fn_to_datetime(self, args):
        value = self._node(args[0])
        if value.kind == DATETIME:
            return value
        if value.kind == DATE:
            return Sql(f"({value.sql})::timestamp", DATETIME)
        if len(args) > 1 and self._static(args[1]) not in (None, ""):
            raise NotTranslatable("to_datetime() with a format")
        return Sql(f"q2vt.to_datetime({self.to_text(value)})", DATETIME)

    def _fn_to_time(self, args):
        value = self._node(args[0])
        if value.kind == TIME:
            return value
        if value.kind == DATETIME:
            return Sql(f"({value.sql})::time", TIME)
        if len(args) > 1 and self._static(args[1]) not in (None, ""):
            raise NotTranslatable("to_time() with a format")
        return Sql(f"q2vt.to_time({self.to_text(value)})", TIME)

    # --- math ---

    def _math1(self, sql_fn: str, args, kind=REAL):
        return Sql(f"{sql_fn}({self.to_num(self._node(args[0]))})", kind)

    def _fn_abs(self, args):
        value = self._node(args[0])
        if value.kind == INT:
            return Sql(f"abs({value.sql})", INT)
        return Sql(f"abs({self.to_num(value)})", REAL)

    def _fn_round(self, args):
        value = self._node(args[0])
        if len(args) > 1:
            places = self._node(args[1])
            return Sql(f"q2vt.round_to({self.to_num(value)}, {self.to_int(places)})", REAL)
        if value.kind == INT:
            return value
        return Sql(f"q2vt.qround({self.to_num(value)})", INT)

    def _fn_floor(self, args):
        return self._math1("floor", args)

    def _fn_ceil(self, args):
        return self._math1("ceil", args)

    def _fn_sqrt(self, args):
        return Sql(f"q2vt.sqrt({self.to_num(self._node(args[0]))})", REAL)

    def _fn_exp(self, args):
        return self._math1("exp", args)

    def _fn_ln(self, args):
        return Sql(f"q2vt.ln({self.to_num(self._node(args[0]))})", REAL)

    def _fn_log10(self, args):
        return Sql(f"q2vt.log10({self.to_num(self._node(args[0]))})", REAL)

    def _fn_log(self, args):
        base, value = self._args(args, 2)
        return Sql(f"q2vt.logb({self.to_num(base)}, {self.to_num(value)})", REAL)

    def _fn_pi(self, args):
        return Sql("pi()", REAL)

    def _fn_sin(self, args):
        return self._math1("sin", args)

    def _fn_cos(self, args):
        return self._math1("cos", args)

    def _fn_tan(self, args):
        return self._math1("tan", args)

    def _fn_asin(self, args):
        return Sql(f"q2vt.asin({self.to_num(self._node(args[0]))})", REAL)

    def _fn_acos(self, args):
        return Sql(f"q2vt.acos({self.to_num(self._node(args[0]))})", REAL)

    def _fn_atan(self, args):
        return self._math1("atan", args)

    def _fn_atan2(self, args):
        y, x = self._args(args, 2)
        return Sql(f"atan2({self.to_num(y)}, {self.to_num(x)})", REAL)

    def _fn_degrees(self, args):
        return self._math1("degrees", args)

    def _fn_radians(self, args):
        return self._math1("radians", args)

    def _minmax(self, fn: str, args):
        values = self._args(args)
        if all(v.kind == INT for v in values):
            return Sql(f"{fn}({', '.join(v.sql for v in values)})", INT)
        return Sql(f"{fn}({', '.join(self.to_num(v) for v in values)})", REAL)

    def _fn_max(self, args):
        if len(args) == 1:
            raise NotTranslatable("the max() aggregate")
        return self._minmax("GREATEST", args)

    def _constant_aggregate(self, name: str, args):
        """An aggregate of a constant (over the layer, by any group or filter):
        that constant wherever the layer has rows, as QGIS gives it."""
        value = self._static(args[0]) if args else None
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise NotTranslatable(f"the {name}() aggregate (over the whole layer)")
        return self._literal(value)

    def _fn_maximum(self, args):
        return self._constant_aggregate("maximum", args)

    def _fn_minimum(self, args):
        return self._constant_aggregate("minimum", args)

    def _fn_mean(self, args):
        return self._constant_aggregate("mean", args)

    def _fn_median(self, args):
        return self._constant_aggregate("median", args)

    def _fn_majority(self, args):
        return self._constant_aggregate("majority", args)

    def _fn_minority(self, args):
        return self._constant_aggregate("minority", args)

    def _fn_min(self, args):
        if len(args) == 1:
            raise NotTranslatable("the min() aggregate")
        return self._minmax("LEAST", args)

    def _fn_clamp(self, args):
        low, value, high = self._args(args, 3)
        if all(v.kind == INT for v in (low, value, high)):
            return Sql(f"LEAST(GREATEST({value.sql}, {low.sql}), {high.sql})", INT)
        return Sql(f"LEAST(GREATEST({self.to_num(value)}, {self.to_num(low)}), {self.to_num(high)})", REAL)

    def _fn_scale_linear(self, args):
        v, dmin, dmax, rmin, rmax = (self.to_num(a) for a in self._args(args, 5))
        return Sql(f"q2vt.scale_linear({v}, {dmin}, {dmax}, {rmin}, {rmax})", REAL)

    def _fn_scale_exp(self, args):
        v, dmin, dmax, rmin, rmax, e = (self.to_num(a) for a in self._args(args, 6))
        return Sql(f"q2vt.scale_exp({v}, {dmin}, {dmax}, {rmin}, {rmax}, {e})", REAL)

    def _fn_scale_polynomial(self, args):
        return self._fn_scale_exp(args)

    def _fn_scale_exponential(self, args):
        v, dmin, dmax, rmin, rmax, e = (self.to_num(a) for a in self._args(args, 6))
        return Sql(f"q2vt.scale_exponential({v}, {dmin}, {dmax}, {rmin}, {rmax}, {e})", REAL)

    def _fn_rand(self, args):
        raise NotTranslatable("rand() (a new value at each drawing)")

    # --- text ---

    def _text1(self, sql_fn: str, args, kind=TEXT):
        return Sql(f"{sql_fn}({self.to_text(self._node(args[0]))})", kind)

    def _fn_upper(self, args):
        return self._text1("q2vt.upper", args)

    def _fn_lower(self, args):
        return self._text1("q2vt.lower", args)

    def _fn_title(self, args):
        return self._text1("q2vt.title", args)

    def _fn_trim(self, args):
        return self._text1("q2vt.trim", args)

    def _fn_ltrim(self, args):
        value = self._node(args[0])
        chars = self.to_text(self._node(args[1])) if len(args) > 1 else "' '"
        return Sql(f"ltrim({self.to_text(value)}, {chars})", TEXT)

    def _fn_rtrim(self, args):
        value = self._node(args[0])
        chars = self.to_text(self._node(args[1])) if len(args) > 1 else "' '"
        return Sql(f"rtrim({self.to_text(value)}, {chars})", TEXT)

    def _fn_length(self, args):
        value = self._node(args[0])
        if value.kind == GEOM:
            return Sql(f"ST_Length({value.sql})", REAL)
        return Sql(f"char_length({self.to_text(value)})", INT)

    def _fn_char_length(self, args):
        return Sql(f"char_length({self.to_text(self._node(args[0]))})", INT)

    def _fn_substr(self, args):
        value = self._node(args[0])
        start = self.to_int(self._node(args[1]))
        length = self.to_int(self._node(args[2])) if len(args) > 2 else "NULL::bigint"
        return Sql(f"q2vt.substr({self.to_text(value)}, {start}, {length})", TEXT)

    def _fn_left(self, args):
        value, n = self._args(args, 2)
        return Sql(f"q2vt.left({self.to_text(value)}, {self.to_int(n)})", TEXT)

    def _fn_right(self, args):
        value, n = self._args(args, 2)
        return Sql(f"q2vt.right({self.to_text(value)}, {self.to_int(n)})", TEXT)

    def _fn_replace(self, args):
        values = self._args(args)
        if len(values) == 2:
            raise NotTranslatable("replace() with a map")
        value, before, after = values
        if element_kind(before.kind) is not None:
            if element_kind(after.kind) is None:
                return Sql(f"q2vt.replace_each({self.to_text(value)}, {self._text_array(before)}, "
                           f"ARRAY[{self.to_text(after)}])", TEXT)
            return Sql(f"q2vt.replace_each({self.to_text(value)}, {self._text_array(before)}, "
                       f"{self._text_array(after)})", TEXT)
        return Sql(f"replace({self.to_text(value)}, {self.to_text(before)}, {self.to_text(after)})", TEXT)

    def _fn_regexp_replace(self, args):
        value, pattern, replacement = self._args(args, 3)
        return Sql(f"q2vt.regexp_replace({self.to_text(value)}, {self.to_text(pattern)}, "
                   f"{self.to_text(replacement)})", TEXT)

    def _fn_regexp_substr(self, args):
        value, pattern = self._args(args, 2)
        return Sql(f"q2vt.regexp_substr({self.to_text(value)}, {self.to_text(pattern)})", TEXT)

    def _fn_regexp_match(self, args):
        value, pattern = self._args(args, 2)
        return Sql(f"q2vt.regexp_match({self.to_text(value)}, {self.to_text(pattern)})", INT)

    def _fn_regexp_matches(self, args):
        value, pattern = self._args(args, 2)
        return Sql(f"q2vt.regexp_matches({self.to_text(value)}, {self.to_text(pattern)})", array_of(TEXT))

    def _fn_strpos(self, args):
        value, part = self._args(args, 2)
        return Sql(f"strpos({self.to_text(value)}, {self.to_text(part)})", INT)

    def _fn_concat(self, args):
        values = self._args(args)
        return Sql(f"concat({', '.join(self.to_text(v) for v in values)})", TEXT)

    def _fn_format(self, args):
        values = self._args(args)
        texts = ", ".join(self.to_text(v) for v in values[1:])
        rest = f", VARIADIC ARRAY[{texts}]::text[]" if texts else ", VARIADIC ARRAY[]::text[]"
        return Sql(f"q2vt.format({self.to_text(values[0])}{rest})", TEXT)

    def _fn_lpad(self, args):
        value, width, fill = self._args(args, 3)
        return Sql(f"q2vt.lpad({self.to_text(value)}, {self.to_int(width)}, {self.to_text(fill)})", TEXT)

    def _fn_rpad(self, args):
        value, width, fill = self._args(args, 3)
        return Sql(f"q2vt.rpad({self.to_text(value)}, {self.to_int(width)}, {self.to_text(fill)})", TEXT)

    def _fn_char(self, args):
        return Sql(f"q2vt.chr({self.to_int(self._node(args[0]))})", TEXT)

    def _fn_ascii(self, args):
        return Sql(f"ascii({self.to_text(self._node(args[0]))})", INT)

    def _fn_format_number(self, args):
        value = self._node(args[0])
        places = self.to_int(self._node(args[1])) if len(args) > 1 else "0"
        if len(args) > 2 and self._static(args[2]) not in (None, ""):
            raise NotTranslatable("format_number() in a named language")
        return Sql(f"q2vt.format_number({self.to_num(value)}, {places})", TEXT)

    def _fn_format_date(self, args):
        value = self._node(args[0])
        pattern = self._static(args[1])
        if not isinstance(pattern, str):
            raise NotTranslatable("format_date() with a computed format")
        if len(args) > 2 and self._static(args[2]) not in (None, ""):
            raise NotTranslatable("format_date() in a named language")
        if value.kind in (TEXT, COLOR):
            value = Sql(f"q2vt.to_datetime({value.sql})", DATETIME)
        elif value.kind == DATE:
            value = Sql(f"({value.sql})::timestamp", DATETIME)
        elif value.kind == TIME:
            value = Sql(f"(date '2000-01-01' + {value.sql})", DATETIME)
        if value.kind not in (DATETIME, NULL):
            raise NotTranslatable("format_date() of a non-date")
        return Sql(f"to_char({value.sql}, {quote_literal(_qt_date_format(pattern))})", TEXT)

    def _fn_repeat(self, args):
        value, count = self._args(args, 2)
        return Sql(f"repeat({self.to_text(value)}, GREATEST({self.to_int(count)}, 0)::integer)", TEXT)

    def _fn_wordwrap(self, args):
        raise NotTranslatable("wordwrap()")

    def _fn_soundex(self, args):
        raise NotTranslatable("soundex()")

    def _fn_levenshtein(self, args):
        raise NotTranslatable("levenshtein()")

    # --- dates ---

    def _fn_now(self, args):
        return Sql("LOCALTIMESTAMP(3)", DATETIME)

    def _date_part(self, part: str, args):
        value = self._node(args[0])
        if value.kind in (TEXT, COLOR):
            value = Sql(f"q2vt.to_datetime({value.sql})", DATETIME)
        if value.kind not in (DATE, DATETIME, TIME, NULL):
            raise NotTranslatable(f"{part} of a non-date")
        return Sql(f"(extract({part} FROM {value.sql}))::bigint", INT)

    def _fn_year(self, args):
        return self._date_part("year", args)

    def _fn_month(self, args):
        return self._date_part("month", args)

    def _fn_day(self, args):
        return self._date_part("day", args)

    def _fn_hour(self, args):
        return self._date_part("hour", args)

    def _fn_minute(self, args):
        return self._date_part("minute", args)

    def _fn_day_of_week(self, args):
        return self._date_part("dow", args)

    def _fn_week(self, args):
        return self._date_part("week", args)

    def _fn_make_date(self, args):
        y, m, d = (self.to_int(a) for a in self._args(args, 3))
        return Sql(f"q2vt.make_date({y}, {m}, {d})", DATE)

    def _fn_make_datetime(self, args):
        y, m, d, h, mi, s = self._args(args, 6)
        return Sql(f"q2vt.make_datetime({self.to_int(y)}, {self.to_int(m)}, {self.to_int(d)}, "
                   f"{self.to_int(h)}, {self.to_int(mi)}, {self.to_num(s)})", DATETIME)

    def _fn_epoch(self, args):
        value = self._node(args[0])
        if value.kind not in (DATE, DATETIME, NULL):
            raise NotTranslatable("epoch() of a non-date")
        return Sql(f"(extract(epoch FROM ({value.sql})::timestamp) * 1000)::bigint", INT)

    # --- arrays ---

    def _array(self, values: List[Sql]) -> Sql:
        kind = self.common_kind(values) if values else TEXT
        if kind == NULL:
            kind = TEXT
        if kind == GEOM:
            return Sql(f"ARRAY[{', '.join(v.sql for v in values)}]::geometry[]", array_of(GEOM))
        pg = _PG_TYPE.get(kind)
        if pg is None:
            raise NotTranslatable("arrays of this kind")
        items = ", ".join(self.as_kind(v, kind) if v.kind != NULL else "NULL" for v in values)
        return Sql(f"ARRAY[{items}]::{pg}[]", array_of(kind))

    def _fn_array(self, args):
        return self._array(self._args(args))

    def _need_array(self, value: Sql) -> str:
        element = element_kind(value.kind)
        if element is None:
            if value.kind == NULL:
                return TEXT
            raise NotTranslatable("an array function on a non-array")
        return element

    def _fn_array_length(self, args):
        value = self._node(args[0])
        self._need_array(value)
        return Sql(f"COALESCE(cardinality({value.sql}), 0)", INT)

    def _fn_array_get(self, args):
        container, index = self._args(args, 2)
        element = self._need_array(container)
        return Sql(f"q2vt.array_get({container.sql}, {self.to_int(index)})", element)

    def _fn_array_first(self, args):
        value = self._node(args[0])
        return Sql(f"({value.sql})[1]", self._need_array(value))

    def _fn_array_last(self, args):
        value = self._node(args[0])
        return Sql(f"({value.sql})[cardinality({value.sql})]", self._need_array(value))

    def _fn_array_contains(self, args):
        container, item = self._args(args, 2)
        element = self._need_array(container)
        return Sql(f"COALESCE({self.as_kind(item, element)} = ANY({container.sql}), false)", TVL)

    def _fn_array_find(self, args):
        container, item = self._args(args, 2)
        element = self._need_array(container)
        return Sql(f"COALESCE(array_position({container.sql}, {self.as_kind(item, element)}) - 1, -1)", INT)

    def _fn_array_to_string(self, args):
        values = self._args(args)
        container = values[0]
        self._need_array(container)
        delimiter = self.to_text(values[1]) if len(values) > 1 else "','"
        empty = self.to_text(values[2]) if len(values) > 2 else "''"
        return Sql(f"array_to_string({self._text_array(container)}, {delimiter}, {empty})", TEXT)

    def _fn_string_to_array(self, args):
        values = self._args(args)
        delimiter = self.to_text(values[1]) if len(values) > 1 else "','"
        empty = self.to_text(values[2]) if len(values) > 2 else "''"
        return Sql(f"q2vt.string_to_array({self.to_text(values[0])}, {delimiter}, {empty})", array_of(TEXT))

    def _fn_array_cat(self, args):
        values = self._args(args)
        kinds = {self._need_array(v) for v in values}
        if len(kinds) == 1:
            element = kinds.pop()
            return Sql(" || ".join(f"({v.sql})" for v in values), array_of(element))
        return Sql(" || ".join(f"({self._text_array(v)})" for v in values), array_of(TEXT))

    def _fn_array_append(self, args):
        container, item = self._args(args, 2)
        element = self._need_array(container)
        return Sql(f"array_append({container.sql}, {self.as_kind(item, element)})", container.kind)

    def _fn_array_prepend(self, args):
        container, item = self._args(args, 2)
        element = self._need_array(container)
        return Sql(f"array_prepend({self.as_kind(item, element)}, {container.sql})", container.kind)

    def _fn_array_distinct(self, args):
        value = self._node(args[0])
        self._need_array(value)
        return Sql(f"q2vt.array_distinct({value.sql})", value.kind)

    def _fn_array_reverse(self, args):
        value = self._node(args[0])
        self._need_array(value)
        return Sql(f"ARRAY(SELECT e FROM unnest({value.sql}) WITH ORDINALITY AS u(e, o) ORDER BY o DESC)",
                   value.kind)

    def _fn_array_sort(self, args):
        value = self._node(args[0])
        element = self._need_array(value)
        ascending = self._static(args[1]) if len(args) > 1 else True
        order = "ASC" if ascending in (True, None, 1) else "DESC"
        collate = ' COLLATE "C"' if element == TEXT else ""
        return Sql(f"ARRAY(SELECT e FROM unnest({value.sql}) AS u(e) ORDER BY e{collate} {order})", value.kind)

    def _fn_array_slice(self, args):
        container, start, stop = self._args(args, 3)
        self._need_array(container)
        return Sql(f"q2vt.array_slice({container.sql}, {self.to_int(start)}, {self.to_int(stop)})",
                   container.kind)

    def _fn_generate_series(self, args):
        values = self._args(args)
        step = values[2] if len(values) > 2 else Sql("1", INT)
        if all(v.kind == INT for v in values[:2]) and step.kind == INT:
            return Sql(f"ARRAY(SELECT generate_series({values[0].sql}, {values[1].sql}, {step.sql}))",
                       array_of(INT))
        return Sql(f"ARRAY(SELECT generate_series(({self.to_num(values[0])})::numeric, "
                   f"({self.to_num(values[1])})::numeric, ({self.to_num(step)})::numeric)::double precision)",
                   array_of(REAL))

    def _fn_array_foreach(self, args):
        container = self._node(args[0])
        element = self._need_array(container)
        item = self._bind("element", Sql("u.e", element), args[1])
        result = self._array_select(item, f"unnest({container.sql}) WITH ORDINALITY AS u(e, o)", "ORDER BY u.o")
        return result

    def _fn_array_filter(self, args):
        container = self._node(args[0])
        element = self._need_array(container)
        keep = self._bind("element", Sql("u.e", element), args[1])
        limit = f" LIMIT {self.to_int(self._node(args[2]))}" if len(args) > 2 and self._static(args[2]) else ""
        return Sql(f"ARRAY(SELECT u.e FROM unnest({container.sql}) WITH ORDINALITY AS u(e, o) "
                   f"WHERE {self.truth(keep)} ORDER BY u.o{limit})", container.kind)

    def _array_select(self, item: Sql, source: str, order: str) -> Sql:
        if item.kind == GEOM:
            return Sql(f"ARRAY(SELECT {item.sql} FROM {source} {order})", array_of(GEOM))
        if item.kind == NULL:
            return Sql(f"ARRAY(SELECT NULL::text FROM {source} {order})", array_of(TEXT))
        pg = _PG_TYPE.get(item.kind)
        if pg is None:
            raise NotTranslatable("arrays of this kind")
        return Sql(f"ARRAY(SELECT ({item.sql})::{pg} FROM {source} {order})", array_of(item.kind))

    # --- colours (as QGIS's "r,g,b,a" text) ---

    def _fn_color_rgb(self, args):
        r, g, b = (self.to_num(a) for a in self._args(args, 3))
        return Sql(f"q2vt.color_rgba({r}, {g}, {b}, 255)", COLOR)

    def _fn_color_rgba(self, args):
        r, g, b, a = (self.to_num(v) for v in self._args(args, 4))
        return Sql(f"q2vt.color_rgba({r}, {g}, {b}, {a})", COLOR)

    def _fn_color_hsl(self, args):
        h, s, l = (self.to_num(a) for a in self._args(args, 3))
        return Sql(f"q2vt.color_hsla({h}, {s}, {l}, 255)", COLOR)

    def _fn_color_hsla(self, args):
        h, s, l, a = (self.to_num(v) for v in self._args(args, 4))
        return Sql(f"q2vt.color_hsla({h}, {s}, {l}, {a})", COLOR)

    def _fn_color_hsv(self, args):
        h, s, v = (self.to_num(a) for a in self._args(args, 3))
        return Sql(f"q2vt.color_hsva({h}, {s}, {v}, 255)", COLOR)

    def _fn_color_hsva(self, args):
        h, s, v, a = (self.to_num(x) for x in self._args(args, 4))
        return Sql(f"q2vt.color_hsva({h}, {s}, {v}, {a})", COLOR)

    def _fn_color_cmyk(self, args):
        c, m, y, k = (self.to_num(x) for x in self._args(args, 4))
        return Sql(f"q2vt.color_cmyka({c}, {m}, {y}, {k}, 255)", COLOR)

    def _fn_color_cmyka(self, args):
        c, m, y, k, a = (self.to_num(x) for x in self._args(args, 5))
        return Sql(f"q2vt.color_cmyka({c}, {m}, {y}, {k}, {a})", COLOR)

    def _fn_color_part(self, args):
        color, part = self._args(args, 2)
        return Sql(f"q2vt.color_part({self.to_text(color)}, {self.to_text(part)})", REAL)

    def _fn_set_color_part(self, args):
        color, part, value = self._args(args, 3)
        return Sql(f"q2vt.set_color_part({self.to_text(color)}, {self.to_text(part)}, {self.to_num(value)})",
                   COLOR)

    def _fn_darker(self, args):
        color, factor = self._args(args, 2)
        return Sql(f"q2vt.darker({self.to_text(color)}, {self.to_num(factor)})", COLOR)

    def _fn_lighter(self, args):
        color, factor = self._args(args, 2)
        return Sql(f"q2vt.lighter({self.to_text(color)}, {self.to_num(factor)})", COLOR)

    def _fn_color_mix_rgb(self, args):
        a, b, ratio = self._args(args, 3)
        return Sql(f"q2vt.color_mix({self.to_text(a)}, {self.to_text(b)}, {self.to_num(ratio)})", COLOR)

    def _fn_color_mix(self, args):
        return self._fn_color_mix_rgb(args)

    def _fn_ramp_color(self, args):
        name = self._static(args[0])
        value = self.to_num(self._node(args[1]))
        ramp = None
        if isinstance(name, str):
            ramp = QgsStyle.defaultStyle().colorRamp(name)
        if ramp is None:
            raise NotTranslatable("ramp_color() of a ramp not in the style library")
        # The ramp sampled finely (QGIS interpolates its own stops; 256
        # samples are the same colour to the eye and in 8-bit values).
        samples = []
        for i in range(257):
            c = ramp.color(i / 256)
            samples.append(f"{c.red()},{c.green()},{c.blue()},{c.alpha()}")
        array = "ARRAY[" + ", ".join(quote_literal(s) for s in samples) + "]"
        return Sql(f"q2vt.ramp_sample({array}, {value})", COLOR)

    def _fn_project_color(self, args):
        raise NotTranslatable("project_color()")

    # --- geometry ---

    def _geom(self, node) -> Sql:
        return self.to_geom(self._node(node))

    def _same_srid(self, a: Sql, b: Sql) -> Tuple[str, str, Optional[int]]:
        """Two geometries in one SRID: one made from coordinates (none) takes
        the other's, as QGIS ignores CRSs in these functions."""
        if a.srid == b.srid:
            return a.sql, b.sql, a.srid
        if a.srid is None:
            return f"ST_SetSRID({a.sql}, {b.srid})", b.sql, b.srid
        if b.srid is None:
            return a.sql, f"ST_SetSRID({b.sql}, {a.srid})", a.srid
        return a.sql, f"ST_SetSRID({b.sql}, {a.srid})", a.srid

    def _g1(self, sql_fn: str, args, kind=GEOM, extra: str = ""):
        g = self._geom(args[0])
        return Sql(f"{sql_fn}({g.sql}{extra})", kind, g.srid if kind == GEOM else None)

    def _g2(self, sql_fn: str, args, kind=GEOM):
        a, b = self._geom(args[0]), self._geom(args[1])
        x, y, srid = self._same_srid(a, b)
        return Sql(f"{sql_fn}({x}, {y})", kind, srid if kind == GEOM else None)

    def _fn_dollar_area(self, args):
        return Sql(f"ST_Area({self.geometry.sql})", REAL)

    def _fn_dollar_length(self, args):
        return Sql(f"ST_Length({self.geometry.sql})", REAL)

    def _fn_dollar_perimeter(self, args):
        return Sql(f"ST_Perimeter({self.geometry.sql})", REAL)

    def _fn_dollar_x(self, args):
        return Sql(f"q2vt.x({self.geometry.sql})", REAL)

    def _fn_dollar_y(self, args):
        return Sql(f"q2vt.y({self.geometry.sql})", REAL)

    def _fn_x_at(self, args):
        return Sql(f"ST_X(q2vt.point_n({self.geometry.sql}, {self.to_int(self._node(args[0]))} + 1))", REAL)

    def _fn_y_at(self, args):
        return Sql(f"ST_Y(q2vt.point_n({self.geometry.sql}, {self.to_int(self._node(args[0]))} + 1))", REAL)

    def _fn_area(self, args):
        return self._g1("ST_Area", args, REAL)

    def _fn_perimeter(self, args):
        return self._g1("ST_Perimeter", args, REAL)

    def _fn_x(self, args):
        return self._g1("q2vt.x", args, REAL)

    def _fn_y(self, args):
        return self._g1("q2vt.y", args, REAL)

    def _fn_z(self, args):
        return self._g1("ST_Z", args, REAL)

    def _fn_m(self, args):
        return self._g1("ST_M", args, REAL)

    def _fn_x_min(self, args):
        return self._g1("ST_XMin", args, REAL)

    def _fn_x_max(self, args):
        return self._g1("ST_XMax", args, REAL)

    def _fn_y_min(self, args):
        return self._g1("ST_YMin", args, REAL)

    def _fn_y_max(self, args):
        return self._g1("ST_YMax", args, REAL)

    def _fn_bounds_width(self, args):
        g = self._geom(args[0])
        return Sql(f"(ST_XMax({g.sql}) - ST_XMin({g.sql}))", REAL)

    def _fn_bounds_height(self, args):
        g = self._geom(args[0])
        return Sql(f"(ST_YMax({g.sql}) - ST_YMin({g.sql}))", REAL)

    def _fn_centroid(self, args):
        return self._g1("q2vt.centroid", args)

    def _fn_point_on_surface(self, args):
        return self._g1("q2vt.point_on_surface", args)

    def _fn_pole_of_inaccessibility(self, args):
        g = self._geom(args[0])
        tolerance = self.to_num(self._node(args[1]))
        return Sql(f"q2vt.pole_of_inaccessibility({g.sql}, {tolerance})", GEOM, g.srid)

    def _fn_boundary(self, args):
        return self._g1("q2vt.boundary", args)

    def _fn_bounds(self, args):
        return self._g1("ST_Envelope", args)

    def _fn_envelope(self, args):
        return self._g1("ST_Envelope", args)

    def _fn_convex_hull(self, args):
        return self._g1("ST_ConvexHull", args)

    def _fn_buffer(self, args):
        g = self._geom(args[0])
        distance = self.to_num(self._node(args[1]))
        segments = self.to_int(self._node(args[2])) if len(args) > 2 else "8"
        defaults = {3: ("round",), 4: ("round",), 5: (2, 2.0)}
        for i, accepted in defaults.items():
            if len(args) > i and str(self._static(args[i])).lower() not in [str(a) for a in accepted]:
                raise NotTranslatable("buffer() with cap or join styles")
        return Sql(f"q2vt.buffer({g.sql}, {distance}, {segments})", GEOM, g.srid)

    def _fn_single_sided_buffer(self, args):
        g = self._geom(args[0])
        distance = self.to_num(self._node(args[1]))
        segments = self.to_int(self._node(args[2])) if len(args) > 2 else "8"
        return Sql(f"q2vt.single_sided_buffer({g.sql}, {distance}, {segments})", GEOM, g.srid)

    def _fn_offset_curve(self, args):
        g = self._geom(args[0])
        distance = self.to_num(self._node(args[1]))
        segments = self.to_int(self._node(args[2])) if len(args) > 2 else "8"
        join = self.to_int(self._node(args[3])) if len(args) > 3 else "1"
        miter = self.to_num(self._node(args[4])) if len(args) > 4 else "2.0"
        return Sql(f"q2vt.offset_curve({g.sql}, {distance}, {segments}, {join}, {miter})", GEOM, g.srid)

    def _fn_intersection(self, args):
        return self._g2("q2vt.intersection", args)

    def _fn_difference(self, args):
        return self._g2("q2vt.difference", args)

    def _fn_sym_difference(self, args):
        return self._g2("ST_SymDifference", args)

    def _fn_combine(self, args):
        return self._g2("ST_Union", args)

    def _fn_intersects(self, args):
        return self._g2("ST_Intersects", args, TVL)

    def _fn_contains(self, args):
        return self._g2("ST_Contains", args, TVL)

    def _fn_within(self, args):
        return self._g2("ST_Within", args, TVL)

    def _fn_touches(self, args):
        return self._g2("ST_Touches", args, TVL)

    def _fn_crosses(self, args):
        return self._g2("ST_Crosses", args, TVL)

    def _fn_overlaps(self, args):
        return self._g2("ST_Overlaps", args, TVL)

    def _fn_disjoint(self, args):
        return self._g2("ST_Disjoint", args, TVL)

    def _fn_intersects_bbox(self, args):
        a, b = self._geom(args[0]), self._geom(args[1])
        x, y, _ = self._same_srid(a, b)
        return Sql(f"({x} && {y})", TVL)

    def _fn_distance(self, args):
        return self._g2("ST_Distance", args, REAL)

    def _fn_shortest_line(self, args):
        return self._g2("ST_ShortestLine", args)

    def _fn_hausdorff_distance(self, args):
        return self._g2("ST_HausdorffDistance", args, REAL)

    def _fn_transform(self, args):
        g = self._geom(args[0])
        source, target = self._static(args[1]), self._static(args[2])
        src, dst = _epsg(source), _epsg(target)
        if src is None or dst is None:
            raise NotTranslatable("transform() between CRSs other than EPSG codes")
        sql = g.sql if g.srid == src else f"ST_SetSRID({g.sql}, {src})"
        return Sql(f"q2vt.transform({sql}, {dst})", GEOM, dst)

    def _fn_make_point(self, args):
        values = self._args(args, 2)
        coords = ", ".join(self.to_num(v) for v in values[:4])
        fn = {2: "ST_MakePoint", 3: "ST_MakePoint", 4: "ST_MakePoint"}[min(len(values), 4)]
        return Sql(f"{fn}({coords})", GEOM)

    def _fn_make_point_m(self, args):
        x, y, m = self._args(args, 3)
        return Sql(f"ST_MakePointM({self.to_num(x)}, {self.to_num(y)}, {self.to_num(m)})", GEOM)

    def _fn_make_line(self, args):
        values = self._args(args)
        if len(values) == 1 and element_kind(values[0].kind) == GEOM:
            return Sql(f"q2vt.make_line({values[0].sql})", GEOM)
        geoms = [self.to_geom(v) for v in values]
        srid = self._srid_of(geoms)
        return Sql(f"q2vt.make_line(ARRAY[{', '.join(g.sql for g in geoms)}]::geometry[])", GEOM, srid)

    def _fn_make_polygon(self, args):
        values = [self.to_geom(v) for v in self._args(args)]
        rings = ", ".join(v.sql for v in values[1:])
        holes = f", ARRAY[{rings}]::geometry[]" if rings else ""
        return Sql(f"ST_MakePolygon(q2vt.as_ring({values[0].sql}){holes})", GEOM, values[0].srid)

    def _fn_make_circle(self, args):
        center = self._geom(args[0])
        radius = self.to_num(self._node(args[1]))
        segments = self.to_int(self._node(args[2])) if len(args) > 2 else "36"
        return Sql(f"q2vt.make_circle({center.sql}, {radius}, {segments})", GEOM, center.srid)

    def _fn_geom_from_wkt(self, args):
        return Sql(f"q2vt.geom_from_wkt({self.to_text(self._node(args[0]))})", GEOM)

    def _fn_geom_from_wkb(self, args):
        raise NotTranslatable("geom_from_wkb()")

    def _fn_geom_to_wkt(self, args):
        g = self._geom(args[0])
        precision = self.to_int(self._node(args[1])) if len(args) > 1 else "8"
        return Sql(f"q2vt.geom_to_wkt({g.sql}, {precision})", TEXT)

    def _fn_is_empty(self, args):
        g = self._geom(args[0])
        return Sql(f"ST_IsEmpty({g.sql})", TVL)

    def _fn_is_empty_or_null(self, args):
        g = self._node(args[0])
        if g.kind == NULL:
            return Sql("true", TVL)
        g = self.to_geom(g)
        return Sql(f"(({g.sql}) IS NULL OR ST_IsEmpty({g.sql}))", TVL)

    def _fn_is_valid(self, args):
        return self._g1("ST_IsValid", args, TVL)

    def _fn_is_closed(self, args):
        return self._g1("q2vt.is_closed", args, TVL)

    def _fn_make_valid(self, args):
        return self._g1("ST_MakeValid", args)

    def _fn_start_point(self, args):
        return self._g1("q2vt.start_point", args)

    def _fn_end_point(self, args):
        return self._g1("q2vt.end_point", args)

    def _fn_point_n(self, args):
        g = self._geom(args[0])
        return Sql(f"q2vt.point_n({g.sql}, {self.to_int(self._node(args[1]))})", GEOM, g.srid)

    def _fn_num_points(self, args):
        return self._g1("ST_NPoints", args, INT)

    def _fn_num_geometries(self, args):
        return self._g1("q2vt.num_geometries", args, INT)

    def _fn_geometry_n(self, args):
        g = self._geom(args[0])
        return Sql(f"ST_GeometryN({g.sql}, {self.to_int(self._node(args[1]))}::integer)", GEOM, g.srid)

    def _fn_num_rings(self, args):
        return self._g1("ST_NRings", args, INT)

    def _fn_num_interior_rings(self, args):
        return self._g1("ST_NumInteriorRings", args, INT)

    def _fn_exterior_ring(self, args):
        return self._g1("ST_ExteriorRing", args)

    def _fn_interior_ring_n(self, args):
        g = self._geom(args[0])
        return Sql(f"ST_InteriorRingN({g.sql}, {self.to_int(self._node(args[1]))}::integer)", GEOM, g.srid)

    def _fn_nodes_to_points(self, args):
        g = self._geom(args[0])
        ignore_closing = self.truth(self._node(args[1])) if len(args) > 1 else "false"
        return Sql(f"q2vt.nodes_to_points({g.sql}, {ignore_closing})", GEOM, g.srid)

    def _fn_collect_geometries(self, args):
        values = self._args(args)
        if len(values) == 1 and element_kind(values[0].kind) == GEOM:
            return Sql(f"q2vt.collect({values[0].sql})", GEOM)
        geoms = [self.to_geom(v) for v in values]
        return Sql(f"q2vt.collect(ARRAY[{', '.join(g.sql for g in geoms)}]::geometry[])", GEOM,
                   self._srid_of(geoms))

    def _fn_line_interpolate_point(self, args):
        g = self._geom(args[0])
        return Sql(f"q2vt.line_interpolate_point({g.sql}, {self.to_num(self._node(args[1]))})", GEOM, g.srid)

    def _fn_line_interpolate_angle(self, args):
        g = self._geom(args[0])
        return Sql(f"q2vt.line_interpolate_angle({g.sql}, {self.to_num(self._node(args[1]))})", REAL)

    def _fn_line_locate_point(self, args):
        a, b = self._geom(args[0]), self._geom(args[1])
        x, y, _ = self._same_srid(a, b)
        return Sql(f"q2vt.line_locate_point({x}, {y})", REAL)

    def _fn_angle_at_vertex(self, args):
        g = self._geom(args[0])
        return Sql(f"q2vt.angle_at_vertex({g.sql}, {self.to_int(self._node(args[1]))})", REAL)

    def _fn_line_merge(self, args):
        return self._g1("ST_LineMerge", args)

    def _fn_reverse(self, args):
        return self._g1("ST_Reverse", args)

    def _fn_simplify(self, args):
        g = self._geom(args[0])
        return Sql(f"ST_Simplify({g.sql}, {self.to_num(self._node(args[1]))})", GEOM, g.srid)

    def _fn_simplify_vw(self, args):
        g = self._geom(args[0])
        return Sql(f"ST_SimplifyVW({g.sql}, {self.to_num(self._node(args[1]))})", GEOM, g.srid)

    def _fn_smooth(self, args):
        g = self._geom(args[0])
        iterations = self.to_int(self._node(args[1])) if len(args) > 1 else "1"
        for i, default in ((2, 0.25), (3, -1), (4, 180)):
            if len(args) > i and self._static(args[i]) not in (default, float(default)):
                raise NotTranslatable("smooth() with an offset, minimum length or angle")
        return Sql(f"ST_ChaikinSmoothing({g.sql}, LEAST({iterations}, 5)::integer, true)", GEOM, g.srid)

    def _fn_translate(self, args):
        g = self._geom(args[0])
        dx, dy = self.to_num(self._node(args[1])), self.to_num(self._node(args[2]))
        return Sql(f"ST_Translate({g.sql}, {dx}, {dy})", GEOM, g.srid)

    def _fn_rotate(self, args):
        g = self._geom(args[0])
        angle = self.to_num(self._node(args[1]))
        if len(args) > 2 and self._node(args[2]).kind != NULL:
            center = self._geom(args[2])
            _, c, _ = self._same_srid(g, center)
            return Sql(f"ST_Rotate({g.sql}, -radians({angle}), {c})", GEOM, g.srid)
        return Sql(f"ST_Rotate({g.sql}, -radians({angle}), ST_Centroid({g.sql}))", GEOM, g.srid)

    def _fn_scale(self, args):
        g = self._geom(args[0])
        sx, sy = self.to_num(self._node(args[1])), self.to_num(self._node(args[2]))
        if len(args) > 3 and self._node(args[3]).kind != NULL:
            center = self._geom(args[3])
            _, c, _ = self._same_srid(g, center)
        else:
            c = f"ST_Centroid({g.sql})"
        return Sql(f"ST_Scale({g.sql}, ST_MakePoint({sx}, {sy}), {c})", GEOM, g.srid)

    def _fn_azimuth(self, args):
        return self._g2("q2vt.azimuth", args, REAL)

    def _fn_project(self, args):
        g = self._geom(args[0])
        distance = self.to_num(self._node(args[1]))
        azimuth = self.to_num(self._node(args[2])) if len(args) > 2 else "pi() / 2"
        if len(args) > 3:
            raise NotTranslatable("project() with an elevation")
        return Sql(f"ST_Translate({g.sql}, {distance} * sin({azimuth}), {distance} * cos({azimuth}))",
                   GEOM, g.srid)

    def _fn_flip_coordinates(self, args):
        return self._g1("ST_FlipCoordinates", args)

    def _fn_force_rhr(self, args):
        return self._g1("ST_ForcePolygonCW", args)

    def _fn_force_polygon_cw(self, args):
        return self._g1("ST_ForcePolygonCW", args)

    def _fn_force_polygon_ccw(self, args):
        return self._g1("ST_ForcePolygonCCW", args)

    def _fn_minimal_circle(self, args):
        g = self._geom(args[0])
        segments = self.to_int(self._node(args[1])) if len(args) > 1 else "36"
        return Sql(f"ST_MinimumBoundingCircle({g.sql}, GREATEST(({segments}) / 4, 1)::integer)", GEOM, g.srid)

    def _fn_oriented_bbox(self, args):
        return self._g1("ST_OrientedEnvelope", args)

    def _fn_concave_hull(self, args):
        g = self._geom(args[0])
        ratio = self.to_num(self._node(args[1]))
        holes = self.truth(self._node(args[2])) if len(args) > 2 else "false"
        return Sql(f"ST_ConcaveHull({g.sql}, {ratio}, {holes})", GEOM, g.srid)

    def _fn_delaunay_triangulation(self, args):
        return self._g1("ST_DelaunayTriangles", args)

    def _fn_straight_distance_2d(self, args):
        g = self._geom(args[0])
        return Sql(f"ST_Distance(q2vt.start_point({g.sql}), q2vt.end_point({g.sql}))", REAL)

    def _fn_sinuosity(self, args):
        g = self._geom(args[0])
        return Sql(f"q2vt.div(ST_Length({g.sql}), ST_Distance(q2vt.start_point({g.sql}), "
                   f"q2vt.end_point({g.sql})))", REAL)

    def _fn_geometry_type(self, args):
        return self._g1("q2vt.geometry_type", args, TEXT)

    def _fn_close_line(self, args):
        return self._g1("q2vt.close_line", args)

    def _fn_segments_to_lines(self, args):
        return self._g1("q2vt.segments_to_lines", args)

    def _fn_extend(self, args):
        g = self._geom(args[0])
        start, end = self.to_num(self._node(args[1])), self.to_num(self._node(args[2]))
        return Sql(f"q2vt.extend({g.sql}, {start}, {end})", GEOM, g.srid)

    def _fn_main_angle(self, args):
        return self._g1("q2vt.main_angle", args, REAL)

    def _fn_inclination(self, args):
        raise NotTranslatable("inclination()")


def _epsg(authid) -> Optional[int]:
    if isinstance(authid, str):
        match = re.fullmatch(r"\s*EPSG:(\d+)\s*", authid, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


# Qt date format tokens and their PostgreSQL to_char equivalents, longest first.
_QT_DATE_TOKENS = [
    ("dddd", "FMDay"), ("ddd", "Dy"), ("dd", "DD"), ("d", "FMDD"),
    ("MMMM", "FMMonth"), ("MMM", "Mon"), ("MM", "MM"), ("M", "FMMM"),
    ("yyyy", "YYYY"), ("yy", "YY"),
    ("HH", "HH24"), ("H", "FMHH24"), ("hh", "HH24"), ("h", "FMHH24"),
    ("mm", "MI"), ("m", "FMMI"), ("ss", "SS"), ("s", "FMSS"),
    ("zzz", "MS"), ("z", "FMMS"), ("AP", "AM"), ("ap", "am"), ("A", "AM"), ("a", "am"),
]


def _qt_date_format(pattern: str) -> str:
    """A Qt date format (QDateTime::toString) as a to_char() format."""
    out, i = [], 0
    while i < len(pattern):
        if pattern[i] == "'":
            end = pattern.find("'", i + 1)
            if end < 0:
                end = len(pattern)
            text = pattern[i + 1:end] or "'"
            out.append('"' + text.replace('"', '\\"') + '"')
            i = end + 1
            continue
        for token, pg in _QT_DATE_TOKENS:
            if pattern.startswith(token, i):
                out.append(pg)
                i += len(token)
                break
        else:
            ch = pattern[i]
            out.append(f'"{ch}"' if ch.isalpha() else ch)
            i += 1
    return "".join(out)
