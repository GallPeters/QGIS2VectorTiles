"""
live_sql.py

LiveSqlWriter — writes q2vt_live.sql, the script that makes the export's
PostGIS layers live: run once in their database (QGIS's DB Manager, pgAdmin
or psql), it creates a schema whose tile function draws those layers'
tiles from the tables as they are at each request, as the export draws
them into the static tiles. A tile server (Martin, pg_tileserv) serves the
function; the web map reads those layers from it (the viewers' liveTiles
setting), the others from the export's PMTiles file.

The script does in SQL what the export does for these layers:

  * base rows (RulesExporter phases 1-2): each feature's geometry repaired,
    in EPSG:3857, cut into parts, lines and polygons simplified by
    _DATA_SIMPLIFICATION_TOLERANCE, its largest part flagged;
  * rule datasets (phase 3): each rule's filter, calculated fields (labels,
    data-defined properties, marker angles) and geometry (label points,
    centroid fills, marker points, geometry generators), clipped to the
    export extent; a feature's parts joined again, merged features,
    inverted polygons and merged label lines as the export makes them;
  * tiles (GDALTilesGenerator): each dataset at its zoom levels (lines,
    fills and labels also at the zoom below), simplified and small polygons
    left out as there, point clusters grouped at each zoom by QGIS's
    algorithm.

Every QGIS expression goes through PgCompiler (pg_expression.py). A layer
with anything the SQL can't compute as QGIS does stays static; the
script's header lists those layers and why.

Depends on: config, flattened_rule, ddp_fetcher, pg_expression, live_sql_library
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from qgis.core import (
    QgsDataSourceUri, QgsExpressionContextUtils, QgsProject, QgsProviderRegistry, QgsRectangle, QgsVectorLayer,
)

from ..utils.config import (
    _DATA_SIMPLIFICATION_TOLERANCE, _FIELD_PREFIX, _SIMPLIFICATION, _SIMPLIFICATION_LOW,
    _SIMPLIFICATION_LOW_ZOOM, _TILES_MAX_ZOOM,
)
from ..utils.flattened_rule import (
    CLUSTER_SIZE_FIELD, GROUP_ZOOM_FIELD, TILE_LAYERS_KEY, group_of, group_sources,
)
from .ddp_fetcher import COLOR_RAW_EXPRESSIONS
from .live_sql_library import library_sql
from .pg_expression import (
    BOOL, DATE, DATETIME, GEOM, INT, NULL, REAL, TEXT, TIME,
    NotTranslatable, PgCompiler, Sql, quote_ident, quote_literal,
)

# The style metadata key describing the live tile layers (the viewers draw
# them from the live tile server when one is set).
LIVE_KEY = "q2vt:live"
LIVE_FILE = "q2vt_live.sql"

_WORLD = 2 * 20037508.342789244
_TILE_PIXELS = 512
_MIN_POLYGON_PIXELS = 0.25
# Tile extent and buffer in tile units, as the GDAL MVT driver writes them.
_EXTENT = 4096
_BUFFER = 80
_ORIG_ID = f"{_FIELD_PREFIX}_orig_id"
_BIGGEST = f"{_FIELD_PREFIX}_biggest"
_DESCRIPTION = f"{_FIELD_PREFIX}_description"
_LABEL = f"{_FIELD_PREFIX}_label"
_PART_VARIABLE = "q2vt_part"
# A shared dataset's member number (RulesExporter.RULE_FIELD).
RULE_FIELD = f"{_FIELD_PREFIX}_rule"
# QMetaType codes -> kinds (others are read as text).
_QTYPE_KIND = {1: BOOL, 2: INT, 3: INT, 4: INT, 5: INT, 6: REAL, 38: REAL, 10: TEXT, 14: DATE,
               16: DATETIME, 15: TIME}
_KIND_PG = {INT: "bigint", REAL: "double precision", BOOL: "boolean", DATE: "date",
            DATETIME: "timestamp", TIME: "time", TEXT: "text"}
# Output field types (QMetaType codes) -> their SQL types in the tiles.
_OUT_PG = {1: "boolean", 2: "bigint", 3: "bigint", 4: "bigint", 5: "bigint", 6: "double precision",
           10: "text"}


def _enum_int(value) -> int:
    return int(getattr(value, "value", value))


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return (slug or "map")[:40]


@dataclass
class LiveLayer:
    """A PostGIS layer, as the script needs it (taken on QGIS's main thread)."""
    layer_id: str
    name: str
    database: str
    relation: str                        # "schema"."table", or (query) AS alias
    geometry_column: str
    key_columns: List[str]
    subset: str                          # The layer's own filter, in PostgreSQL.
    srid: Optional[int]
    family: Optional[int]                # 0 points, 1 lines, 2 polygons
    fields: List[Tuple[str, str, int]]   # name, kind, QMetaType code
    value_maps: Dict[str, Dict[str, str]] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    renderer: str = ""
    # Whether its rows have a physical order (a table's ctid): QGIS reads
    # them in it, and draws and groups them in it.
    physical_order: bool = False

    @classmethod
    def of(cls, layer: QgsVectorLayer) -> Optional["LiveLayer"]:
        if layer.providerType() != "postgres":
            return None
        uri = QgsDataSourceUri(layer.source())
        table = uri.table()
        if table.startswith("("):
            relation = f"{table} AS q2vt_query"
        else:
            relation = (f"{quote_ident(uri.schema())}." if uri.schema() else "") + quote_ident(table)
        authid = layer.crs().authid()
        srid = int(authid.split(":")[1]) if authid.upper().startswith("EPSG:") else None
        fields, value_maps = [], {}
        for i, f in enumerate(layer.fields()):
            code = _enum_int(f.type())
            fields.append((f.name(), _QTYPE_KIND.get(code, TEXT), code))
            setup = layer.editorWidgetSetup(i)
            if setup.type() == "ValueMap":
                mapping = {}
                config = setup.config().get("map")
                items = config if isinstance(config, list) else [config] if isinstance(config, dict) else []
                for item in items:
                    for shown, stored in (item or {}).items():
                        mapping[str(stored)] = str(shown)
                if mapping:
                    value_maps[f.name()] = mapping
        # The variables the export's rule passes have (global and project
        # scopes, RulesExporter._export_rules_parallel); others are NULL there.
        variables = {}
        for scope in (QgsExpressionContextUtils.globalScope(),
                      QgsExpressionContextUtils.projectScope(QgsProject.instance())):
            for name in scope.variableNames():
                value = scope.variable(name)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    variables[name] = value
        family = {0: 0, 1: 1, 2: 2}.get(_enum_int(layer.geometryType()))
        keys = [k.strip() for k in uri.keyColumn().replace('"', "").split(",") if k.strip()]
        physical = False
        if not table.startswith("("):
            physical = cls._is_table(layer, uri)
        return cls(
            layer_id=layer.id(), name=layer.name(),
            database=f"{uri.host()}:{uri.port()}/{uri.database()}",
            relation=relation, geometry_column=uri.geometryColumn(), key_columns=keys,
            subset=layer.subsetString() or uri.sql() or "", srid=srid, family=family,
            fields=fields, value_maps=value_maps, variables=variables,
            renderer=layer.renderer().type() if layer.renderer() else "",
            physical_order=physical,
        )

    @staticmethod
    def _is_table(layer: QgsVectorLayer, uri: QgsDataSourceUri) -> bool:
        """Whether the layer reads a table (or materialized view), whose rows
        have a ctid; asked of its database through QGIS's connection."""
        try:
            metadata = QgsProviderRegistry.instance().providerMetadata("postgres")
            connection = metadata.createConnection(layer.source(), {})
            name = (quote_ident(uri.schema()) + "." if uri.schema() else "") + quote_ident(uri.table())
            rows = connection.executeSql(
                f"SELECT relkind FROM pg_class WHERE oid = to_regclass({quote_literal(name)})")
            return bool(rows) and rows[0][0] in ("r", "m", "p")
        except Exception:  # noqa: BLE001  (key order then)
            return False


@dataclass
class _Dataset:
    """One rule dataset's live SQL: its function and tiled columns."""
    name: str
    function: str
    columns: List[Tuple[str, str]]       # tiled columns: name, SQL type
    zooms: Tuple[int, int]               # the tile zooms holding it
    last_zoom: int                       # its own highest zoom (may pass the tiles')
    margin: float = 0.125                # rows read around a tile, in tile widths
    grouped: Optional[dict] = None       # cluster metadata (see group_sources)
    group_source: Optional[str] = None   # the dataset whose points are grouped
    cluster: bool = False                # the cluster symbols' own dataset
    join_lines: bool = False             # lines joined below the highest zoom (_vrt_layer_block)


class LiveSqlWriter:
    """Write q2vt_live.sql for the export's PostGIS layers."""

    def __init__(self, project_name: str, extent: QgsRectangle, rule_groups,
                 live_layers: Dict[str, LiveLayer], style: dict):
        self.slug = _slug(project_name)
        self.schema = f"q2vt_{self.slug}"
        self.function = f"{self.slug}_tiles"
        self.extent = extent
        self.groups = rule_groups
        self.layers = live_layers
        self.style = style
        self.style_text = str(style)
        self.static_reasons: Dict[str, str] = {}

    # ------------------------------------------------------------------
    def write(self, path: str) -> Optional[dict]:
        """Write the script; returns the style metadata for the live layers
        ({layers, schema, function}), or None if no layer is live."""
        if not self.layers:
            return None
        shared = (self.style.get("metadata") or {}).get(TILE_LAYERS_KEY, {})
        grouped = group_sources(self.style.get("layers", []))
        styled = {l.get("source-layer") for l in self.style.get("layers", []) if l.get("source-layer")}
        by_layer: Dict[str, List[Any]] = {}
        for grp in self.groups:
            if grp.layer_id in self.layers:
                by_layer.setdefault(grp.layer_id, []).append(grp)
        self._directed = self._directed_layers(shared)
        first_zoom = min((g.zoom_range[0] for g in self.groups), default=0)
        last_zoom = min(max((g.zoom_range[1] for g in self.groups), default=_TILES_MAX_ZOOM), _TILES_MAX_ZOOM)

        bodies: List[str] = []
        datasets: List[_Dataset] = []
        for layer_id, groups in by_layer.items():
            layer = self.layers[layer_id]
            try:
                sql, sets = self._layer(layer, groups, shared, grouped, styled, first_zoom)
                tile_check = self._tile_function(sets, shared, last_zoom)  # Fails here, not later.
                del tile_check
            except NotTranslatable as e:
                self.static_reasons[layer.name] = str(e)
                continue
            bodies.append(sql)
            datasets.extend(sets)
        tiles = [self._tile_function(datasets, shared, last_zoom)] if datasets else []
        self._write_file(path, bodies, tiles)
        if not datasets:
            return None
        names = sorted({shared.get(d.name, d.name) for d in datasets})
        return {"layers": names, "schema": self.schema, "function": self.function, "file": LIVE_FILE}

    def _extent_box(self) -> str:
        e = self.extent
        return f"ST_MakeEnvelope({e.xMinimum()!r}, {e.yMinimum()!r}, {e.xMaximum()!r}, {e.yMaximum()!r}, 3857)"

    def _directed_layers(self, shared: Dict[str, str]) -> set:
        """Datasets whose lines stay as they are in the tiles (drawn along
        their direction, or labelled): GDALTilesGenerator._directed."""
        directed = set()
        for layer_def in self.style.get("layers", []):
            paint = layer_def.get("paint", {})
            if layer_def.get("type") == "symbol" or (layer_def.get("type") == "line" and any(
                    paint.get(key) not in (None, 0) for key in ("line-offset", "line-pattern", "line-gradient"))):
                directed.add(layer_def.get("source-layer"))
        return directed | {name for name, target in shared.items() if target in directed}

    # ------------------------------------------------------------------
    # One source layer
    # ------------------------------------------------------------------
    def _layer(self, layer: LiveLayer, groups, shared, grouped, styled, first_zoom):
        if layer.srid is None:
            raise NotTranslatable("its CRS has no EPSG code")
        if layer.family is None:
            raise NotTranslatable("it has no geometry")
        # Unique per layer (its id's hash), readable (its name), within
        # PostgreSQL's 63-character names.
        digest = hashlib.md5(layer.layer_id.encode("utf-8")).hexdigest()[:8]
        base = f"{self.schema}.{quote_ident('base_' + _slug(layer.name)[:40] + '_' + digest)}"
        parts = [self._base_function(layer, base)]
        sets: List[_Dataset] = []
        for grp in groups:
            if grp.output_dataset not in styled and shared.get(grp.output_dataset) not in styled:
                continue  # Not in the style: the export wrote no rows for it.
            source = group_of(grp.output_dataset, grouped)
            meta = grouped.get(source) if source else None
            if meta is not None and meta.get("role") != "cluster":
                raise NotTranslatable("point displacement renderers")
            sql, dataset = self._dataset(layer, base, grp, meta, source, first_zoom)
            parts.append(sql)
            sets.append(dataset)
        names = {d.name for d in sets}
        for d in sets:
            if d.grouped is not None:
                if d.group_source not in names:
                    raise NotTranslatable("a point cluster renderer whose cluster symbol has no tiles")
                parts.append(self._grouped_function(d, next(s for s in sets if s.name == d.group_source)))
        return "\n".join(parts), sets

    def _compiler(self, layer: LiveLayer) -> PgCompiler:
        fields = {name: (f"b.{quote_ident(name)}", kind) for name, kind, _ in layer.fields}
        fields[_ORIG_ID] = (f"b.{quote_ident(_ORIG_ID)}", TEXT)
        fields[_BIGGEST] = ("b.q2vt_biggest", INT)
        variables = {"id": Sql("b.q2vt_rank", REAL)}
        constants = {k: v for k, v in layer.variables.items() if k not in ("geometry", "feature", "id")}
        return PgCompiler(fields, "b.geom", 3857, variables, layer.value_maps, constants)

    def _base_function(self, layer: LiveLayer, base: str) -> str:
        """The layer's base rows over an area (EPSG:3857), as phases 1-2 make them."""
        g = quote_ident(layer.geometry_column)
        numeric_key = next((k for k, kind, _ in layer.fields if [k] == layer.key_columns and kind in (INT, REAL)),
                           None)
        if layer.key_columns:
            key = " || '|' || ".join(f"t.{quote_ident(k)}::text" for k in layer.key_columns)
        else:
            key = "t.ctid::text" if layer.physical_order else "md5(ST_AsEWKB(t." + quote_ident(layer.geometry_column) + ")::text)"
        if layer.physical_order:
            # The rows' physical order, as QGIS reads (and draws) them.
            rank = "((t.ctid::text::point)[0] * 65536 + (t.ctid::text::point)[1])"
        elif numeric_key:
            rank = f"t.{quote_ident(numeric_key)}::double precision"
        else:
            rank = "NULL::double precision"
        own = [(n, k) for n, k, _ in layer.fields if n != layer.geometry_column]
        select = "".join(
            f", t.{quote_ident(n)}" if k in _KIND_PG and k != TEXT
            else f", t.{quote_ident(n)}::text AS {quote_ident(n)}" for n, k in own)
        types = "".join(f", {quote_ident(n)} {_KIND_PG.get(k, 'text')}" for n, k in own)
        out = "".join(f", p.{quote_ident(n)}" for n, _ in own)
        subset = f" AND ({layer.subset})" if layer.subset else ""
        dim = layer.family
        size = {0: "0", 1: "ST_Length(p.part)", 2: "ST_Area(p.part)"}[dim]
        simplify = "p.part" if dim == 0 else f"ST_Simplify(p.part, {float(_DATA_SIMPLIFICATION_TOLERANCE)!r})"
        return f"""
-- Base rows of the "{layer.name}" layer: repaired, in EPSG:3857, in parts.
CREATE OR REPLACE FUNCTION {base}(area geometry)
RETURNS TABLE ({quote_ident(_ORIG_ID)} text, q2vt_rank double precision, q2vt_part integer, q2vt_biggest bigint{types}, geom geometry)
LANGUAGE sql STABLE PARALLEL SAFE AS $f$
  WITH s AS (
    SELECT {key} AS q2vt_key, {rank} AS q2vt_rank{select},
           q2vt.base_geometry(t.{g}, {layer.srid}, {dim}) AS geom
    FROM {layer.relation} t
    WHERE t.{g} && q2vt.area_in(area, {layer.srid}){subset}
  ), p AS (
    SELECT s.*, d.geom AS part, COALESCE(d.path[1], 1) AS part_no
    FROM s CROSS JOIN LATERAL ST_Dump(s.geom) d
  )
  SELECT p.q2vt_key, p.q2vt_rank, p.part_no,
         (row_number() OVER (PARTITION BY p.q2vt_key ORDER BY {size} DESC, p.part_no) = 1)::int::bigint{out},
         {simplify}
  FROM p
$f$;
"""

    # ------------------------------------------------------------------
    # One rule dataset
    # ------------------------------------------------------------------
    def _dataset(self, layer: LiveLayer, base: str, grp, meta, source, first_zoom):
        compiler = self._compiler(layer)
        # The output's dimension: 0 points, 1 lines, 2 polygons (QGIS's
        # geometry type, which the export flips only for its algorithm).
        family = _enum_int(grp.geometry_target)
        if family not in (0, 1, 2):
            raise NotTranslatable("a dataset of no geometry type")
        columns, member_exprs = self._fields(layer, grp)
        tiled = [(n, t) for n, t in columns if self._is_tiled(n)]

        geometry = compiler.compile(grp.geometry_expression)
        if geometry.kind not in (GEOM, NULL):
            raise NotTranslatable("a geometry expression giving no geometry")
        geom_sql = geometry.sql if geometry.kind == GEOM else "NULL::geometry"
        if geometry.kind == GEOM and geometry.srid is None:
            geom_sql = f"ST_SetSRID({geom_sql}, 3857)"
        elif geometry.kind == GEOM and geometry.srid != 3857:
            geom_sql = f"q2vt.transform({geom_sql}, 3857)"
        e = self.extent
        box = f"ST_MakeEnvelope({e.xMinimum()!r}, {e.yMinimum()!r}, {e.xMaximum()!r}, {e.yMaximum()!r}, 3857)"
        order_columns = []
        for i, (expr, ascending, nulls_first) in enumerate(grp.order_by):
            value = compiler.compile(expr)
            sql = compiler.to_num(value) if value.kind in (INT, REAL, BOOL) else f'{compiler.to_text(value)} COLLATE "C"'
            kind = "double precision" if value.kind in (INT, REAL, BOOL) else "text"
            order_columns.append((f"q2vt_o{i}", sql, kind, ascending, nulls_first))

        selects, point_selects = [], []
        for index, member in enumerate(grp.members):
            where = compiler.condition(member.filter_expression) if member.filter_expression else "true"
            if grp.keep_biggest_part:
                where = f"({where}) AND b.q2vt_biggest = 1"
            values = [self._value(layer, compiler, member_exprs[index].get(n), t) + f" AS {quote_ident(n)}"
                      for n, t in tiled]
            values.append(f"b.{quote_ident(_ORIG_ID)}")
            values += [f"({sql})::{kind} AS {name}" for name, sql, kind, _, _ in order_columns]
            selects.append(f"""
    SELECT {', '.join(values)}, b.q2vt_rank, b.{quote_ident(_ORIG_ID)} AS q2vt_key, {index} AS q2vt_member,
           b.q2vt_part, COALESCE(p.path[1], 1) AS q2vt_sub, ST_Force2D(p.geom) AS geom
    FROM base b
    CROSS JOIN LATERAL (SELECT {geom_sql} AS g) gx
    CROSS JOIN LATERAL ST_Dump(q2vt.clip_to(gx.g, {box})) p
    WHERE ({where}) AND ST_Dimension(p.geom) = {family}""")
            # The same rows without their fields (the costly part), for grouping.
            lite = [f"b.{quote_ident(_ORIG_ID)}"] + [f"({sql})::{kind} AS {name}"
                                                     for name, sql, kind, _, _ in order_columns]
            point_selects.append(selects[-1].replace(f"SELECT {', '.join(values)},", f"SELECT {', '.join(lite)},", 1))
        order = "".join(f"r.{name} {'ASC' if asc else 'DESC'} NULLS {'FIRST' if nf else 'LAST'}, "
                        for name, _, _, asc, nf in order_columns)
        order += "r.q2vt_rank, r.q2vt_key, r.q2vt_member, r.q2vt_part, r.q2vt_sub"
        cols = "".join(f"r.{quote_ident(n)}, " for n, _ in tiled)
        mode = self._merge_mode(grp, family, layer)
        if mode == "parts":
            final = (f"SELECT {cols}r.{quote_ident(_ORIG_ID)}, CASE WHEN count(*) = 1 THEN (array_agg(r.geom))[1] "
                     f"ELSE ST_Multi(ST_Collect(r.geom ORDER BY r.q2vt_n)) END AS geom, min(r.q2vt_n) AS q2vt_n "
                     f"FROM ordered r GROUP BY {cols}r.{quote_ident(_ORIG_ID)}")
        elif mode in ("merge", "lines", "invert"):
            union = "ST_LineMerge(ST_Union(r.geom))" if family == 1 else "ST_Union(r.geom)"
            if mode == "invert":
                w, h = e.width() * 0.05, e.height() * 0.05
                pad = (f"ST_MakeEnvelope({e.xMinimum() - w!r}, {e.yMinimum() - h!r}, "
                       f"{e.xMaximum() + w!r}, {e.yMaximum() + h!r}, 3857)")
                union = f"q2vt.difference({pad}, ST_Union(r.geom))"
            group = f" GROUP BY {cols[:-2]}" if tiled else ""
            final = (f"SELECT {cols}min(r.{quote_ident(_ORIG_ID)}) AS {quote_ident(_ORIG_ID)}, {union} AS geom, "
                     f"min(r.q2vt_n) AS q2vt_n FROM ordered r{group}")
            if mode == "lines":
                final = (f"SELECT {cols.replace('r.', 'm.')}m.{quote_ident(_ORIG_ID)}, d.geom, m.q2vt_n "
                         f"FROM ({final}) m CROSS JOIN LATERAL ST_Dump(m.geom) d WHERE ST_Dimension(d.geom) = 1")
        else:
            final = f"SELECT {cols}r.{quote_ident(_ORIG_ID)}, r.geom, r.q2vt_n FROM ordered r"
        function = f"{self.schema}.{quote_ident(grp.output_dataset)}"
        types = "".join(f"{quote_ident(n)} {_OUT_PG.get(t, 'text')}, " for n, t in tiled)
        rows = "\n    UNION ALL".join(selects)
        points_sql = ""
        if meta is not None:
            # A grouped layer's points, in drawing order, without fields.
            point_rows = "\n    UNION ALL".join(point_selects)
            points_sql = f"""
CREATE OR REPLACE FUNCTION {self.schema}.{quote_ident(grp.output_dataset + '__points')}(area geometry)
RETURNS TABLE ({quote_ident(_ORIG_ID)} text, geom geometry, q2vt_sort bigint)
LANGUAGE sql STABLE PARALLEL SAFE AS $f$
  WITH base AS (SELECT * FROM {base}(area)),
  rows AS ({point_rows}
  )
  SELECT r.{quote_ident(_ORIG_ID)}, r.geom, row_number() OVER (ORDER BY {order}) AS q2vt_sort FROM rows r
$f$;
"""
        sql = f"""
-- The {grp.label}: {grp.output_dataset}
CREATE OR REPLACE FUNCTION {function}(area geometry, keys text[] DEFAULT NULL)
RETURNS TABLE ({types}{quote_ident(_ORIG_ID)} text, geom geometry, q2vt_sort bigint)
LANGUAGE sql STABLE PARALLEL SAFE AS $f$
  WITH base AS (SELECT * FROM {base}(area) b WHERE keys IS NULL OR b.{quote_ident(_ORIG_ID)} = ANY(keys)),
  rows AS ({rows}
  ),
  ordered AS (SELECT r.*, row_number() OVER (ORDER BY {order}) AS q2vt_n FROM rows r)
  SELECT f.* FROM ({final}) f ORDER BY f.q2vt_n
$f$;
""" + points_sql
        low, high = grp.zoom_range
        grouped = meta is not None
        if not grouped and (grp.rule_type == 1 or (grp.rule_type == 0 and family in (1, 2))):
            low = max(low - 1, first_zoom)  # Also in the zoom below's tiles (_layer_zooms).
        generator = re.search(r"transform\(|buffer\(|translate\(|offset_curve\(|project\(", grp.geometry_expression)
        return sql, _Dataset(
            name=grp.output_dataset, function=function,
            columns=[(n, _OUT_PG.get(t, "text")) for n, t in tiled],
            zooms=(min(low, _TILES_MAX_ZOOM), min(high, _TILES_MAX_ZOOM)), last_zoom=high,
            margin=1.0 if generator else 0.125, grouped=meta, group_source=source,
            cluster=grouped and source == grp.output_dataset,
            join_lines=grp.rule_type == 0 and family == 1 and grp.output_dataset not in self._directed)

    def _fields(self, layer: LiveLayer, grp) -> Tuple[List[Tuple[str, int]], List[Dict[str, str]]]:
        """The dataset's fields and each member's expressions, as
        RulesExporter._dataset_fields makes them."""
        columns: List[Tuple[str, int]] = []
        names = set()

        def add(name: str, field_type) -> None:
            if name not in names:
                names.add(name)
                columns.append((name, 10 if field_type is None else _enum_int(field_type)))

        exprs: List[Dict[str, str]] = [{} for _ in grp.members]
        if grp.tagged:
            add(RULE_FIELD, 2)
        for index, member in enumerate(grp.members):
            if grp.tagged:
                exprs[index][RULE_FIELD] = str(index)
            for field_type, expr, name in member.expression_fields:
                add(name, field_type)
                exprs[index].setdefault(name, expr)
        for name, kind, code in layer.fields:
            if name == layer.geometry_column or "ogc_fid" in name.lower():
                continue
            if grp.copied_fields is None or name in grp.copied_fields:
                add(name, code if code in _OUT_PG else 10)
                for e in exprs:
                    e.setdefault(name, f'"{name}"')
        return columns, exprs

    def _value(self, layer: LiveLayer, compiler: PgCompiler, expr, field_type: int) -> str:
        pg = _OUT_PG.get(field_type, "text")
        if not isinstance(expr, str) or not expr.strip():
            return f"NULL::{pg}"
        raw = COLOR_RAW_EXPRESSIONS.get(expr)
        if raw is not None:  # A colour: as #RRGGBBAA, as the tiles hold them.
            return f"q2vt.color_hex({compiler.to_text(compiler.compile(raw))})"
        if f"@{_PART_VARIABLE}" in expr:
            part = self._compiler(layer)
            part.variables[_PART_VARIABLE] = Sql("p.geom", GEOM, 3857)
            return f"({part.value(expr, field_type)})::{pg}"
        return f"({compiler.value(expr, field_type)})::{pg}"

    def _is_tiled(self, name: str) -> bool:
        if name in (_ORIG_ID, _BIGGEST, _DESCRIPTION):
            return False
        if f"{_FIELD_PREFIX}_property_" in name:
            return name in self.style_text
        return True

    @staticmethod
    def _merge_mode(grp, family: int, layer: LiveLayer) -> str:
        """As RulesExporter._collect_parts, _merge_mode and _merge_label_lines."""
        if grp.merge_lines:
            return "lines"
        if grp.rule_type != 0:
            return "none"
        if any(name.startswith(_LABEL) for m in grp.members for _, _, name in m.expression_fields):
            return "none"
        if layer.renderer == "mergedFeatureRenderer" and family in (1, 2):
            return "merge"
        if layer.renderer == "invertedPolygonRenderer" and family == 2:
            return "invert"
        return "parts"

    # ------------------------------------------------------------------
    # Point clusters
    # ------------------------------------------------------------------
    def _grouped_function(self, d: _Dataset, source: _Dataset) -> str:
        """A grouped dataset's rows at a zoom: the cluster symbols' (groups of
        several points: their centre, counting them) or the other point
        datasets' (their points outside such groups)."""
        distance = (d.grouped or {}).get("distance") or {}
        search = (f"{float(distance['m'])!r}" if distance.get("m")
                  else f"{float(distance.get('px', 0))!r} * {_WORLD!r} / ({_TILE_PIXELS} * 2 ^ gz)")
        types = "".join(f"{quote_ident(n)} {t}, " for n, t in d.columns)
        cols = "".join(f"m.{quote_ident(n)}, " for n, _ in d.columns)
        groups = f"""
  WITH pts AS (
    SELECT s.*, row_number() OVER (ORDER BY s.q2vt_sort) AS q2vt_i,
           ST_X(ST_GeometryN(ST_Multi(s.geom), 1)) AS px, ST_Y(ST_GeometryN(ST_Multi(s.geom), 1)) AS py
    FROM {self.schema}.{quote_ident(source.name + '__points')}(CASE WHEN ST_XMax(area) - ST_XMin(area) + 16 * ({search}) > {0.5 * max(self.extent.width(), self.extent.height())!r}
      THEN {self._extent_box()} ELSE ST_Expand(area, 8 * ({search})) END) s
  ), g AS (
    SELECT q2vt.cluster_groups(array_agg(px ORDER BY q2vt_i), array_agg(py ORDER BY q2vt_i), {search}) AS gi
    FROM pts
  ), m AS (SELECT p.*, g.gi[p.q2vt_i] AS q2vt_g FROM pts p, g),
  big AS (
    SELECT q2vt_g, count(*) AS n, avg(px) AS cx, avg(py) AS cy, min(q2vt_i) AS fi
    FROM m GROUP BY q2vt_g HAVING count(*) > 1
  )"""
        name = f"{self.schema}.{quote_ident(d.name + '__grouped')}"
        if d.cluster:
            return f"""
CREATE OR REPLACE FUNCTION {name}(area geometry, gz integer)
RETURNS TABLE ({types}{quote_ident(CLUSTER_SIZE_FIELD)} bigint, geom geometry, q2vt_sort bigint)
LANGUAGE sql STABLE PARALLEL SAFE AS $f${groups}
  , firsts AS (
    SELECT m.{quote_ident(_ORIG_ID)} AS k, m.q2vt_sort AS head_sort, m.px, m.py, big.n, big.cx, big.cy
    FROM big JOIN m ON m.q2vt_i = big.fi
  ),
  rows AS (
    -- The fields of each cluster's first point only (that point's own row).
    SELECT DISTINCT ON (f.k) f.head_sort, f.n, f.cx, f.cy{''.join(', d.' + quote_ident(n) for n, _ in d.columns)}
    FROM firsts f JOIN {d.function}(
        (SELECT ST_Expand(ST_SetSRID(ST_Extent(ST_MakePoint(px, py))::geometry, 3857), 1) FROM firsts),
        ARRAY(SELECT k FROM firsts)) d
      ON d.{quote_ident(_ORIG_ID)} = f.k
    ORDER BY f.k, ST_Distance(ST_GeometryN(ST_Multi(d.geom), 1), ST_SetSRID(ST_MakePoint(f.px, f.py), 3857))
  )
  SELECT {cols.replace('m.', 'r.')}r.n, ST_SetSRID(ST_MakePoint(r.cx, r.cy), 3857), r.head_sort
  FROM rows r ORDER BY r.head_sort
$f$;
"""
        own = "".join(f"o.{quote_ident(n)}, " for n, _ in d.columns)
        return f"""
CREATE OR REPLACE FUNCTION {name}(area geometry, gz integer)
RETURNS TABLE ({types}geom geometry, q2vt_sort bigint)
LANGUAGE sql STABLE PARALLEL SAFE AS $f${groups},
  clustered AS (
    SELECT m.{quote_ident(_ORIG_ID)} AS k, round(m.px::numeric, 3) AS x, round(m.py::numeric, 3) AS y
    FROM m JOIN big ON m.q2vt_g = big.q2vt_g
  )
  , singles AS (
    -- The tile's points outside groups: only their fields are computed.
    SELECT q.{quote_ident(_ORIG_ID)} AS k FROM {self.schema}.{quote_ident(d.name + '__points')}(area) q
    WHERE NOT EXISTS (
      SELECT 1 FROM clustered c WHERE c.k = q.{quote_ident(_ORIG_ID)}
        AND c.x = round(ST_X(ST_GeometryN(ST_Multi(q.geom), 1))::numeric, 3)
        AND c.y = round(ST_Y(ST_GeometryN(ST_Multi(q.geom), 1))::numeric, 3))
  )
  SELECT {own}o.geom, o.q2vt_sort FROM {d.function}(area, ARRAY(SELECT DISTINCT k FROM singles)) o
  WHERE NOT EXISTS (
    SELECT 1 FROM clustered c WHERE c.k = o.{quote_ident(_ORIG_ID)}
      AND c.x = round(ST_X(ST_GeometryN(ST_Multi(o.geom), 1))::numeric, 3)
      AND c.y = round(ST_Y(ST_GeometryN(ST_Multi(o.geom), 1))::numeric, 3))
  ORDER BY o.q2vt_sort
$f$;
"""

    # ------------------------------------------------------------------
    # The tile function
    # ------------------------------------------------------------------
    def _tile_function(self, datasets: List[_Dataset], shared: Dict[str, str], last_zoom: int) -> str:
        by_tile: Dict[str, List[_Dataset]] = {}
        for d in datasets:
            by_tile.setdefault(shared.get(d.name, d.name), []).append(d)
        blocks = []
        for tile_layer, members in by_tile.items():
            columns: Dict[str, str] = {}
            for d in members:
                for n, t in d.columns:
                    columns.setdefault(n, t)
                if d.cluster:
                    columns.setdefault(CLUSTER_SIZE_FIELD, "bigint")
                if d.grouped is not None and d.last_zoom > _TILES_MAX_ZOOM:
                    columns.setdefault(GROUP_ZOOM_FIELD, "bigint")
            blocks.append(self._tile_block(tile_layer, members, columns))
        return f"""
-- The tiles: each layer at its zoom levels, as the export makes them.
CREATE OR REPLACE FUNCTION {self.schema}.{quote_ident(self.function)}(z integer, x integer, y integer)
RETURNS bytea
LANGUAGE plpgsql STABLE PARALLEL SAFE AS $f$
DECLARE
  env geometry := ST_TileEnvelope(z, x, y);
  width double precision := {_WORLD!r} / 2 ^ z;
  tolerance double precision := CASE WHEN z >= {last_zoom} THEN 0
    WHEN z <= {_SIMPLIFICATION_LOW_ZOOM} THEN {_SIMPLIFICATION_LOW} ELSE {_SIMPLIFICATION} END * width / {_EXTENT};
  min_area double precision := CASE WHEN z >= {last_zoom} THEN NULL
    ELSE {_MIN_POLYGON_PIXELS} * (width / {_TILE_PIXELS}) ^ 2 END;
  last_zoom integer := {last_zoom};
  tile bytea := '';
  layer bytea;
BEGIN
{''.join(blocks)}
  RETURN tile;
END
$f$;
"""

    def _tile_block(self, tile_layer: str, members: List[_Dataset], columns: Dict[str, str]) -> str:
        """One tile layer: its datasets' rows (each at its own zooms) in one
        ST_AsMVT, a layer name appearing once per tile."""
        geom = ("ST_AsMVTGeom(CASE WHEN tolerance > 0 AND ST_Dimension(r.geom) > 0 "
                f"THEN ST_Simplify(r.geom, tolerance, true) ELSE r.geom END, env, {_EXTENT}, {_BUFFER}, true)")
        parts = []
        for index, d in enumerate(members):
            own = {n for n, _ in d.columns}
            extra = {}
            if d.cluster:
                own.add(CLUSTER_SIZE_FIELD)
            low, high = d.zooms
            if d.grouped is None:
                source = f"{d.function}(ST_Expand(env, width * {d.margin!r})) r"
            else:
                last = max(d.last_zoom, high)
                if d.last_zoom > _TILES_MAX_ZOOM:
                    zooms = f"generate_series(z, CASE WHEN z >= {_TILES_MAX_ZOOM} THEN {last} ELSE z END)"
                    # Only the highest tiles' copies (one per zoom beyond) say theirs.
                    extra[GROUP_ZOOM_FIELD] = f"CASE WHEN gz.zoom >= {_TILES_MAX_ZOOM} THEN gz.zoom END"
                else:
                    zooms = "z"
                source = (f"(SELECT {zooms} AS zoom) gz CROSS JOIN LATERAL "
                          f"{self.schema}.{quote_ident(d.name + '__grouped')}(ST_Expand(env, width * 0.125), gz.zoom) r")
            values = []
            for name, pg in columns.items():
                if name in extra:
                    values.append(f"({extra[name]})::{pg} AS {quote_ident(name)}")
                elif name in own:
                    values.append(f"r.{quote_ident(name)}")
                else:
                    values.append(f"NULL::{pg} AS {quote_ident(name)}")
            if d.join_lines:
                # Below the highest zoom, lines alike in every tiled field are
                # joined where they meet within cells a tile wide.
                keys = "".join(f"r.{quote_ident(n)}, " for n, _ in d.columns)
                source = (f"(SELECT {keys}ST_LineMerge(ST_Collect(r.geom)) AS geom, min(r.q2vt_sort) AS q2vt_sort FROM {source} GROUP BY {keys}"
                          f"CASE WHEN z >= last_zoom THEN r.q2vt_sort END, "
                          f"floor(ST_XMin(r.geom) / width), floor(ST_YMin(r.geom) / width)) r")
            parts.append(
                f"SELECT {geom} AS geom{''.join(', ' + v for v in values)}, {index} AS q2vt_d, r.q2vt_sort "
                f"FROM {source} WHERE z BETWEEN {low} AND {high} AND q2vt.big_enough(r.geom, min_area)")
        name = quote_literal(tile_layer)
        low = min(d.zooms[0] for d in members)
        high = max(d.zooms[1] for d in members)
        union = "\n        UNION ALL ".join(parts)
        cols = "".join(f", u.{quote_ident(n)}" for n in columns)
        return f"""
  IF z BETWEEN {low} AND {high} THEN
    BEGIN
      SELECT ST_AsMVT(t, {name}, {_EXTENT}, 'geom') INTO layer FROM (
        SELECT u.geom{cols} FROM (
        {union}) u WHERE u.geom IS NOT NULL ORDER BY u.q2vt_d, u.q2vt_sort) t;
      tile := tile || COALESCE(layer, ''::bytea);
    EXCEPTION WHEN OTHERS THEN
      RAISE WARNING 'q2vt: % left out of tile %/%/%: %', {name}, z, x, y, SQLERRM;
    END;
  END IF;
"""

    # ------------------------------------------------------------------
    def _write_file(self, path: str, bodies: List[str], tiles: List[str]) -> None:
        live = sorted({l.name for l in self.layers.values()} - set(self.static_reasons))
        lines = [
            "-- QGIS2VectorTiles: live tiles for this export's PostGIS layers.",
            "--",
            "-- Run this script once in the layers' database (QGIS: Database > DB Manager >",
            "-- SQL Window; pgAdmin: Query Tool; or psql -f q2vt_live.sql). It creates the",
            f"-- schemas q2vt (shared functions) and {self.schema} (this map's), and changes",
            "-- none of your tables. Run it again after exporting the map again.",
            "--",
            f"-- The tiles: {self.schema}.{self.function}(z, x, y), served by a tile server, e.g.",
            f"--   Martin:      https://<server>/{self.function}/{{z}}/{{x}}/{{y}}",
            f"--   pg_tileserv: https://<server>/{self.schema}.{self.function}/{{z}}/{{x}}/{{y}}.pbf",
            "-- Then set liveTiles to that address in the web map's index.html (or open the",
            "-- web map with ?live=<that address>).",
            "--",
            "-- Needs PostgreSQL 12+ with PostGIS 3.2+. Each tile is made from the tables at",
            "-- its request: a tile server's cache would keep serving older tiles (Martin:",
            "-- cache_size_mb: 0, or a short-lived HTTP cache in front, keeps them current).",
            "--",
            "-- Live layers (drawn from the tables as they are at each request):",
        ]
        lines += [f"--   {name}" for name in live] or ["--   (none)"]
        if self.static_reasons:
            lines.append("-- Layers left static (drawn from the export's tiles only):")
            lines += [f"--   {name}: {reason}" for name, reason in sorted(self.static_reasons.items())]
        e = self.extent
        lines.append(f"-- Extent (EPSG:3857): {e.xMinimum():.1f}, {e.yMinimum():.1f}, {e.xMaximum():.1f}, "
                     f"{e.yMaximum():.1f}: features beyond it are left out, as in the export.")
        databases = sorted({l.database for name, l in ((l.name, l) for l in self.layers.values())
                            if name in live})
        if len(databases) > 1:
            lines.append("-- Note: the live layers come from several databases (" + ", ".join(databases)
                         + "): run the script in each; a layer's functions work in its own.")
        script = "\n".join(lines) + "\n\n"
        if bodies:
            script += ("BEGIN;\nCREATE SCHEMA IF NOT EXISTS q2vt;\n"
                       f"DROP SCHEMA IF EXISTS {self.schema} CASCADE;\nCREATE SCHEMA {self.schema};\n")
            script += library_sql() + _EXPORT_LIBRARY + "\n".join(bodies) + "\n".join(tiles) + "\nCOMMIT;\n"
        # Newlines written as they are: those inside SQL texts are labels' own.
        with open(path, "w", encoding="utf-8", newline=chr(10)) as f:
            f.write(script)


# The export's own steps (base rows, extent clip, clusters).
_EXPORT_LIBRARY = r"""
-- A source geometry as the export's base rows hold it: 2D, curves as
-- lines, repaired keeping parts of its own dimension, in EPSG:3857.
CREATE OR REPLACE FUNCTION q2vt.base_geometry(g geometry, srid int, dim int) RETURNS geometry
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE fixed geometry;
BEGIN
  IF g IS NULL THEN RETURN NULL; END IF;
  g := ST_Force2D(ST_CurveToLine(g));
  IF ST_SRID(g) = 0 THEN g := ST_SetSRID(g, srid); END IF;
  IF dim > 0 AND NOT ST_IsValid(g) THEN
    fixed := ST_MakeValid(g, 'method=linework');
    g := CASE WHEN ST_Dimension(fixed) = dim AND GeometryType(fixed) <> 'GEOMETRYCOLLECTION'
              THEN fixed ELSE ST_CollectionExtract(fixed, dim + 1) END;
    IF g IS NULL OR ST_IsEmpty(g) THEN RETURN NULL; END IF;
  END IF;
  RETURN ST_Transform(g, 3857);
EXCEPTION WHEN OTHERS THEN RETURN NULL;
END $$;

-- An area (EPSG:3857) in a layer's CRS, to find its rows by: within the
-- Web Mercator world (beyond it, longitudes would wrap to the other side),
-- its edges densified so that another projection's bends stay inside.
CREATE OR REPLACE FUNCTION q2vt.area_in(area geometry, srid int) RETURNS geometry
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE w double precision := 20037508.342789244; box geometry;
BEGIN
  box := ST_MakeEnvelope(GREATEST(ST_XMin(area), -w), GREATEST(ST_YMin(area), -w),
                         LEAST(ST_XMax(area), w), LEAST(ST_YMax(area), w), 3857);
  IF srid = 3857 THEN RETURN box; END IF;
  RETURN ST_Envelope(ST_Transform(ST_Segmentize(box, GREATEST(ST_XMax(box) - ST_XMin(box), ST_YMax(box) - ST_YMin(box)) / 16), srid));
EXCEPTION WHEN OTHERS THEN RETURN ST_Transform(box, srid);
END $$;

-- The part of a geometry inside the export extent (QgsGeometry::clipped).
CREATE OR REPLACE FUNCTION q2vt.clip_to(g geometry, box geometry) RETURNS geometry
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN
  IF g IS NULL OR ST_IsEmpty(g) THEN RETURN NULL; END IF;
  IF ST_SRID(g) <> ST_SRID(box) THEN g := ST_SetSRID(g, ST_SRID(box)); END IF;
  IF ST_Covers(box, ST_Envelope(g)) THEN RETURN g; END IF;
  IF NOT (g && box) THEN RETURN NULL; END IF;
  RETURN ST_ClipByBox2D(g, box);
EXCEPTION WHEN OTHERS THEN
  BEGIN RETURN ST_Intersection(g, box); EXCEPTION WHEN OTHERS THEN RETURN NULL; END;
END $$;

-- QGIS's point groups (QgsPointDistanceRenderer): the points, in drawing
-- order, each join the group whose first point lies within the distance
-- (a square) and whose centre is nearest, else start one. Returns each
-- point's group number. Groups are found through a grid of cells the
-- distance wide, keyed by their first point's cell, as QGIS finds them.
CREATE OR REPLACE FUNCTION q2vt.cluster_groups(xs double precision[], ys double precision[], search double precision)
RETURNS int[]
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE
  n int := COALESCE(cardinality(xs), 0);
  cell double precision := CASE WHEN search > 0 THEN search ELSE 1 END;
  x0 double precision; y0 double precision; cols int; rows int;
  head int[]; nxt int[]; ox double precision[]; oy double precision[];
  mx double precision[]; my double precision[]; cnt int[]; result int[];
  groups int := 0; i int; cx int; cy int; gx int; gy int; g int; best int; best_d double precision; d double precision;
  slot int;
BEGIN
  IF n = 0 THEN RETURN ARRAY[]::int[]; END IF;
  SELECT min(v), max(v) INTO x0, d FROM unnest(xs) v;
  cols := floor((d - x0) / cell)::int + 3;
  SELECT min(v), max(v) INTO y0, d FROM unnest(ys) v;
  rows := floor((d - y0) / cell)::int + 3;
  x0 := x0 - cell; y0 := y0 - cell;
  head := array_fill(0, ARRAY[cols * rows]);
  nxt := array_fill(0, ARRAY[n]); ox := array_fill(0::double precision, ARRAY[n]);
  oy := ox; mx := ox; my := ox; cnt := array_fill(0, ARRAY[n]); result := array_fill(0, ARRAY[n]);
  FOR i IN 1 .. n LOOP
    IF xs[i] IS NULL OR ys[i] IS NULL THEN CONTINUE; END IF;
    cx := floor((xs[i] - x0) / cell)::int; cy := floor((ys[i] - y0) / cell)::int;
    best := 0; best_d := NULL;
    FOR gx IN GREATEST(cx - 1, 0) .. LEAST(cx + 1, cols - 1) LOOP
      FOR gy IN GREATEST(cy - 1, 0) .. LEAST(cy + 1, rows - 1) LOOP
        g := head[gy * cols + gx + 1];
        WHILE g > 0 LOOP
          IF abs(ox[g] - xs[i]) <= search AND abs(oy[g] - ys[i]) <= search THEN
            d := (mx[g] - xs[i]) ^ 2 + (my[g] - ys[i]) ^ 2;
            IF best_d IS NULL OR d < best_d OR (d = best_d AND g < best) THEN best := g; best_d := d; END IF;
          END IF;
          g := nxt[g];
        END LOOP;
      END LOOP;
    END LOOP;
    IF best = 0 THEN
      groups := groups + 1; best := groups;
      slot := cy * cols + cx + 1;
      nxt[best] := head[slot]; head[slot] := best;
      ox[best] := xs[i]; oy[best] := ys[i]; mx[best] := xs[i]; my[best] := ys[i]; cnt[best] := 1;
    ELSE
      mx[best] := (mx[best] * cnt[best] + xs[i]) / (cnt[best] + 1);
      my[best] := (my[best] * cnt[best] + ys[i]) / (cnt[best] + 1);
      cnt[best] := cnt[best] + 1;
    END IF;
    result[i] := best;
  END LOOP;
  RETURN result;
END $$;
"""
