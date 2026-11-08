"""
live_sql_library.py

The q2vt schema's helper functions, written at the top of q2vt_live.sql:
QGIS's ways of computing values (pg_expression.PgCompiler calls them), in
PostgreSQL. Each gives NULL where QGIS gives NULL (or fails on one value):
an SQL error would fail the whole tile.

library_sql() returns the SQL; colour names are QColor's own (QGIS reads
named colours through it).

Depends on: nothing of the plugin's
"""

from qgis.PyQt.QtGui import QColor

_LIBRARY = r"""
-- ---------------------------------------------------------------------
-- Numbers and text, as QGIS converts them
-- ---------------------------------------------------------------------

-- A text's number (QVariant::toDouble): NULL if it is not one.
CREATE OR REPLACE FUNCTION q2vt.num(t text) RETURNS double precision
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN t ~ '^\s*[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?\s*$'
              THEN trim(t)::double precision END
$$;

-- A whole number's text (to_int('3') = 3; to_int('3.7') is NULL in QGIS).
CREATE OR REPLACE FUNCTION q2vt.text_int(t text) RETURNS bigint
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN t ~ '^\s*[-+]?[0-9]{1,18}\s*$' THEN trim(t)::bigint END
$$;

-- QVariant::toBool of a text: false if empty, '0' or 'false'.
CREATE OR REPLACE FUNCTION q2vt.truth(t text) RETURNS boolean
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN t IS NULL THEN NULL ELSE NOT (lower(t) IN ('', '0', 'false')) END
$$;

-- Rounding half away from zero (qRound64, std::round).
CREATE OR REPLACE FUNCTION q2vt.qround(x double precision) RETURNS bigint
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN x IS NULL OR x <> x OR abs(x) > 9.2e18 THEN NULL
              ELSE (sign(x) * floor(abs(x) + 0.5))::bigint END
$$;

CREATE OR REPLACE FUNCTION q2vt.round_to(x double precision, places bigint) RETURNS double precision
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN x IS NULL OR places IS NULL OR x <> x THEN NULL
              ELSE sign(x * power(10::double precision, places))
                   * floor(abs(x * power(10::double precision, places)) + 0.5)
                   / power(10::double precision, places) END
$$;

CREATE OR REPLACE FUNCTION q2vt.div(a double precision, b double precision) RETURNS double precision
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT CASE WHEN b = 0 THEN NULL ELSE a / b END $$;

CREATE OR REPLACE FUNCTION q2vt.intdiv(a double precision, b double precision) RETURNS bigint
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN b = 0 OR abs(a / b) > 9.2e18 THEN NULL ELSE floor(a / b)::bigint END
$$;

CREATE OR REPLACE FUNCTION q2vt.imod(a bigint, b bigint) RETURNS bigint
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT CASE WHEN b = 0 THEN NULL ELSE a % b END $$;

CREATE OR REPLACE FUNCTION q2vt.fmod(a double precision, b double precision) RETURNS double precision
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN b = 0 THEN NULL ELSE a - trunc(a / b) * b END
$$;

CREATE OR REPLACE FUNCTION q2vt.pow(a double precision, b double precision) RETURNS double precision
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN
  RETURN power(a, b);
EXCEPTION WHEN OTHERS THEN RETURN NULL;
END $$;

CREATE OR REPLACE FUNCTION q2vt.sqrt(x double precision) RETURNS double precision
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT CASE WHEN x >= 0 THEN sqrt(x) END $$;

CREATE OR REPLACE FUNCTION q2vt.ln(x double precision) RETURNS double precision
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT CASE WHEN x > 0 THEN ln(x) END $$;

CREATE OR REPLACE FUNCTION q2vt.log10(x double precision) RETURNS double precision
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT CASE WHEN x > 0 THEN log(x) END $$;

CREATE OR REPLACE FUNCTION q2vt.logb(base double precision, x double precision) RETURNS double precision
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN x > 0 AND base > 0 AND base <> 1 THEN ln(x) / ln(base) END
$$;

CREATE OR REPLACE FUNCTION q2vt.asin(x double precision) RETURNS double precision
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT CASE WHEN x BETWEEN -1 AND 1 THEN asin(x) END $$;

CREATE OR REPLACE FUNCTION q2vt.acos(x double precision) RETURNS double precision
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT CASE WHEN x BETWEEN -1 AND 1 THEN acos(x) END $$;

CREATE OR REPLACE FUNCTION q2vt.scale_linear(v double precision, dmin double precision, dmax double precision,
    rmin double precision, rmax double precision) RETURNS double precision
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN dmin = dmax THEN NULL
              WHEN v <= LEAST(dmin, dmax) AND dmin < dmax THEN rmin
              WHEN v >= GREATEST(dmin, dmax) AND dmin < dmax THEN rmax
              ELSE (rmax - rmin) / (dmax - dmin) * (LEAST(GREATEST(v, LEAST(dmin, dmax)), GREATEST(dmin, dmax)) - dmin) + rmin END
$$;

CREATE OR REPLACE FUNCTION q2vt.scale_exp(v double precision, dmin double precision, dmax double precision,
    rmin double precision, rmax double precision, e double precision) RETURNS double precision
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN dmin = dmax OR e <= 0 THEN NULL
              WHEN v <= dmin THEN rmin WHEN v >= dmax THEN rmax
              ELSE (rmax - rmin) * power((v - dmin) / (dmax - dmin), e) + rmin END
$$;

CREATE OR REPLACE FUNCTION q2vt.scale_exponential(v double precision, dmin double precision, dmax double precision,
    rmin double precision, rmax double precision, e double precision) RETURNS double precision
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN dmin = dmax OR e <= 0 THEN NULL
              WHEN v <= dmin THEN rmin WHEN v >= dmax THEN rmax
              ELSE (rmax - rmin) / (power(e, dmax - dmin) - 1) * (power(e, v - dmin) - 1) + rmin END
$$;

-- A number as QGIS writes it (QString::number(x, 'g', shortest)): its
-- shortest digits, in whichever of decimal or scientific notation is
-- shorter (1e+06, 120000, 0.001, 1e-04).
CREATE OR REPLACE FUNCTION q2vt.qstr(x double precision) RETURNS text
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE SET extra_float_digits = 1 AS $$
DECLARE
  s text; neg boolean; mant text; e int := 0; digits text; decpt int; dot int;
  fixed text; sci text; ex int;
BEGIN
  IF x IS NULL THEN RETURN NULL; END IF;
  IF x <> x THEN RETURN 'nan'; END IF;
  IF x = 'Infinity'::double precision THEN RETURN 'inf'; END IF;
  IF x = '-Infinity'::double precision THEN RETURN '-inf'; END IF;
  IF x = 0 THEN RETURN '0'; END IF;
  s := x::text;
  neg := left(s, 1) = '-';
  IF neg THEN s := substr(s, 2); END IF;
  IF position('e' in s) > 0 THEN
    mant := split_part(s, 'e', 1); e := split_part(s, 'e', 2)::int;
  ELSE
    mant := s;
  END IF;
  dot := position('.' in mant);
  IF dot = 0 THEN decpt := length(mant); digits := mant;
  ELSE decpt := dot - 1; digits := replace(mant, '.', ''); END IF;
  decpt := decpt + e;
  -- Leading zeros leave the digits, moving the point; trailing ones go.
  WHILE left(digits, 1) = '0' AND length(digits) > 1 LOOP
    digits := substr(digits, 2); decpt := decpt - 1;
  END LOOP;
  digits := rtrim(digits, '0');
  IF digits = '' THEN RETURN '0'; END IF;
  IF decpt <= 0 THEN
    fixed := '0.' || repeat('0', -decpt) || digits;
  ELSIF decpt >= length(digits) THEN
    fixed := digits || repeat('0', decpt - length(digits));
  ELSE
    fixed := left(digits, decpt) || '.' || substr(digits, decpt + 1);
  END IF;
  ex := decpt - 1;
  sci := left(digits, 1) || CASE WHEN length(digits) > 1 THEN '.' || substr(digits, 2) ELSE '' END
         || 'e' || CASE WHEN ex < 0 THEN '-' ELSE '+' END || lpad(abs(ex)::text, 2, '0');
  RETURN CASE WHEN neg THEN '-' ELSE '' END || CASE WHEN length(sci) < length(fixed) THEN sci ELSE fixed END;
END $$;

-- upper() / lower() over all of Unicode, as QGIS (Qt) does, whatever the
-- database's locale: through ICU's root collation where the server has it.
DO $do$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_collation WHERE collname = 'und-x-icu') THEN
    EXECUTE $f$CREATE OR REPLACE FUNCTION q2vt.upper(t text) RETURNS text
      LANGUAGE sql IMMUTABLE PARALLEL SAFE AS 'SELECT upper(t COLLATE "und-x-icu")'$f$;
    EXECUTE $f$CREATE OR REPLACE FUNCTION q2vt.lower(t text) RETURNS text
      LANGUAGE sql IMMUTABLE PARALLEL SAFE AS 'SELECT lower(t COLLATE "und-x-icu")'$f$;
  ELSE
    EXECUTE $f$CREATE OR REPLACE FUNCTION q2vt.upper(t text) RETURNS text
      LANGUAGE sql IMMUTABLE PARALLEL SAFE AS 'SELECT upper(t)'$f$;
    EXECUTE $f$CREATE OR REPLACE FUNCTION q2vt.lower(t text) RETURNS text
      LANGUAGE sql IMMUTABLE PARALLEL SAFE AS 'SELECT lower(t)'$f$;
  END IF;
END $do$;

-- title(): each word (between spaces) in lower case with a capital first letter.
CREATE OR REPLACE FUNCTION q2vt.title(t text) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT string_agg(q2vt.upper(left(w, 1)) || q2vt.lower(substr(w, 2)), ' ' ORDER BY n)
  FROM regexp_split_to_table(t, ' ') WITH ORDINALITY AS s(w, n)
$$;

CREATE OR REPLACE FUNCTION q2vt.trim(t text) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT regexp_replace(t, '^\s+|\s+$', '', 'g') $$;

-- substr(): 1-based, a negative start counting from the end, a negative
-- length leaving that many characters off the end.
CREATE OR REPLACE FUNCTION q2vt.substr(t text, start bigint, len bigint) RETURNS text
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE f bigint; n bigint := char_length(t);
BEGIN
  IF t IS NULL OR start IS NULL THEN RETURN NULL; END IF;
  IF start < 0 THEN f := GREATEST(n + start, 0);
  ELSIF start > 0 THEN f := start - 1;
  ELSE f := 0; END IF;
  IF len IS NULL THEN RETURN substr(t, (f + 1)::int); END IF;
  IF len < 0 THEN len := GREATEST(n + len - f, 0); END IF;
  RETURN substr(t, (f + 1)::int, len::int);
END $$;

CREATE OR REPLACE FUNCTION q2vt.left(t text, n bigint) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT CASE WHEN n < 0 THEN t ELSE left(t, n::int) END $$;

CREATE OR REPLACE FUNCTION q2vt.right(t text, n bigint) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT CASE WHEN n < 0 THEN t ELSE right(t, n::int) END $$;

CREATE OR REPLACE FUNCTION q2vt.replace_each(t text, befores text[], afters text[]) RETURNS text
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE i int;
BEGIN
  IF t IS NULL THEN RETURN NULL; END IF;
  IF cardinality(afters) <> 1 AND cardinality(afters) <> cardinality(befores) THEN RETURN NULL; END IF;
  FOR i IN 1 .. COALESCE(cardinality(befores), 0) LOOP
    t := replace(t, befores[i], CASE WHEN cardinality(afters) = 1 THEN afters[1] ELSE afters[i] END);
  END LOOP;
  RETURN t;
END $$;

-- A regular expression of QGIS (PCRE) for PostgreSQL: word boundaries,
-- Unicode letter and digit classes, multi-line anchors.
CREATE OR REPLACE FUNCTION q2vt.re(p text) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT replace(replace(replace(replace(replace(replace(p,
         '\b', '\y'), '\B', '\Y'), '\p{L}', '[[:alpha:]]'), '\p{N}', '[[:digit:]]'),
         '(?m)', '(?n)'), '\p{Lu}', '[[:upper:]]')
$$;

CREATE OR REPLACE FUNCTION q2vt.regexp_replace(t text, p text, r text) RETURNS text
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN
  RETURN regexp_replace(t, q2vt.re(p), r, 'g');
EXCEPTION WHEN OTHERS THEN RETURN NULL;
END $$;

CREATE OR REPLACE FUNCTION q2vt.regexp_substr(t text, p text) RETURNS text
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN
  IF t IS NULL OR p IS NULL THEN RETURN NULL; END IF;
  RETURN COALESCE(substring(t FROM q2vt.re(p)), '');
EXCEPTION WHEN OTHERS THEN RETURN NULL;
END $$;

CREATE OR REPLACE FUNCTION q2vt.regexp_match(t text, p text) RETURNS bigint
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE m text;
BEGIN
  IF t IS NULL OR p IS NULL THEN RETURN NULL; END IF;
  m := substring(t FROM '(' || q2vt.re(p) || ')');
  IF m IS NULL THEN RETURN 0; END IF;
  RETURN strpos(t, m);
EXCEPTION WHEN OTHERS THEN RETURN NULL;
END $$;

CREATE OR REPLACE FUNCTION q2vt.regexp_matches(t text, p text) RETURNS text[]
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN
  RETURN regexp_match(t, q2vt.re(p));
EXCEPTION WHEN OTHERS THEN RETURN NULL;
END $$;

-- format('%1 of %2', ...): QString::arg markers.
CREATE OR REPLACE FUNCTION q2vt.format(f text, VARIADIC args text[]) RETURNS text
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE i int;
BEGIN
  IF f IS NULL THEN RETURN NULL; END IF;
  FOR i IN REVERSE COALESCE(cardinality(args), 0) .. 1 LOOP
    f := replace(f, '%' || i, COALESCE(args[i], ''));
  END LOOP;
  RETURN f;
END $$;

CREATE OR REPLACE FUNCTION q2vt.lpad(t text, w bigint, fill text) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN w < 0 THEN t WHEN fill = '' THEN left(t, w::int) ELSE lpad(t, w::int, fill) END
$$;

CREATE OR REPLACE FUNCTION q2vt.rpad(t text, w bigint, fill text) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN w < 0 THEN t WHEN fill = '' THEN left(t, w::int) ELSE rpad(t, w::int, fill) END
$$;

CREATE OR REPLACE FUNCTION q2vt.chr(n bigint) RETURNS text
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN
  RETURN chr(n::int);
EXCEPTION WHEN OTHERS THEN RETURN NULL;
END $$;

-- format_number(): rounded half away from zero, with thousands separators.
CREATE OR REPLACE FUNCTION q2vt.format_number(x double precision, places bigint) RETURNS text
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE v numeric; s text; whole text; frac text; neg boolean; out text := '';
BEGIN
  IF x IS NULL OR x <> x THEN RETURN NULL; END IF;
  places := GREATEST(COALESCE(places, 0), 0);
  v := round(x::numeric, places::int);
  neg := v < 0;
  s := abs(v)::text;
  whole := split_part(s, '.', 1); frac := split_part(s, '.', 2);
  WHILE length(whole) > 3 LOOP
    out := ',' || right(whole, 3) || out; whole := left(whole, length(whole) - 3);
  END LOOP;
  out := whole || out;
  IF places > 0 THEN out := out || '.' || rpad(frac, places::int, '0'); END IF;
  RETURN CASE WHEN neg THEN '-' ELSE '' END || out;
END $$;

-- ---------------------------------------------------------------------
-- Dates
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION q2vt.to_date(t text) RETURNS date
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN RETURN left(trim(t), 10)::date; EXCEPTION WHEN OTHERS THEN RETURN NULL; END $$;

CREATE OR REPLACE FUNCTION q2vt.to_datetime(t text) RETURNS timestamp
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN RETURN trim(t)::timestamp; EXCEPTION WHEN OTHERS THEN RETURN NULL; END $$;

CREATE OR REPLACE FUNCTION q2vt.to_time(t text) RETURNS time
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN RETURN trim(t)::time; EXCEPTION WHEN OTHERS THEN RETURN NULL; END $$;

CREATE OR REPLACE FUNCTION q2vt.make_date(y bigint, m bigint, d bigint) RETURNS date
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN RETURN make_date(y::int, m::int, d::int); EXCEPTION WHEN OTHERS THEN RETURN NULL; END $$;

CREATE OR REPLACE FUNCTION q2vt.make_datetime(y bigint, mo bigint, d bigint, h bigint, mi bigint, s double precision)
RETURNS timestamp
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN RETURN make_timestamp(y::int, mo::int, d::int, h::int, mi::int, s);
EXCEPTION WHEN OTHERS THEN RETURN NULL; END $$;

-- ---------------------------------------------------------------------
-- Arrays (QGIS's are 0-based, negative indexes counting from the end)
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION q2vt.array_get(a anyarray, i bigint) RETURNS anyelement
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN i >= 0 THEN a[i + 1] ELSE a[cardinality(a) + i + 1] END
$$;

CREATE OR REPLACE FUNCTION q2vt.array_slice(a anyarray, s bigint, e bigint) RETURNS anyarray
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT a[(CASE WHEN s < 0 THEN cardinality(a) + s ELSE s END) + 1
           : (CASE WHEN e < 0 THEN cardinality(a) + e ELSE e END) + 1]
$$;

CREATE OR REPLACE FUNCTION q2vt.array_distinct(a anyarray) RETURNS anyarray
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT array_agg(e ORDER BY o) FROM (
    SELECT DISTINCT ON (e) e, o FROM unnest(a) WITH ORDINALITY AS u(e, o) ORDER BY e, o) d
$$;

CREATE OR REPLACE FUNCTION q2vt.string_to_array(t text, d text, empty text) RETURNS text[]
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT ARRAY(SELECT CASE WHEN e = '' THEN empty ELSE e END
               FROM unnest(string_to_array(t, d)) WITH ORDINALITY AS u(e, o) ORDER BY o)
$$;

-- ---------------------------------------------------------------------
-- Colours, as QGIS's "r,g,b,a" text
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION q2vt.c8(x double precision) RETURNS int
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT LEAST(GREATEST(q2vt.qround(x), 0), 255)::int $$;

CREATE OR REPLACE FUNCTION q2vt.color_rgba(r double precision, g double precision, b double precision,
    a double precision) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN r IS NULL OR g IS NULL OR b IS NULL OR a IS NULL THEN NULL
         ELSE concat_ws(',', q2vt.c8(r), q2vt.c8(g), q2vt.c8(b), q2vt.c8(a)) END
$$;

CREATE OR REPLACE FUNCTION q2vt.color_hsla(h double precision, s double precision, l double precision,
    a double precision) RETURNS text
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE c double precision; x double precision; m double precision; hh double precision;
        r double precision := 0; g double precision := 0; b double precision := 0;
BEGIN
  IF h IS NULL OR s IS NULL OR l IS NULL OR a IS NULL THEN RETURN NULL; END IF;
  s := LEAST(GREATEST(s, 0), 100) / 100; l := LEAST(GREATEST(l, 0), 100) / 100;
  hh := (h - 360 * floor(h / 360)) / 60;
  c := (1 - abs(2 * l - 1)) * s; x := c * (1 - abs(hh - 2 * floor(hh / 2) - 1)); m := l - c / 2;
  IF hh < 1 THEN r := c; g := x; ELSIF hh < 2 THEN r := x; g := c; ELSIF hh < 3 THEN g := c; b := x;
  ELSIF hh < 4 THEN g := x; b := c; ELSIF hh < 5 THEN r := x; b := c; ELSE r := c; b := x; END IF;
  RETURN q2vt.color_rgba((r + m) * 255, (g + m) * 255, (b + m) * 255, a);
END $$;

CREATE OR REPLACE FUNCTION q2vt.color_hsva(h double precision, s double precision, v double precision,
    a double precision) RETURNS text
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE c double precision; x double precision; m double precision; hh double precision;
        r double precision := 0; g double precision := 0; b double precision := 0;
BEGIN
  IF h IS NULL OR s IS NULL OR v IS NULL OR a IS NULL THEN RETURN NULL; END IF;
  s := LEAST(GREATEST(s, 0), 100) / 100; v := LEAST(GREATEST(v, 0), 100) / 100;
  hh := (h - 360 * floor(h / 360)) / 60;
  c := v * s; x := c * (1 - abs(hh - 2 * floor(hh / 2) - 1)); m := v - c;
  IF hh < 1 THEN r := c; g := x; ELSIF hh < 2 THEN r := x; g := c; ELSIF hh < 3 THEN g := c; b := x;
  ELSIF hh < 4 THEN g := x; b := c; ELSIF hh < 5 THEN r := x; b := c; ELSE r := c; b := x; END IF;
  RETURN q2vt.color_rgba((r + m) * 255, (g + m) * 255, (b + m) * 255, a);
END $$;

CREATE OR REPLACE FUNCTION q2vt.color_cmyka(c double precision, m double precision, y double precision,
    k double precision, a double precision) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT q2vt.color_rgba(255 * (1 - c / 100) * (1 - k / 100), 255 * (1 - m / 100) * (1 - k / 100),
                         255 * (1 - y / 100) * (1 - k / 100), a)
$$;

-- The colour a text names, as QGIS reads it (QgsSymbolLayerUtils::decodeColor):
-- "r,g,b[,a]", #rgb, #rrggbb, #aarrggbb, rgb(...), rgba(...), names.
-- Returns {r, g, b, a}; NULL if it names none.
CREATE OR REPLACE FUNCTION q2vt.color_values(t text) RETURNS int[]
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE s text; p text[]; h text; n text;
BEGIN
  IF t IS NULL THEN RETURN NULL; END IF;
  s := lower(trim(t));
  IF s ~ '^[0-9]{1,3}\s*,\s*[0-9]{1,3}\s*,\s*[0-9]{1,3}(\s*,\s*[0-9]{1,3})?$' THEN
    p := regexp_split_to_array(s, '\s*,\s*');
    RETURN ARRAY[LEAST(p[1]::int, 255), LEAST(p[2]::int, 255), LEAST(p[3]::int, 255),
                 LEAST(COALESCE(p[4], '255')::int, 255)];
  END IF;
  IF s ~ '^#?[0-9a-f]{3}$' THEN
    h := ltrim(s, '#');
    RETURN ARRAY[('x' || repeat(substr(h, 1, 1), 2))::bit(8)::int, ('x' || repeat(substr(h, 2, 1), 2))::bit(8)::int,
                 ('x' || repeat(substr(h, 3, 1), 2))::bit(8)::int, 255];
  END IF;
  IF s ~ '^#?[0-9a-f]{6}$' THEN
    h := ltrim(s, '#');
    RETURN ARRAY[('x' || substr(h, 1, 2))::bit(8)::int, ('x' || substr(h, 3, 2))::bit(8)::int,
                 ('x' || substr(h, 5, 2))::bit(8)::int, 255];
  END IF;
  IF s ~ '^#[0-9a-f]{8}$' THEN
    h := substr(s, 2);
    RETURN ARRAY[('x' || substr(h, 3, 2))::bit(8)::int, ('x' || substr(h, 5, 2))::bit(8)::int,
                 ('x' || substr(h, 7, 2))::bit(8)::int, ('x' || substr(h, 1, 2))::bit(8)::int];
  END IF;
  IF s ~ '^rgba?\(' THEN
    p := regexp_match(s, '^rgba?\(\s*([0-9.]+)(%?)\s*,\s*([0-9.]+)(%?)\s*,\s*([0-9.]+)(%?)\s*(,\s*([0-9.]+)\s*)?\)$');
    IF p IS NULL THEN RETURN NULL; END IF;
    RETURN ARRAY[
      q2vt.c8(p[1]::double precision * CASE WHEN p[2] = '%' THEN 2.55 ELSE 1 END),
      q2vt.c8(p[3]::double precision * CASE WHEN p[4] = '%' THEN 2.55 ELSE 1 END),
      q2vt.c8(p[5]::double precision * CASE WHEN p[6] = '%' THEN 2.55 ELSE 1 END),
      CASE WHEN p[8] IS NULL THEN 255 ELSE q2vt.c8(p[8]::double precision * 255) END];
  END IF;
  n := regexp_replace(s, '\s', '', 'g');
  IF n = 'transparent' THEN RETURN ARRAY[0, 0, 0, 0]; END IF;
  RETURN (SELECT ARRAY[c.r, c.g, c.b, 255] FROM q2vt.color_names c WHERE c.name = n);
EXCEPTION WHEN OTHERS THEN RETURN NULL;
END $$;

-- A colour as the tiles hold it: #RRGGBBAA.
CREATE OR REPLACE FUNCTION q2vt.color_hex(t text) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN v IS NULL THEN NULL ELSE '#' || upper(lpad(to_hex(v[1]), 2, '0') || lpad(to_hex(v[2]), 2, '0')
         || lpad(to_hex(v[3]), 2, '0') || lpad(to_hex(v[4]), 2, '0')) END
  FROM (SELECT q2vt.color_values(t) AS v) c
$$;

CREATE OR REPLACE FUNCTION q2vt.color_part(t text, part text) RETURNS double precision
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE v int[]; r double precision; g double precision; b double precision;
        mx double precision; mn double precision; d double precision; h double precision;
BEGIN
  v := q2vt.color_values(t);
  IF v IS NULL THEN RETURN NULL; END IF;
  r := v[1] / 255.0; g := v[2] / 255.0; b := v[3] / 255.0;
  mx := GREATEST(r, g, b); mn := LEAST(r, g, b); d := mx - mn;
  IF d = 0 THEN h := -1;
  ELSIF mx = r THEN h := 60 * (((g - b) / d)::numeric % 6);
  ELSIF mx = g THEN h := 60 * ((b - r) / d + 2);
  ELSE h := 60 * ((r - g) / d + 4); END IF;
  IF h < 0 AND h <> -1 THEN h := h + 360; END IF;
  RETURN CASE lower(part)
    WHEN 'red' THEN v[1] WHEN 'green' THEN v[2] WHEN 'blue' THEN v[3] WHEN 'alpha' THEN v[4]
    WHEN 'hue' THEN CASE WHEN h < 0 THEN -1 ELSE round(h) END
    WHEN 'saturation' THEN CASE WHEN mx = 0 THEN 0 ELSE round(100 * d / mx) END
    WHEN 'value' THEN round(100 * mx)
    WHEN 'hsl_hue' THEN CASE WHEN h < 0 THEN -1 ELSE round(h) END
    WHEN 'hsl_saturation' THEN CASE WHEN d = 0 THEN 0 ELSE round(100 * d / (1 - abs(mx + mn - 1))) END
    WHEN 'lightness' THEN round(100 * (mx + mn) / 2)
    WHEN 'cyan' THEN CASE WHEN mx = 0 THEN 0 ELSE round(100 * (1 - r - (1 - mx)) / mx) END
    WHEN 'magenta' THEN CASE WHEN mx = 0 THEN 0 ELSE round(100 * (1 - g - (1 - mx)) / mx) END
    WHEN 'yellow' THEN CASE WHEN mx = 0 THEN 0 ELSE round(100 * (1 - b - (1 - mx)) / mx) END
    WHEN 'black' THEN round(100 * (1 - mx))
    END;
END $$;

CREATE OR REPLACE FUNCTION q2vt.set_color_part(t text, part text, x double precision) RETURNS text
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE v int[];
BEGIN
  v := q2vt.color_values(t);
  IF v IS NULL OR x IS NULL THEN RETURN NULL; END IF;
  CASE lower(part)
    WHEN 'red' THEN v[1] := q2vt.c8(x);
    WHEN 'green' THEN v[2] := q2vt.c8(x);
    WHEN 'blue' THEN v[3] := q2vt.c8(x);
    WHEN 'alpha' THEN v[4] := q2vt.c8(x);
    WHEN 'hue', 'saturation', 'value' THEN
      RETURN q2vt.color_hsva(CASE WHEN lower(part) = 'hue' THEN x ELSE GREATEST(q2vt.color_part(t, 'hue'), 0) END,
                             CASE WHEN lower(part) = 'saturation' THEN x ELSE q2vt.color_part(t, 'saturation') END,
                             CASE WHEN lower(part) = 'value' THEN x ELSE q2vt.color_part(t, 'value') END, v[4]);
    WHEN 'lightness', 'hsl_hue', 'hsl_saturation' THEN
      RETURN q2vt.color_hsla(CASE WHEN lower(part) = 'hsl_hue' THEN x ELSE GREATEST(q2vt.color_part(t, 'hsl_hue'), 0) END,
                             CASE WHEN lower(part) = 'hsl_saturation' THEN x ELSE q2vt.color_part(t, 'hsl_saturation') END,
                             CASE WHEN lower(part) = 'lightness' THEN x ELSE q2vt.color_part(t, 'lightness') END, v[4]);
    ELSE RETURN NULL;
  END CASE;
  RETURN concat_ws(',', v[1], v[2], v[3], v[4]);
END $$;

-- QColor::darker / lighter: the HSV value divided / multiplied by factor %.
CREATE OR REPLACE FUNCTION q2vt.darker(t text, f double precision) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN f <= 0 THEN t ELSE q2vt.set_color_part(t, 'value', q2vt.color_part(t, 'value') * 100 / f) END
$$;

CREATE OR REPLACE FUNCTION q2vt.lighter(t text, f double precision) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN f <= 0 THEN t ELSE q2vt.set_color_part(t, 'value', LEAST(q2vt.color_part(t, 'value') * f / 100, 100)) END
$$;

CREATE OR REPLACE FUNCTION q2vt.color_mix(a text, b text, ratio double precision) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN x IS NULL OR y IS NULL OR ratio IS NULL THEN NULL ELSE
    q2vt.color_rgba(x[1] * (1 - ratio) + y[1] * ratio, x[2] * (1 - ratio) + y[2] * ratio,
                    x[3] * (1 - ratio) + y[3] * ratio, x[4] * (1 - ratio) + y[4] * ratio) END
  FROM (SELECT q2vt.color_values(a) AS x, q2vt.color_values(b) AS y) c
$$;

CREATE OR REPLACE FUNCTION q2vt.ramp_sample(samples text[], v double precision) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN v IS NULL THEN NULL
         ELSE samples[(q2vt.qround(LEAST(GREATEST(v, 0), 1) * (cardinality(samples) - 1)) + 1)::int] END
$$;

-- ---------------------------------------------------------------------
-- Geometry
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION q2vt.x(g geometry) RETURNS double precision
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN g IS NULL OR ST_IsEmpty(g) THEN NULL
              WHEN GeometryType(g) = 'POINT' THEN ST_X(g) ELSE ST_X(ST_Centroid(g)) END
$$;

CREATE OR REPLACE FUNCTION q2vt.y(g geometry) RETURNS double precision
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN g IS NULL OR ST_IsEmpty(g) THEN NULL
              WHEN GeometryType(g) = 'POINT' THEN ST_Y(g) ELSE ST_Y(ST_Centroid(g)) END
$$;

CREATE OR REPLACE FUNCTION q2vt.centroid(g geometry) RETURNS geometry
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN RETURN ST_Centroid(g); EXCEPTION WHEN OTHERS THEN RETURN NULL; END $$;

CREATE OR REPLACE FUNCTION q2vt.point_on_surface(g geometry) RETURNS geometry
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN RETURN ST_PointOnSurface(g);
EXCEPTION WHEN OTHERS THEN
  BEGIN RETURN ST_PointOnSurface(ST_MakeValid(g)); EXCEPTION WHEN OTHERS THEN RETURN NULL; END;
END $$;

-- QGIS's pole of inaccessibility: of a multi-polygon, its part's whose
-- pole lies farthest inside (PostGIS's largest inscribed circle's centre).
CREATE OR REPLACE FUNCTION q2vt.pole_of_inaccessibility(g geometry, tolerance double precision) RETURNS geometry
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE best geometry; best_r double precision := -1; part geometry; c record;
BEGIN
  IF g IS NULL OR ST_IsEmpty(g) OR ST_Dimension(g) <> 2 THEN RETURN NULL; END IF;
  FOR part IN SELECT (ST_Dump(g)).geom LOOP
    IF ST_Dimension(part) = 2 AND NOT ST_IsEmpty(part) THEN
      c := ST_MaximumInscribedCircle(part, GREATEST(tolerance, 1e-9));
      IF c.radius > best_r THEN best_r := c.radius; best := c.center; END IF;
    END IF;
  END LOOP;
  RETURN best;
EXCEPTION WHEN OTHERS THEN RETURN ST_PointOnSurface(g);
END $$;

CREATE OR REPLACE FUNCTION q2vt.boundary(g geometry) RETURNS geometry
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN RETURN ST_Boundary(g); EXCEPTION WHEN OTHERS THEN RETURN NULL; END $$;

CREATE OR REPLACE FUNCTION q2vt.buffer(g geometry, d double precision, segments bigint) RETURNS geometry
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN RETURN ST_Buffer(g, d, 'quad_segs=' || GREATEST(segments, 1));
EXCEPTION WHEN OTHERS THEN RETURN NULL; END $$;

CREATE OR REPLACE FUNCTION q2vt.single_sided_buffer(g geometry, d double precision, segments bigint) RETURNS geometry
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN RETURN ST_Buffer(g, abs(d), 'quad_segs=' || GREATEST(segments, 1) || ' side=' || CASE WHEN d >= 0 THEN 'left' ELSE 'right' END);
EXCEPTION WHEN OTHERS THEN RETURN NULL; END $$;

CREATE OR REPLACE FUNCTION q2vt.offset_curve(g geometry, d double precision, segments bigint, join_style bigint,
    miter double precision) RETURNS geometry
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN
  RETURN ST_OffsetCurve(g, d, 'quad_segs=' || GREATEST(segments, 1) || ' join='
         || CASE join_style WHEN 2 THEN 'mitre mitre_limit=' || miter WHEN 3 THEN 'bevel' ELSE 'round' END);
EXCEPTION WHEN OTHERS THEN RETURN NULL; END $$;

CREATE OR REPLACE FUNCTION q2vt.intersection(a geometry, b geometry) RETURNS geometry
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN RETURN ST_Intersection(a, b);
EXCEPTION WHEN OTHERS THEN
  BEGIN RETURN ST_Intersection(ST_MakeValid(a), ST_MakeValid(b)); EXCEPTION WHEN OTHERS THEN RETURN NULL; END;
END $$;

CREATE OR REPLACE FUNCTION q2vt.difference(a geometry, b geometry) RETURNS geometry
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN RETURN ST_Difference(a, b);
EXCEPTION WHEN OTHERS THEN
  BEGIN RETURN ST_Difference(ST_MakeValid(a), ST_MakeValid(b)); EXCEPTION WHEN OTHERS THEN RETURN NULL; END;
END $$;

CREATE OR REPLACE FUNCTION q2vt.transform(g geometry, srid int) RETURNS geometry
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN RETURN ST_Transform(g, srid); EXCEPTION WHEN OTHERS THEN RETURN NULL; END $$;

CREATE OR REPLACE FUNCTION q2vt.geom_from_wkt(t text) RETURNS geometry
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN RETURN ST_GeomFromText(t); EXCEPTION WHEN OTHERS THEN RETURN NULL; END $$;

-- WKT as QGIS writes it: "LineString (0 0, 1 1)".
CREATE OR REPLACE FUNCTION q2vt.geom_to_wkt(g geometry, places bigint) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT replace(regexp_replace(regexp_replace(ST_AsText(g, places::int),
    '(MULTIPOLYGON|MULTILINESTRING|MULTIPOINT|POLYGON|LINESTRING|POINT|GEOMETRYCOLLECTION)\s*',
    '\1 ', 'g'), '(\S)\(', '\1 (', 'g'), ',', ', ')
$$;

CREATE OR REPLACE FUNCTION q2vt.make_line(points geometry[]) RETURNS geometry
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN RETURN ST_MakeLine(ARRAY(SELECT p FROM unnest(points) WITH ORDINALITY AS u(p, o) WHERE p IS NOT NULL ORDER BY o));
EXCEPTION WHEN OTHERS THEN RETURN NULL; END $$;

CREATE OR REPLACE FUNCTION q2vt.as_ring(g geometry) RETURNS geometry
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN ST_IsClosed(g) THEN g ELSE ST_AddPoint(g, ST_StartPoint(g)) END
$$;

CREATE OR REPLACE FUNCTION q2vt.make_circle(c geometry, r double precision, segments bigint) RETURNS geometry
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT ST_Buffer(c, r, 'quad_segs=' || GREATEST(ceil(segments / 4.0)::int, 1))
$$;

CREATE OR REPLACE FUNCTION q2vt.is_closed(g geometry) RETURNS boolean
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN ST_Dimension(g) = 1 THEN ST_IsClosed(g) END
$$;

-- Vertices of any geometry, in order (1-based; negative from the end).
CREATE OR REPLACE FUNCTION q2vt.point_n(g geometry, n bigint) RETURNS geometry
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT d.geom FROM ST_DumpPoints(g) WITH ORDINALITY AS d(path, geom, o)
  WHERE o = CASE WHEN n < 0 THEN ST_NPoints(g) + n + 1 ELSE n END
$$;

CREATE OR REPLACE FUNCTION q2vt.start_point(g geometry) RETURNS geometry
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT q2vt.point_n(g, 1) $$;

CREATE OR REPLACE FUNCTION q2vt.end_point(g geometry) RETURNS geometry
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT q2vt.point_n(g, -1) $$;

CREATE OR REPLACE FUNCTION q2vt.num_geometries(g geometry) RETURNS bigint
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT CASE WHEN g IS NULL THEN NULL ELSE COALESCE(ST_NumGeometries(g), 1) END $$;

CREATE OR REPLACE FUNCTION q2vt.nodes_to_points(g geometry, ignore_closing boolean) RETURNS geometry
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT ST_Collect(d.geom ORDER BY o) FROM ST_DumpPoints(g) WITH ORDINALITY AS d(path, geom, o)
  WHERE NOT COALESCE(ignore_closing, false) OR ST_Dimension(g) = 0 OR NOT EXISTS (
    SELECT 1 FROM (SELECT (ST_Dump(CASE WHEN ST_Dimension(g) = 2 THEN ST_Boundary(g) ELSE g END)).geom AS ring) r
    WHERE ST_IsClosed(r.ring) AND ST_NPoints(r.ring) > 1 AND ST_Equals(d.geom, ST_EndPoint(r.ring))
      AND d.path[array_length(d.path, 1)] = ST_NPoints(r.ring))
$$;

CREATE OR REPLACE FUNCTION q2vt.collect(parts geometry[]) RETURNS geometry
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN
  IF parts IS NULL OR cardinality(array_remove(parts, NULL)) = 0 THEN RETURN NULL; END IF;
  RETURN ST_Collect(array_remove(parts, NULL));
EXCEPTION WHEN OTHERS THEN RETURN NULL;
END $$;

-- The lines a geometry is drawn along: its lines, or its polygons' rings.
CREATE OR REPLACE FUNCTION q2vt.lines_of(g geometry) RETURNS SETOF geometry
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT (ST_Dump(CASE WHEN ST_Dimension(g) = 2 THEN ST_Boundary(g) ELSE g END)).geom
$$;

-- The point at a distance along the geometry's lines, one after another
-- (QgsGeometry::interpolate); NULL beyond its length.
CREATE OR REPLACE FUNCTION q2vt.line_interpolate_point(g geometry, d double precision) RETURNS geometry
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE line geometry; len double precision; walked double precision := 0;
BEGIN
  IF g IS NULL OR d IS NULL OR d < 0 OR ST_Dimension(g) < 1 THEN RETURN NULL; END IF;
  FOR line IN SELECT q2vt.lines_of(g) LOOP
    len := ST_Length(line);
    IF d <= walked + len THEN
      IF len = 0 THEN RETURN ST_StartPoint(line); END IF;
      RETURN ST_LineInterpolatePoint(line, LEAST(GREATEST((d - walked) / len, 0), 1));
    END IF;
    walked := walked + len;
  END LOOP;
  RETURN NULL;
EXCEPTION WHEN OTHERS THEN RETURN NULL;
END $$;

-- A segment's angle, clockwise from north, in radians (QgsGeometryUtils::lineAngle).
CREATE OR REPLACE FUNCTION q2vt.line_angle(a geometry, b geometry) RETURNS double precision
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN a1 < 0 THEN a1 + 2 * pi() ELSE a1 END
  FROM (SELECT atan2(ST_X(b) - ST_X(a), ST_Y(b) - ST_Y(a)) AS a1) s
$$;

CREATE OR REPLACE FUNCTION q2vt.average_angle(a1 double precision, a2 double precision) RETURNS double precision
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN r < 0 THEN r + 2 * pi() ELSE r END
  FROM (SELECT atan2((sin(a1) + sin(a2)) / 2, (cos(a1) + cos(a2)) / 2) AS r) s
$$;

-- The line's direction at a distance along it (degrees clockwise from
-- north); at a vertex, between its two segments (QgsGeometry::interpolateAngle).
CREATE OR REPLACE FUNCTION q2vt.line_interpolate_angle(g geometry, d double precision) RETURNS double precision
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE line geometry; n int; i int; walked double precision := 0; seg double precision;
        a geometry; b geometry; prev_angle double precision; angle double precision;
BEGIN
  IF g IS NULL OR d IS NULL OR d < 0 OR ST_Dimension(g) < 1 THEN RETURN NULL; END IF;
  FOR line IN SELECT q2vt.lines_of(g) LOOP
    n := ST_NPoints(line);
    prev_angle := NULL;
    FOR i IN 1 .. n - 1 LOOP
      a := ST_PointN(line, i); b := ST_PointN(line, i + 1);
      seg := ST_Distance(a, b);
      angle := q2vt.line_angle(a, b);
      IF d = walked AND prev_angle IS NOT NULL THEN
        RETURN degrees(q2vt.average_angle(prev_angle, angle));
      END IF;
      IF d < walked + seg OR (d = walked AND prev_angle IS NULL) THEN
        RETURN degrees(angle);
      END IF;
      walked := walked + seg;
      prev_angle := angle;
    END LOOP;
    IF d = walked AND prev_angle IS NOT NULL THEN
      IF ST_IsClosed(line) AND n > 2 THEN
        RETURN degrees(q2vt.average_angle(prev_angle, q2vt.line_angle(ST_PointN(line, 1), ST_PointN(line, 2))));
      END IF;
      RETURN degrees(prev_angle);
    END IF;
  END LOOP;
  RETURN NULL;
EXCEPTION WHEN OTHERS THEN RETURN NULL;
END $$;

CREATE OR REPLACE FUNCTION q2vt.line_locate_point(g geometry, p geometry) RETURNS double precision
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE line geometry; walked double precision := 0; best double precision; best_d double precision; dist double precision;
BEGIN
  IF g IS NULL OR p IS NULL OR ST_Dimension(g) < 1 THEN RETURN NULL; END IF;
  FOR line IN SELECT q2vt.lines_of(g) LOOP
    dist := ST_Distance(line, p);
    IF best_d IS NULL OR dist < best_d THEN
      best_d := dist; best := walked + ST_LineLocatePoint(line, ST_ClosestPoint(line, p)) * ST_Length(line);
    END IF;
    walked := walked + ST_Length(line);
  END LOOP;
  RETURN best;
EXCEPTION WHEN OTHERS THEN RETURN NULL;
END $$;

CREATE OR REPLACE FUNCTION q2vt.angle_at_vertex(g geometry, v bigint) RETURNS double precision
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE n bigint := ST_NPoints(g); i bigint; a double precision; b double precision;
BEGIN
  IF g IS NULL OR ST_Dimension(g) < 1 THEN RETURN NULL; END IF;
  i := CASE WHEN v < 0 THEN n + v ELSE v END + 1;
  IF i < 1 OR i > n THEN RETURN NULL; END IF;
  IF i > 1 THEN a := q2vt.line_angle(q2vt.point_n(g, i - 1), q2vt.point_n(g, i)); END IF;
  IF i < n THEN b := q2vt.line_angle(q2vt.point_n(g, i), q2vt.point_n(g, i + 1)); END IF;
  IF a IS NULL THEN RETURN degrees(b); END IF;
  IF b IS NULL THEN RETURN degrees(a); END IF;
  RETURN degrees(q2vt.average_angle(a, b));
EXCEPTION WHEN OTHERS THEN RETURN NULL;
END $$;

CREATE OR REPLACE FUNCTION q2vt.azimuth(a geometry, b geometry) RETURNS double precision
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
BEGIN RETURN ST_Azimuth(a, b); EXCEPTION WHEN OTHERS THEN RETURN NULL; END $$;

CREATE OR REPLACE FUNCTION q2vt.geometry_type(g geometry) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE ST_Dimension(g) WHEN 0 THEN 'Point' WHEN 1 THEN 'Line' WHEN 2 THEN 'Polygon' END
$$;

CREATE OR REPLACE FUNCTION q2vt.close_line(g geometry) RETURNS geometry
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT ST_Collect(CASE WHEN ST_IsClosed(l) THEN l ELSE ST_AddPoint(l, ST_StartPoint(l)) END)
  FROM (SELECT (ST_Dump(g)).geom AS l) d WHERE ST_Dimension(g) = 1
$$;

CREATE OR REPLACE FUNCTION q2vt.segments_to_lines(g geometry) RETURNS geometry
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT ST_Collect(s.geom ORDER BY s.path) FROM ST_DumpSegments(g) AS s
$$;

CREATE OR REPLACE FUNCTION q2vt.extend(g geometry, s double precision, e double precision) RETURNS geometry
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE n int; a double precision; b double precision;
BEGIN
  IF GeometryType(g) <> 'LINESTRING' THEN RETURN NULL; END IF;
  n := ST_NPoints(g);
  a := q2vt.line_angle(ST_PointN(g, 2), ST_PointN(g, 1));
  b := q2vt.line_angle(ST_PointN(g, n - 1), ST_PointN(g, n));
  g := ST_SetPoint(g, 0, ST_Translate(ST_PointN(g, 1), s * sin(a), s * cos(a)));
  RETURN ST_SetPoint(g, n - 1, ST_Translate(ST_PointN(g, n), e * sin(b), e * cos(b)));
EXCEPTION WHEN OTHERS THEN RETURN NULL;
END $$;

CREATE OR REPLACE FUNCTION q2vt.main_angle(g geometry) RETURNS double precision
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE r geometry; a geometry; b geometry; c geometry; ang double precision;
BEGIN
  r := ST_ExteriorRing(ST_OrientedEnvelope(g));
  a := ST_PointN(r, 1); b := ST_PointN(r, 2); c := ST_PointN(r, 3);
  IF ST_Distance(a, b) >= ST_Distance(b, c) THEN ang := q2vt.line_angle(a, b);
  ELSE ang := q2vt.line_angle(b, c); END IF;
  ang := degrees(ang);
  RETURN CASE WHEN ang >= 180 THEN ang - 180 ELSE ang END;
EXCEPTION WHEN OTHERS THEN RETURN NULL;
END $$;

-- Polygons of a tile smaller than a share of a pixel are left out below
-- the highest zoom; lines and points stay.
CREATE OR REPLACE FUNCTION q2vt.big_enough(g geometry, min_area double precision) RETURNS boolean
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT min_area IS NULL OR ST_Dimension(g) < 2 OR ST_Area(g) >= min_area
$$;
"""


def library_sql() -> str:
    """The q2vt schema's helper functions and colour names."""
    names = []
    for name in QColor.colorNames():
        color = QColor(name)
        names.append(f"('{name.lower()}', {color.red()}, {color.green()}, {color.blue()})")
    colors = (
        "DROP TABLE IF EXISTS q2vt.color_names;\n"
        "CREATE TABLE q2vt.color_names (name text PRIMARY KEY, r int, g int, b int);\n"
        "INSERT INTO q2vt.color_names VALUES\n  " + ",\n  ".join(names) + ";\n"
    )
    return colors + _LIBRARY
