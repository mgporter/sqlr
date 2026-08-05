"""The type lattice, and the mapping from written type names onto it.

Shared by `schema_resolution` (which reads type names out of casts) and `declared`
(which reads them out of yml), because the two have to agree: a `cast(x as varchar)` and
a `data_type: varchar(50)` must land on the same `ResolvedTypeName` or every comparison
between inference and declaration is noise.
"""

from typing import Literal

ResolvedTypeName = Literal[
    "integer",
    "decimal",
    "float",
    "number",
    "numeric",
    "string",
    "date",
    "timestamp",
    "boolean",
    "unknown",
]
"""A concrete type, or a family that covers several.

Most evidence pins down that a column holds a number without saying which kind: `x * 12`
and `x > 5` are equally true of an INT, a DECIMAL and a DOUBLE. Rather than pick one and
be wrong, those resolve to a family:

- `number`  = `integer` | `decimal`  (exact)
- `numeric` = `integer` | `decimal` | `float`

Only evidence that names a type - a cast, or a function with a fixed return type - yields
a concrete one.
"""

# The concrete types each `ResolvedTypeName` covers. Widening picks the narrowest entry
# whose cover is a superset of both inputs, so `integer` and `decimal` meet at `number`.
TYPE_COVER: dict[ResolvedTypeName, frozenset[str]] = {
    "integer": frozenset({"integer"}),
    "decimal": frozenset({"decimal"}),
    "float": frozenset({"float"}),
    "number": frozenset({"integer", "decimal"}),
    "numeric": frozenset({"integer", "decimal", "float"}),
    "string": frozenset({"string"}),
    "date": frozenset({"date"}),
    "timestamp": frozenset({"timestamp"}),
    "boolean": frozenset({"boolean"}),
    "unknown": frozenset(),
}

# Narrowest first, so the first superset found is the least upper bound.
WIDENING_ORDER: list[ResolvedTypeName] = [
    "integer",
    "decimal",
    "float",
    "string",
    "date",
    "timestamp",
    "boolean",
    "number",
    "numeric",
]


def widen(left: ResolvedTypeName, right: ResolvedTypeName) -> ResolvedTypeName:
    """The narrowest type covering both. `unknown` when nothing does."""
    if left == right:
        return left
    if left == "unknown" or right == "unknown":
        return "unknown"
    covered = TYPE_COVER[left] | TYPE_COVER[right]
    for candidate in WIDENING_ORDER:
        if covered <= TYPE_COVER[candidate]:
            return candidate
    return "unknown"


def compatible(left: ResolvedTypeName, right: ResolvedTypeName) -> bool:
    """True when some concrete type satisfies both.

    `integer` and `number` are compatible - an INT is a number. `string` and `numeric`
    are not, and that disagreement is what a divergence diagnostic reports. `unknown`
    covers nothing, so it is compatible with nothing; callers check for it first.
    """
    return bool(TYPE_COVER[left] & TYPE_COVER[right])


TYPE_NAMES: dict[str, ResolvedTypeName] = {
    "INT": "integer",
    "INT2": "integer",
    "INT4": "integer",
    "INT8": "integer",
    "INT16": "integer",
    "INT32": "integer",
    "INT64": "integer",
    "INTEGER": "integer",
    "BIGINT": "integer",
    "SMALLINT": "integer",
    "TINYINT": "integer",
    "MEDIUMINT": "integer",
    "SERIAL": "integer",
    "BIGSERIAL": "integer",
    "DECIMAL": "decimal",
    "NUMERIC": "decimal",
    "NUMBER": "decimal",
    "MONEY": "decimal",
    "BIGDECIMAL": "decimal",
    "FLOAT": "float",
    "FLOAT4": "float",
    "FLOAT8": "float",
    "FLOAT64": "float",
    "DOUBLE": "float",
    "REAL": "float",
    "VARCHAR": "string",
    "NVARCHAR": "string",
    "TEXT": "string",
    "CHAR": "string",
    "NCHAR": "string",
    "BPCHAR": "string",
    "STRING": "string",
    "UUID": "string",
    "DATE": "date",
    "TIMESTAMP": "timestamp",
    "TIMESTAMPTZ": "timestamp",
    "TIMESTAMPLTZ": "timestamp",
    "TIMESTAMPNTZ": "timestamp",
    "TIMESTAMP_TZ": "timestamp",
    "TIMESTAMP_LTZ": "timestamp",
    "TIMESTAMP_NTZ": "timestamp",
    "DATETIME": "timestamp",
    "BOOLEAN": "boolean",
    "BOOL": "boolean",
}


def normalize_type_name(raw: str) -> str:
    """Strip parameters and whitespace from a written type name.

    Declared types arrive as people write them - `varchar(50)`, `decimal(10, 2)`,
    `NUMBER(38,0)`, `timestamp with time zone`. Only the head matters here; precision and
    scale are a generator's problem, not a compatibility one.
    """
    name = raw.strip().upper()
    if "(" in name:
        name = name.split("(", 1)[0]
    if "[" in name:
        name = name.split("[", 1)[0]
    # `timestamp with time zone` / `timestamp without time zone`.
    if name.startswith("TIMESTAMP WITH"):
        return "TIMESTAMPTZ"
    if name.startswith("TIMESTAMP WITHOUT"):
        return "TIMESTAMP"
    return name.strip().replace(" ", "_")


def resolve_type_name(raw: str | None) -> ResolvedTypeName:
    """Map a written type name onto the lattice. `unknown` when unrecognised."""
    if not raw:
        return "unknown"
    return TYPE_NAMES.get(normalize_type_name(raw), "unknown")
