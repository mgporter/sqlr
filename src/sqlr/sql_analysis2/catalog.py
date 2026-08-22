"""Step 0 - the signature catalog.

One table, two consumers, and the split matters:

- `expression_metadata` (in `annotate.py`) installs an annotator only for the keys in
  `ANNOTATION_GAP_KEYS`. sqlglot already types most of what is here, `TYPED_DIVISION` and
  DECIMAL precision included, and overriding it would trade a correct answer for ours. The
  catalog stays a gap-filler for *return* types.
- the fact walk (in `facts.py`) consults every entry, because sqlglot's metadata says what a
  call returns and never what it accepts. Argument families are the whole reason this file
  is bigger than the three gap entries.

Operators are entries like any other. `a = b`, `a + b` and `coalesce(a, b)` are overloaded
signatures, so every rule about them falls out of overload selection instead of being three
hand-written tables.

## Common first, dialect on top

`upper(number_column)` is an error in every engine, so the signature for it belongs to no
engine in particular. `CATALOG_COMMON` holds the operators and the standard-SQL functions;
`CATALOG_BY_DIALECT` holds only what an engine genuinely does differently or has to itself.

**A dialect entry replaces the common entry for its key outright; it never appends
overloads.** Appending would mean a dialect could only ever widen what a key accepts, so it
could not express *narrower* - and narrowing is usually why a key needs a dialect entry at
all. Replacement is also the only rule a reader can predict from looking at one file.

## Two things every signature has to get right

**Argument order is sqlglot's node order, not the written order.** `date_trunc('day', ts)`
parses to `TimestampTrunc(this=ts, unit='day')`, so its signature reads `(TEMPORAL, ANY)`.
Writing it the way the SQL reads produces a confident error on valid SQL.

**Name the shallowest family that is really required.** A signature saying `DECIMAL` where the
engine accepts any numeric turns `round(int_col)` into an error. `NUMERIC` is almost always
what a numeric parameter means.
"""

from dataclasses import dataclass
from typing import NamedTuple

from sqlglot import exp

from sqlr.sql_analysis2.families import FamilyName

type DialectName = str
type CatalogKey = str
"""What a call is looked up by: a function's `sql_name()`, an `Anonymous` call's written
name uppercased, or an operator's SQL symbol. Never a class name - those change between
sqlglot versions and do not exist for `Anonymous` at all."""

type ReturnMarker = str
"""What a call returns. One of:

- a concrete type name - `VARCHAR`, `BIGINT`, `ARRAY<VARCHAR>`.
- `@argN` - *the result is argument N's type*. An identity link between the call and that
  argument, which is what carries a type backwards through arithmetic.
- `@family(argN)` - *the result is in argument N's family*, but not necessarily its type.
  `SUM(INT)` is `HUGEINT` in DuckDB and `NUMBER(38,0)` in Snowflake; it is certainly numeric
  because its argument is.
- `@element` - unwrap `ARRAY<T>` to `T`.
"""


class ReturnLink(NamedTuple):
    """A return marker read as a link: which argument the result is bound to, and how."""

    argument_index: int
    family_only: bool
    """True for `@family(argN)`: the two ends share a family, not a type. An inferred column
    takes the family and never the concrete type."""


def _return_link_of(marker: ReturnMarker) -> ReturnLink | None:
    if marker.startswith("@family(arg") and marker.endswith(")"):
        return ReturnLink(int(marker[len("@family(arg") : -1]), family_only=True)
    if marker.startswith("@arg"):
        return ReturnLink(int(marker[len("@arg") :]), family_only=False)
    return None


@dataclass(frozen=True)
class Sig:
    """One overload."""

    params: tuple[FamilyName, ...]
    returns: ReturnMarker
    variadic: bool = False
    """Whether the last parameter repeats. `coalesce(a, b, c)` is one signature, not one
    per arity."""

    def accepts_arity(self, count: int) -> bool:
        if self.variadic:
            return count >= len(self.params)
        return count == len(self.params)

    def family_at(self, index: int) -> FamilyName:
        """The family constraining one argument position, repeating the last parameter for
        a variadic signature."""
        if index < len(self.params):
            return self.params[index]
        return self.params[-1] if self.variadic and self.params else "ANY"

    @property
    def return_link(self) -> ReturnLink | None:
        """The argument this signature's result is bound to, when it is bound to one.

        `returns="@arg0"` is not merely a note for the annotator: it says the result *is*
        that argument's value-type, which is a link, and reading it as one is what lets a
        claim on `amount * 2` reach `amount`.
        """
        return _return_link_of(self.returns)


# ------------------------------------------------------------------------- operators
OPERATOR_SQL_NAMES: dict[type[exp.Expr], CatalogKey] = {
    exp.EQ: "=",
    exp.NEQ: "!=",
    exp.GT: ">",
    exp.LT: "<",
    exp.GTE: ">=",
    exp.LTE: "<=",
    exp.Add: "+",
    exp.Sub: "-",
    exp.Mul: "*",
    exp.Div: "/",
    exp.Mod: "%",
    exp.DPipe: "||",
    exp.And: "AND",
    exp.Or: "OR",
    exp.Not: "NOT",
    exp.Like: "LIKE",
    exp.ILike: "ILIKE",
}
"""The one place in the catalog that names sqlglot classes, and it has no alternative:
operators are `exp.Binary`, not `exp.Func`, so they have no `sql_name()` and appear in no
`FUNCTION_BY_NAME` map. These classes are core to the parser and older than any dialect
module, so the coupling the rest of the design avoids is not a real risk here."""


_COMPARISON_SIGNATURES = [Sig(params=("@T", "@T"), returns="BOOLEAN")]
"""Both operands share one domain, whichever it is. That is a link, not a claim - it types
an undeclared column from whatever it is compared against, and claims nothing when neither
side is known."""

_OPERATORS: dict[CatalogKey, list[Sig]] = {
    "=": _COMPARISON_SIGNATURES,
    "!=": _COMPARISON_SIGNATURES,
    ">": _COMPARISON_SIGNATURES,
    "<": _COMPARISON_SIGNATURES,
    ">=": _COMPARISON_SIGNATURES,
    "<=": _COMPARISON_SIGNATURES,
    # `date + 7` and `ts + interval` are real overloads, not autocasts. Enumerating them is
    # what stops `+` from claiming NUMERIC on a date column - the whole point of putting
    # operators in the catalog rather than writing an arithmetic rule.
    "+": [
        Sig(params=("NUMERIC", "NUMERIC"), returns="@arg0"),
        Sig(params=("TEMPORAL", "NUMERIC"), returns="@arg0"),
        Sig(params=("NUMERIC", "TEMPORAL"), returns="@arg1"),
        Sig(params=("TEMPORAL", "INTERVAL"), returns="@arg0"),
        Sig(params=("INTERVAL", "TEMPORAL"), returns="@arg1"),
        Sig(params=("INTERVAL", "INTERVAL"), returns="@arg0"),
    ],
    "-": [
        Sig(params=("NUMERIC", "NUMERIC"), returns="@arg0"),
        Sig(params=("TEMPORAL", "NUMERIC"), returns="@arg0"),
        Sig(params=("TEMPORAL", "INTERVAL"), returns="@arg0"),
        Sig(params=("TEMPORAL", "TEMPORAL"), returns="BIGINT"),
        Sig(params=("INTERVAL", "INTERVAL"), returns="@arg0"),
    ],
    "*": [
        Sig(params=("NUMERIC", "NUMERIC"), returns="@arg0"),
        Sig(params=("INTERVAL", "NUMERIC"), returns="@arg0"),
        Sig(params=("NUMERIC", "INTERVAL"), returns="@arg1"),
    ],
    "/": [Sig(params=("NUMERIC", "NUMERIC"), returns="DOUBLE")],
    "%": [Sig(params=("NUMERIC", "NUMERIC"), returns="@arg0")],
    "||": [
        Sig(params=("STRING", "STRING"), returns="VARCHAR"),
        Sig(params=("ARRAY", "ARRAY"), returns="@arg0"),
    ],
    "AND": [Sig(params=("BOOLEAN", "BOOLEAN"), returns="BOOLEAN")],
    "OR": [Sig(params=("BOOLEAN", "BOOLEAN"), returns="BOOLEAN")],
    "NOT": [Sig(params=("BOOLEAN",), returns="BOOLEAN")],
    "LIKE": [Sig(params=("STRING", "STRING"), returns="BOOLEAN")],
    "ILIKE": [Sig(params=("STRING", "STRING"), returns="BOOLEAN")],
}


# ------------------------------------------------------------- standard SQL functions
_STANDARD_FUNCTIONS: dict[CatalogKey, list[Sig]] = {
    # strings
    "UPPER": [Sig(params=("STRING",), returns="VARCHAR")],
    "LOWER": [Sig(params=("STRING",), returns="VARCHAR")],
    "INITCAP": [Sig(params=("STRING",), returns="VARCHAR")],
    "TRIM": [
        Sig(params=("STRING",), returns="VARCHAR"),
        Sig(params=("STRING", "STRING"), returns="VARCHAR"),
    ],
    "LTRIM": [
        Sig(params=("STRING",), returns="VARCHAR"),
        Sig(params=("STRING", "STRING"), returns="VARCHAR"),
    ],
    "RTRIM": [
        Sig(params=("STRING",), returns="VARCHAR"),
        Sig(params=("STRING", "STRING"), returns="VARCHAR"),
    ],
    "LENGTH": [Sig(params=("STRING",), returns="BIGINT")],
    "CONCAT": [Sig(params=("STRING",), returns="VARCHAR", variadic=True)],
    "CONCAT_WS": [Sig(params=("STRING",), returns="VARCHAR", variadic=True)],
    # Only the first argument is a string; the offsets are not. Splitting by position is
    # what v1's STRING_FIRST_ARGUMENT_FUNCTIONS existed for, and a signature says it once.
    "SUBSTRING": [
        Sig(params=("STRING", "NUMERIC"), returns="VARCHAR"),
        Sig(params=("STRING", "NUMERIC", "NUMERIC"), returns="VARCHAR"),
    ],
    "LEFT": [Sig(params=("STRING", "NUMERIC"), returns="VARCHAR")],
    "RIGHT": [Sig(params=("STRING", "NUMERIC"), returns="VARCHAR")],
    "REPLACE": [Sig(params=("STRING", "STRING", "STRING"), returns="VARCHAR")],
    "LPAD": [
        Sig(params=("STRING", "NUMERIC"), returns="VARCHAR"),
        Sig(params=("STRING", "NUMERIC", "STRING"), returns="VARCHAR"),
    ],
    "RPAD": [
        Sig(params=("STRING", "NUMERIC"), returns="VARCHAR"),
        Sig(params=("STRING", "NUMERIC", "STRING"), returns="VARCHAR"),
    ],
    # numbers
    "ROUND": [
        Sig(params=("NUMERIC",), returns="@arg0"),
        Sig(params=("NUMERIC", "NUMERIC"), returns="@arg0"),
    ],
    "ABS": [Sig(params=("NUMERIC",), returns="@arg0")],
    "CEIL": [Sig(params=("NUMERIC",), returns="@arg0")],
    "FLOOR": [Sig(params=("NUMERIC",), returns="@arg0")],
    "SIGN": [Sig(params=("NUMERIC",), returns="BIGINT")],
    "SQRT": [Sig(params=("NUMERIC",), returns="DOUBLE")],
    "EXP": [Sig(params=("NUMERIC",), returns="DOUBLE")],
    "LN": [Sig(params=("NUMERIC",), returns="DOUBLE")],
    "LOG": [
        Sig(params=("NUMERIC",), returns="DOUBLE"),
        Sig(params=("NUMERIC", "NUMERIC"), returns="DOUBLE"),
    ],
    "POWER": [Sig(params=("NUMERIC", "NUMERIC"), returns="DOUBLE")],
    # aggregates. `SUM` is the reason `@family` exists: `SUM(INT)` is HUGEINT in DuckDB and
    # NUMBER(38,0) in Snowflake, so it is not `@arg0` - but it is certainly numeric because
    # its argument is, and that is what an undeclared column needs to hear.
    "SUM": [Sig(params=("NUMERIC",), returns="@family(arg0)")],
    "AVG": [Sig(params=("NUMERIC",), returns="DOUBLE")],
    "MIN": [Sig(params=("ANY",), returns="@arg0")],
    "MAX": [Sig(params=("ANY",), returns="@arg0")],
    "COUNT": [Sig(params=("ANY",), returns="BIGINT", variadic=True)],
    # type-transparent: every argument shares one domain with every other, so these link
    # rather than claim. `coalesce(a.name, b.name)` types either from the other.
    "COALESCE": [Sig(params=("@T",), returns="@arg0", variadic=True)],
    "NULLIF": [Sig(params=("@T", "@T"), returns="@arg0")],
    "GREATEST": [Sig(params=("@T",), returns="@arg0", variadic=True)],
    "LEAST": [Sig(params=("@T",), returns="@arg0", variadic=True)],
}


CATALOG_COMMON: dict[CatalogKey, list[Sig]] = {**_STANDARD_FUNCTIONS, **_OPERATORS}
"""What every engine agrees on. `upper(number_column)` is an error in DuckDB, Snowflake and
Spark alike, so the signature that says so belongs to none of them in particular."""


# --------------------------------------------------------------- per-dialect additions
_DUCKDB_FUNCTIONS: dict[CatalogKey, list[Sig]] = {
    # The unit arrives as an `exp.Var`, which carries no type, so it is ANY rather than
    # STRING - a claim about it would be a claim about a keyword.
    "TIMESTAMP_TRUNC": [Sig(params=("TEMPORAL", "ANY"), returns="TIMESTAMP")],
    "DATE_TRUNC": [Sig(params=("TEMPORAL", "ANY"), returns="DATE")],
    "SPLIT": [Sig(params=("STRING", "STRING"), returns="ARRAY<VARCHAR>")],
    "LIST_EXTRACT": [Sig(params=("ARRAY", "INTEGER"), returns="@element")],
    "LIST_VALUE": [Sig(params=("@T",), returns="ARRAY<VARCHAR>", variadic=True)],
    "EPOCH": [Sig(params=("TEMPORAL",), returns="DOUBLE")],
}

_SNOWFLAKE_FUNCTIONS: dict[CatalogKey, list[Sig]] = {
    "ZEROIFNULL": [Sig(params=("NUMERIC",), returns="@arg0")],
    "NVL": [Sig(params=("@T", "@T"), returns="@arg0")],
    "IFF": [Sig(params=("BOOLEAN", "@T", "@T"), returns="@arg1")],
    "TO_VARCHAR": [Sig(params=("ANY",), returns="VARCHAR")],
    "TO_NUMBER": [Sig(params=("ANY",), returns="DECIMAL(38,0)")],
    "ARRAY_CONSTRUCT": [Sig(params=("@T",), returns="ARRAY<VARCHAR>", variadic=True)],
    # Snowflake's DIV0 is the one place `/` differs enough to be worth writing down.
    "DIV0": [Sig(params=("NUMERIC", "NUMERIC"), returns="DOUBLE")],
}

_SPARK_FUNCTIONS: dict[CatalogKey, list[Sig]] = {
    "DATE_ADD": [Sig(params=("TEMPORAL", "INTEGER"), returns="DATE")],
    "DATE_SUB": [Sig(params=("TEMPORAL", "INTEGER"), returns="DATE")],
    "EXPLODE": [Sig(params=("ARRAY",), returns="@element")],
    "ARRAY_CONTAINS": [Sig(params=("ARRAY", "ANY"), returns="BOOLEAN")],
    # Spark's `concat` is not string-only: it concatenates arrays too, which is narrower
    # than the common entry would allow to be contradicted.
    "CONCAT": [
        Sig(params=("STRING",), returns="VARCHAR", variadic=True),
        Sig(params=("ARRAY",), returns="@arg0", variadic=True),
    ],
}


CATALOG_BY_DIALECT: dict[DialectName, dict[CatalogKey, list[Sig]]] = {
    "duckdb": _DUCKDB_FUNCTIONS,
    "snowflake": _SNOWFLAKE_FUNCTIONS,
    "spark": _SPARK_FUNCTIONS,
}


CATALOG_IS_COMPLETE: dict[DialectName, bool] = {
    "duckdb": False,
    "snowflake": False,
    "spark": False,
}
"""Whether a dialect's catalog enumerates every function the engine has.

Gates `unknown-function` and nothing else. "This function does not exist" is only a truthful
statement from an exhaustive list, and none of these lists is exhaustive - they are
hand-written gap-fillers, and `duckdb_functions()` generation is deferred. With the flag
False the finding is dormant.

`function-arity` and `contradicted-type` are deliberately not gated: both reason from an
entry that *is* present, so a thin catalog makes them quiet rather than wrong.
"""


ANNOTATION_GAP_KEYS: dict[DialectName, frozenset[CatalogKey]] = {
    # Where sqlglot's own metadata has a hole. `date_trunc` parses to TimestampTrunc in
    # DuckDB but only DateTrunc is registered; `split` and `list_extract` are unregistered
    # outright. Everything else in the catalog is consulted for argument families and left
    # to sqlglot to annotate.
    "duckdb": frozenset({"TIMESTAMP_TRUNC", "SPLIT", "LIST_EXTRACT"}),
    "snowflake": frozenset({"ZEROIFNULL", "DIV0"}),
    "spark": frozenset(),
}


def signatures_for_dialect(dialect_name: DialectName) -> dict[CatalogKey, list[Sig]]:
    """The common catalog with one dialect's entries layered over it.

    A dialect entry *replaces* the common one for its key. Never merged: a dialect layer
    exists to say something different, and appending overloads could only ever widen.
    """
    return {**CATALOG_COMMON, **CATALOG_BY_DIALECT.get(dialect_name, {})}


def dialect_has_its_own_layer(dialect_name: DialectName) -> bool:
    """Whether this dialect was written down at all, or is being served common-only."""
    return dialect_name in CATALOG_BY_DIALECT
