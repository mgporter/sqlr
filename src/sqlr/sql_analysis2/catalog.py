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
hand-written tables - see the revision section of `type_check_plan.md`.

Signatures are written against **sqlglot's node argument order, not the written order**:
`date_trunc('day', ts)` parses to `TimestampTrunc(this=ts, unit='day')`, so its signature
reads `(TEMPORAL, ANY)`. Writing it the way the SQL reads produces a confident error on
valid SQL.
"""

from dataclasses import dataclass

from sqlglot import exp

type DialectName = str
type CatalogKey = str
"""What a call is looked up by: a function's `sql_name()`, an `Anonymous` call's written
name uppercased, or an operator's SQL symbol. Never a class name - those change between
sqlglot versions and do not exist for `Anonymous` at all."""

type FamilyName = str
"""A parameter's accepted domain. One of the `FAMILIES` keys in `annotate.py`, or:

- `ANY` - accepts anything, and says nothing about what was passed.
- `@T`, `@U` - a type variable. Accepts anything, but two positions sharing one variable
  are constrained to each other, which is what produces a *link* fact rather than a claim.
"""

type ReturnMarker = str
"""A concrete type name, or `@arg0`/`@argN` for "whatever that argument was", or
`@element` to unwrap `ARRAY<T>` to `T`."""


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


# ------------------------------------------------------------------------- functions
_DUCKDB_FUNCTIONS: dict[CatalogKey, list[Sig]] = {
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
    # Only the first argument is a string; the offsets are not. Splitting by position is
    # what v1's STRING_FIRST_ARGUMENT_FUNCTIONS existed for, and a signature says it once.
    "SUBSTRING": [
        Sig(params=("STRING", "NUMERIC"), returns="VARCHAR"),
        Sig(params=("STRING", "NUMERIC", "NUMERIC"), returns="VARCHAR"),
    ],
    "LEFT": [Sig(params=("STRING", "NUMERIC"), returns="VARCHAR")],
    "RIGHT": [Sig(params=("STRING", "NUMERIC"), returns="VARCHAR")],
    "REPLACE": [Sig(params=("STRING", "STRING", "STRING"), returns="VARCHAR")],
    "SPLIT": [Sig(params=("STRING", "STRING"), returns="ARRAY<VARCHAR>")],
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
    # temporal. The unit arrives as an `exp.Var`, which carries no type, so it is ANY
    # rather than STRING - a claim about it would be a claim about a keyword.
    "TIMESTAMP_TRUNC": [Sig(params=("TEMPORAL", "ANY"), returns="TIMESTAMP")],
    "DATE_TRUNC": [Sig(params=("TEMPORAL", "ANY"), returns="DATE")],
    # arrays
    "LIST_EXTRACT": [Sig(params=("ARRAY", "NUMERIC"), returns="@element")],
    # type-transparent: every argument shares one domain with every other, so these link
    # rather than claim. `coalesce(a.name, b.name)` types either from the other.
    "COALESCE": [Sig(params=("@T",), returns="@arg0", variadic=True)],
    "NULLIF": [Sig(params=("@T", "@T"), returns="@arg0")],
    "GREATEST": [Sig(params=("@T",), returns="@arg0", variadic=True)],
    "LEAST": [Sig(params=("@T",), returns="@arg0", variadic=True)],
}


CATALOG: dict[DialectName, dict[CatalogKey, list[Sig]]] = {
    "duckdb": {**_DUCKDB_FUNCTIONS, **_OPERATORS},
}


ANNOTATION_GAP_KEYS: dict[DialectName, frozenset[CatalogKey]] = {
    # Where sqlglot's own metadata has a hole. `date_trunc` parses to TimestampTrunc in
    # DuckDB but only DateTrunc is registered; `split` and `list_extract` are unregistered
    # outright. Everything else in the catalog is consulted for argument families and left
    # to sqlglot to annotate.
    "duckdb": frozenset({"TIMESTAMP_TRUNC", "SPLIT", "LIST_EXTRACT"}),
}
