"""The type lattice: a tree of families, and the three questions asked of it.

A flat set of families cannot say that `round` accepts any numeric while `list_extract`
accepts only an integer index, and it cannot answer "are these two types the same kind of
thing" at more than one granularity. So families are a tree, every concrete type maps to a
leaf of it, and every type decision in the pipeline is one of three operations:

- `in_family`            - does this type satisfy this family? Three-valued, and the single
  most important contract in the design.
- `nearest_common_family` - what do these two types have in common? `ANY` means nothing,
  which is the conflict signal.
- `widest_type_in`       - what does an inferred family become in the schema?

**Precision and scale are recorded, never checked.** `DECIMAL(10,2)` against `DECIMAL(38,9)`
is not a finding and `VARCHAR(20)` against `VARCHAR(100)` is not a finding. The parameters
exist so a report can print what the user wrote and so fixture generation can size a column
later. Nothing here reads them, and adding a check on them is a deliberate act rather than a
natural extension of anything below.
"""

from __future__ import annotations

from typing import Iterable

from sqlglot import exp

type FamilyName = str
"""A node of the lattice, or `ANY`, or a `@T`-style type variable.

`ANY` accepts anything and says nothing about what was passed. A variable accepts anything
too, but two positions sharing one variable are constrained to each other - which is what
produces a *link* fact rather than a claim, and the difference between `a = b` (types either
column from the other) and `a = anything` (says nothing at all).
"""

ANY: FamilyName = "ANY"


FAMILY_PARENT: dict[FamilyName, FamilyName | None] = {
    ANY: None,
    "NUMERIC": ANY,
    "INTEGER": "NUMERIC",
    "DECIMAL": "NUMERIC",
    "FLOAT": "NUMERIC",
    "STRING": ANY,
    "TEMPORAL": ANY,
    "DATE": "TEMPORAL",
    "TIME": "TEMPORAL",
    "TIMESTAMP": "TEMPORAL",
    "BOOLEAN": ANY,
    "INTERVAL": ANY,
    "BINARY": ANY,
    "ARRAY": ANY,
    "STRUCT": ANY,
}
"""The lattice, written as child -> parent because that is the direction every query walks.

Depth stops where the distinction stops being one a signature needs. There is no
`TIMESTAMPTZ` node under `TIMESTAMP`: no catalog entry accepts a timestamp with a zone and
rejects one without, and a node nothing branches on is a node that only ever produces
findings nobody asked for.
"""


def _leaf_families() -> dict[exp.DType, FamilyName]:
    """Every concrete sqlglot type mapped to the deepest family that describes it.

    Built from sqlglot's own type-set constants rather than a hand list, so a type added by
    a later sqlglot lands in the right family without this file changing. Order matters:
    the narrow sets are written first and win, since `NUMERIC_TYPES` contains every integer
    and every decimal too.
    """
    leaves: dict[exp.DType, FamilyName] = {}

    def assign(types: Iterable[exp.DType], family: FamilyName) -> None:
        for dtype in types:
            leaves.setdefault(dtype, family)

    assign(exp.DataType.INTEGER_TYPES, "INTEGER")
    assign(exp.DataType.FLOAT_TYPES, "FLOAT")
    assign(exp.DataType.REAL_TYPES, "DECIMAL")  # REAL_TYPES minus the floats above
    assign(exp.DataType.NUMERIC_TYPES, "NUMERIC")  # anything numeric the three missed

    assign(exp.DataType.TEXT_TYPES, "STRING")

    assign({exp.DType.DATE, exp.DType.DATE32}, "DATE")
    assign({exp.DType.TIME, exp.DType.TIMETZ}, "TIME")
    assign(exp.DataType.TEMPORAL_TYPES, "TIMESTAMP")  # the datetimes and timestamps

    assign(exp.DataType.ARRAY_TYPES, "ARRAY")
    assign(exp.DataType.STRUCT_TYPES, "STRUCT")

    assign({exp.DType.BOOLEAN}, "BOOLEAN")
    assign({exp.DType.INTERVAL}, "INTERVAL")
    assign(
        {
            dtype
            for dtype in exp.DType
            if dtype.name in ("BINARY", "VARBINARY", "BLOB", "BYTES")
        },
        "BINARY",
    )
    return leaves


LEAF_FAMILY_OF_TYPE: dict[exp.DType, FamilyName] = _leaf_families()


WIDEST_TYPE_IN_FAMILY: dict[FamilyName, str] = {
    "NUMERIC": "DECIMAL(38,9)",
    "INTEGER": "BIGINT",
    "DECIMAL": "DECIMAL(38,9)",
    "FLOAT": "DOUBLE",
    "STRING": "VARCHAR",
    "TEMPORAL": "TIMESTAMP",
    "DATE": "DATE",
    "TIME": "TIME",
    "TIMESTAMP": "TIMESTAMP",
    "BOOLEAN": "BOOLEAN",
    "INTERVAL": "INTERVAL",
    "BINARY": "VARBINARY",
    "ARRAY": "ARRAY<VARCHAR>",
}
"""What an inferred family becomes in the schema sqlglot re-annotates with.

The *widest* member, never a plausible-looking narrow one. `where amount > 0` proves the
column is numeric and proves nothing about its width, so inferring `INT` from it would
falsely contradict `amount * 1.5` three CTEs later. A wide type contradicts nothing, and the
family - which is what a report prints and what fixture generation reads - is recorded
separately and stays exact.

`STRUCT` and `ANY` are deliberately absent: neither has a member wide enough to stand for the
rest, and a column inferred into one of them is better left UNKNOWN.
"""


def is_a_type_variable(family: FamilyName) -> bool:
    """Whether a parameter is `@T`-style: accepts anything, but binds to its twin."""
    return family.startswith("@")


def family_of_type(dtype: exp.DataType | None) -> FamilyName | None:
    """The deepest family a concrete type belongs to, or None when nothing can say.

    None for an unknown type and for a type in no family at all - a user-defined type, or a
    name the lattice does not recognise. Both mean "no judgement", never "wrong".
    """
    if dtype is None or dtype.is_type(exp.DType.UNKNOWN):
        return None
    return LEAF_FAMILY_OF_TYPE.get(dtype.this)


def ancestry_of(family: FamilyName) -> list[FamilyName]:
    """A family and every family above it, narrowest first, ending at `ANY`."""
    chain: list[FamilyName] = []
    current: FamilyName | None = family
    while current is not None:
        chain.append(current)
        current = FAMILY_PARENT.get(current)
    return chain


def root_family_of(family: FamilyName) -> FamilyName:
    """The top-level family a node sits under. `INTEGER` -> `NUMERIC`, `DATE` -> `TEMPORAL`.

    What an *untyped literal* proves. `where amount > 0` says the column is numeric; it says
    nothing about whether it is an integer, and inferring `BIGINT` from it would falsely
    contradict `amount * 1.5` three CTEs later. A literal that carries a written type -
    `date '2024-01-01'`, a cast - is not an `exp.Literal` at all and keeps its exact type.
    """
    chain = ancestry_of(family)
    return chain[-2] if len(chain) >= 2 else family


def family_satisfies(family: FamilyName, required: FamilyName) -> bool:
    """Whether `family` is `required` or sits beneath it. `DECIMAL` satisfies `NUMERIC`."""
    return required in ancestry_of(family)


def in_family(dtype: exp.DataType | None, family: FamilyName) -> bool | None:
    """True / False / None. **None means unknown, so no judgement is made.**

    `False` means "definitely wrong, report it". Conflating the two is how a checker gets a
    reputation for lying, and every false positive found while building this traced back to
    the distinction. The lattice changed what `True` means; it did not change what `None`
    means.
    """
    if family == ANY or is_a_type_variable(family):
        return True
    leaf = family_of_type(dtype)
    if leaf is None:
        return None  # <- NOT False
    return family_satisfies(leaf, family)


def nearest_common_family(left: FamilyName, right: FamilyName) -> FamilyName:
    """The deepest family both sit beneath. `ANY` means they have nothing in common.

    This is the compatibility test: two stated types conflict **iff their nearest common
    family is `ANY`**. `int_col = decimal_col` is ordinary SQL and not a defect, and
    reporting it is how a checker gets switched off - which costs more than the bug it would
    have caught. The cases worth reporting, a string compared to a number or a boolean used
    as text, all reach `ANY`.
    """
    above_right = set(ancestry_of(right))
    for family in ancestry_of(left):
        if family in above_right:
            return family
    return ANY


def widest_type_in(family: FamilyName) -> str | None:
    """The type name an inferred family becomes, or None when the family has no stand-in."""
    return WIDEST_TYPE_IN_FAMILY.get(family)


def describe_families(families: frozenset[FamilyName] | set[FamilyName]) -> str:
    """`NUMERIC`, `NUMERIC or INTERVAL` - in a stable order, since a set has none."""
    names = sorted(families)
    if not names:
        return ANY
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} or {names[-1]}"
