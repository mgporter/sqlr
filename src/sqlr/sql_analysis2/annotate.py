"""Step 0 - the signature catalog, wired into sqlglot's annotate_types.

sqlglot dispatches annotation on a plain dict keyed by expression class
(`Dialect.EXPRESSION_METADATA`). That dict is the whole extension point. We build a
*copy* of it with our entries layered on top and hand it to `annotate_types(
expression_metadata=...)`, so nothing global is mutated (plan Q6).

Only the keys in `ANNOTATION_GAP_KEYS` are installed. The rest of the catalog exists for
`facts.py`, which needs to know what a call *accepts* - something sqlglot's metadata never
says. Installing an annotator for a function sqlglot already types correctly would replace a
right answer with ours.

The family test in here is the single most important function in the design. It is
three-valued on purpose: `False` means "definitely wrong, report it" and `None` means
"cannot say, stay quiet". Conflating them is how a checker gets a reputation for lying.
"""

from __future__ import annotations

from typing import Any, Callable, cast

from sqlglot import exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.optimizer.annotate_types import TypeAnnotator
from sqlglot.typing import ExprMetadataType

from sqlr.sql_analysis2.catalog import (
    ANNOTATION_GAP_KEYS,
    CATALOG,
    OPERATOR_SQL_NAMES,
    CatalogKey,
    DialectName,
    FamilyName,
    Sig,
)

Annotator = Callable[[TypeAnnotator, exp.Expr], None]

# Families map onto sqlglot's own type-set constants. "ANY" and type variables are handled
# in `in_family`, since neither is a set of types.
FAMILIES: dict[FamilyName, set[exp.DType]] = {
    "NUMERIC": set(exp.DataType.NUMERIC_TYPES),
    "STRING": set(exp.DataType.TEXT_TYPES),
    "TEMPORAL": set(exp.DataType.TEMPORAL_TYPES),
    "BOOLEAN": {exp.DType.BOOLEAN},
    "ARRAY": {exp.DType.ARRAY},
    "INTERVAL": {exp.DType.INTERVAL},
}

FAMILY_DEFAULT_TYPE: dict[FamilyName, str] = {
    "NUMERIC": "BIGINT",
    "STRING": "VARCHAR",
    "TEMPORAL": "TIMESTAMP",
    "BOOLEAN": "BOOLEAN",
    "ARRAY": "ARRAY<VARCHAR>",
    "INTERVAL": "INTERVAL",
}
"""What an inferred family becomes when nothing more precise is available. A link carrying
a concrete type is preferred over these - `DATE` stays `DATE` rather than widening to
`TIMESTAMP` - so this only applies to claims, which name a family and nothing else."""


def is_a_type_variable(family: FamilyName) -> bool:
    """Whether a parameter is `@T`-style: accepts anything, but binds to its twin.

    Two positions of one signature sharing a variable are constrained to each other. That
    constraint is a *link* fact, and it is the only thing separating `a = b` (which types
    either column from the other) from `a = anything` (which would say nothing at all).
    """
    return family.startswith("@")


def catalog_key(node: exp.Expr) -> CatalogKey | None:
    """The name we look the catalog up by. Never a class name - except for operators, which
    have no other name; see `OPERATOR_SQL_NAMES`."""
    if isinstance(node, exp.Anonymous):
        return node.name.upper()
    operator = OPERATOR_SQL_NAMES.get(type(node))
    if operator is not None:
        return operator
    if isinstance(node, exp.Func):
        return node.sql_name()
    return None


def signatures_for_dialect(dialect_name: DialectName) -> dict[CatalogKey, list[Sig]]:
    return CATALOG.get(dialect_name, {})


def signatures_of_call(
    node: exp.Expr, signatures: dict[CatalogKey, list[Sig]]
) -> list[Sig]:
    """Every overload the catalog holds for this call, or an empty list."""
    key = catalog_key(node)
    return signatures.get(key, []) if key is not None else []


def arguments_of_call(node: exp.Expr) -> list[exp.Expr]:
    """Positional arguments of a call, in sqlglot's *node* arg order.

    Not the written order: date_trunc('day', ts) parses to
    TimestampTrunc(this=ts, unit='day'). Catalog signatures follow this order.

    Operators go through the same path - `exp.Binary` declares `this` and `expression`, so
    `a + b` yields `[a, b]` with no special case.
    """
    out: list[exp.Expr] = []
    if isinstance(node, exp.Anonymous):
        names = ["expressions"]
    else:
        names = list(node.arg_types)  # declaration order, stable
    for name in names:
        val = node.args.get(name)
        # anything that is not an Expr is a flag (`big_int`) or a unit string, not an argument
        if isinstance(val, list):
            out.extend(a for a in cast("list[Any]", val) if isinstance(a, exp.Expr))
        elif isinstance(val, exp.Expr):
            out.append(val)
    return out


def in_family(dtype: exp.DataType | None, family: FamilyName) -> bool | None:
    """True / False / None. None means unknown, so no judgement is made.

    False means "definitely wrong, report it". Conflating the two is how a checker
    gets a reputation for lying.
    """
    if family == "ANY" or is_a_type_variable(family):
        return True
    if dtype is None or dtype.is_type(exp.DType.UNKNOWN):
        return None  # <- NOT False
    return dtype.this in FAMILIES.get(family, set())


def family_of_type(dtype: exp.DataType | None) -> FamilyName | None:
    """Which family a concrete type belongs to, or None when it belongs to none.

    The inverse of `in_family`, and the direction a link needs: a link says two nodes share
    a domain, so the known end has to name the domain it is in before the unknown end can
    take it.
    """
    if dtype is None or dtype.is_type(exp.DType.UNKNOWN):
        return None
    for family, types in FAMILIES.items():
        if dtype.this in types:
            return family
    return None


def candidate_overloads(node: exp.Expr, signatures: list[Sig]) -> list[Sig]:
    """The overloads whose arity this call could satisfy.

    Arity is separated from types on purpose: a call with no arity match is an arity error
    and must never also be reported as a type error against a signature it was never going
    to match.
    """
    count = len(arguments_of_call(node))
    return [sig for sig in signatures if sig.accepts_arity(count)]


def pick_overload(
    node: exp.Expr, signatures: dict[CatalogKey, list[Sig]]
) -> tuple[Sig | None, list[exp.Expr]]:
    """First overload no argument definitively contradicts."""
    args = arguments_of_call(node)
    for sig in candidate_overloads(node, signatures_of_call(node, signatures)):
        if any(
            in_family(argument.type, sig.family_at(index)) is False
            for index, argument in enumerate(args)
        ):
            continue  # definitely the wrong overload
        return sig, args
    return None, args


def resolve_return_type(
    sig: Sig, args: list[exp.Expr], dialect_name: DialectName
) -> exp.DataType | exp.DType:
    """Turn a signature's `returns` marker into a concrete type."""
    if sig.returns == "@element":
        arr = args[0].type if args else None
        if arr is not None and arr.is_type(exp.DType.ARRAY) and arr.expressions:
            element = arr.expressions[0]
            return element if isinstance(element, exp.DataType) else exp.DType.UNKNOWN
        return exp.DType.UNKNOWN
    if sig.returns.startswith("@arg"):
        i = int(sig.returns[len("@arg") :])
        arg_type = args[i].type if i < len(args) else None
        return arg_type if arg_type is not None else exp.DType.UNKNOWN
    return exp.DataType.build(sig.returns, dialect=dialect_name)


def _make_annotator(
    signatures: dict[CatalogKey, list[Sig]],
    dialect_name: DialectName,
    fallback: Annotator | None = None,
) -> Annotator:
    """One annotator closed over the catalog for this dialect.

    sqlglot calls annotators bottom-up (children first), so every argument's `.type`
    is already set when we run.
    """

    def _annotate(annotator: TypeAnnotator, node: exp.Expr) -> None:
        sig, args = pick_overload(node, signatures)
        if sig is not None:
            # _set_type is sqlglot's documented extension contract for annotators
            annotator._set_type(  # pyright: ignore[reportPrivateUsage]
                node, resolve_return_type(sig, args, dialect_name)
            )
        elif fallback is not None:
            fallback(annotator, node)  # keep sqlglot's own behaviour for uncatalogued calls
        else:
            annotator._set_type(node, exp.DType.UNKNOWN)  # pyright: ignore[reportPrivateUsage]

    return _annotate


def expression_metadata(dialect_name: DialectName) -> ExprMetadataType:
    """The dialect's metadata dict, copied, with our gap entries layered on top.

    Pass the result to `annotate_types(expression_metadata=...)`. The dialect itself is
    never modified, so two runs with different dialects cannot interfere.

    Only `ANNOTATION_GAP_KEYS` are installed. Operator keys could not be installed even by
    mistake - `exp.FUNCTION_BY_NAME` holds no `"+"` - which is why they can share one
    catalog with the functions.
    """
    dialect = Dialect.get_or_raise(dialect_name)
    signatures = signatures_for_dialect(dialect_name)
    metadata: ExprMetadataType = dict(dialect.EXPRESSION_METADATA)

    mapped = 0
    for key in ANNOTATION_GAP_KEYS.get(dialect_name, frozenset()):
        cls = exp.FUNCTION_BY_NAME.get(key)  # sqlglot's own "ROUND" -> exp.Round map
        if cls is not None and not issubclass(cls, exp.Anonymous):
            metadata[cls] = {"annotator": _make_annotator(signatures, dialect_name)}
            mapped += 1

    # Everything with no sqlglot class arrives as Anonymous; route it by written name.
    # Chain to the dialect's own Anonymous annotator so registered UDF types survive.
    prior = cast("Annotator | None", metadata.get(exp.Anonymous, {}).get("annotator"))
    metadata[exp.Anonymous] = {
        "annotator": _make_annotator(signatures, dialect_name, fallback=prior)
    }
    return metadata
