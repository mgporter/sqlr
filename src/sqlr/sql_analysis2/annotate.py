"""Step 0 - the signature catalog, wired into sqlglot's annotate_types.

sqlglot dispatches annotation on a plain dict keyed by expression class
(`Dialect.EXPRESSION_METADATA`). That dict is the whole extension point. We build a
*copy* of it with our entries layered on top and hand it to `annotate_types(
expression_metadata=...)`, so nothing global is mutated.

Only the keys in `ANNOTATION_GAP_KEYS` are installed. The rest of the catalog exists for
`facts.py`, which needs to know what a call *accepts* - something sqlglot's metadata never
says. Installing an annotator for a function sqlglot already types correctly would replace a
right answer with ours.

The lattice itself lives in `families.py`; this module is the wiring between it, the
catalog, and sqlglot.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, cast

from sqlglot import exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.optimizer.annotate_types import TypeAnnotator
from sqlglot.typing import ExprMetadataType

from sqlr.sql_analysis2.catalog import (
    ANNOTATION_GAP_KEYS,
    CATALOG_IS_COMPLETE,
    OPERATOR_SQL_NAMES,
    CatalogKey,
    DialectName,
    Sig,
    dialect_has_its_own_layer,
    signatures_for_dialect,
)
from sqlr.sql_analysis2.families import in_family

logger = logging.getLogger(__name__)

Annotator = Callable[[TypeAnnotator, exp.Expr], None]

__all__ = [
    "arguments_of_call",
    "candidate_overloads",
    "catalog_key",
    "expression_metadata",
    "pick_overload",
    "signatures_for_dialect",
    "signatures_of_call",
    "surviving_overloads",
    "unknown_function_findings_are_trustworthy",
]


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


def candidate_overloads(node: exp.Expr, signatures: list[Sig]) -> list[Sig]:
    """The overloads whose arity this call could satisfy.

    Arity is separated from types on purpose: a call with no arity match is an arity error
    and must never also be reported as a type error against a signature it was never going
    to match.
    """
    count = len(arguments_of_call(node))
    return [sig for sig in signatures if sig.accepts_arity(count)]


def surviving_overloads(node: exp.Expr, signatures: list[Sig]) -> list[Sig]:
    """Arity candidates minus the ones some argument definitively contradicts.

    `is False`, never `None`: an argument of unknown type rules nothing out, which is what
    keeps an undeclared column from silently narrowing the overload set to one that then
    speaks confidently about it.

    This is what makes a return-marker link possible on an overloaded operator. `*` has
    three overloads returning `@arg0`, `@arg0` and `@arg1`, which disagree - but `amount * 2`
    rules out `(NUMERIC, INTERVAL)` because `2` is not an interval, and the two survivors
    agree.
    """
    args = arguments_of_call(node)
    return [
        sig
        for sig in candidate_overloads(node, signatures)
        if not any(
            in_family(argument.type, sig.family_at(index)) is False
            for index, argument in enumerate(args)
        )
    ]


def pick_overload(
    node: exp.Expr, signatures: dict[CatalogKey, list[Sig]]
) -> tuple[Sig | None, list[exp.Expr]]:
    """First overload no argument definitively contradicts."""
    surviving = surviving_overloads(node, signatures_of_call(node, signatures))
    return (surviving[0] if surviving else None), arguments_of_call(node)


def unknown_function_findings_are_trustworthy(dialect_name: DialectName) -> bool:
    """Whether `unknown-function` may be reported for this dialect.

    "This function does not exist" is only truthful from an exhaustive list, and every
    catalog here is a hand-written gap-filler. See `CATALOG_IS_COMPLETE`.
    """
    return CATALOG_IS_COMPLETE.get(dialect_name, False)


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

    link = sig.return_link
    if link is not None:
        # `@family(argN)` names no concrete type, so annotation can only pass the argument's
        # own type through. The *family* half of it is a fact, read by `facts.py`.
        arg_type = args[link.argument_index].type if link.argument_index < len(args) else None
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

    if not dialect_has_its_own_layer(dialect_name):
        # Never silently: a dialect served common-only still checks `upper(number)`, but it
        # knows none of the engine's own functions, and a reader seeing few findings needs
        # to know which of the two reasons they are looking at.
        logger.warning(
            "no dialect catalog for %s; using the common signatures only "
            "(%d entries). Engine-specific functions will not be checked.",
            dialect_name,
            len(signatures),
        )

    mapped = 0
    for key in ANNOTATION_GAP_KEYS.get(dialect_name, frozenset()):
        cls = exp.FUNCTION_BY_NAME.get(key)  # sqlglot's own "ROUND" -> exp.Round map
        if cls is not None and not issubclass(cls, exp.Anonymous):
            metadata[cls] = {"annotator": _make_annotator(signatures, dialect_name)}
            mapped += 1
    logger.info(
        "catalog for %s: %d signatures, %d annotation gaps installed",
        dialect_name,
        len(signatures),
        mapped,
    )

    # Everything with no sqlglot class arrives as Anonymous; route it by written name.
    # Chain to the dialect's own Anonymous annotator so registered UDF types survive.
    prior = cast("Annotator | None", metadata.get(exp.Anonymous, {}).get("annotator"))
    metadata[exp.Anonymous] = {
        "annotator": _make_annotator(signatures, dialect_name, fallback=prior)
    }
    return metadata
