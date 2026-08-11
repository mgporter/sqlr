"""Step 0 — the signature catalog, wired into sqlglot's annotate_types.

sqlglot dispatches annotation on a plain dict keyed by expression class
(`Dialect.EXPRESSION_METADATA`). That dict is the whole extension point. We build a
*copy* of it with our entries layered on top and hand it to `annotate_types(
expression_metadata=...)`, so nothing global is mutated (plan Q6).
"""

from __future__ import annotations

from typing import Any, Callable, cast

from sqlglot import exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.optimizer.annotate_types import TypeAnnotator
from sqlglot.typing import ExprMetadataType

from sqlr.sql_analysis2.catalog import CATALOG, Sig

Annotator = Callable[[TypeAnnotator, exp.Expr], None]

# Families map onto sqlglot's own type-set constants. "ANY" is handled in in_family.
FAMILIES: dict[str, set[exp.DType]] = {
    "NUMERIC": set(exp.DataType.NUMERIC_TYPES),
    "STRING": set(exp.DataType.TEXT_TYPES),
    "TEMPORAL": set(exp.DataType.TEMPORAL_TYPES),
    "BOOLEAN": {exp.DType.BOOLEAN},
    "ARRAY": {exp.DType.ARRAY},
}


def catalog_key(node: exp.Expr) -> str | None:
    """The name we look the catalog up by. Never a class name — see plan step 0."""
    if isinstance(node, exp.Anonymous):
        return node.name.upper()
    if isinstance(node, exp.Func):
        return node.sql_name()
    return None


def func_args(node: exp.Func) -> list[exp.Expr]:
    """Positional arguments of a call, in sqlglot's *node* arg order.

    Not the written order: date_trunc('day', ts) parses to
    TimestampTrunc(this=ts, unit='day'). Catalog signatures follow this order.
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


def in_family(dtype: exp.DataType | None, family: str) -> bool | None:
    """True / False / None. None means unknown, so no judgement is made.

    False means "definitely wrong, report it". Conflating the two is how a checker
    gets a reputation for lying.
    """
    if family == "ANY":
        return True
    if dtype is None or dtype.is_type(exp.DType.UNKNOWN):
        return None  # <- NOT False
    return dtype.this in FAMILIES[family]


def pick_overload(
    node: exp.Expr, sigs_by_key: dict[str, list[Sig]]
) -> tuple[Sig | None, list[exp.Expr]]:
    """First overload no argument definitively contradicts."""
    if not isinstance(node, exp.Func):
        return None, []
    args = func_args(node)
    for sig in sigs_by_key.get(catalog_key(node) or "", []):
        if len(sig.params) != len(args):
            continue
        if any(in_family(a.type, f) is False for a, f in zip(args, sig.params)):
            continue  # definitely the wrong overload
        return sig, args
    return None, args


def resolve_return_type(
    sig: Sig, args: list[exp.Expr], dialect_name: str
) -> exp.DataType | exp.DType:
    """Turn a signature's `returns` marker into a concrete type."""
    if sig.returns == "@element":
        arr = args[0].type if args else None
        if arr is not None and arr.is_type(exp.DType.ARRAY) and arr.expressions:
            element = arr.expressions[0]
            return element if isinstance(element, exp.DataType) else exp.DType.UNKNOWN
        return exp.DType.UNKNOWN
    if sig.returns.startswith("@arg"):
        i = int(sig.returns[len("@arg"):])
        arg_type = args[i].type if i < len(args) else None
        return arg_type if arg_type is not None else exp.DType.UNKNOWN
    return exp.DataType.build(sig.returns, dialect=dialect_name)


def _make_annotator(
    sigs_by_key: dict[str, list[Sig]],
    dialect_name: str,
    fallback: Annotator | None = None,
) -> Annotator:
    """One annotator closed over the catalog for this dialect.

    sqlglot calls annotators bottom-up (children first), so every argument's `.type`
    is already set when we run.
    """

    def _annotate(annotator: TypeAnnotator, node: exp.Expr) -> None:
        sig, args = pick_overload(node, sigs_by_key)
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


def expression_metadata(dialect_name: str) -> ExprMetadataType:
    """The dialect's metadata dict, copied, with our catalog entries layered on top.

    Pass the result to `annotate_types(expression_metadata=...)`. The dialect itself is
    never modified, so two runs with different dialects cannot interfere.
    """
    dialect = Dialect.get_or_raise(dialect_name)
    sigs_by_key = CATALOG.get(dialect_name, {})
    metadata: ExprMetadataType = dict(dialect.EXPRESSION_METADATA)

    mapped = 0
    for key in sigs_by_key:
        cls = exp.FUNCTION_BY_NAME.get(key)  # sqlglot's own "ROUND" -> exp.Round map
        if cls is not None and not issubclass(cls, exp.Anonymous):
            metadata[cls] = {"annotator": _make_annotator(sigs_by_key, dialect_name)}
            mapped += 1

    # Everything with no sqlglot class arrives as Anonymous; route it by written name.
    # Chain to the dialect's own Anonymous annotator so registered UDF types survive.
    prior = cast("Annotator | None", metadata.get(exp.Anonymous, {}).get("annotator"))
    metadata[exp.Anonymous] = {
        "annotator": _make_annotator(sigs_by_key, dialect_name, fallback=prior)
    }
    return metadata
