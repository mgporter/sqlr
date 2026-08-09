"""Minimal signature-catalog type checker layered on sqlglot. DuckDB dialect."""
from __future__ import annotations

from dataclasses import dataclass

from sqlglot import exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.optimizer.scope import traverse_scope

DIALECT = "duckdb"

# ------------------------------------------------------------------ catalog
@dataclass(frozen=True)
class Sig:
    params: tuple[str, ...]   # families: NUMERIC STRING TEMPORAL BOOLEAN ARRAY ANY
    returns: str              # a type string, or "@element" / "@argN"

CATALOG: dict[str, list[Sig]] = {
    "UPPER":           [Sig(("STRING",), "VARCHAR")],
    "SPLIT":           [Sig(("STRING", "STRING"), "ARRAY<VARCHAR>")],
    "LIST_EXTRACT":    [Sig(("ARRAY", "NUMERIC"), "@element")],
    "ROUND":           [Sig(("NUMERIC", "NUMERIC"), "@arg0"), Sig(("NUMERIC",), "@arg0")],
    # NOTE: params are in sqlglot's *node* arg order, not the order written in SQL.
    # date_trunc('day', ts) parses to TimestampTrunc(this=ts, unit='day').
    "TIMESTAMP_TRUNC": [Sig(("TEMPORAL", "STRING"), "@arg0")],
}

FAMILIES: dict[str, set] = {
    "NUMERIC":  set(exp.DataType.NUMERIC_TYPES),
    "STRING":   set(exp.DataType.TEXT_TYPES),
    "TEMPORAL": set(exp.DataType.TEMPORAL_TYPES),
    "BOOLEAN":  {exp.DType.BOOLEAN},
    "ARRAY":    {exp.DType.ARRAY},
}
FAMILY_DEFAULT = {"STRING": "VARCHAR", "NUMERIC": "DOUBLE",
                  "TEMPORAL": "TIMESTAMP", "BOOLEAN": "BOOLEAN", "ARRAY": "ARRAY<VARCHAR>"}


def catalog_key(node: exp.Expr) -> str | None:
    if isinstance(node, exp.Anonymous):
        return node.name.upper()
    if isinstance(node, exp.Func):
        return node.sql_name()
    return None


def func_args(node: exp.Func) -> list[exp.Expr]:
    """Positional arguments of a call, in sqlglot's node arg order."""
    if isinstance(node, exp.Anonymous):
        return list(node.args.get("expressions") or [])
    out: list[exp.Expr] = []
    for name in node.arg_types:
        val = node.args.get(name)
        if val is None or isinstance(val, (str, bool)):
            continue
        out.extend(val if isinstance(val, list) else [val])
    return out


def in_family(dtype: exp.DataType | None, family: str) -> bool | None:
    """True / False / None. None means unknown, so no judgement is made."""
    if family == "ANY":
        return True
    if dtype is None or dtype.is_type(exp.DType.UNKNOWN):
        return None
    return dtype.this in FAMILIES[family]


# ------------------------------------------------- forward: catalog -> types
def _resolve_return(sig: Sig, args: list[exp.Expr]) -> exp.DataType:
    if sig.returns == "@element":
        arr = args[0].type if args else None
        if arr is not None and arr.is_type(exp.DType.ARRAY) and arr.expressions:
            return arr.expressions[0]
        return exp.DType.UNKNOWN.into_expr()
    if sig.returns.startswith("@arg"):
        i = int(sig.returns[4:])
        return args[i].type if i < len(args) else exp.DType.UNKNOWN.into_expr()
    return exp.DataType.build(sig.returns, dialect=DIALECT)


def pick_overload(node: exp.Func) -> tuple[Sig | None, list[exp.Expr]]:
    """First overload no argument definitively contradicts."""
    args = func_args(node)
    for sig in CATALOG.get(catalog_key(node) or "", []):
        if len(sig.params) != len(args):
            continue
        if any(in_family(a.type, f) is False for a, f in zip(args, sig.params)):
            continue
        return sig, args
    return None, args


def _annotate(annotator, node: exp.Func) -> None:
    sig, args = pick_overload(node)
    annotator._set_type(node, _resolve_return(sig, args) if sig else exp.DType.UNKNOWN)


def install_catalog(dialect_name: str = DIALECT) -> None:
    """Register one annotator per catalog entry, on this dialect only."""
    d = Dialect.get_or_raise(dialect_name)
    for key in CATALOG:
        cls = exp.FUNCTION_BY_NAME.get(key)
        if cls is not None:
            d.EXPRESSION_METADATA[cls] = {"annotator": _annotate}
    # everything with no sqlglot class arrives as Anonymous; route it by written name
    d.EXPRESSION_METADATA[exp.Anonymous] = {"annotator": _annotate}


# ------------------------------------------- backward: catalog -> source types
def source_columns(tree: exp.Expr) -> dict[tuple[str, str], exp.Column]:
    """Every qualified column that reads a real table, keyed by (table, column)."""
    found: dict[tuple[str, str], exp.Column] = {}
    for scope in traverse_scope(tree):
        for col in scope.columns:
            source = scope.sources.get(col.table)
            if isinstance(source, exp.Table):
                found.setdefault((source.name, col.name), col)
    return found


def backward_evidence(tree: exp.Expr) -> dict[tuple[str, str], str]:
    """Read the catalog in reverse: an argument position constrains what feeds it."""
    real = source_columns(tree)
    by_node = {id(c): k for k, c in real.items()}
    evidence: dict[tuple[str, str], str] = {}
    for node in tree.walk():
        if not isinstance(node, exp.Func):
            continue
        for sig in CATALOG.get(catalog_key(node) or "", []):
            args = func_args(node)
            if len(sig.params) != len(args):
                continue
            for arg, family in zip(args, sig.params):
                key = by_node.get(id(arg))
                if key and family != "ANY" and in_family(arg.type, family) is None:
                    evidence.setdefault(key, family)
            break
    return evidence


def widen_schema(schema: dict, evidence: dict[tuple[str, str], str]) -> dict:
    out = {t: dict(cols) for t, cols in schema.items()}
    for (table, column), family in evidence.items():
        if out.get(table, {}).get(column, "UNKNOWN").upper() == "UNKNOWN":
            out.setdefault(table, {})[column] = FAMILY_DEFAULT[family]
    return out


# ---------------------------------------------------------------- diagnostics
@dataclass
class Finding:
    code: str
    message: str
    start: int | None
    end: int | None


def name_span(node: exp.Expr) -> tuple[int | None, int | None]:
    """Just the token sqlglot positioned on this node - for a call, its name."""
    return node.meta.get("start"), node.meta.get("end")


def span(node: exp.Expr) -> tuple[int | None, int | None]:
    """Hull of every positioned leaf beneath a node. Same idea as sqlr's Positions."""
    starts = [n.meta["start"] for n in node.walk() if n.meta.get("start") is not None]
    ends = [n.meta["end"] for n in node.walk() if n.meta.get("end") is not None]
    return (min(starts) if starts else None, max(ends) if ends else None)


def check(tree: exp.Expr) -> list[Finding]:
    out: list[Finding] = []
    for node in tree.walk():
        if not isinstance(node, exp.Func):
            continue
        key = catalog_key(node)
        sigs = CATALOG.get(key or "")
        if not sigs:
            if isinstance(node, exp.Anonymous):
                s, e = name_span(node)
                out.append(Finding("SQLR-F001", f"unknown function {node.name!r} in dialect {DIALECT}", s, e))
            continue

        args = func_args(node)
        if all(len(sig.params) != len(args) for sig in sigs):
            arities = sorted({len(s.params) for s in sigs})
            s, e = name_span(node)
            out.append(Finding("SQLR-F002",
                               f"{key} takes {arities} arguments, {len(args)} given", s, e))
            continue

        sig, _ = pick_overload(node)
        if sig is None:
            for cand in sigs:
                if len(cand.params) != len(args):
                    continue
                for i, (arg, family) in enumerate(zip(args, cand.params)):
                    if in_family(arg.type, family) is False:
                        s, e = span(arg)
                        out.append(Finding(
                            "SQLR-F003",
                            f"{key} argument {i + 1} expects {family}, got {arg.type.sql(DIALECT)}",
                            s, e))
                break
    return out
