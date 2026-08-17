"""Step 7 - the only step with an opinion. Everything before it infers.

Three things go wrong, and they are not the same kind of thing:

- **F001 unknown function** and **F002 arity** are structural. They need no types at all,
  they are about the *call*, and they are found by walking the tree.
- **a contradicted claim** is about the *value* flowing into something. It is found by
  reading the facts, because a fact is already exactly "this value must be family F, and
  here is where the SQL says so".

The old F003 was the second of those written as a tree walk, which is why it has no code of
its own here: `findings_for_contradicted_claims` covers `round(status, 2)`,
`where amount`, `'abc' + 5` and everything else in one loop.

Messages are built here rather than in `reporting.py` because a claim's wording is inseparable
from the fact that produced it - the finding is not a rendering of a structure, it *is* the
structure read aloud.

The checker never looks at SQL text, only at `node.type`. Everything upstream - declared
yml, inference, sqlglot's own tables, cross-CTE propagation - has already been flattened
into one uniform annotated tree.
"""

from __future__ import annotations

from sqlglot import exp

from sqlr.sql_analysis2.annotate import (
    arguments_of_call,
    candidate_overloads,
    catalog_key,
    in_family,
    signatures_for_dialect,
    signatures_of_call,
)
from sqlr.sql_analysis2.catalog import DialectName, Sig
from sqlr.sql_analysis2.facts import Facts, TypeClaim
from sqlr.sql_analysis2.infer import ColumnTypeConflict, TypeEvidence
from sqlr.sql_analysis2.reporting import TypeFinding
from sqlr.sql_analysis2.resolve import is_declared
from sqlr.sql_analysis2.sourcedoc import Positions, SourceSpan
from sqlr.sql_analysis2.types import ColumnName, ColumnTypeName, TableName


def span_of_call_name(node: exp.Expr, positions: Positions) -> SourceSpan | None:
    """Where the function's *name* is written.

    A call's hull covers its arguments and stops one character short of the closing paren,
    because punctuation carries no position - underlining `zeroifnull(amount` for an unknown
    function reads as a bug in the tool. sqlglot puts the name token's offsets on the call
    node itself, so when they are there they are exactly right.
    """
    start = node.meta.get("start")
    end = node.meta.get("end")
    if isinstance(start, int) and isinstance(end, int):
        return positions.span(start, end + 1)  # sqlglot's `end` is inclusive
    return positions.span_of(node)


def _describe_arities(signatures: list[Sig]) -> str:
    """`2`, `1 or 2`, `at least 1` - the arities a call could have had."""
    if any(sig.variadic for sig in signatures):
        return f"at least {min(len(sig.params) for sig in signatures)}"
    arities = sorted({len(sig.params) for sig in signatures})
    if len(arities) == 1:
        return str(arities[0])
    return f"{', '.join(str(n) for n in arities[:-1])} or {arities[-1]}"


def findings_for_unknown_functions(
    tree: exp.Expr, positions: Positions, dialect_name: DialectName
) -> list[TypeFinding]:
    """A call the dialect does not have.

    Only `Anonymous` calls qualify. A function sqlglot parsed into a class of its own exists
    *somewhere*, so its absence from the catalog says our catalog is thin, not that the SQL
    is wrong - reporting those would make every uncatalogued function a user-facing error.
    """
    signatures = signatures_for_dialect(dialect_name)
    return [
        TypeFinding(
            code="unknown-function",
            message=f"unknown function '{node.name}' in dialect {dialect_name}",
            span=span_of_call_name(node, positions),
            context_span=positions.span_of(node),
            function=node.name.upper(),
        )
        for node in tree.find_all(exp.Anonymous)
        if not signatures_of_call(node, signatures)
    ]


def findings_for_calls_with_wrong_arity(
    tree: exp.Expr, positions: Positions, dialect_name: DialectName
) -> list[TypeFinding]:
    """A catalogued call given the wrong number of arguments.

    Checked before any type comparison and reported instead of one, never beside it: a
    one-argument call must not also be told its argument contradicts a two-argument
    signature it was never going to match.
    """
    signatures = signatures_for_dialect(dialect_name)
    findings: list[TypeFinding] = []
    for node in tree.walk():
        overloads = signatures_of_call(node, signatures)
        if not overloads or candidate_overloads(node, overloads):
            continue
        key = catalog_key(node)
        given = len(arguments_of_call(node))
        findings.append(
            TypeFinding(
                code="function-arity",
                message=(
                    f"{key} takes {_describe_arities(overloads)} arguments, {given} given"
                ),
                span=span_of_call_name(node, positions),
                context_span=positions.span_of(node),
                function=key,
            )
        )
    return findings


def claim_is_contradicted(claim: TypeClaim, actual: exp.DataType) -> bool:
    """Whether a value definitely satisfies none of the families claimed of it.

    Every family has to answer `False`. One `None` anywhere - an unknown type, an
    uncatalogued family - and the claim stays silent, which is the whole three-valued
    discipline in one line.
    """
    return all(in_family(actual, family) is False for family in claim.families)


def _declared_type_of_claim(
    claim: TypeClaim,
    declared_types_per_table: dict[TableName, dict[ColumnName, ColumnTypeName]],
) -> ColumnTypeName | None:
    """What the user declared for the column this claim landed on, if anything."""
    key = claim.site.source_table_column
    if key is None:
        return None
    table, column = key
    written = declared_types_per_table.get(table, {}).get(column)
    return written if is_declared(written) else None


def _contradiction_message(
    claim: TypeClaim, actual: exp.DataType, declared: ColumnTypeName | None
) -> str:
    """`ROUND argument 1 expects NUMERIC, but 'status' is declared varchar (TEXT)`.

    Naming the declaration when there is one is the difference between a message about the
    SQL and a message about the disagreement: the type did not appear from nowhere, the
    user wrote it, and that is where the fix goes.
    """
    wants = f"{claim.because.describe()} expects {claim.describe_families()}"
    if declared is not None:
        return (
            f"{wants}, but {claim.site.describe()} is declared "
            f"{declared} ({actual.sql()})"
        )
    return f"{wants}, got {actual.sql()}"


def findings_for_contradicted_claims(
    facts: Facts,
    declared_types_per_table: dict[TableName, dict[ColumnName, ColumnTypeName]],
) -> list[TypeFinding]:
    """Every claim the value it is about definitely does not satisfy.

    `is False`, never `None`. A claim about a value of unknown type produces nothing at all
    - that is what stops one uncatalogued function from blaming every column above it, and
    every false positive found while building this traced back to the distinction.
    """
    findings: list[TypeFinding] = []
    for claim in facts.type_claims:
        actual = claim.site.node.type
        if actual is None or not claim_is_contradicted(claim, actual):
            continue
        findings.append(
            TypeFinding(
                code="contradicted-type",
                message=_contradiction_message(
                    claim, actual, _declared_type_of_claim(claim, declared_types_per_table)
                ),
                span=claim.site.span,
                context_span=claim.site.span,
                function=claim.because.call,
                argument_index=claim.because.argument_index,
                column_name=claim.site.column.name if claim.site.column else None,
            )
        )
    return findings


def _conflict_message(conflict: ColumnTypeConflict, here: TypeEvidence) -> str:
    """One site's half of a disagreement, with the other sites named.

    Every site gets its own finding because every site is a place the user has to look:
    the fix is either a declaration or one of these usages, and which one is theirs to say.
    """
    elsewhere = [
        f"{item.family} at {item.span}" if item.span is not None else item.family
        for item in conflict.evidence
        if item.family != here.family
    ]
    return (
        f"column '{conflict.column}' of '{conflict.table}' has no declared type and its "
        f"usage disagrees: {here.family} here ({here.detail}), "
        f"{' and '.join(dict.fromkeys(elsewhere))}; declare its type or fix the usage"
    )


def findings_for_columns_with_conflicting_facts(
    conflicts: list[ColumnTypeConflict],
) -> list[TypeFinding]:
    """An undeclared column used as two different things.

    An error, not a warning. Engine autocasting is not something sqlr relies on, so
    `upper(x)` beside `x > 5` is a defect rather than a dialect feature - and the column
    stays UNKNOWN either way, so nothing downstream inherits a guess.
    """
    return [
        TypeFinding(
            code="conflicting-usage",
            message=_conflict_message(conflict, item),
            span=item.span,
            context_span=item.span,
            column_name=conflict.column,
        )
        for conflict in conflicts
        for item in conflict.evidence
    ]
