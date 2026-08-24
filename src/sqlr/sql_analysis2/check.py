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

from sqlr.declared.types import DeclaredColumn, DeclaredRelation
from sqlr.sql_analysis2.annotate import (
    arguments_of_call,
    candidate_overloads,
    catalog_key,
    signatures_for_dialect,
    signatures_of_call,
    unknown_function_findings_are_trustworthy,
)
from sqlr.sql_analysis2.catalog import DialectName, Sig
from sqlr.sql_analysis2.families import ANY, family_of_type, in_family, nearest_common_family
from sqlr.sql_analysis2.facts import Facts, TypeClaim
from sqlr.sql_analysis2.infer import (
    ColumnTypeConflict,
    Inference,
    TypeEvidence,
    TypeStrength,
)
from sqlr.sql_analysis2.reporting import TypeFinding, FindingSeverity
from sqlr.sql_analysis2.resolve import declared_data_type, is_declared
from sqlr.sql_analysis2.sourcedoc import Positions, SourceSpan
from sqlr.sql_analysis2.types import (
    ColumnName,
    ColumnTypeAnnotation,
    ColumnTypeName,
    RelationKey,
)


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

    Gated on the dialect's catalog being complete, for the same reason one step further out:
    "this function does not exist" is only a truthful statement from an exhaustive list, and
    every catalog sqlr ships is a hand-written gap-filler. See `CATALOG_IS_COMPLETE`.
    """
    if not unknown_function_findings_are_trustworthy(dialect_name):
        return []
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
    declared_types_per_relation: dict[RelationKey, dict[ColumnName, ColumnTypeName]],
) -> ColumnTypeName | None:
    """What the user declared for the column this claim landed on, if anything."""
    key = claim.site.source_table_column
    if key is None:
        return None
    table, column = key
    written = declared_types_per_relation.get(table, {}).get(column)
    return written if is_declared(written) else None


def _contradiction_message(
    claim: TypeClaim, actual: exp.DataType, declared: ColumnTypeName | None
) -> str:
    """`ROUND argument 1 expects NUMERIC, but 'status' is declared varchar (TEXT)`.

    Three shapes for one finding, because the fix is in a different place each time:

    - **declared** - the user wrote the type in a yml, and that is where the fix goes. Naming
      the declaration is the difference between a message about the SQL and a message about
      the disagreement.
    - **a named value with a computed type** - `'amount' is TEXT here`. Nothing to edit in a
      yml; the expression that produced it is upstream in this file.
    - **anything else** - a literal, an expression with no name. The type is all there is to
      say.
    """
    wants = f"{claim.because.describe()} expects {claim.describe_families()}"
    if declared is not None:
        return (
            f"{wants}, but {claim.site.describe()} is declared "
            f"{declared} ({actual.sql()})"
        )
    if claim.site.column is not None:
        return f"{wants}, but {claim.site.describe()} is {actual.sql()} here"
    return f"{wants}, got {actual.sql()}"


def findings_for_contradicted_claims(
    facts: Facts,
    declared_types_per_relation: dict[RelationKey, dict[ColumnName, ColumnTypeName]],
    inference: Inference,
) -> list[TypeFinding]:
    """Every claim the value it is about definitely does not satisfy.

    `is False`, never `None`. A claim about a value of unknown type produces nothing at all
    - that is what stops one uncatalogued function from blaming every column above it, and
    every false positive found while building this traced back to the distinction.

    **Only against a `stated` value.** An inferred type was derived from this very claim set,
    so if the claims disagreed the column is unresolved and there is nothing to contradict;
    and a value in a component that *did* conflict is already reported once per site by
    `findings_for_columns_with_conflicting_facts`. Either way a second finding on the same
    span would be noise rather than information.
    """
    findings: list[TypeFinding] = []
    for claim in facts.type_claims:
        actual = claim.site.node.type
        strength: TypeStrength | None = inference.strength_of_node.get(id(claim.site.node))
        if strength is not None and strength != "stated":
            continue
        if actual is None or not claim_is_contradicted(claim, actual):
            continue
        findings.append(
            TypeFinding(
                code="contradicted-type",
                message=_contradiction_message(
                    claim, actual, _declared_type_of_claim(claim, declared_types_per_relation)
                ),
                span=claim.site.span,
                context_span=claim.site.span,
                function=claim.because.call,
                argument_index=claim.because.argument_index,
                column_name=claim.site.column.name if claim.site.column else None,
            )
        )
    return findings


# ------------------------------------- the file's own output against its own declaration
def _declared_columns_by_name(
    declaration: DeclaredRelation,
) -> dict[ColumnName, DeclaredColumn]:
    return {column.name.lower(): column for column in declaration.columns}


def _where_declared(declaration: DeclaredRelation) -> str:
    """`'mydatabase.ods.employee' at project/sources.yml:108` - the entry to go and edit."""
    return f"'{declaration.relation_name}' at {declaration.where}"


def _where_declared_column(
    declaration: DeclaredRelation, column: DeclaredColumn
) -> str:
    """The same, pointing at one column's own line instead of at the table entry.

    A finding about a declared column sends the reader to the line they have to change,
    which for a table with twenty columns is not the line the entry starts on. Falls back to
    the entry when the column carries no position.
    """
    located = column.name_span or column.span
    if located is None:
        return _where_declared(declaration)
    base = declaration.label or str(declaration.path)
    return f"'{declaration.relation_name}' at {base}:{located.start_line + 1}"


def _finding_for_a_projected_column_no_declaration_covers(
    projected: ColumnTypeAnnotation, declaration: DeclaredRelation
) -> TypeFinding:
    """A column this file produces that its declaration leaves out.

    An error for the same reason reading an undeclared column off a closed table is one:
    a declaration without `declaration_is_partial` is the *complete* column list, so a file
    producing a column it omits contradicts it. Letting this pass would make one wrong yml
    fatal in every file that reads the relation and silent in the file that builds it.
    """
    return TypeFinding(
        code="projected-column-not-declared",
        message=(
            f"column '{projected.name}' is projected by this file, but "
            f"{_where_declared(declaration)} does not declare it; declare it there, or "
            f"set 'declaration_is_partial' to 'true'"
        ),
        span=projected.span,
        context_span=projected.span,
        column_name=projected.name,
    )


def _finding_for_a_declared_column_the_file_does_not_produce(
    declared: DeclaredColumn, declaration: DeclaredRelation, projection_is_complete: bool
) -> TypeFinding:
    """A column the declaration promises that the projection list does not carry.

    Only an error when the projection list is the whole answer. A list built by expanding a
    `*` over a table nobody declares is a lower bound - the columns this file happens to
    name - so the column may well be produced and simply be invisible from here, and
    reporting that as a defect would blame the yml for a gap in a different yml.
    """
    if projection_is_complete:
        return TypeFinding(
            code="declared-column-not-projected",
            message=(
                f"column '{declared.name}' is declared for "
                f"{_where_declared_column(declaration, declared)}, "
                f"but this file does not project it"
            ),
            column_name=declared.name,
        )
    return TypeFinding(
        code="declared-column-not-projected",
        severity="warning",
        message=(
            f"column '{declared.name}' is declared for "
            f"{_where_declared_column(declaration, declared)}, and "
            f"this file's projection list is a lower bound - it expands a '*' over a table "
            f"nobody declares - so nothing here can say whether it is produced; declare "
            f"that table's columns, or project the column explicitly"
        ),
        column_name=declared.name,
    )


def _finding_for_a_projected_type_that_contradicts_the_declaration(
    projected: ColumnTypeAnnotation,
    declared: DeclaredColumn,
    declaration: DeclaredRelation,
    dialect_name: DialectName,
) -> TypeFinding | None:
    """The declared type of an output column against the type this file gives it.

    Two stated types conflict **iff their nearest common family is `ANY`** - the same test
    every other type comparison in the pipeline makes, so `decimal(10,2)` declared against a
    computed `DECIMAL(38,9)` is not a finding and a `varchar` against a `DECIMAL` is.

    Severity splits on where the projected type came from, exactly as a conflicting-usage
    finding does. **Computed** or **declared** means two written-down things disagree and one
    is wrong: an error. **Inferred** means sqlr read a family off how an undeclared column is
    used and that guess fought the declaration: a warning, whose fix is usually to declare
    the column being read.

    Nothing is reported when either side is unknown, and nothing when the declared type is
    one the dialect cannot parse - that is `unrecognized-declared-type`'s finding to make.
    """
    if projected.provenance == "unknown":
        return None
    declared_type = declared_data_type(declared.written_type, dialect_name)
    declared_family = family_of_type(declared_type)
    projected_family = family_of_type(projected.type)
    if declared_family is None or projected_family is None:
        return None
    if nearest_common_family(declared_family, projected_family) != ANY:
        return None

    inferred = projected.provenance == "inferred"
    severity: FindingSeverity = "warning" if inferred else "error"
    return TypeFinding(
        code="contradicted-declaration",
        severity=severity,
        message=(
            f"column '{declared.name}' of "
            f"{_where_declared_column(declaration, declared)} is declared "
            f"{declared.written_type} ({declared_type.sql() if declared_type else '?'}), "
            f"but this file "
            + (
                f"infers {projected.type_name} for it from usage; declare the column it "
                f"reads, or fix the declaration"
                if inferred
                else f"projects {projected.type_name}"
            )
        ),
        span=projected.span,
        context_span=projected.span,
        column_name=declared.name,
    )


def findings_for_a_projection_that_disagrees_with_its_declaration(
    projected: list[ColumnTypeAnnotation],
    declaration: DeclaredRelation | None,
    dialect_name: DialectName,
    projection_is_complete: bool = True,
) -> list[TypeFinding]:
    """What this file produces, held against what its yml says it produces.

    The one check that reads a declaration of the *model* rather than of a table it reads,
    and the only place that declaration has any effect on this file at all: it never types
    anything here, because a declaration of a model's output cannot describe the columns
    feeding it. Every type on this side was inferred or computed from the SQL, and the
    comparison is between those and the yml.

    A declaration with no columns is read as no declaration, exactly as it is for a source
    table: naming a relation without listing its columns says nothing about them.

    The projected side is walked in projection order and the declared side after it, so a
    reader sees the file's own columns in the order they wrote them before the ones only
    the yml knows about.
    """
    if declaration is None or not declaration.columns:
        return []

    declared_columns = _declared_columns_by_name(declaration)
    findings: list[TypeFinding] = []
    for column in projected:
        declared = declared_columns.get(column.name.lower())
        if declared is None:
            if not declaration.declaration_is_partial:
                findings.append(
                    _finding_for_a_projected_column_no_declaration_covers(
                        column, declaration
                    )
                )
            continue
        finding = _finding_for_a_projected_type_that_contradicts_the_declaration(
            column, declared, declaration, dialect_name
        )
        if finding is not None:
            findings.append(finding)

    produced = {column.name.lower() for column in projected}
    findings.extend(
        _finding_for_a_declared_column_the_file_does_not_produce(
            declared, declaration, projection_is_complete
        )
        for declared in declaration.columns
        if declared.name.lower() not in produced
    )
    return findings


def _conflict_message(conflict: ColumnTypeConflict, here: TypeEvidence) -> str:
    """One site's half of a disagreement, with the other sites named.

    Every site gets its own finding because every site is a place the user has to look: the
    fix is either a declaration or one of these usages, and which one is theirs to say.

    The two strengths need two sentences, not one with a severity attached. A stated
    conflict is an accusation about the SQL - two types are written down and one is wrong. An
    inferred conflict is an admission about sqlr - it guessed twice and the guesses fought,
    and declaring the column ends the argument.
    """
    mine = here.describe_family()
    elsewhere = [
        f"{item.describe_family()} at {item.span}"
        if item.span is not None
        else item.describe_family()
        for item in conflict.evidence
        if item.describe_family() != mine
    ]
    others = " and ".join(dict.fromkeys(elsewhere))

    if conflict.strength == "stated":
        return (
            f"{conflict.describe_subject()} is used as {mine} here ({here.detail}), "
            f"but as {others}; these cannot both be true and sqlr does not assume the "
            f"engine will cast between them"
        )
    return (
        f"{conflict.describe_subject()} has no declared type and its usage disagrees: "
        f"{mine} here ({here.detail}), {others}; declare its type or fix the usage"
    )


def findings_for_columns_with_conflicting_facts(
    conflicts: list[ColumnTypeConflict],
) -> list[TypeFinding]:
    """A value used as two different things, reported once per site.

    Severity is the conflict's strength, and the difference is real. Two *stated* types -
    a declaration, a computed expression, a literal - means two things were written down and
    one of them is wrong, which is an **error**. Two *inferred* families means sqlr derived
    both from usage and they fought, which is a **warning** about sqlr's confidence rather
    than an accusation about the SQL.

    Either way the column stays UNKNOWN. UNKNOWN is absorbing in sqlglot, so everything
    downstream goes quiet instead of inheriting a coin-flip.
    """
    return [
        TypeFinding(
            code="conflicting-usage",
            message=_conflict_message(conflict, item),
            severity="error" if conflict.strength == "stated" else "warning",
            span=item.span,
            context_span=item.span,
            column_name=conflict.column,
        )
        for conflict in conflicts
        for item in conflict.evidence
    ]
