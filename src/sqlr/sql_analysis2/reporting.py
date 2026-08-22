"""Turning what the pipeline found into something with a place in the source.

A finding carries spans rather than a formatted line, so the same value serves the CLI and
an editor that wants to underline the offending text.
"""

from typing import Literal

from pydantic import BaseModel
from sqlglot import exp

from sqlr.sql_analysis2.sourcedoc import Positions, SourceSpan
from sqlr.sql_analysis2.types import (
    AmbiguousColumn,
    ColumnName,
    ColumnTypeName,
    DuplicateProjection,
    GuessedColumn,
    ParsedColumn,
    ProjectionSite,
    RelationKey,
    ScopeKind,
    UnresolvableColumn,
    UnresolvableRelation,
)
from sqlr.typemap import resolve_type_name

type FindingCode = Literal[
    "unresolvable-column",
    "dot-access-unsupported",
    "structured-column-declared-scalar",
    "ambiguous-column",
    "column-without-source",
    "duplicate-projected-column",
]

type FindingSeverity = Literal["error", "warning"]
"""How much a finding costs. An `error` means the statement cannot be checked further; a
`warning` means it can, but the reader should know what was assumed to get there."""


class ColumnFinding(BaseModel):
    """One reportable problem, and where in the SQL it was written.

    Two spans, because an editor wants both: `span` covers the offending name, which is
    what to underline, and `access_span` covers the whole read it appeared in.
    """

    code: FindingCode
    column_name: ColumnName
    message: str
    severity: FindingSeverity = "error"
    span: SourceSpan | None = None
    access_span: SourceSpan | None = None

    def where(self) -> str:
        """`:10:3`, or nothing when the node carries no position."""
        return f":{self.span}" if self.span is not None else ""


type TypeFindingCode = Literal[
    "unknown-function",
    "function-arity",
    "contradicted-type",
    "conflicting-usage",
]
"""What went wrong with a *value*, as opposed to with a column's attribution.

`unknown-function` and `function-arity` are structural - they need no types at all and are
about the call. The other two are the two readings of one fact: a claim contradicted by a
type that is already known, and claims that disagree about a column with no type at all.
"""


class TypeFinding(BaseModel):
    """One reportable problem with a value, and where in the SQL it was written.

    Not `FunctionFinding`: most of these are about the value flowing into a call rather
    than about the call itself, and some involve no call at all. `function` and
    `argument_index` are filled when a call is what made the claim.
    """

    code: TypeFindingCode
    message: str
    severity: FindingSeverity = "error"
    span: SourceSpan | None = None
    context_span: SourceSpan | None = None
    function: str | None = None
    argument_index: int | None = None
    column_name: ColumnName | None = None

    def where(self) -> str:
        """`:10:3`, or nothing when the node carries no position."""
        return f":{self.span}" if self.span is not None else ""


def span_of_access(column: exp.Column, positions: Positions) -> SourceSpan | None:
    """The span of the read a column sits inside - `x.field`, `x['field']`.

    sqlglot puts no position on a closing bracket, so the hull stops one character short
    of it; that character is added back here.
    """
    parent = column.parent
    if not isinstance(parent, (exp.Dot, exp.Bracket)):
        return None
    span = positions.span_of(parent)
    if span is None:
        return None
    if isinstance(parent, exp.Bracket) and positions.text[span.end : span.end + 1] == "]":
        return positions.span(span.start, span.end + 1)
    return span


def written_text_of(node: exp.Expr, positions: Positions) -> str:
    """The name as the user typed it, falling back to the name sqlglot holds.

    Worth the lookup because the probe normalises identifiers to the dialect's case, and
    Snowflake's is upper: a message built from the tree calls the user's `updated_at`
    `UPDATED_AT`, and then lists the declared columns beside it in the lower case the yml
    used. One sentence, two spellings of the same convention, and neither is what the reader
    can search their file for.
    """
    # A `Column`'s hull covers its qualifier too, so spanning the node would quote
    # `src.nonsense` where the sentence is about `nonsense`. The identifier under it is the
    # name on its own.
    named = node.this if isinstance(node, exp.Column) and node.this is not None else node
    span = positions.span_of(named)
    if span is None:
        return node.name
    return positions.text[span.start : span.end] or node.name


def _describe_relation(relation: UnresolvableRelation) -> str:
    """`table 'mydatabase.myschema.raw_address'`, `CTE 'ranked'` - what a reader calls it."""
    kinds = {"table": "table", "cte": "CTE", "derived": "derived table"}
    return f"{kinds.get(relation.kind, 'relation')} '{relation.display_name}'"


def _declare_it_clause(relations: list[UnresolvableRelation]) -> str:
    """` Declare it on 'x' at schema.yml:12, or ...` - the fix, when a yml holds one.

    Names the declared relation and not only the file, because the two are often different
    things: a column failing against a CTE has to be declared on whatever that CTE reads
    through its `*`, and a reader sent to the file alone still has to guess which entry.

    Empty for a relation closed by its own projection list. A CTE that writes out its
    columns has no declaration behind it, and telling that reader to set
    `declaration_is_partial` sends them to a file with nothing in it to change.
    """
    entries = " or ".join(
        dict.fromkeys(
            f"'{entry.relation_name}' at {entry.where}"
            for relation in relations
            for entry in relation.closing_declarations
        )
    )
    if not entries:
        return ""
    return f" Declare it on {entries}, or set 'declaration_is_partial' to 'true' there."


def _unresolvable_message(
    unresolvable: UnresolvableColumn, positions: Positions
) -> str:
    """Why one column belongs to nothing, in terms the reader can act on.

    Three reasons, three sentences, because they have three different fixes:

    - `no_such_source` is a typo in the qualifier, and no yml would change that.
    - `not_projected` is a name every relation in scope rules out: a mistyped column, or a
      declaration that is complete when it should be partial. When a declaration is what
      closed the relation, the sentence names the file and line to edit - the reader cannot
      find it otherwise, because the yml that made the read an error is not the yml the
      relation was written in.
    - `several_undeclared_sources` is not a mistake in the SQL at all. Nothing can place the
      name because two or more relations leave their columns undeclared.
    """
    name = written_text_of(unresolvable.column, positions)
    relations = unresolvable.relations

    if unresolvable.reason == "several_undeclared_sources":
        listed = _as_a_list_of_names([r.display_name for r in relations])
        return (
            f"column '{name}' has no source alias and could come from {listed}. Qualify "
            "it, or declare the columns of the table that owns it."
        )

    if unresolvable.reason == "no_such_source" or not relations:
        qualifier = unresolvable.column.table
        if qualifier:
            return (
                f"column '{name}' is qualified with '{qualifier.lower()}', which matches "
                "no relation in this statement"
            )
        return f"column '{name}' could not be resolved to any source"

    # `not_projected`. One relation is the ordinary case - a qualified column, or a bare one
    # over a single source; several means a bare column over a join where every relation
    # ruled it out, and each of them is a place the reader might have meant it to come from.
    if len(relations) == 1:
        return (
            f"found undeclared column '{name}' in {_describe_relation(relations[0])}."
            f"{_declare_it_clause(relations)}"
        )

    # No column lists here, unlike the one-relation case. A bare name over a join can be
    # ruled out by several relations at once, and printing what each of them declares means
    # printing most of the schema to say one thing.
    return (
        f"column '{name}' is projected by none of the relations in scope. Declare it on a "
        "source, or set 'declaration_is_partial' to 'true' for one of the relations."
    )


def findings_for_unresolvable_columns(
    columns: list[UnresolvableColumn], positions: Positions
) -> list[ColumnFinding]:
    """A column no relation can own - a typo, a missing join, a complete declaration that
    omits a column the SQL reads, or a name no undeclared table can be credited with."""
    return [
        ColumnFinding(
            code="unresolvable-column",
            column_name=unresolvable.column.name,
            message=_unresolvable_message(unresolvable, positions),
            span=positions.span_of(unresolvable.column),
        )
        for unresolvable in columns
    ]


def _as_a_list_of_names(names: list[str]) -> str:
    """`'a'`, `'a' and 'b'`, `'a', 'b' and 'c'` - a list a person can read aloud."""
    quoted = [f"'{name}'" for name in names]
    if len(quoted) <= 1:
        return "".join(quoted)
    return f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def findings_for_ambiguous_columns(
    ambiguous: list[AmbiguousColumn], positions: Positions
) -> list[ColumnFinding]:
    """A column two relations could each own.

    There is no tie-break to apply and no default worth printing, so the message offers the
    candidates and asks for the fix rather than announcing a pick - and which fix that is
    depends on why the tie happened. A name two relations both project needs a qualifier. A
    name arriving through a star over a join of undeclared tables cannot be qualified into
    existence: one of those tables has to declare its columns.
    """
    findings: list[ColumnFinding] = []
    for column, candidate_sources, kind in ambiguous:
        if kind == "star_over_join":
            message = (
                f"column '{column.name}' arrives through a '*' over "
                f"{_as_a_list_of_names(candidate_sources)}, "
                f"{'neither' if len(candidate_sources) == 2 else 'none'} of which "
                "declares its columns, so nothing says which one owns it; declare one of "
                "them, or set star_over_join_behavior to 'guess'"
            )
        else:
            message = (
                f"column '{column.name}' is ambiguous: "
                f"{_as_a_list_of_names(candidate_sources)} "
                f"{'both' if len(candidate_sources) == 2 else 'all'} project it; "
                "qualify it with a source alias"
            )
        findings.append(
            ColumnFinding(
                code="ambiguous-column",
                column_name=column.name,
                message=message,
                span=positions.span_of(column),
            )
        )
    return findings


def findings_for_columns_without_a_source(
    guessed: list[GuessedColumn], positions: Positions
) -> list[ColumnFinding]:
    """A bare column attributed while some other source could still have owned it.

    The attribution stands - it is the best answer available - so the message names it.
    Saying which source was *not* ruled out is what makes the warning actionable: it is
    the difference between "add a qualifier" and "declare that table's columns".
    """
    return [
        ColumnFinding(
            code="column-without-source",
            column_name=column.name,
            severity="warning",
            message=(
                f"column '{column.name}' has no source alias and "
                f"{_as_a_list_of_names(open_sources)} "
                f"{'does' if len(open_sources) == 1 else 'do'} not declare "
                f"{'its' if len(open_sources) == 1 else 'their'} columns; "
                f"reading it from '{resolved_source}'"
            ),
            span=positions.span_of(column),
        )
        for column, resolved_source, open_sources in guessed
    ]


def findings_for_columns_read_with_unsupported_dot_notation(
    columns: list[exp.Column], positions: Positions, dialect_name: str
) -> list[ColumnFinding]:
    """One finding per read site.

    The message covers both readings of the same tree, because nothing distinguishes them
    once sqlglot's `_convert_columns_to_dots` has run: the name is either a mistyped
    source or a structured column read with syntax the dialect does not have.
    """
    return [
        ColumnFinding(
            code="dot-access-unsupported",
            column_name=column.name,
            message=(
                f"'{column.name}' matches no source, and {dialect_name} does not read a "
                "dotted name as a field of a structured column"
            ),
            span=positions.span_of(column),
            access_span=span_of_access(column, positions),
        )
        for column in columns
    ]


def _describe_scope(name: str, kind: ScopeKind) -> str:
    """`CTE 'src'`, `the final projection` - a scope named the way a reader would say it."""
    if kind == "final":
        return "the final projection"
    if kind == "branch":
        return "a set operation branch"
    return f"{'CTE' if kind == 'cte' else 'derived table'} '{name}'"


def _describe_projection_site(site: ProjectionSite) -> str:
    """How one projection came to carry the name, and where to look for it."""
    if site.from_star:
        return f"expanded from the '*' at {site.span}" if site.span else "expanded from a '*'"
    return f"written at {site.span}" if site.span else "written"


def findings_for_duplicate_projected_columns(
    duplicates: list[DuplicateProjection],
) -> list[ColumnFinding]:
    """A scope projecting one name twice - usually a star overlapping written columns.

    The span points at a written occurrence when there is one, because that is the half a
    user can delete; a pair of star-expanded duplicates has only the stars to point at.
    Either way the message lists every site, since fixing it means knowing which two
    projections collided.
    """
    findings: list[ColumnFinding] = []
    for scope_name, scope_kind, column_name, sites in duplicates:
        written = [site for site in sites if not site.from_star and site.span is not None]
        located = written or [site for site in sites if site.span is not None]
        findings.append(
            ColumnFinding(
                code="duplicate-projected-column",
                column_name=column_name,
                message=(
                    f"column '{column_name}' is projected {len(sites)} times by "
                    f"{_describe_scope(scope_name, scope_kind)} "
                    f"({', '.join(_describe_projection_site(site) for site in sites)}); "
                    "a relation cannot have two columns with the same name"
                ),
                span=located[0].span if located else None,
            )
        )
    return findings


def findings_for_columns_declared_as_scalar_but_read_as_structured(
    declared: dict[RelationKey, dict[ColumnName, ColumnTypeName]],
    columns_per_relation: dict[RelationKey, dict[ColumnName, ParsedColumn]],
    positions: Positions,
) -> list[ColumnFinding]:
    """A column read as `x.field` or `x['field']` cannot hold a scalar.

    Only a declaration that lands on the lattice is contradicted: every `ResolvedTypeName`
    is scalar, so `varchar` is a mistake, while `json`, `struct(...)`, `map` and `variant`
    all resolve to `unknown` and say nothing. An unrecognised name resolves to `unknown`
    too, and is somebody else's finding.
    """
    findings: list[ColumnFinding] = []
    for table_name, columns in columns_per_relation.items():
        for column in columns.values():
            if not column.requires_structured_type:
                continue
            written = declared.get(table_name, {}).get(column.name)
            if written is None or resolve_type_name(written) == "unknown":
                continue
            message = (
                f"column '{column.name}' of '{table_name}' is declared "
                f"'{written}', but is read as a structured value; declare it as "
                "a struct, json, map or variant type"
            )
            for _, node in column.structured_access:
                findings.append(
                    ColumnFinding(
                        code="structured-column-declared-scalar",
                        column_name=column.name,
                        message=message,
                        span=positions.span_of(node),
                        access_span=span_of_access(node, positions),
                    )
                )
    return findings


def findings_without_exact_duplicates(
    findings: list[ColumnFinding],
) -> list[ColumnFinding]:
    """The same finding reported twice is one finding.

    A column can be reached by more than one path through the resolution - the same read
    is unresolvable in a CTE and again in the query that selects from it - and each path
    builds its own finding. Nothing distinguishes them once built, so only the first
    occurrence is kept; two findings that differ in any field, position included, are two
    separate problems and both survive.
    """
    seen: set[tuple[object, ...]] = set()
    unique: list[ColumnFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.column_name,
            finding.message,
            finding.span,
            finding.access_span,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return unique


def print_findings(findings: list[ColumnFinding], relative_path: str) -> None:
    """The plain-text view. An editor consumes the findings themselves instead."""
    for finding in findings_without_exact_duplicates(findings):
        print(
            f"{finding.severity}: {relative_path}{finding.where()}: {finding.message}"
        )
