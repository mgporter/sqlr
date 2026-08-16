"""Steps 1-3: parse, gap-fill the declared schema, qualify.

The `qualify-schema` command is exactly this much of the pipeline, and `validate-schema`
is this plus annotation. The goal of the three steps together is one sentence: **every
column names the source it reads from, and every star has become a real projection list.**
Nothing downstream can be done before that is true.

One statement per file. A model is one projection - the same rule dbt applies - so a file
holding two statements has no single answer to "what does this model project", and is
reported rather than half-analysed.
"""

import logging
from pathlib import Path
from typing import Literal, NamedTuple, cast

import sqlglot
from sqlglot import ParseError, exp
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema

from sqlr.config.types import SqlrConfig
from sqlr.declared.types import DeclaredSchemas
from sqlr.selection.types import Model
from sqlr.sql_analysis2.reporting import (
    ColumnFinding,
    findings_for_ambiguous_columns,
    findings_for_columns_declared_as_scalar_but_read_as_structured,
    findings_for_columns_read_with_unsupported_dot_notation,
    findings_for_columns_without_a_source,
    findings_for_unresolvable_columns,
    findings_without_exact_duplicates,
)
from sqlr.sql_analysis2.resolve import (
    declared_columns,
    get_declared_types_per_table,
    is_declared,
    resolve_columns_to_source_tables,
)
from sqlr.sql_analysis2.sourcedoc import Positions, SourceDoc, token_offsets_of
from sqlr.sql_analysis2.types import (
    ColumnName,
    ColumnTypeName,
    ResolvedColumns,
    TableName,
)

logger = logging.getLogger(__name__)

DEFAULT_DIALECT = "duckdb"

type ScopeKind = Literal["cte", "derived", "final", "branch"]
"""What a scope is, for a reader looking at a printed table.

- `cte`     - a named `WITH` term.
- `derived` - a subquery in a FROM or JOIN, named by its alias.
- `final`   - the statement's own projection, the one the model is.
- `branch`  - one arm of a set operation, or a scope with no name of its own.
"""

type ColumnQualifierOrigin = Literal["written", "inferred", "star"]
"""Where a column's source attribution came from.

- `written`  - the user qualified it themselves.
- `inferred` - the user wrote a bare name and step 2 attributed it.
- `star`     - the column does not appear in the SQL at all; step 3 materialised it by
  expanding a star against the gap-filled schema.
"""


class ColumnReference(NamedTuple):
    """One `exp.Column` in a qualified tree, described for a reader rather than a pass."""

    name: ColumnName
    source_alias: TableName
    """The qualifier the column carries after step 3 - an alias, not necessarily a table."""
    source_name: TableName | None
    """The real table behind the alias, when the source is a table and is named
    differently. None when the alias *is* the name, or when the source is a CTE."""
    source_kind: ScopeKind | Literal["table", "unknown"]
    origin: ColumnQualifierOrigin
    declared: bool
    """Whether the user gave this column a type in a yml. False for anything a CTE
    produces: a CTE column's type is computed, and nobody declares it."""


class ScopeColumns(NamedTuple):
    """Every column reference in one scope, split by whether the scope projects it.

    The split is the useful one to read: the projected list is what this scope *hands
    onwards*, and the rest is what it consulted to build them - join keys, filters,
    grouping. A name can appear in both, and does not mean the same thing in each.
    """

    name: str
    kind: ScopeKind
    projected: list[ColumnReference]
    non_projected: list[ColumnReference]


class QualifiedStatement(NamedTuple):
    """The statement after step 3, with everything steps 4-7 need to work from."""

    qualified: exp.Expr
    """Mutated in place from the parsed statement: every column qualified, stars expanded."""
    scopes: list[Scope]
    """Built once here. The scope graph is the single most re-derived thing in the
    pipeline, so it is threaded through rather than rebuilt per step."""
    mapped_schema: Schema
    """The gap-filled schema as sqlglot wants it. Shared by `qualify` and `annotate_types`
    so neither has to build a `MappingSchema` from a bare dict again."""
    declared_types_per_table: dict[TableName, dict[ColumnName, ColumnTypeName]]
    """The same schema as plain data - what step 5 widens and what `needs_inference` reads."""
    resolved: ResolvedColumns
    dialect_name: str


class QualifiedModel(NamedTuple):
    """One model's trip through steps 1-3, whether or not it survived them.

    A model with no `statement` did not reach the end of step 3; `errors` and `findings`
    together say why. Both are always populated, because a file that fails step 3 has more
    to tell the reader than a file that fails to parse.
    """

    model: Model
    source: SourceDoc
    positions: Positions
    findings: list[ColumnFinding]
    """Column-level problems, each with a span into `source`."""
    errors: list[str]
    """File-level problems with nowhere to point: a parse failure, more than one statement,
    a `qualify` that raised."""
    statement: QualifiedStatement | None

    @property
    def has_errors(self) -> bool:
        """Whether this model failed. A warning-severity finding is not a failure."""
        return bool(self.errors) or any(
            finding.severity == "error" for finding in self.findings
        )


def any_model_has_errors(results: list[QualifiedModel]) -> bool:
    """Whether the run should exit non-zero."""
    return any(result.has_errors for result in results)


# ---------------------------------------------------------------------- steps 1-3
def qualify_schema(
    cfg: SqlrConfig, declared: DeclaredSchemas, models: list[Model]
) -> list[QualifiedModel]:
    """Resolve every column in every selected model to the source it reads from.

    One result per model, in selection order, including the models that failed - a caller
    reporting on a run needs to say what happened to each file, not only to the ones that
    worked.
    """
    dialect_name = cfg.general.sql_dialect or DEFAULT_DIALECT

    # Declarations are read once for the run: they do not vary per model.
    declared_schema = declared_columns(declared)

    return [
        qualify_one_model(
            model,
            declared_schema,
            dialect_name,
            cfg.general.warn_on_column_without_source,
        )
        for model in models
    ]


def qualify_one_model(
    model: Model,
    declared_schema: dict[TableName, dict[ColumnName, ColumnTypeName]],
    dialect_name: str,
    warn_on_column_without_source: bool = True,
) -> QualifiedModel:
    """Steps 1-3 for one file. Never raises: every failure comes back in the result."""
    sql = Path(model.path).read_text()
    source = SourceDoc(path=model.path, text=sql)

    # One index for the whole document. sqlglot's character offsets are absolute, so it
    # stays valid however the statement is rewritten beneath it.
    positions = Positions(sql)

    def failed(errors: list[str], findings: list[ColumnFinding] | None = None) -> QualifiedModel:
        return QualifiedModel(
            model=model,
            source=source,
            positions=positions,
            findings=findings or [],
            errors=errors,
            statement=None,
        )

    # Step 1: parse.
    try:
        statements = sqlglot.parse(sql, read=dialect_name)
    except ParseError as e:
        return failed([str(e)])

    parsed = [statement for statement in statements if statement is not None]
    if not parsed:
        return failed(["no statements found in the SQL file"])
    if len(parsed) > 1:
        # A model is one projection, so there is no answer to "what does this model
        # produce" here. Checking the last statement and calling it the model's schema was
        # the old behaviour and it silently described the wrong thing.
        return failed(
            [
                f"file contains {len(parsed)} statements; a model must be a single "
                "statement"
            ]
        )

    statement = parsed[0]
    logger.info("parsed %s", model.relative_path)

    # Step 2a: a probe qualification resolves every column to the source it reads from.
    # This is necessary to get the sources for columns without an alias but which MUST
    # come from a source because no other source has that column.
    # E.g.: with t2 as (select distinct id from mytable)
    #       select a from t1 join t2 on t1.id = t2.id;
    # Here, 'a' must come from t1, since t2 only has 'id'.
    try:
        resolved = resolve_columns_to_source_tables(
            statement, dialect_name, declared_schema=declared_schema
        )
    except OptimizeError as e:
        return failed([str(e)])

    # Everything wrong with the resolution is reported before step 3. `qualify` describes
    # the tree it rewrote rather than the SQL that was written, so its message for the
    # same mistake is strictly harder to act on than the one built here.
    findings = [
        *findings_for_unresolvable_columns(resolved.unresolvable_columns, positions),
        *findings_for_ambiguous_columns(resolved.ambiguous_columns, positions),
        *findings_for_columns_read_with_unsupported_dot_notation(
            resolved.columns_read_with_unsupported_dot_notation, positions, dialect_name
        ),
    ]
    if warn_on_column_without_source:
        findings += findings_for_columns_without_a_source(
            resolved.guessed_columns, positions
        )

    # Step 2b: match the resolved tables against the source declarations to fill in
    # declared type information.
    declared_types_per_table = get_declared_types_per_table(
        declared_schema, resolved.columns_per_table, resolved.source_table_names
    )
    findings += findings_for_columns_declared_as_scalar_but_read_as_structured(
        declared_schema, resolved.columns_per_table, positions
    )
    findings = findings_without_exact_duplicates(findings)

    # Two findings have to stop the statement. An unresolvable column belongs to no table,
    # so no fabricated schema can cover it and `qualify` raises. An ambiguous one belongs
    # to two, and committing to either would hand `qualify` an attribution that is as
    # likely wrong as right - every type inferred downstream would inherit the choice. A
    # dotted read and a guessed column are both harvested despite their findings, so the
    # rest of the statement is still checked.
    if resolved.unresolvable_columns or resolved.ambiguous_columns:
        return failed([], findings)

    for table, columns in declared_types_per_table.items():
        declared_count = sum(1 for name in columns.values() if is_declared(name))
        logger.info(
            "%s: %d of %d columns declared, %d gap-filled UNKNOWN",
            table,
            declared_count,
            len(columns),
            len(columns) - declared_count,
        )

    # Built once and threaded through: `qualify` and `annotate_types` each construct a
    # MappingSchema from a bare dict otherwise. The cast is the one place our
    # `{table: {column: type}}` meets sqlglot's invariant `dict[str, object]`.
    mapped_schema = ensure_schema(
        cast("dict[str, object]", declared_types_per_table), dialect=dialect_name
    )

    # Step 3: qualify. Every column names its relation, `select *` becomes a real
    # projection list. Prerequisite for all type inference, and 39% of the runtime.
    try:
        qualified = qualify(statement, schema=mapped_schema, dialect=dialect_name)
    except OptimizeError as e:
        return failed([str(e)], findings)

    logger.debug(
        "qualified %s:\n%s", model.relative_path, qualified.sql(dialect_name, pretty=True)
    )

    return QualifiedModel(
        model=model,
        source=source,
        positions=positions,
        findings=findings,
        errors=[],
        statement=QualifiedStatement(
            qualified=qualified,
            scopes=traverse_scope(qualified),
            mapped_schema=mapped_schema,
            declared_types_per_table=declared_types_per_table,
            resolved=resolved,
            dialect_name=dialect_name,
        ),
    )


# ------------------------------------------------------- reading the qualified tree
def name_and_kind_of_scope(scope: Scope) -> tuple[str, ScopeKind]:
    """What to call a scope in a report.

    A scope has no name of its own; what names it is the thing that holds it, so the
    answer comes from the parent node. An unnamed one is a set-operation arm, which is
    reported as a branch rather than given an invented name.
    """
    parent = scope.expression.parent
    if isinstance(parent, exp.CTE):
        return parent.alias, "cte"
    if isinstance(parent, exp.Subquery):
        return parent.alias or "<subquery>", "derived"
    if isinstance(parent, exp.SetOperation):
        return "<branch>", "branch"
    return "<final>", "final"


def is_in_the_projection_list(column: exp.Column, scope: Scope) -> bool:
    """Whether this column is part of what the scope selects, rather than how.

    Walks up to the child of the scope's own expression and asks which argument it sits
    in: `expressions` is the projection list, anything else is a FROM, WHERE, GROUP BY or
    ORDER BY. Walking rather than checking the immediate parent is what makes
    `upper(t.status)` count as projected.
    """
    node: exp.Expr | None = column
    while node is not None and node.parent is not scope.expression:
        node = node.parent
    return node is not None and node.arg_key == "expressions"


def _describe_source(
    column: exp.Column, scope: Scope
) -> tuple[TableName | None, ScopeKind | Literal["table", "unknown"]]:
    """The real relation behind a column's qualifier, and what kind of thing it is."""
    source = scope.sources.get(column.table)
    if isinstance(source, exp.Table):
        return (source.name if source.name != column.table else None), "table"
    if isinstance(source, Scope):
        return None, name_and_kind_of_scope(source)[1]
    return None, "unknown"


def column_references_of_scope(
    scope: Scope, statement: QualifiedStatement
) -> ScopeColumns:
    """Every column reference in one scope, deduplicated and split by projection.

    Deduplicated because the count is not the point: `qualify` clones a projection into
    GROUP BY, and a filter can name the same column twice, so a row per occurrence would
    say more about the rewrite than about the SQL. Two rows that agree on name, source and
    provenance are one fact.
    """
    written_bare = statement.resolved.offsets_of_columns_written_without_a_source
    projected: dict[ColumnReference, None] = {}
    non_projected: dict[ColumnReference, None] = {}

    for column in scope.columns:
        source_name, source_kind = _describe_source(column, scope)
        offsets = token_offsets_of(column)
        if offsets is None:
            # Nothing under it carries a position, so it was not written: step 3 built it
            # by expanding a star against the schema.
            origin: ColumnQualifierOrigin = "star"
        elif offsets in written_bare:
            origin = "inferred"
        else:
            origin = "written"

        declared_table = source_name or column.table
        reference = ColumnReference(
            name=column.name,
            source_alias=column.table,
            source_name=source_name,
            source_kind=source_kind,
            origin=origin,
            declared=source_kind == "table"
            and is_declared(
                statement.declared_types_per_table.get(declared_table.lower(), {}).get(
                    column.name.lower()
                )
            ),
        )
        bucket = projected if is_in_the_projection_list(column, scope) else non_projected
        bucket[reference] = None

    name, kind = name_and_kind_of_scope(scope)
    return ScopeColumns(
        name=name,
        kind=kind,
        projected=list(projected),
        non_projected=list(non_projected),
    )


def column_references_per_scope(statement: QualifiedStatement) -> list[ScopeColumns]:
    """Every scope that reads a column, in the order sqlglot resolves them.

    That order is dependency order - a CTE before whatever selects from it - so the final
    projection comes last, which is where a reader looks for it. Scopes reading no columns
    at all are dropped: a set-operation wrapper has none of its own and an empty table
    under its name only asks the reader to work out why it is empty.
    """
    scopes = [
        column_references_of_scope(scope, statement) for scope in statement.scopes
    ]
    return [scope for scope in scopes if scope.projected or scope.non_projected]
