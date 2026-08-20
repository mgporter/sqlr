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
from collections.abc import Sequence
from typing import Literal, NamedTuple, Protocol, cast

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
    findings_for_duplicate_projected_columns,
    findings_for_unresolvable_columns,
    findings_without_exact_duplicates,
)
from sqlr.sql_analysis2.output_names import (
    is_a_name_meaning_unnamed,
    output_column_namer,
)
from sqlr.sql_analysis2.resolve import (
    declared_columns,
    get_declared_types_per_table,
    is_declared,
    resolve_columns_to_source_tables,
)
from sqlr.sql_analysis2.sourcedoc import (
    Positions,
    SourceDoc,
    SourceSpan,
    token_offsets_of,
)
from sqlr.sql_analysis2.types import (
    ColumnName,
    ColumnTypeName,
    DuplicateProjection,
    ProjectionSite,
    ResolvedColumns,
    ScopeKind,
    TableName,
)

logger = logging.getLogger(__name__)

DEFAULT_DIALECT = "duckdb"

type ColumnQualifierOrigin = Literal["written", "inferred", "star"]
"""Where a column's source attribution came from.

- `written`  - the user qualified it themselves.
- `inferred` - the user wrote a bare name and step 2 attributed it.
- `star`     - the column does not appear in the SQL at all; step 3 materialised it by
  expanding a star against the gap-filled schema.
"""


type SourceKind = ScopeKind | Literal["table", "unknown"]


class ColumnSource(NamedTuple):
    """Where a column reads from, named the two ways a reader needs it."""

    alias: TableName
    """The qualifier the column carries after step 3 - an alias, not necessarily a table."""
    name: TableName | None
    """The real table behind the alias, when the source is a table and is named
    differently. None when the alias *is* the name, or when the source is a CTE."""
    kind: SourceKind


class ColumnReference(NamedTuple):
    """One `exp.Column` in a qualified tree, described for a reader rather than a pass."""

    name: ColumnName
    source: ColumnSource
    origin: ColumnQualifierOrigin
    declared: bool
    """Whether the user gave this column a type in a yml. False for anything a CTE
    produces: a CTE column's type is computed, and nobody declares it."""


class ProjectedColumn(NamedTuple):
    """One column a scope *outputs* - a row of its schema, not a read of its input.

    Keyed by the name it is known by downstream, which is the alias when there is one.
    That is what separates `upper(first_name) as first_name_upper` from the `first_name`
    it reads: they are two different columns of the relation, and collapsing them loses
    one of the outputs entirely.
    """

    name: ColumnName
    """Case-sensitive: a downstream reference has to quote it exactly."""
    engine_named: bool
    """True when nobody named this projection and the engine's own rule supplied the name
    (`upper(first_name)` in DuckDB, `UPPER(FIRST_NAME)` in Snowflake, `upper` in Postgres).
    Dialect-dependent by nature - see `output_names.py`."""
    origin: ColumnQualifierOrigin
    declared: bool
    reads: list[ColumnReference]
    """Every column of this scope's sources the projection reads, deduplicated. Empty for
    a projection that reads none, such as `1 + 1`."""
    column: ColumnReference | None
    """Set when the projection is nothing but a column, aliased or not - the case where
    the output *is* an input column rather than something computed from one."""


class ScopeColumns(NamedTuple):
    """What one scope outputs, and what else it had to read to get there.

    Two different kinds of thing, deliberately: `projected` is the scope's schema, one
    entry per output column in projection order, and `non_projected` is the columns
    consulted along the way - join keys, filters, grouping - that no output carries.
    """

    name: str
    kind: ScopeKind
    projected: list[ProjectedColumn]
    non_projected: list[ColumnReference]
    """One entry per `(name, source alias)`, and never a column some output already *is*:
    a downstream pass reading these as two sets must not have to subtract one from the
    other. A column read only inside a computed projection stays here, because no output
    column carries its name."""


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
    engine_named_projections: dict[int, ColumnName]
    """What the engine calls each projection nobody named, keyed by the id of its
    expression node. Captured before step 3, which is the only time it can be: see
    `engine_names_of_unaliased_projections`. Reporting-only - the tree itself carries
    sqlglot's `_col_1` placeholders, and nothing in steps 4-7 reads a projection's name."""
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


class ModelResult(Protocol):
    """Anything a command can exit non-zero over.

    Structural rather than a base class: `QualifiedModel` and the annotated result of steps
    4-7 answer the same question and are otherwise unrelated - one is a stage of the other,
    not a subtype of it.
    """

    @property
    def has_errors(self) -> bool: ...


def any_model_has_errors(results: Sequence[ModelResult]) -> bool:
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

    print(resolved)

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
    print()
    print(mapped_schema.column_names("raw_address"))

    # Two things about the projection lists that only exist before step 3. Where the stars
    # are, because step 3 expands them away and a duplicate they cause can only be
    # explained by pointing back at them; and what the engine calls the projections nobody
    # named, because step 3 relabels those `_col_1`.
    star_spans = spans_of_star_projections(statement, positions)
    engine_names = engine_names_of_unaliased_projections(
        statement, positions, dialect_name
    )

    # Step 3: qualify. Every column names its relation, `select *` becomes a real
    # projection list. Prerequisite for all type inference, and 39% of the runtime.
    try:
        qualified = qualify(statement, schema=mapped_schema, dialect=dialect_name)
        print()
        print(qualified)
    except OptimizeError as e:
        return failed([str(e)], findings)

    logger.debug(
        "qualified %s:\n%s", model.relative_path, qualified.sql(dialect_name, pretty=True)
    )

    scopes = traverse_scope(qualified)

    # A scope projecting one name twice is impossible SQL, and only visible now: it is
    # usually a star overlapping columns written out beside it, so it does not exist until
    # the star has expanded. It stops the model for the same reason an ambiguous column
    # does - a later reference to the name has two answers, and nothing can pick. sqlglot
    # will not say so: it quietly stops expanding stars over the broken relation instead,
    # which surfaces as an empty projection in some other scope entirely.
    duplicates = [
        duplicate
        for scope in scopes
        for duplicate in duplicate_projections_of_scope(
            scope, positions, star_spans, engine_names
        )
    ]
    if duplicates:
        return failed([], findings + findings_for_duplicate_projected_columns(duplicates))

    return QualifiedModel(
        model=model,
        source=source,
        positions=positions,
        findings=findings,
        errors=[],
        statement=QualifiedStatement(
            qualified=qualified,
            scopes=scopes,
            mapped_schema=mapped_schema,
            declared_types_per_table=declared_types_per_table,
            resolved=resolved,
            engine_named_projections=engine_names,
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


def qualifier_of_a_star_projection(projection: exp.Expr) -> TableName | None | Literal[False]:
    """The source a `*` projection is qualified by, or False when it is not a star.

    `False` rather than a raised exception because the three answers are all ordinary:
    `t.*` is qualified, `*` is not, and `upper(x)` is not a star at all. None and False
    would otherwise be the same value for two opposite facts.
    """
    if isinstance(projection, exp.Star):
        return None
    if isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star):
        return projection.table or None
    return False


def spans_of_star_projections(
    statement: exp.Expr, positions: Positions
) -> dict[int, dict[TableName | None, SourceSpan]]:
    """Where each select's stars are written, keyed by the id of the select node.

    Recorded *before* step 3, because step 3 is what removes them: an expanded projection
    carries no trace of the `*` it came from, and the star node is gone from the tree by
    the time anyone notices the duplicate it caused.

    Keying on `id()` is safe here and only here - the caller holds the statement alive
    across both halves of the call, and `qualify` rewrites a select's arguments rather
    than replacing the select itself. A miss degrades to a message without a position.
    """
    out: dict[int, dict[TableName | None, SourceSpan]] = {}
    for select in statement.find_all(exp.Select):
        stars: dict[TableName | None, SourceSpan] = {}
        for projection in select.expressions:
            qualifier = qualifier_of_a_star_projection(projection)
            if qualifier is False:
                continue
            span = positions.span_of(projection)
            key = qualifier.lower() if qualifier is not None else None
            if span is not None and key not in stars:
                stars[key] = span
        if stars:
            out[id(select)] = stars
    return out


def engine_names_of_unaliased_projections(
    statement: exp.Expr, positions: Positions, dialect_name: str
) -> dict[int, ColumnName]:
    """Name every projection nobody named, keyed by the id of its expression node.

    Recorded *before* step 3 for two reasons. `qualify_outputs` labels these `_col_1`,
    which no engine produces and which nothing can tell apart from a hand-written alias
    afterwards; and the name an engine gives is derived from the projection *as written*,
    not from the qualified, case-folded rewrite of it.

    Only unaliased non-column projections need it. A written alias survives step 3 intact,
    and a bare column is named after the column - both correctly, and both already folded
    the way the dialect folds identifiers.

    Keying on `id()` is safe here for the same reason as in `spans_of_star_projections`:
    the caller holds the statement across both halves of the call, and `qualify` rewrites
    a projection's contents rather than replacing the node.
    """
    name_it = output_column_namer(dialect_name)
    out: dict[int, ColumnName] = {}
    for select in statement.find_all(exp.Select):
        for projection in select.expressions:
            if isinstance(projection, (exp.Alias, exp.Column)):
                continue
            if qualifier_of_a_star_projection(projection) is not False:
                continue
            span = positions.span_of(projection)
            out[id(projection)] = name_it(
                projection, positions.text[span.start : span.end] if span else None
            )
    return out


def output_name_of(projection: exp.Expr, engine_names: dict[int, ColumnName]) -> ColumnName:
    """The name a projection of a qualified select produces downstream.

    sqlglot's alias where the user or the column supplied one, and the engine's own rule
    where neither did - never `_col_1`, which is sqlglot's placeholder for exactly the
    case `engine_names` answers.
    """
    inner = projection.this if isinstance(projection, exp.Alias) else projection
    return engine_names.get(id(inner), projection.alias_or_name)


def _projection_site(
    projection: exp.Expr, stars: dict[TableName | None, SourceSpan]
) -> ProjectionSite:
    """Where one projection of a qualified select came from.

    An expanded projection carries no token position - nothing under it was ever typed -
    which is what separates it from one the user wrote. The star it came from is then
    matched by the source it reads, so `select a.*, b.*` blames the right one.
    """
    if token_offsets_of(projection) is not None:
        return ProjectionSite(from_star=False, span=None)

    column = projection.find(exp.Column)
    qualifier = column.table.lower() if column is not None and column.table else None
    span = stars.get(qualifier) or stars.get(None)
    if span is None and stars:
        span = next(iter(stars.values()))
    return ProjectionSite(from_star=True, span=span)


def duplicate_projections_of_scope(
    scope: Scope,
    positions: Positions,
    star_spans: dict[int, dict[TableName | None, SourceSpan]],
    engine_names: dict[int, ColumnName],
) -> list[DuplicateProjection]:
    """Output names this scope projects more than once.

    Two kinds of projection are skipped rather than counted, both because their "name" is
    not one. A star that did not expand: `select a.*, b.*` over undeclared tables leaves
    two projections both named `*`, and they are one unexpanded star each. And a name the
    engine uses to mean *unnamed*: Postgres calls both `a + b` and `1` `?column?`, returns
    them side by side without complaint, and only objects when something references one.
    """
    select = scope.expression
    if not isinstance(select, exp.Select):
        return []

    stars = star_spans.get(id(select), {})
    sites_per_name: dict[ColumnName, list[ProjectionSite]] = {}
    for projection in select.selects:
        if qualifier_of_a_star_projection(projection) is not False:
            continue
        name = output_name_of(projection, engine_names)
        if is_a_name_meaning_unnamed(name):
            continue
        site = _projection_site(projection, stars)
        if not site.from_star:
            site = ProjectionSite(from_star=False, span=positions.span_of(projection))
        sites_per_name.setdefault(name, []).append(site)

    name, kind = name_and_kind_of_scope(scope)
    return [
        DuplicateProjection(
            scope_name=name, scope_kind=kind, column_name=column_name, sites=sites
        )
        for column_name, sites in sites_per_name.items()
        if len(sites) > 1
    ]


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


def _source_of(column: exp.Column, scope: Scope) -> ColumnSource:
    """The relation behind a column's qualifier, and what kind of thing it is."""
    source = scope.sources.get(column.table)
    if isinstance(source, exp.Table):
        name = source.name if source.name != column.table else None
        return ColumnSource(alias=column.table, name=name, kind="table")
    if isinstance(source, Scope):
        return ColumnSource(
            alias=column.table, name=None, kind=name_and_kind_of_scope(source)[1]
        )
    return ColumnSource(alias=column.table, name=None, kind="unknown")


_ORIGIN_PRECEDENCE: dict[ColumnQualifierOrigin, int] = {
    "written": 0,
    "inferred": 1,
    "star": 2,
}
"""Which reading of one column to report when a scope has several.

The most explicit wins. `select t.*, upper(t.name) from t` reads `t.name` twice, once
through the star and once written out; calling it star-expanded would say the name appears
nowhere in the file, which is the one thing a reader could check and find false.
"""


def column_reference_of(
    column: exp.Column, scope: Scope, statement: QualifiedStatement
) -> ColumnReference:
    """Describe one `exp.Column` node: where it reads from, and how it got there."""
    source = _source_of(column, scope)

    offsets = token_offsets_of(column)
    if offsets is None:
        # Nothing under it carries a position, so it was not written: step 3 built it by
        # expanding a star against the schema.
        origin: ColumnQualifierOrigin = "star"
    elif offsets in statement.resolved.offsets_of_columns_written_without_a_source:
        origin = "inferred"
    else:
        origin = "written"

    declared_table = source.name or source.alias
    return ColumnReference(
        name=column.name,
        source=source,
        origin=origin,
        declared=source.kind == "table"
        and is_declared(
            statement.declared_types_per_table.get(declared_table.lower(), {}).get(
                column.name.lower()
            )
        ),
    )


def _projected_column(
    projection: exp.Expr, scope: Scope, statement: QualifiedStatement
) -> ProjectedColumn:
    """One entry of a scope's output schema.

    `reads` is restricted to columns of *this* scope, so a scalar subquery in the
    projection contributes its own scope's reads to that scope and not to this one.
    """
    inner = projection.this if isinstance(projection, exp.Alias) else projection
    own_columns = {id(column) for column in scope.columns}

    reads: dict[tuple[ColumnName, TableName], ColumnReference] = {}
    for column in projection.find_all(exp.Column):
        if id(column) not in own_columns:
            continue
        reference = column_reference_of(column, scope, statement)
        seen = reads.get((reference.name, reference.source.alias))
        if seen is None or (
            _ORIGIN_PRECEDENCE[reference.origin] < _ORIGIN_PRECEDENCE[seen.origin]
        ):
            reads[(reference.name, reference.source.alias)] = reference

    # A projection that is nothing but a column *is* that column, renamed or not. Anything
    # computed from one is a new column that merely reads it.
    column_read = (
        column_reference_of(inner, scope, statement) if isinstance(inner, exp.Column) else None
    )
    name = output_name_of(projection, statement.engine_named_projections)
    return ProjectedColumn(
        name=name,
        engine_named=id(inner) in statement.engine_named_projections,
        origin=column_read.origin if column_read is not None else "written",
        declared=column_read.declared if column_read is not None else False,
        reads=list(reads.values()),
        column=column_read,
    )


def scope_columns_of(scope: Scope, statement: QualifiedStatement) -> ScopeColumns:
    """One scope's output schema, and the columns it read that no output carries.

    The projection list is taken as written, in order and without deduplication: two
    outputs of the same name is impossible SQL and has already stopped the model, so
    anything reaching here is a real schema.

    `non_projected` holds one entry per `(name, source alias)` - `qualify` clones a
    projection into GROUP BY and a filter can name the same column twice, so a row per
    occurrence would describe the rewrite rather than the SQL. An entry is dropped when
    some output column *is* that column: `select src.id from src where src.id > 0` reads
    one column and projects it. A column read only inside a computed projection stays,
    because no output carries its name.
    """
    select = scope.expression
    projected = (
        [
            _projected_column(projection, scope, statement)
            for projection in select.selects
            if qualifier_of_a_star_projection(projection) is False
        ]
        if isinstance(select, exp.Select)
        else []
    )
    projected_columns = {
        (entry.column.name, entry.column.source.alias)
        for entry in projected
        if entry.column is not None
    }

    non_projected: dict[tuple[ColumnName, TableName], ColumnReference] = {}
    for column in scope.columns:
        if is_in_the_projection_list(column, scope):
            continue
        reference = column_reference_of(column, scope, statement)
        key = (reference.name, reference.source.alias)
        if key in projected_columns:
            continue
        non_projected.setdefault(key, reference)

    name, kind = name_and_kind_of_scope(scope)
    return ScopeColumns(
        name=name,
        kind=kind,
        projected=projected,
        non_projected=list(non_projected.values()),
    )


def columns_per_scope(statement: QualifiedStatement) -> list[ScopeColumns]:
    """Every scope with columns to report, in the order sqlglot resolves them.

    That order is dependency order - a CTE before whatever selects from it - so the final
    projection comes last, which is where a reader looks for it. Scopes with nothing in
    either list are dropped: a set-operation wrapper has no columns of its own, and an
    empty table under its name only asks the reader to work out why it is empty.
    """
    scopes = [scope_columns_of(scope, statement) for scope in statement.scopes]
    return [scope for scope in scopes if scope.projected or scope.non_projected]
