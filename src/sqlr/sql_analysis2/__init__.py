"""The typing pipeline. Steps are numbered as in `type_check_plan.md`.

Only steps 0-4 are wired. Steps 5-7 (backward evidence, re-annotate, check) are marked
where they belong and do nothing yet.
"""

import logging
from pathlib import Path
from typing import cast

import sqlglot
from pydantic import BaseModel
from sqlglot import ParseError, exp
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import ensure_schema
from sqlglot.typing import ExprMetadataType

from sqlr.config.types import SqlrConfig
from sqlr.declared.types import DeclaredSchemas
from sqlr.selection.types import Model
from sqlr.sql_analysis2.annotate import expression_metadata, func_args
from sqlr.sql_analysis2.reporting import (
    findings_for_ambiguous_columns,
    findings_for_columns_declared_as_scalar_but_read_as_structured,
    findings_for_columns_read_with_unsupported_dot_notation,
    findings_for_columns_without_a_source,
    findings_for_unresolvable_columns,
    print_findings,
)
from sqlr.sql_analysis2.sourcedoc import (
    Positions,
    SourceDoc,
    SourceSpan,
    token_offsets_of,
)
from sqlr.sql_analysis2.types import (
    AmbiguousColumn,
    ColumnName,
    ColumnTypeName,
    GuessedColumn,
    ParsedColumn,
    ResolvedColumns,
    StructuredAccessKind,
    TableName,
)

logger = logging.getLogger(__name__)

DEFAULT_DIALECT = "duckdb"




class TypeFact(BaseModel):
    span: SourceSpan
    type: str


# ---------------------------------------------------------------- step 2: schema
def declared_columns(declared: DeclaredSchemas) -> dict[TableName, dict[ColumnName, ColumnTypeName]]:
    """Every declared source table, keyed by the bare table name, lowercased.

    Bare name only: sqlglot's schema dict is keyed the way the SQL writes the table, and
    `qualify` lowercases identifiers. A declaration that writes `db.schema.table` matches
    here on its last part - a known simplification, see plan step 3 and Q7 before any of
    this reaches a report.
    """
    out: dict[TableName, dict[ColumnName, ColumnTypeName]] = {}
    for table in declared.sources.values():
        out[table.table_name.lower()] = {
            column.name.lower(): column.written_type for column in table.columns
        }
    return out


DIALECTS_WITH_DOT_FIELD_ACCESS = frozenset(
    {"duckdb", "spark", "databricks", "bigquery", "hive", "trino", "presto", "athena"}
)
"""Dialects that read `a.b` as a field of the structured column `a` when `a` names no
source. Postgres wants `(a).b` and Snowflake wants `a:b`, so for them a dotted name that
matches no source is a mistake instead."""


def dialect_parses_unresolvable_aliases_as_json_columns(dialect_name: str) -> bool:
    """Whether a qualifier naming no source is legitimately a structured column read.

    Internal for now; a candidate for `GeneralConfig` if a dialect ever needs overriding.
    """
    return dialect_name.lower() in DIALECTS_WITH_DOT_FIELD_ACCESS


def structured_access_of(column: exp.Column) -> StructuredAccessKind | None:
    """The access one column node carries, or None when it is read as a plain value."""
    parent = column.parent
    if parent is None or parent.args.get("this") is not column:
        return None
    if isinstance(parent, exp.Dot):
        return "dot_field"
    if isinstance(parent, exp.Bracket):
        key = parent.expressions[0] if parent.expressions else None
        if isinstance(key, exp.Literal) and key.is_string:
            return "bracket_key"
        return "bracket_index"
    return None


def sources_a_bare_column_could_read(scope: Scope) -> list[tuple[TableName, exp.Table | Scope]]:
    """The sources an unqualified column in this scope could actually be reading from.

    `Scope.sources` also holds every CTE the statement defines, whether or not this scope
    selects from it. `selected_sources` is what a FROM or JOIN actually brought in, and
    `lateral_sources` what a lateral added - the same pair sqlglot's own `Resolver`
    consults when it decides which names are unambiguous.
    """
    sources: list[tuple[TableName, exp.Table | Scope]] = [
        (name, source) for name, (_, source) in scope.selected_sources.items()
    ]
    already_listed = {name for name, _ in sources}
    sources.extend(
        (name, source)
        for name, source in scope.lateral_sources.items()
        if name not in already_listed
    )
    return sources


def known_column_names_of_source(
    source: exp.Table | Scope,
    declared_schema: dict[TableName, dict[ColumnName, ColumnTypeName]],
) -> set[ColumnName] | None:
    """Every column a source is known to have, or None when its column set is unknown.

    None is not "owns nothing" - it is "no answer available", and the two lead to opposite
    verdicts about an unqualified column.

    A CTE or derived table answers for itself: its projections are the whole set. A real
    table answers only through `sources.yml`, and a declaration carrying at least one
    column is read as the *complete* list rather than a sample - that is the rule this
    whole check rests on. A table that is undeclared, or declared with no columns, could
    own anything.
    """
    if isinstance(source, exp.Table):
        declared = declared_schema.get(source.name.lower())
        return set(declared) if declared else None

    named_selects = getattr(source.expression, "named_selects", None)
    if not named_selects or "*" in named_selects:
        # A derived table projecting a star cannot be enumerated without a schema, which
        # the probe deliberately does not have.
        return None
    return {name.lower() for name in named_selects}


def sources_owning_and_sources_open_for(
    column_name: ColumnName,
    scope: Scope,
    declared_schema: dict[TableName, dict[ColumnName, ColumnTypeName]],
) -> tuple[list[TableName], list[TableName]]:
    """Split this scope's sources by what they say about one unqualified column name.

    Returns the sources *known* to own the name and the sources whose column set is
    unknown, both sorted. A source known not to own it appears in neither: it has been
    ruled out, and cannot make an attribution either wrong or uncertain.
    """
    owning: list[TableName] = []
    open_sources: list[TableName] = []
    for source_name, source in sources_a_bare_column_could_read(scope):
        known = known_column_names_of_source(source, declared_schema)
        if known is None:
            open_sources.append(source_name)
        elif column_name in known:
            owning.append(source_name)
    return sorted(owning), sorted(open_sources)


def judge_a_column_written_without_a_source(
    column: exp.Column,
    scope: Scope,
    declared_schema: dict[TableName, dict[ColumnName, ColumnTypeName]],
) -> AmbiguousColumn | GuessedColumn | None:
    """How much the probe's attribution of one unqualified column can be trusted.

    The column has already been attributed - `column.table` is the probe's answer. This
    only grades it, by counting the sources that were *not* ruled out:

    - two or more sources known to own the name: an `AmbiguousColumn`. Nothing can break
      the tie, so the statement stops.
    - exactly one candidate: certain, and nothing is returned. A lone source with an
      unknown column set counts here - if every other source in the scope is known not to
      own the name, the attribution is forced rather than guessed.
    - several candidates, at least one of them a source with an unknown column set: a
      `GuessedColumn`. The attribution stands and the reader is told what was assumed.

    A source known *not* to own the name is ruled out and counts for nothing - which is
    what keeps a partially declared table from turning every column it omits into noise.
    That table is still the probe's fallback and still absorbs those columns; a declaration
    is read as complete only when deciding this question, never when resolving one.
    """
    owning, open_sources = sources_owning_and_sources_open_for(
        column.name.lower(), scope, declared_schema
    )
    if len(owning) >= 2:
        return AmbiguousColumn(column=column, candidate_sources=owning)
    if open_sources and len(owning) + len(open_sources) > 1:
        return GuessedColumn(
            column=column, resolved_source=column.table, open_sources=open_sources
        )
    return None


def resolve_columns_to_source_tables(
    statement: exp.Expr,
    dialect_name: str,
    allow_unresolvable_aliases_as_structured_columns: bool | None = None,
    declared_schema: dict[TableName, dict[ColumnName, ColumnTypeName]] | None = None,
) -> ResolvedColumns:
    """Attribute every column to the source it reads from, by letting sqlglot do it.

    Qualify a *throwaway copy* against an **empty** schema. That is the whole trick: with
    no schema every real table reports zero known columns, so it becomes the
    `infer_schema` fallback target, while CTEs and derived tables still report their own
    projections. Bare columns land on the one table that could own them, and only a
    genuinely ambiguous name is left unqualified. Passing the *declared* schema instead
    would defeat it - a partially declared table stops being the fallback target and its
    undeclared columns fail to resolve.

    `Resolver.get_table` also brings join-context disambiguation, USING expansion, set-op
    and lateral derivation and alias shadowing; re-deriving those here would be a worse
    copy. The copy is discarded - only the harvested names are wanted.

    sqlglot's `_convert_columns_to_dots` rewrites any qualifier naming no source into a
    struct field read, so `select ghots.id from test` comes back as a column `ghots` of
    `test`. The rewrite is not dialect-gated, so this gates it:
    `allow_unresolvable_aliases_as_structured_columns` false makes such a read a finding
    rather than a column. It only fires when a lone source can absorb the name; with two
    sources the qualifier resolves to nothing and is unresolvable instead.

    `declared_schema` is judgement, never input: it decides whether an unqualified column
    was attributed with certainty, and is *not* handed to `qualify`. Passing it as the
    probe's schema would undo everything above - a partially declared table would stop
    being the `infer_schema` fallback target, and every column it does not declare would
    fail to resolve. Omit it and the certainty verdicts are simply not made.
    """
    if allow_unresolvable_aliases_as_structured_columns is None:
        allow_unresolvable_aliases_as_structured_columns = (
            dialect_parses_unresolvable_aliases_as_json_columns(dialect_name)
        )

    # Which columns were written without a qualifier, recorded before the probe rewrites
    # them. Token offsets survive `copy()` and survive `qualify` - the synthesised table
    # identifier it adds carries none of its own - so the hull of a qualified column is
    # still the hull of the name the user typed, and identifies it across the two trees.
    offsets_of_columns_written_without_a_source = {
        offsets
        for column in statement.find_all(exp.Column)
        if not column.table and (offsets := token_offsets_of(column)) is not None
    }

    print(offsets_of_columns_written_without_a_source)

    probe = qualify(
        statement.copy(),
        dialect=dialect_name,
        schema={},
        infer_schema=True,
        # A column qualified against a table whose columns are unknown is not an error.
        allow_partial_qualification=True,
        # A star cannot expand against an empty schema, and the probe does not need it to.
        expand_stars=False,
        # Unresolvable columns are reported by us, with a span; they must not raise here.
        validate_qualify_columns=False,
        quote_identifiers=False,
    )

    columns_per_table: dict[TableName, dict[ColumnName, ParsedColumn]] = {}
    unresolvable_columns: list[exp.Column] = []
    columns_read_with_unsupported_dot_notation: list[exp.Column] = []
    ambiguous_columns: list[AmbiguousColumn] = []
    guessed_columns: list[GuessedColumn] = []
    # `qualify` can clone a projection into GROUP BY or ORDER BY, so one written column can
    # arrive here twice. It is one mistake either way, and deserves one finding.
    offsets_already_judged: set[tuple[int, int]] = set()

    for scope in traverse_scope(probe):
        for column in scope.columns:
            if not column.table:
                # Still bare after qualification. Against an empty schema two tables that
                # both own the name are indistinguishable from two that both lack it, so
                # sqlglot gives up on either - and only the declarations can say which
                # happened. Without them, the older reading stands.
                offsets = token_offsets_of(column)
                print(offsets)
                if (
                    declared_schema is not None
                    and offsets is not None
                    and offsets not in offsets_already_judged
                ):
                    offsets_already_judged.add(offsets)
                    owning, _ = sources_owning_and_sources_open_for(
                        column.name.lower(), scope, declared_schema
                    )
                    if len(owning) >= 2:
                        ambiguous_columns.append(
                            AmbiguousColumn(column=column, candidate_sources=owning)
                        )
                        continue
                unresolvable_columns.append(column)
                continue

            source = scope.sources.get(column.table)
            if source is None:
                # A qualifier naming no source in this scope - a mistyped table alias.
                # See the docstring: this only survives when the scope has more than one
                # source, because a lone source absorbs the name as a struct read.
                unresolvable_columns.append(column)
                continue

            offsets = token_offsets_of(column)
            if (
                declared_schema is not None
                and offsets in offsets_of_columns_written_without_a_source
                and offsets not in offsets_already_judged
            ):
                offsets_already_judged.add(offsets)
                verdict = judge_a_column_written_without_a_source(
                    column, scope, declared_schema
                )
                if isinstance(verdict, AmbiguousColumn):
                    # Attributing it at all would fabricate a declaration slot on a table
                    # that may not own the column.
                    ambiguous_columns.append(verdict)
                    continue
                if isinstance(verdict, GuessedColumn):
                    guessed_columns.append(verdict)

            access = structured_access_of(column)
            if access == "dot_field" and not allow_unresolvable_aliases_as_structured_columns:
                columns_read_with_unsupported_dot_notation.append(column)

            if isinstance(source, exp.Table):
                columns = columns_per_table.setdefault(source.name.lower(), {})
                name = column.name.lower()
                parsed = columns.setdefault(
                    name, ParsedColumn(name=name, structured_access=[])
                )
                if access is not None:
                    parsed.structured_access.append((access, column))
            # Anything else is a CTE or derived table, whose columns are its own
            # projections - nothing to fabricate a declaration for.

    return ResolvedColumns(
        columns_per_table=columns_per_table,
        unresolvable_columns=unresolvable_columns,
        columns_read_with_unsupported_dot_notation=columns_read_with_unsupported_dot_notation,
        ambiguous_columns=ambiguous_columns,
        guessed_columns=guessed_columns,
    )


def get_declared_types_per_table(
    declared: dict[TableName, dict[ColumnName, ColumnTypeName]],
    column_names_per_table: dict[TableName, dict[ColumnName, ParsedColumn]],
) -> dict[TableName, dict[ColumnName, ColumnTypeName]]:
    """Complete the schema's *column set* so `qualify` does not raise.

    Declared types where the user wrote them, UNKNOWN everywhere else. Not optional:
    `qualify` validates that every column resolves, and every built-in escape
    (`allow_partial_qualification`, `infer_schema`) still raises. See plan step 2.

    The "typo trap" this used to warn about - a misspelled column silently invented here
    as a real one - is now caught upstream: a name that resolves to no source at all comes
    back in `ResolvedColumns.unresolvable_columns` and the statement is reported before
    reaching this point. A name misspelled into a *declared* table's slot is still
    invented, and still needs the declared column set checked against the fabricated one.
    """
    return {
        table_name: {
            column: declared.get(table_name, {}).get(column, "UNKNOWN")
            for column in sorted(columns)
        }
        for table_name, columns in column_names_per_table.items()
    }


def needs_inference(schema: dict[TableName, dict[ColumnName, ColumnTypeName]]) -> bool:
    """Whether steps 5 and 6 have anything to do. O(columns), not O(nodes)."""
    return any(t.upper() == "UNKNOWN" for cols in schema.values() for t in cols.values())


# ------------------------------------------------------------------- diagnostics
def log_coverage(tree: exp.Expr) -> None:
    """Coverage debt: a Func whose arguments are all typed but whose result is UNKNOWN is
    a catalog gap, never a user error. That second number is the one that matters."""
    total = 0
    typed = 0
    gaps: list[str] = []
    for node in tree.walk():
        total += 1
        node_type = node.type
        if node_type is not None and not node_type.is_type(exp.DType.UNKNOWN):
            typed += 1
        elif isinstance(node, exp.Func) and node.is_type(exp.DType.UNKNOWN):
            args = func_args(node)
            if args and all(
                a.type is not None and not a.type.is_type(exp.DType.UNKNOWN) for a in args
            ):
                gaps.append(node.name if isinstance(node, exp.Anonymous) else node.sql_name())
    logger.info(
        "typed %d/%d nodes; %d UNKNOWN with fully-typed arguments (catalog gaps)",
        typed,
        total,
        len(gaps),
    )
    if gaps:
        logger.debug("catalog gaps: %s", ", ".join(sorted(set(gaps))))


def log_projections(tree: exp.Expr) -> None:
    """Per-scope name -> type table. More useful than dumping an annotated tree, and a
    fraction of the size."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    for scope in traverse_scope(tree):
        select = scope.expression
        if not isinstance(select, exp.Select):
            continue
        parent = select.parent
        name = parent.alias if isinstance(parent, exp.CTE) else "<final>"
        logger.debug("-- %s", name)
        for projection in select.selects:
            logger.debug("   %-20s %s", projection.alias_or_name, projection.type)


# ---------------------------------------------------------------------- pipeline
def validate_schema(
    cfg: SqlrConfig, declared: DeclaredSchemas, models: list[Model]
) -> None:
    dialect_name = cfg.general.sql_dialect or DEFAULT_DIALECT

    # Step 0: the annotation metadata, built once per run rather than per file. It is a
    # copy of the dialect's own map with our catalog layered on top; the dialect itself is
    # never mutated, so two runs with different dialects cannot interfere (plan Q6).
    metadata = expression_metadata(dialect_name)

    # Declarations are read once too - they do not vary per model.
    declared_schema = declared_columns(declared)

    for model in models:
        sql = Path(model.path).read_text()
        source = SourceDoc(path=model.path, text=sql)

        # One index for the whole document: sqlglot's character offsets are absolute, so
        # they stay valid across every statement in the file.
        positions = Positions(sql)

        # Step 1: parse.
        try:
            statements = sqlglot.parse(sql, read=dialect_name)
        except ParseError as e:
            print(f"error: {model.relative_path}: {e}")
            continue

        parsed = [s for s in statements if s is not None]
        if not parsed:
            print(f"error: {model.relative_path}: no statements found in the SQL file")
            continue
        if len(parsed) > 1:
            # Plan Q4: run the steps per statement, keep the warning.
            print(
                f"warning: {model.relative_path}: file contains multiple statements; "
                "each is checked on its own"
            )

        logger.info("parsed %s: %d statements", model.relative_path, len(parsed))

        for statement in parsed:
            _check_statement(
                statement,
                declared_schema,
                dialect_name,
                metadata,
                model,
                source,
                positions,
                cfg.general.warn_on_column_without_source,
            )


def _check_statement(
    statement: exp.Expr,
    declared_schema: dict[TableName, dict[ColumnName, ColumnTypeName]],
    dialect_name: str,
    metadata: ExprMetadataType,
    model: Model,
    source: SourceDoc,
    positions: Positions,
    warn_on_column_without_source: bool = True,
) -> None:

    # Step 2a: a probe qualification resolves every column to the source it reads from.
    # This is necessary to get the sources for columns without an alias but which MUST
    # come from a source because no other source has that column.
    # E.g.: with t2 as (select distinct id from mytable) 
    #       select a from t1 join t2 on t1.id = t2.id;
    # Here, 'a' must come from either t1, since t2 only has 'id'.
    try:
        resolved = resolve_columns_to_source_tables(
            statement, dialect_name, declared_schema=declared_schema
        )
    except OptimizeError as e:
        print(f"error: {model.relative_path}: {e}")
        return

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
        declared_schema, resolved.columns_per_table
    )
    findings += findings_for_columns_declared_as_scalar_but_read_as_structured(
        declared_schema, resolved.columns_per_table, positions
    )
    print_findings(findings, str(model.relative_path))

    # Two findings have to stop the statement. An unresolvable column belongs to no table,
    # so no fabricated schema can cover it and `qualify` raises. An ambiguous one belongs
    # to two, and committing to either would hand `qualify` an attribution that is as
    # likely wrong as right - every type inferred downstream would inherit the choice. A
    # dotted read and a guessed column are both harvested despite their findings, so the
    # rest of the statement is still checked.
    if resolved.unresolvable_columns or resolved.ambiguous_columns:
        return

    logger.debug("declared_types_per_table: %s", declared_types_per_table)

    for table, columns in declared_types_per_table.items():
        declared_count = sum(1 for t in columns.values() if t.upper() != "UNKNOWN")
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
        print(f"error: {model.relative_path}: {e}")
        return
    logger.debug("qualified %s:\n%s", model.relative_path, qualified.sql(dialect_name, pretty=True))

    # Step 3b: the non-type analysis (build_graph / Resolver / extract_facts) belongs
    # here, fed this same qualified tree. Not wired yet.

    # Step 4: annotate, pass 1. Types every expression from what is currently known, and
    # establishes what is *already* known, which step 5 depends on.
    annotated = annotate_types(
        qualified, schema=mapped_schema, expression_metadata=metadata, dialect=dialect_name
    )

    log_coverage(annotated)
    log_projections(annotated)

    # Steps 5 and 6 are skipped when the schema has no UNKNOWN slots - proven no-op.
    if needs_inference(declared_types_per_table):
        logger.debug("schema has UNKNOWN slots; steps 5-6 run here")

    # TODO step 7: check the annotated tree and report findings. `positions` turns a node
    # into a SourceSpan and `source` slices the text under it.
    _ = (source, positions)
