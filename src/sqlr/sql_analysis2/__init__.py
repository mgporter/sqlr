"""The typing pipeline. Steps are numbered as in `type_check_plan.md`.

Only steps 0-4 are wired. Steps 5-7 (backward evidence, re-annotate, check) are marked
where they belong and do nothing yet.
"""

import logging
from pathlib import Path
from typing import NamedTuple, cast

import sqlglot
from pydantic import BaseModel
from sqlglot import ParseError, exp
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope
from sqlglot.schema import ensure_schema
from sqlglot.typing import ExprMetadataType

from sqlr.config.types import SqlrConfig
from sqlr.declared.types import DeclaredSchemas
from sqlr.selection.types import Model
from sqlr.sql_analysis2.annotate import expression_metadata, func_args
from sqlr.sql_analysis2.sourcedoc import Positions, SourceDoc, SourceSpan

logger = logging.getLogger(__name__)

DEFAULT_DIALECT = "duckdb"




class TypeFact(BaseModel):
    span: SourceSpan
    type: str

type TableName = str
type ColumnName = str
type ColumnTypeName = str

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


class ResolvedColumns(NamedTuple):
    """What the probe pass learned: which columns each real table is asked for, and
    which columns belong to no source at all."""

    columns_per_table: dict[TableName, set[ColumnName]]
    unresolvable_columns: list[exp.Column]


def resolve_columns_to_source_tables(
    statement: exp.Expr, dialect_name: str
) -> ResolvedColumns:
    """Attribute every column to the source it reads from, by letting sqlglot do it.

    The naive version of this - "a bare column belongs to the table when the scope has
    exactly one source" - drops every bare column in a scope that joins a table to a CTE,
    even though such a query is perfectly resolvable: a CTE's column set is its own
    projection list, knowable without any schema, so a name absent from it can only have
    come from the table. `Resolver.get_table` already implements exactly that rule, plus
    join-context disambiguation, USING expansion, set-op and lateral column derivation,
    and column-alias shadowing. Re-deriving any of it here would be a second, worse copy.

    So the resolution is delegated: qualify a *throwaway copy* against an **empty** schema.
    That is the whole trick. With no schema, every real table reports zero known columns,
    which makes it the `infer_schema` fallback target, while CTEs and derived tables still
    report theirs. Bare columns then land on the one table that could own them, and only a
    genuinely ambiguous name (two undeclared tables, say) is left unqualified.

    Passing the *declared* schema here instead would defeat it: a partially declared table
    reports a non-empty column set, stops being the fallback target, and its undeclared
    columns fail to resolve again - the exact bug this replaces.

    The copy is discarded. `qualify` rewrites the tree it is given (alias references get
    expanded, stars stay folded), and only the harvested names are wanted.

    One blind spot, and it is sqlglot's rather than ours: `_convert_columns_to_dots`
    reinterprets any qualifier that names no source as a STRUCT or JSON field lookup, so
    `select ghots.id from test` is read as field `id` of a struct column named `ghots` and
    the typo comes back as a real column of `test`. That rewrite is not dialect-gated -
    Postgres and Snowflake, which do not accept bare dotted field access at all, behave
    the same here. It only bites when a lone source is available to absorb the name; with
    two sources the qualifier resolves to nothing and is reported.
    """
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

    columns_per_table: dict[TableName, set[ColumnName]] = {}
    unresolvable_columns: list[exp.Column] = []

    for scope in traverse_scope(probe):
        for column in scope.columns:
            print(column.table, column.name)
            if not column.table:
                # Still bare after qualification: no source can own it.
                unresolvable_columns.append(column)
                continue

            source = scope.sources.get(column.table)
            if isinstance(source, exp.Table):
                columns_per_table.setdefault(source.name.lower(), set()).add(
                    column.name.lower()
                )
            elif source is None:
                # A qualifier naming no source in this scope - a mistyped table alias.
                # See the docstring: this only survives when the scope has more than one
                # source, because a lone source absorbs the name as a struct read.
                unresolvable_columns.append(column)
            # Anything else is a CTE or derived table, whose columns are its own
            # projections - nothing to fabricate a declaration for.

    print()
    print("columns_per_table", columns_per_table)
    print("unresolvable_columns", unresolvable_columns)

    return ResolvedColumns(
        columns_per_table=columns_per_table, unresolvable_columns=unresolvable_columns
    )


def get_declared_types_per_table(
    declared: dict[TableName, dict[ColumnName, ColumnTypeName]],
    column_names_per_table: dict[TableName, set[ColumnName]],
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
                statement, declared_schema, dialect_name, metadata, model, source, positions
            )


def _check_statement(
    statement: exp.Expr,
    declared_schema: dict[TableName, dict[ColumnName, ColumnTypeName]],
    dialect_name: str,
    metadata: ExprMetadataType,
    model: Model,
    source: SourceDoc,
    positions: Positions,
) -> None:
    
    # Step 2a: a probe qualification resolves every column to the source it reads from.
    try:
        resolved = resolve_columns_to_source_tables(statement, dialect_name)
    except OptimizeError as e:
        print(f"error: {model.relative_path}: {e}")
        return

    # A column that resolves to nothing is a user error - a typo, or a missing join. It is
    # reported here rather than being invented as a real column of some table below.
    if resolved.unresolvable_columns:
        for column in resolved.unresolvable_columns:
            span = positions.span_of(column)
            location = f":{span}" if span is not None else ""
            print(
                f"error: {model.relative_path}{location}: column '{column.name}' "
                "could not be resolved to any source"
            )
        return

    # Step 2b: match the resolved tables against the source declarations to fill in
    # declared type information.
    declared_types_per_table = get_declared_types_per_table(
        declared_schema, resolved.columns_per_table
    )
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
