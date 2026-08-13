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
from sqlglot.optimizer.scope import traverse_scope
from sqlglot.schema import ensure_schema
from sqlglot.typing import ExprMetadataType

from sqlr.config.types import SqlrConfig
from sqlr.declared.types import DeclaredSchemas
from sqlr.selection.types import Model
from sqlr.sql_analysis2.annotate import expression_metadata, func_args
from sqlr.sql_analysis2.reporting import (
    findings_for_columns_declared_as_scalar_but_read_as_structured,
    findings_for_columns_read_with_unsupported_dot_notation,
    findings_for_unresolvable_columns,
    print_findings,
)
from sqlr.sql_analysis2.sourcedoc import Positions, SourceDoc, SourceSpan
from sqlr.sql_analysis2.types import (
    ColumnName,
    ColumnTypeName,
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


def resolve_columns_to_source_tables(
    statement: exp.Expr,
    dialect_name: str,
    allow_unresolvable_aliases_as_structured_columns: bool | None = None,
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
    """
    if allow_unresolvable_aliases_as_structured_columns is None:
        allow_unresolvable_aliases_as_structured_columns = (
            dialect_parses_unresolvable_aliases_as_json_columns(dialect_name)
        )

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

    for scope in traverse_scope(probe):
        for column in scope.columns:
            if not column.table:
                # Still bare after qualification: no source can own it.
                unresolvable_columns.append(column)
                continue

            source = scope.sources.get(column.table)
            if source is None:
                # A qualifier naming no source in this scope - a mistyped table alias.
                # See the docstring: this only survives when the scope has more than one
                # source, because a lone source absorbs the name as a struct read.
                unresolvable_columns.append(column)
                continue

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

    # print()
    # print("columns_per_table", columns_per_table)
    # print()
    # print("unresolvable_columns", unresolvable_columns)
    # print()
    # print(
    #     "columns_read_with_unsupported_dot_notation",
    #     columns_read_with_unsupported_dot_notation,
    # )

    return ResolvedColumns(
        columns_per_table=columns_per_table,
        unresolvable_columns=unresolvable_columns,
        columns_read_with_unsupported_dot_notation=columns_read_with_unsupported_dot_notation,
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
    # This is necessary to get the sources for columns without an alias but which MUST
    # come from a source because no other source has that column.
    # E.g.: with t2 as (select distinct id from mytable) 
    #       select a from t1 join t2 on t1.id = t2.id;
    # Here, 'a' must come from either t1, since t2 only has 'id'.
    try:
        resolved = resolve_columns_to_source_tables(statement, dialect_name)
    except OptimizeError as e:
        print(f"error: {model.relative_path}: {e}")
        return

    # Everything wrong with the resolution is reported before step 3. `qualify` describes
    # the tree it rewrote rather than the SQL that was written, so its message for the
    # same mistake is strictly harder to act on than the one built here.
    findings = [
        *findings_for_unresolvable_columns(resolved.unresolvable_columns, positions),
        *findings_for_columns_read_with_unsupported_dot_notation(
            resolved.columns_read_with_unsupported_dot_notation, positions, dialect_name
        ),
    ]

    # Step 2b: match the resolved tables against the source declarations to fill in
    # declared type information.
    declared_types_per_table = get_declared_types_per_table(
        declared_schema, resolved.columns_per_table
    )
    findings += findings_for_columns_declared_as_scalar_but_read_as_structured(
        declared_schema, resolved.columns_per_table, positions
    )
    print_findings(findings, str(model.relative_path))

    # An unresolvable column is the one finding that has to stop the statement: it belongs
    # to no table, so no fabricated schema can cover it and `qualify` raises. A dotted read
    # is harvested despite its finding, so the rest of the statement is still checked.
    if resolved.unresolvable_columns:
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
