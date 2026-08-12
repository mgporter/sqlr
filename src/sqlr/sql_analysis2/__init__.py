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


def table_columns(scopes: list[Scope]) -> dict[TableName, set[ColumnName]]:
    """Columns read straight off a real table, per table. Runs before qualify.

    The `isinstance(source, exp.Table)` filter is what stops CTE-internal names from
    being invented as columns of a real table.
    """
    out: dict[TableName, set[ColumnName]] = {}
    for scope in scopes:
        sources = scope.sources
        # Only take real tables, not CTEs
        tables: dict[TableName, exp.Table] = {n: s for n, s in sources.items() if isinstance(s, exp.Table)}
        if not tables:
            continue  # scope reads CTEs only; nothing to declare
        for column in scope.columns:
            # If the scope contains a single table (len(tables) == 1) and there are 
            # no other CTEs (len(sources) is also 1), then we know that all of the columns belong to that table.
            # Otherwise, we try to get the column's alias (column.table)
            name = column.table or (next(iter(tables)) if len(tables) == 1 and len(sources) == 1 else None)
            if name in tables:
                out.setdefault(tables[name].name.lower(), set()).add(column.name.lower())
    return out


def get_declared_types_per_table(declared: dict[TableName, dict[ColumnName, ColumnTypeName]], scopes: list[Scope]) -> \
    dict[TableName, dict[ColumnName, ColumnTypeName]]:
    """Complete the schema's *column set* so `qualify` does not raise.

    Declared types where the user wrote them, UNKNOWN everywhere else. Not optional:
    `qualify` validates that every column resolves, and every built-in escape
    (`allow_partial_qualification`, `infer_schema`) still raises. See plan step 2.

    TODO (plan step 2, "the typo trap"): a misspelled column is invented here as a real
    one. The unresolvable-column check has to run before this, and does not exist yet.
    """

    column_names_per_table = table_columns(scopes)

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
    
    # Step 2: traverse_scope to get columns, resolve them to tables, and then
    # match the tables against the source declarations to fill in declared type information.
    declared_types_per_table = get_declared_types_per_table(declared_schema, traverse_scope(statement))
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
    print(declared_types_per_table)
    mapped_schema = ensure_schema(cast("dict[str, object]", declared_types_per_table), dialect=dialect_name)
    print(mapped_schema.__dict__)

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
