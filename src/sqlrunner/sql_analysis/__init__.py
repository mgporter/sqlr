from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.expressions.core import Expr
from sqlglot.optimizer.scope import traverse_scope

from sqlrunner.sql_analysis.types import ColumnReference, ColumnUsage, Cte, SqlAnalysisResult

DATE_FUNCTION_CLASSES = {
    "DateTrunc",
    "TimestampTrunc",
    "DatetimeTrunc",
    "Extract",
    "DateAdd",
    "DateSub",
    "DateDiff",
    "DatetimeAdd",
    "DatetimeSub",
    "DatetimeDiff",
    "Year",
    "Month",
    "Day",
    "Quarter",
    "Week",
    "StrToDate",
    "StrToTime",
    "TsOrDsToDate",
}

BOOLEAN_CONTEXT_PARENTS = (exp.Not, exp.Where, exp.And, exp.Or, exp.If)
COMPARISON_CLASSES = (exp.GT, exp.LT, exp.GTE, exp.LTE, exp.EQ, exp.NEQ)
ARITHMETIC_CLASSES = (exp.Add, exp.Sub, exp.Mul, exp.Div)


def _qualified_name(table: exp.Table) -> str:
    parts = [part for part in (table.catalog, table.db, table.name) if part]
    return ".".join(parts)


def _number_shape(literals: list[exp.Literal]) -> str:
    if any("." in lit.this or "e" in lit.this.lower() for lit in literals):
        return "float"
    return "int"


def _classify_usage(column: exp.Column) -> list[tuple[str, str | None]]:
    parent = column.parent
    if parent is None:
        return []

    if isinstance(parent, exp.Cast) and parent.this is column:
        return [("cast", parent.to.this.name)]

    if isinstance(parent, COMPARISON_CLASSES):
        other = parent.right if parent.left is column else parent.left
        if isinstance(other, exp.Literal) and other.is_number:
            return [("compared_to_number", _number_shape([other]))]
        return []

    if isinstance(parent, exp.In) and parent.this is column:
        literals = parent.expressions
        if literals and all(isinstance(e, exp.Literal) for e in literals):
            if all(e.is_string for e in literals):
                return [("in_list_strings", None)]
            if all(e.is_number for e in literals):
                return [("in_list_numbers", _number_shape(literals))]
        return []

    if isinstance(parent, (exp.Like, exp.ILike)):
        return [("like", None)]

    if isinstance(parent, ARITHMETIC_CLASSES):
        return [("arithmetic", None)]

    if isinstance(parent, BOOLEAN_CONTEXT_PARENTS) and parent.this is column:
        return [("boolean_context", None)]

    if type(parent).__name__ in DATE_FUNCTION_CLASSES:
        return [("date_function", type(parent).__name__)]

    return []


def _analyze_columns(
    statement: Expr,
) -> tuple[list[ColumnReference], list[ColumnUsage]]:
    references: set[tuple[str, str]] = set()
    usages: list[ColumnUsage] = []

    for scope in traverse_scope(statement): # one scope per select statement

        # scope.sources contains the tables in that scope, AND all of the CTEs that are visible
        # in that scope (regardless of whether the cte is used or not).

        # print("SCOPE", repr(scope.expression))
        print("SCOPE")
        # for alias, source in scope.sources.items():
        #     print("SOURCE")
        #     print(f"  {alias} -> {repr(source)}")

        for col in scope.columns:
            print(f"COLUMN  {col.table}.{col.name} {repr(col.expression)}")

        external_aliases = {
            alias: _qualified_name(source)
            for alias, source in scope.sources.items()
            if isinstance(source, exp.Table)
        }

        single_external = None
        if len(scope.sources) == 1:
            (only_source,) = scope.sources.values()
            if isinstance(only_source, exp.Table):
                single_external = _qualified_name(only_source)

        for column in scope.columns:
            print(column.table)
            if column.table:
                table_name = external_aliases.get(column.table)
            else:
                table_name = single_external
            if table_name is None:
                continue

            references.add((table_name, column.name))

            for kind, detail in _classify_usage(column):
                usages.append(
                    ColumnUsage(table=table_name, column=column.name, kind=kind, detail=detail)  # type: ignore[arg-type]
                )

    sorted_references = [
        ColumnReference(table=table, column=column)
        for table, column in sorted(references)
    ]
    return sorted_references, usages


def analyze_sql(sql: str, dialect: str | None = None) -> SqlAnalysisResult:
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except ParseError as e:
        return SqlAnalysisResult(
            ctes=[], external_sources=[], column_references=[], column_usages=[], errors=[str(e)]
        )

    ctes: list[Cte] = []
    external_names: set[str] = set()
    column_references: list[ColumnReference] = []
    column_usages: list[ColumnUsage] = []

    for statement in statements:
        if statement is None:
            continue

        local_cte_names = {cte_expr.alias for cte_expr in statement.find_all(exp.CTE)}

        for cte_expr in statement.find_all(exp.CTE):
            depends_on = {
                table.name
                for table in cte_expr.this.find_all(exp.Table)
                if table.name in local_cte_names
            }
            ctes.append(Cte(name=cte_expr.alias, depends_on=sorted(depends_on)))

        for table in statement.find_all(exp.Table):
            if table.name in local_cte_names:
                continue
            external_names.add(_qualified_name(table))

        references, usages = _analyze_columns(statement)
        column_references.extend(references)
        column_usages.extend(usages)

    return SqlAnalysisResult(
        ctes=ctes,
        external_sources=sorted(external_names),
        column_references=column_references,
        column_usages=column_usages,
        errors=[],
    )


def analyze_file(path: Path, dialect: str | None = None) -> SqlAnalysisResult:
    return analyze_sql(Path(path).read_text(), dialect=dialect)
