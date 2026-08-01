from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from sqlrunner.sql_analysis.types import Cte, SqlAnalysisResult


def _qualified_name(table: exp.Table) -> str:
    parts = [part for part in (table.catalog, table.db, table.name) if part]
    return ".".join(parts)


def analyze_sql(sql: str, dialect: str | None = None) -> SqlAnalysisResult:
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except ParseError as e:
        return SqlAnalysisResult(ctes=[], external_sources=[], errors=[str(e)])

    ctes: list[Cte] = []
    external_names: set[str] = set()

    for statement in statements:

        if statement is None:
            continue

        # print(statement.sql(pretty=True, dialect=dialect))
        # print(statement.dump())
        print(repr(statement))

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

        print("done parsing statement")

    return SqlAnalysisResult(
        ctes=ctes,
        external_sources=sorted(external_names),
        errors=[],
    )


def analyze_file(path: Path, dialect: str | None = None) -> SqlAnalysisResult:
    return analyze_sql(Path(path).read_text(), dialect=dialect)
