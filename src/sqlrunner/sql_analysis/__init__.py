"""SQL analysis: column lineage across CTEs, derived tables and set operations.

`analyze_sql` returns, for one statement:

- every external source with the columns the statement attributes to it,
- the statement's final projection in SELECT order,
- flat fact lists (usage, predicates, joins, nullability, cardinality) keyed by column
  node, for schema resolution / relationship inference / constraint extraction.

## The pipeline

Each step's output is the next step's input. `_analyze_statement` runs steps 2-5 for one
statement; `analyze_sql` wraps them with step 1 and step 6.

| # | Step            | In                    | Out                 | Where          |
|---|-----------------|-----------------------|---------------------|----------------|
| 1 | parse, qualify  | SQL text              | statements          | this file      |
| 2 | build graph     | statement             | `RelationGraph`     | `relations.py` |
| 3 | build resolver  | `RelationGraph`       | `Resolver`          | `lineage.py`   |
| 4 | extract facts   | graph, resolver       | `Facts`             | `facts.py`     |
| 5 | assemble result | graph, resolver+facts | `SqlAnalysisResult` | `assemble.py`  |
| 6 | merge           | `[SqlAnalysisResult]` | `SqlAnalysisResult` | this file      |

Step 3 is the odd one out: `Resolver` is not a transformation but a lazy, memoizing
service that steps 4 and 5 both query. `*` cannot be expanded without a catalog, so column
references are resolved backwards on demand rather than up front - which is also why
`StarOverJoinAbort` escapes from inside steps 4 and 5. See `lineage.py`.
"""

from pathlib import Path

import sqlglot
from sqlglot.errors import ParseError, SqlglotError
from sqlglot.optimizer.qualify_tables import qualify_tables

from sqlrunner.sql_analysis.assemble import assemble
from sqlrunner.sql_analysis.facts import extract_facts
from sqlrunner.sql_analysis.resolver import (
    Resolver,
    StarOverJoinAbort,
    StarOverJoinBehavior,
)
from sqlrunner.sql_analysis.relations import build_graph
from sqlrunner.sql_analysis.types import (
    CONFIDENCE_RANK,
    SourceTable,
    SqlAnalysisResult,
)

DEFAULT_STAR_OVER_JOIN_BEHAVIOR: StarOverJoinBehavior = "guess"


def analyze_file(
    path: Path,
    dialect: str | None = None,
    star_over_join_behavior: StarOverJoinBehavior = DEFAULT_STAR_OVER_JOIN_BEHAVIOR,
) -> SqlAnalysisResult:
    return analyze_sql(
        Path(path).read_text(),
        dialect=dialect,
        star_over_join_behavior=star_over_join_behavior,
    )


def analyze_sql(
    sql: str,
    dialect: str | None = None,
    star_over_join_behavior: StarOverJoinBehavior = DEFAULT_STAR_OVER_JOIN_BEHAVIOR,
) -> SqlAnalysisResult:
    # Step 1: parse.
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except ParseError as e:
        return SqlAnalysisResult(errors=[str(e)])

    sqlAnalysisResults: list[SqlAnalysisResult] = []

    for statement in statements:
        if statement is None: 
            continue

        try:
            # qualify table names.
            qualified = qualify_tables(statement.copy())

            # Step 2: build the relation graph.
            graph = build_graph(qualified)

        except SqlglotError as e:
            return SqlAnalysisResult(errors=[str(e)])

        # Step 3: build the resolver. Nothing is resolved yet; steps 4 and 5 query it.
        resolver = Resolver(
            graph,
            star_over_join_behavior=star_over_join_behavior,
            dialect=dialect,
        )

        sqlAnalysisResult: SqlAnalysisResult

        try:
            # Step 4 - extract the facts
            facts = extract_facts(graph, resolver)

            # Step 5 - assemble the result
            sqlAnalysisResult = assemble(graph, resolver, facts)
        except StarOverJoinAbort as e:
            sqlAnalysisResult = SqlAnalysisResult(errors=[str(e)])

        sqlAnalysisResults.append(sqlAnalysisResult)

    # Step 6: merge and return.
    return _merge(sqlAnalysisResults)



# ---- step 6: merge -------------------------------------------------------------------


def _merge(results: list[SqlAnalysisResult]) -> SqlAnalysisResult:
    if not results:
        return SqlAnalysisResult()
    if len(results) == 1:
        return results[0]

    merged = SqlAnalysisResult(
        warnings=[
            "file contains multiple statements; projection reflects the last statement"
        ]
    )
    sources: dict[str, SourceTable] = {}

    for result in results:
        merged.relations.extend(result.relations)
        merged.usages.extend(result.usages)
        merged.predicates.extend(result.predicates)
        merged.joins.extend(result.joins)
        merged.nullability.extend(result.nullability)
        merged.cardinality.extend(result.cardinality)
        merged.ambiguities.extend(result.ambiguities)
        merged.errors.extend(result.errors)
        merged.warnings.extend(result.warnings)
        if result.projection:
            merged.projection = result.projection

        for source in result.sources:
            existing = sources.get(source.name)
            if existing is None:
                sources[source.name] = source.model_copy(deep=True)
                continue
            existing.star_expanded = existing.star_expanded or source.star_expanded
            by_name = {column.name: column for column in existing.columns}
            for column in source.columns:
                current = by_name.get(column.name)
                if current is None:
                    by_name[column.name] = column.model_copy()
                elif (
                    CONFIDENCE_RANK[column.confidence]
                    > CONFIDENCE_RANK[current.confidence]
                ):
                    current.confidence = column.confidence
            existing.columns = [by_name[name] for name in sorted(by_name)]

    merged.sources = [sources[name] for name in sorted(sources)]
    return merged
