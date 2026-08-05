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

from sqlr.source import Positions, SourceDoc
from sqlr.sql_analysis.assemble import assemble
from sqlr.sql_analysis.facts import extract_facts
from sqlr.sql_analysis.resolver import (
    Resolver,
    StarOverJoinAbort,
    StarOverJoinBehavior,
)
from sqlr.sql_analysis.relations import build_graph
from sqlr.sql_analysis.types import (
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
        path=Path(path),
    )


def analyze_sql(
    sql: str,
    dialect: str | None = None,
    star_over_join_behavior: StarOverJoinBehavior = DEFAULT_STAR_OVER_JOIN_BEHAVIOR,
    path: Path | None = None,
) -> SqlAnalysisResult:
    source = SourceDoc(path=path, text=sql)

    # One index for the whole document: sqlglot's character offsets are absolute, so they
    # stay valid across every statement in the file.
    positions = Positions(sql)

    # Step 1: parse.
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except ParseError as e:
        return SqlAnalysisResult(source=source, errors=[str(e)])

    sqlAnalysisResults: list[SqlAnalysisResult] = []

    for statement in statements:
        if statement is None:
            continue

        try:
            # qualify table names. `.copy()` and `qualify_tables` both preserve the
            # position metadata on leaf tokens, so spans survive this.
            qualified = qualify_tables(statement.copy())

            # Step 2: build the relation graph.
            graph = build_graph(qualified)

        except SqlglotError as e:
            return SqlAnalysisResult(source=source, errors=[str(e)])

        # Step 3: build the resolver. Nothing is resolved yet; steps 4 and 5 query it.
        resolver = Resolver(
            graph,
            star_over_join_behavior=star_over_join_behavior,
            dialect=dialect,
            positions=positions,
        )

        sqlAnalysisResult: SqlAnalysisResult

        try:
            # Step 4 - extract the facts
            facts = extract_facts(graph, resolver, positions)

            # Step 5 - assemble the result
            sqlAnalysisResult = assemble(graph, resolver, facts, positions, source)
        except StarOverJoinAbort as e:
            sqlAnalysisResult = SqlAnalysisResult(source=source, errors=[str(e)])

        sqlAnalysisResults.append(sqlAnalysisResult)

    # Step 6: merge and return.
    return _merge(sqlAnalysisResults, source)



# ---- step 6: merge -------------------------------------------------------------------


def _merge(
    results: list[SqlAnalysisResult], source: SourceDoc
) -> SqlAnalysisResult:
    if not results:
        return SqlAnalysisResult(source=source)
    if len(results) == 1:
        return results[0]

    merged = SqlAnalysisResult(
        source=source,
        warnings=[
            "file contains multiple statements; projection reflects the last statement"
        ],
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

        for table in result.sources:
            existing = sources.get(table.name)
            if existing is None:
                sources[table.name] = table.model_copy(deep=True)
                continue
            existing.star_expanded = existing.star_expanded or table.star_expanded
            by_name = {column.name: column for column in existing.columns}
            for column in table.columns:
                current = by_name.get(column.name)
                if current is None:
                    by_name[column.name] = column.model_copy(deep=True)
                    continue
                if (
                    CONFIDENCE_RANK[column.confidence]
                    > CONFIDENCE_RANK[current.confidence]
                ):
                    current.confidence = column.confidence
                # Offsets are absolute across the file, so references from a later
                # statement are directly comparable with earlier ones.
                current.references.extend(
                    span for span in column.references if span not in current.references
                )
            existing.columns = [by_name[name] for name in sorted(by_name)]

    merged.sources = [sources[name] for name in sorted(sources)]
    return merged
