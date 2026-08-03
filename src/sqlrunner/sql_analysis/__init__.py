"""SQL analysis: column lineage across CTEs, derived tables and set operations.

`analyze_sql` returns, for one statement:

- every external source with the columns the statement attributes to it,
- the statement's final projection in SELECT order,
- flat fact lists (usage, predicates, joins, nullability, cardinality) keyed by column
  node, for schema resolution / relationship inference / constraint extraction.

`*` cannot be expanded without a catalog, so column references are resolved backwards on
demand. See `lineage.py`.
"""

from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, SqlglotError
from sqlglot.optimizer.qualify_tables import qualify_tables

from sqlrunner.sql_analysis.facts import FactExtractor
from sqlrunner.sql_analysis.lineage import (
    Resolver,
    StarOverJoinAbort,
    StarOverJoinBehavior,
)
from sqlrunner.sql_analysis.relations import OutputCol, RelationGraph, build_graph
from sqlrunner.sql_analysis.types import (
    CONFIDENCE_RANK,
    ColumnOrigin,
    Confidence,
    OutputColumn,
    ProjectedColumn,
    Relation,
    RelationRef,
    SourceColumn,
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
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except ParseError as e:
        return SqlAnalysisResult(errors=[str(e)])

    results = [
        _analyze_statement(statement, dialect, star_over_join_behavior)
        for statement in statements
        if statement is not None
    ]
    return _merge(results)


def _analyze_statement(
    statement: exp.Expr,
    dialect: str | None,
    star_over_join_behavior: StarOverJoinBehavior,
) -> SqlAnalysisResult:
    try:
        qualified = qualify_tables(statement.copy())
        graph = build_graph(qualified)
    except SqlglotError as e:
        return SqlAnalysisResult(errors=[str(e)])

    resolver = Resolver(
        graph,
        star_over_join_behavior=star_over_join_behavior,
        dialect=dialect
    )
    extractor = FactExtractor(graph, resolver)

    try:
        extractor.run()
        relations = _build_relations(graph, resolver)
        projection, projection_warnings = _build_projection(graph, resolver)
    except StarOverJoinAbort as e:
        return SqlAnalysisResult(errors=[str(e)])

    origins: list[ColumnOrigin] = list(extractor.observed)
    for relation in relations:
        for output in relation.outputs:
            origins.extend(output.origins)
    for projected in projection:
        origins.extend(projected.origins)

    return SqlAnalysisResult(
        relations=relations,
        sources=_build_sources(graph, origins),
        projection=projection,
        usages=extractor.usages,
        predicates=extractor.predicates,
        joins=extractor.joins,
        nullability=extractor.nullability,
        cardinality=extractor.cardinality,
        ambiguities=resolver.ambiguities,
        warnings=graph.warnings + projection_warnings,
    )


# ---- assembly ----------------------------------------------------------------------


def _output_column(
    graph: RelationGraph, resolver: Resolver, ref: RelationRef, output: OutputCol
) -> OutputColumn:
    origins: list[ColumnOrigin] = []
    if output.kind != "star" and output.name is not None:
        origins = resolver.resolve(ref, output.name).origins

    return OutputColumn(
        name=output.name,
        ordinal=output.ordinal,
        kind=output.kind,  # type: ignore[arg-type]
        function=output.function,
        origins=origins,
        star_sources=list(output.star_sources),
    )


def _build_relations(graph: RelationGraph, resolver: Resolver) -> list[Relation]:
    relations: list[Relation] = []
    for ref in graph.order:
        info = graph.relations[ref]
        if info.is_table:
            continue
        depends_on = [binding.ref for binding in info.bindings] + list(info.branches)
        relations.append(
            Relation(
                ref=ref,
                is_set_operation=info.is_setop,
                depends_on=depends_on,
                outputs=[
                    _output_column(graph, resolver, ref, output)
                    for output in info.outputs
                ],
                star_sources=list(info.star_sources),
            )
        )
    return relations


def _expand_outputs(
    graph: RelationGraph, ref: RelationRef, seen: frozenset[RelationRef]
) -> list[tuple[RelationRef, OutputCol]] | None:
    """Flatten a relation's SELECT list, splicing in `*` sources recursively.

    Returns None when a `*` reads a relation whose column list is unknown - an external
    table, or a set operation with a starred branch.
    """
    info = graph.get(ref)
    if info is None or info.is_table or ref in seen:
        return None
    if not info.outputs:
        return None

    seen = seen | {ref}
    expanded: list[tuple[RelationRef, OutputCol]] = []

    for output in info.outputs:
        if output.kind != "star":
            expanded.append((ref, output))
            continue
        if not output.star_sources:
            return None
        for source in output.star_sources:
            nested = _expand_outputs(graph, source, seen)
            if nested is None:
                return None
            expanded.extend(nested)

    return expanded


def _build_projection(
    graph: RelationGraph, resolver: Resolver
) -> tuple[list[ProjectedColumn], list[str]]:
    info = graph.get(graph.root)
    if info is None:
        return [], []

    warnings: list[str] = []
    expanded = _expand_outputs(graph, graph.root, frozenset())

    if expanded is None:
        warnings.append(
            "final projection contains a * over a relation with an unknown column list; "
            "projected columns are not exhaustive"
        )
        expanded = [(graph.root, output) for output in info.outputs]

    projection: list[ProjectedColumn] = []
    for ordinal, (owner, output) in enumerate(expanded):
        origins: list[ColumnOrigin] = []
        if output.kind != "star" and output.name is not None:
            origins = resolver.resolve(owner, output.name).origins
        projection.append(
            ProjectedColumn(
                name=output.name,
                ordinal=ordinal,
                kind=output.kind,  # type: ignore[arg-type]
                function=output.function,
                origins=origins,
                star_of=list(output.star_sources),
            )
        )
    return projection, warnings


def _build_sources(
    graph: RelationGraph, origins: list[ColumnOrigin]
) -> list[SourceTable]:
    columns: dict[str, dict[str, Confidence]] = {}
    for origin in origins:
        if origin.node.relation.kind != "table":
            continue
        table = columns.setdefault(origin.node.relation.name, {})
        current = table.get(origin.node.column)
        if current is None or CONFIDENCE_RANK[origin.confidence] > CONFIDENCE_RANK[current]:
            table[origin.node.column] = origin.confidence

    starred = {
        ref.name
        for info in graph.relations.values()
        for ref in info.star_sources
        if ref.kind == "table"
    }

    table_names = sorted(
        {ref.name for ref in graph.relations if ref.kind == "table"} | set(columns)
    )
    return [
        SourceTable(
            name=name,
            columns=[
                SourceColumn(name=column, confidence=confidence)
                for column, confidence in sorted(columns.get(name, {}).items())
            ],
            star_expanded=name in starred,
        )
        for name in table_names
    ]


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
