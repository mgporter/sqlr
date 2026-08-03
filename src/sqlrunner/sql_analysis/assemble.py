"""Step 5 of the pipeline: assemble the public result.

    (RelationGraph, Resolver, Facts) -> SqlAnalysisResult

Three things are built here, all by querying the resolver:

- `relations`: every non-table relation with its outputs resolved to origins,
- `projection`: the statement's final SELECT list, with `*` expanded where possible,
- `sources`: external tables, with every column the statement attributed to them.

`sources` is derived from the union of all origins seen anywhere - the facts' `observed`
list plus the origins on relations and projection - so a column mentioned only in a WHERE
clause still shows up on its table.

See `__init__.py` for the pipeline as a whole.
"""

from __future__ import annotations

from sqlrunner.sql_analysis.facts import Facts
from sqlrunner.sql_analysis.resolver import Resolver
from sqlrunner.sql_analysis.relations import OutputCol, RelationGraph
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


def assemble(
    graph: RelationGraph, resolver: Resolver, facts: Facts
) -> SqlAnalysisResult:
    """Build the result. May raise `StarOverJoinAbort` via lazy resolution."""
    relations = _build_relations(graph, resolver)
    projection, projection_warnings = _build_projection(graph, resolver)

    origins: list[ColumnOrigin] = list(facts.observed)
    for relation in relations:
        for output in relation.outputs:
            origins.extend(output.origins)
    for projected in projection:
        origins.extend(projected.origins)

    return SqlAnalysisResult(
        relations=relations,
        sources=_build_sources(graph, origins),
        projection=projection,
        usages=facts.usages,
        predicates=facts.predicates,
        joins=facts.joins,
        nullability=facts.nullability,
        cardinality=facts.cardinality,
        ambiguities=resolver.ambiguities,
        warnings=graph.warnings + projection_warnings,
    )


# ---- relations -----------------------------------------------------------------------


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
                    _output_column(resolver, ref, output) for output in info.outputs
                ],
                star_sources=list(info.star_sources),
            )
        )
    return relations


def _output_column(
    resolver: Resolver, ref: RelationRef, output: OutputCol
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


# ---- projection ----------------------------------------------------------------------


def _build_projection(
    graph: RelationGraph, resolver: Resolver
) -> tuple[list[ProjectedColumn], list[str]]:
    info = graph.get(graph.root)
    if info is None:
        return [], []

    warnings: list[str] = []
    expanded = graph.expand_outputs(graph.root)

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


# ---- sources -------------------------------------------------------------------------


def _build_sources(
    graph: RelationGraph, origins: list[ColumnOrigin]
) -> list[SourceTable]:
    columns: dict[str, dict[str, Confidence]] = {}
    for origin in origins:
        if origin.node.relation.kind != "table":
            continue
        table = columns.setdefault(origin.node.relation.name, {})
        current = table.get(origin.node.column)
        if (
            current is None
            or CONFIDENCE_RANK[origin.confidence] > CONFIDENCE_RANK[current]
        ):
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
