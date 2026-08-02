"""Turn analysis facts into a per-table schema good enough to generate fixtures from.

Type evidence is weighted; the strongest wins. Columns joined by equality are unified
into one value domain, so `address.person_id` inherits `person.id`'s type and a generator
can produce overlapping values instead of a join that returns nothing.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from sqlrunner.schema_resolution.types import (
    ColumnSchema,
    ProjectedColumnSchema,
    ResolvedType,
    StatementSchema,
    TableSchema,
    TypeSource,
    ValueConstraint,
)
from sqlrunner.sql_analysis.types import (
    ColumnNode,
    NullabilityFact,
    PredicateFact,
    RelationRef,
    SqlAnalysisResult,
    UsageFact,
)

CAST_TYPE_MAP: dict[str, ResolvedType] = {
    "INT": "integer",
    "INTEGER": "integer",
    "BIGINT": "integer",
    "SMALLINT": "integer",
    "TINYINT": "integer",
    "DECIMAL": "decimal",
    "NUMERIC": "decimal",
    "FLOAT": "decimal",
    "DOUBLE": "decimal",
    "REAL": "decimal",
    "VARCHAR": "string",
    "TEXT": "string",
    "CHAR": "string",
    "STRING": "string",
    "NVARCHAR": "string",
    "DATE": "date",
    "TIMESTAMP": "timestamp",
    "TIMESTAMPTZ": "timestamp",
    "DATETIME": "timestamp",
    "BOOLEAN": "boolean",
    "BOOL": "boolean",
}

# Higher confidence wins when a column has conflicting evidence.
USAGE_TYPE_WEIGHT: dict[str, int] = {
    "cast": 100,
    "boolean_context": 90,
    "date_function": 70,
    "in_list_strings": 60,
    "like": 60,
    "compared_to_number": 50,
    "in_list_numbers": 50,
    "arithmetic": 10,
}

NAME_PATTERN_WEIGHT = 5

# Type of a column produced by an expression, keyed by sqlglot class name.
FUNCTION_TYPE_MAP: dict[str, ResolvedType] = {
    "RowNumber": "integer",
    "Rank": "integer",
    "DenseRank": "integer",
    "Ntile": "integer",
    "Count": "integer",
    "Sum": "decimal",
    "Avg": "decimal",
    "Stddev": "decimal",
    "Variance": "decimal",
    "Add": "decimal",
    "Sub": "decimal",
    "Mul": "decimal",
    "Div": "decimal",
    "Concat": "string",
    "DPipe": "string",
    "Lower": "string",
    "Upper": "string",
    "Trim": "string",
    "Substring": "string",
    "CurrentTimestamp": "timestamp",
    "CurrentDate": "date",
    "DateTrunc": "timestamp",
    "TimestampTrunc": "timestamp",
}

NULLABILITY_WEIGHT: dict[str, int] = {
    "is_null_predicate": 3,
    "is_not_null_predicate": 3,
    "coalesce_argument": 2,
    "inner_join_key": 1,
    # `outer_join_padded` describes the join result, not the stored column, so it does
    # not decide source nullability. It stays available on the analysis result.
}


@dataclass
class _TypeEvidence:
    resolved_type: ResolvedType
    weight: int
    source: TypeSource
    evidence: str | None = None


def _type_from_usage(usage: UsageFact) -> tuple[ResolvedType, int] | None:
    weight = USAGE_TYPE_WEIGHT[usage.kind]

    if usage.kind == "cast":
        resolved = CAST_TYPE_MAP.get(usage.detail or "")
        return (resolved, weight) if resolved else None
    if usage.kind == "boolean_context":
        return ("boolean", weight)
    if usage.kind == "date_function":
        return ("date", weight)
    if usage.kind in ("in_list_strings", "like"):
        return ("string", weight)
    if usage.kind in ("compared_to_number", "in_list_numbers"):
        return ("integer" if usage.detail == "int" else "decimal", weight)
    if usage.kind == "arithmetic":
        return ("decimal", weight)
    return None


def _type_from_name(column: str) -> tuple[ResolvedType, str] | None:
    lower = column.lower()
    if lower.startswith("is_"):
        return ("boolean", "is_*")
    if lower.endswith("_id"):
        return ("integer", "*_id")
    if lower.endswith("_at"):
        return ("timestamp", "*_at")
    return None


class _JoinGroups:
    """Union-find over column nodes linked by equi-joins."""

    def __init__(self) -> None:
        self._parent: dict[ColumnNode, ColumnNode] = {}

    def add(self, node: ColumnNode) -> None:
        self._parent.setdefault(node, node)

    def find(self, node: ColumnNode) -> ColumnNode:
        self.add(node)
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[node] != root:
            self._parent[node], node = root, self._parent[node]
        return root

    def union(self, left: ColumnNode, right: ColumnNode) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root

    def groups(self) -> list[list[ColumnNode]]:
        members: dict[ColumnNode, list[ColumnNode]] = defaultdict(list)
        for node in self._parent:
            members[self.find(node)].append(node)
        return [
            sorted(group, key=str) for group in members.values() if len(group) > 1
        ]


def resolve_schema(analysis: SqlAnalysisResult) -> StatementSchema:
    usages_by_node: dict[ColumnNode, list[UsageFact]] = defaultdict(list)
    for usage in analysis.usages:
        usages_by_node[usage.node].append(usage)

    predicates_by_node: dict[ColumnNode, list[PredicateFact]] = defaultdict(list)
    for predicate in analysis.predicates:
        predicates_by_node[predicate.node].append(predicate)

    nullability_by_node: dict[ColumnNode, list[NullabilityFact]] = defaultdict(list)
    for fact in analysis.nullability:
        nullability_by_node[fact.node].append(fact)

    derived_functions = {
        ColumnNode(relation=relation.ref, column=output.name): output.function
        for relation in analysis.relations
        for output in relation.outputs
        if output.kind == "derived" and output.name and output.function
    }

    nodes = _all_nodes(analysis, usages_by_node, predicates_by_node, derived_functions)
    evidence = {
        node: _initial_evidence(node, usages_by_node.get(node, []), derived_functions)
        for node in nodes
    }

    join_groups = _JoinGroups()
    for join in analysis.joins:
        join_groups.union(join.left, join.right)
    groups = join_groups.groups()

    _unify_join_groups(groups, evidence)

    group_index = {
        node: index for index, group in enumerate(groups) for node in group
    }

    tables = _build_tables(
        analysis,
        evidence,
        predicates_by_node,
        nullability_by_node,
        group_index,
    )
    projection = _build_projection(analysis, evidence)

    return StatementSchema(
        tables=tables,
        projection=projection,
        join_groups=[[str(node) for node in group] for group in groups],
    )


def _all_nodes(
    analysis: SqlAnalysisResult,
    usages_by_node: dict[ColumnNode, list[UsageFact]],
    predicates_by_node: dict[ColumnNode, list[PredicateFact]],
    derived_functions: Mapping[ColumnNode, str],
) -> set[ColumnNode]:
    nodes: set[ColumnNode] = set(usages_by_node) | set(predicates_by_node)
    nodes |= set(derived_functions)
    for source in analysis.sources:
        for column in source.columns:
            nodes.add(
                ColumnNode(
                    relation=_table_ref(source.name), column=column.name
                )
            )
    for join in analysis.joins:
        nodes.add(join.left)
        nodes.add(join.right)
    for projected in analysis.projection:
        for origin in projected.origins:
            nodes.add(origin.node)
    return nodes


def _table_ref(name: str) -> RelationRef:
    return RelationRef(kind="table", name=name)


def _initial_evidence(
    node: ColumnNode,
    usages: list[UsageFact],
    derived_functions: Mapping[ColumnNode, str],
) -> _TypeEvidence:
    best: _TypeEvidence | None = None
    for usage in usages:
        candidate = _type_from_usage(usage)
        if candidate is None:
            continue
        resolved, weight = candidate
        if best is None or weight > best.weight:
            best = _TypeEvidence(resolved, weight, "usage", usage.kind)
    if best is not None:
        return best

    function = derived_functions.get(node)
    if function is not None:
        resolved_from_function = FUNCTION_TYPE_MAP.get(function)
        if resolved_from_function is not None:
            return _TypeEvidence(resolved_from_function, 80, "expression", function)

    name_match = _type_from_name(node.column)
    if name_match is not None:
        resolved, pattern = name_match
        return _TypeEvidence(resolved, NAME_PATTERN_WEIGHT, "name_pattern", pattern)

    return _TypeEvidence("unknown", 0, "unknown", None)


def _unify_join_groups(
    groups: list[list[ColumnNode]], evidence: dict[ColumnNode, _TypeEvidence]
) -> None:
    for group in groups:
        known = [
            (evidence[node].weight, node)
            for node in group
            if node in evidence and evidence[node].resolved_type != "unknown"
        ]
        if not known:
            continue
        best_weight, best_node = max(known, key=lambda item: item[0])
        best = evidence[best_node]
        for node in group:
            current = evidence.get(node)
            if current is None or current.weight < best_weight:
                evidence[node] = _TypeEvidence(
                    best.resolved_type,
                    best_weight,
                    "join_group",
                    f"joined to {best_node}",
                )


def _build_tables(
    analysis: SqlAnalysisResult,
    evidence: dict[ColumnNode, _TypeEvidence],
    predicates_by_node: dict[ColumnNode, list[PredicateFact]],
    nullability_by_node: dict[ColumnNode, list[NullabilityFact]],
    group_index: dict[ColumnNode, int],
) -> list[TableSchema]:
    tables: list[TableSchema] = []
    for source in analysis.sources:
        ref = _table_ref(source.name)
        columns: list[ColumnSchema] = []
        for column in source.columns:
            node = ColumnNode(relation=ref, column=column.name)
            resolved = evidence.get(node) or _TypeEvidence("unknown", 0, "unknown", None)
            columns.append(
                ColumnSchema(
                    name=column.name,
                    resolved_type=resolved.resolved_type,
                    source=resolved.source,
                    evidence=resolved.evidence,
                    confidence=column.confidence,
                    nullable=_nullability(nullability_by_node.get(node, [])),
                    constraints=_constraints(predicates_by_node.get(node, [])),
                    join_group=group_index.get(node),
                )
            )
        tables.append(
            TableSchema(
                name=source.name,
                columns=columns,
                star_expanded=source.star_expanded,
            )
        )
    return tables


def _nullability(facts: list[NullabilityFact]) -> bool | None:
    best_weight = 0
    result: bool | None = None
    for fact in facts:
        weight = NULLABILITY_WEIGHT.get(fact.reason, 0)
        if weight > best_weight:
            best_weight = weight
            result = fact.nullable
    return result


def _constraints(facts: list[PredicateFact]) -> list[ValueConstraint]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    constraints: list[ValueConstraint] = []
    for fact in facts:
        key = (fact.operator, tuple(fact.values))
        if key in seen:
            continue
        seen.add(key)
        constraints.append(ValueConstraint(operator=fact.operator, values=fact.values))
    return constraints


def _build_projection(
    analysis: SqlAnalysisResult, evidence: dict[ColumnNode, _TypeEvidence]
) -> list[ProjectedColumnSchema]:
    projection: list[ProjectedColumnSchema] = []
    for projected in analysis.projection:
        resolved_type: ResolvedType = "unknown"
        source: TypeSource = "unknown"

        if projected.kind == "derived" and projected.function:
            from_function = FUNCTION_TYPE_MAP.get(projected.function)
            if from_function is not None:
                resolved_type, source = from_function, "expression"

        if resolved_type == "unknown":
            for origin in projected.origins:
                candidate = evidence.get(origin.node)
                if candidate is not None and candidate.resolved_type != "unknown":
                    resolved_type, source = candidate.resolved_type, candidate.source
                    break

        projection.append(
            ProjectedColumnSchema(
                name=projected.name,
                ordinal=projected.ordinal,
                resolved_type=resolved_type,
                source=source,
                origins=[str(origin.node) for origin in projected.origins],
            )
        )
    return projection
