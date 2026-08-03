"""Turn analysis facts into a per-table schema good enough to generate fixtures from.

Type evidence is weighted; the strongest wins. Columns joined by equality are unified
into one value domain, so `address.person_id` inherits `person.id`'s type and a generator
can produce overlapping values instead of a join that returns nothing.

Evidence that does not name a type resolves to a family rather than a guess - see
`ResolvedType`. Two pieces of evidence of equal weight are widened together, which is also
how a join group with conflicting member types is unified.
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
    LiteralKind,
    NullabilityFact,
    OutputColumn,
    PredicateFact,
    ProjectedColumn,
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
    "FLOAT": "float",
    "DOUBLE": "float",
    "REAL": "float",
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

# Higher confidence wins when a column has conflicting evidence. `cast` is absent on
# purpose: `cast(x as decimal)` fixes the type of the *result*, not of `x`, which could
# have been anything castable. It types the derived column instead, via `cast_type`.
USAGE_TYPE_WEIGHT: dict[str, int] = {
    "boolean_context": 90,
    "date_function": 70,
    "coalesce_default": 65,
    "in_list_strings": 60,
    "like": 60,
    "compared_to_number": 50,
    "in_list_numbers": 50,
    "arithmetic": 10,
}

NAME_PATTERN_WEIGHT = 5
CAST_WEIGHT = 100
EXPRESSION_WEIGHT = 80
LITERAL_ARGUMENT_WEIGHT = 55

# Type of a column produced by an expression, keyed by sqlglot class name. Only entries
# with a return type the dialect fixes are concrete; arithmetic follows its operands.
FUNCTION_TYPE_MAP: dict[str, ResolvedType] = {
    "RowNumber": "integer",
    "Rank": "integer",
    "DenseRank": "integer",
    "Ntile": "integer",
    "Count": "integer",
    "DateDiff": "integer",
    "DatetimeDiff": "integer",
    "TimestampDiff": "integer",
    "Sum": "numeric",
    "Avg": "numeric",
    "Stddev": "numeric",
    "Variance": "numeric",
    "Add": "numeric",
    "Sub": "numeric",
    "Mul": "numeric",
    "Div": "numeric",
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

# Functions that hand back the type of their own arguments, so a literal argument types
# the column beside it. `Case` is absent: its branch values are not direct arguments.
TYPE_TRANSPARENT_FUNCTIONS = {"Coalesce", "Nullif", "Greatest", "Least", "If"}

# The concrete types each `ResolvedType` covers. Widening picks the narrowest entry whose
# cover is a superset of both inputs, so `integer` and `decimal` meet at `number`.
TYPE_COVER: dict[ResolvedType, frozenset[str]] = {
    "integer": frozenset({"integer"}),
    "decimal": frozenset({"decimal"}),
    "float": frozenset({"float"}),
    "number": frozenset({"integer", "decimal"}),
    "numeric": frozenset({"integer", "decimal", "float"}),
    "string": frozenset({"string"}),
    "date": frozenset({"date"}),
    "timestamp": frozenset({"timestamp"}),
    "boolean": frozenset({"boolean"}),
    "unknown": frozenset(),
}

# Narrowest first, so the first superset found is the least upper bound.
WIDENING_ORDER: list[ResolvedType] = [
    "integer",
    "decimal",
    "float",
    "string",
    "date",
    "timestamp",
    "boolean",
    "number",
    "numeric",
]

# A literal proves its neighbour holds a number, never which kind of number.
LITERAL_KIND_TYPE: dict[LiteralKind, ResolvedType] = {
    "int": "numeric",
    "float": "numeric",
    "string": "string",
}


def widen(left: ResolvedType, right: ResolvedType) -> ResolvedType:
    """The narrowest type covering both. `unknown` when nothing does."""
    if left == right:
        return left
    if left == "unknown" or right == "unknown":
        return "unknown"
    covered = TYPE_COVER[left] | TYPE_COVER[right]
    for candidate in WIDENING_ORDER:
        if covered <= TYPE_COVER[candidate]:
            return candidate
    return "unknown"


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
    weight = USAGE_TYPE_WEIGHT.get(usage.kind)
    if weight is None:
        return None

    if usage.kind == "boolean_context":
        return ("boolean", weight)
    if usage.kind == "date_function":
        return ("date", weight)
    if usage.kind in ("in_list_strings", "like"):
        return ("string", weight)
    if usage.kind in ("compared_to_number", "in_list_numbers", "arithmetic"):
        return ("numeric", weight)
    if usage.kind == "coalesce_default":
        resolved = LITERAL_KIND_TYPE.get(usage.detail or "")  # type: ignore[arg-type]
        return (resolved, weight) if resolved is not None else None
    return None


def _type_from_expression(
    output: OutputColumn | ProjectedColumn,
) -> _TypeEvidence | None:
    """Type evidence carried by the expression that produced a derived column.

    The resolver stops at derived columns, so nothing else can type them.
    """
    if output.cast_type:
        resolved = CAST_TYPE_MAP.get(output.cast_type)
        if resolved is not None:
            return _TypeEvidence(
                resolved, CAST_WEIGHT, "expression", f"cast to {output.cast_type}"
            )

    from_function = FUNCTION_TYPE_MAP.get(output.function or "")
    if from_function is not None:
        return _TypeEvidence(
            from_function, EXPRESSION_WEIGHT, "expression", output.function
        )

    if output.function in TYPE_TRANSPARENT_FUNCTIONS:
        from_literals = _type_from_literal_kinds(output.literal_kinds)
        if from_literals is not None:
            return _TypeEvidence(
                from_literals,
                LITERAL_ARGUMENT_WEIGHT,
                "expression",
                f"{output.function} literal",
            )

    return None


def _type_from_literal_kinds(kinds: list[LiteralKind]) -> ResolvedType | None:
    resolved: ResolvedType | None = None
    for kind in kinds:
        current = LITERAL_KIND_TYPE.get(kind)
        if current is None:
            return None
        resolved = current if resolved is None else widen(resolved, current)
    return resolved if resolved != "unknown" else None


def _type_from_name(column: str) -> tuple[ResolvedType, str] | None:
    lower = column.lower()
    if lower.startswith("is_"):
        return ("boolean", "is_*")
    if lower.endswith("_at"):
        return ("timestamp", "*_at")
    if lower.endswith("_timestamp"):
        return ("timestamp", "*_timestamp")
    if lower.endswith("_ts"):
        return ("timestamp", "*_ts")
    if lower.endswith("_name"):
        return ("string", "*_name")
    if lower.endswith("_amount"):
        return ("numeric", "*_amount")
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

    derived_outputs = {
        ColumnNode(relation=relation.ref, column=output.name): output
        for relation in analysis.relations
        for output in relation.outputs
        if output.kind == "derived" and output.name
    }

    nodes = _all_nodes(analysis, usages_by_node, predicates_by_node, derived_outputs)
    evidence = {
        node: _initial_evidence(node, usages_by_node.get(node, []), derived_outputs)
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
    derived_outputs: Mapping[ColumnNode, OutputColumn],
) -> set[ColumnNode]:
    nodes: set[ColumnNode] = set(usages_by_node) | set(predicates_by_node)
    nodes |= set(derived_outputs)
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
    derived_outputs: Mapping[ColumnNode, OutputColumn],
) -> _TypeEvidence:
    best: _TypeEvidence | None = None
    for usage in usages:
        candidate = _type_from_usage(usage)
        if candidate is None:
            continue
        resolved, weight = candidate
        if best is None or weight > best.weight:
            best = _TypeEvidence(resolved, weight, "usage", usage.kind)
        elif weight == best.weight and resolved != best.resolved_type:
            # Two equally good pieces of evidence disagree; keep what both allow.
            best = _TypeEvidence(
                widen(best.resolved_type, resolved),
                weight,
                "usage",
                f"{best.evidence}+{usage.kind}",
            )

    output = derived_outputs.get(node)
    if output is not None:
        from_expression = _type_from_expression(output)
        if from_expression is not None and (
            best is None or from_expression.weight > best.weight
        ):
            best = from_expression

    if best is not None and best.resolved_type != "unknown":
        return best

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

        # Members of a join group have to share one value domain, so a group holding both
        # an `integer` and a `decimal` becomes `number` throughout rather than picking one
        # and generating values the other side can never match.
        unified = evidence[best_node].resolved_type
        for _, node in known:
            unified = widen(unified, evidence[node].resolved_type)

        for node in group:
            current = evidence.get(node)
            if current is not None and current.resolved_type == unified:
                continue
            evidence[node] = _TypeEvidence(
                unified,
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

        from_expression = _type_from_expression(projected)
        if from_expression is not None:
            resolved_type, source = from_expression.resolved_type, from_expression.source

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
