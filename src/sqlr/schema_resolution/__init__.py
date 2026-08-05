"""Turn analysis facts into a per-table schema good enough to generate fixtures from.

Type evidence is weighted; the strongest wins. Columns joined by equality are unified
into one value domain, so `address.person_id` inherits `person.id`'s type and a generator
can produce overlapping values instead of a join that returns nothing.

Evidence that does not name a type resolves to a family rather than a guess - see
`ResolvedTypeName`. Two pieces of evidence of equal weight are widened together, which is also
how a join group with conflicting member types is unified.

**Collection and choice are separate**, and collection never discards. `_collect_evidence`
gathers everything a column's usage implies; `_choose` decides which of it wins. Both the
winner and the losers reach the caller, because "this column is a string" is a much weaker
thing to hand a user than "this column is a string, because of these three comparisons,
at these three places in the file".
"""

from collections import defaultdict
from collections.abc import Mapping
from typing import Literal

from sqlr.schema_resolution.types import (
    ColumnSchema,
    EvidenceKind,
    JoinGroup,
    NullabilityResolution,
    ProjectionSchema,
    ResolvedType,
    ResolvedTypeName,
    StatementSchema,
    TableSchema,
    TypeEvidence,
    ValueConstraint,
)
from sqlr.source import SourceSpan
from sqlr.sql_analysis.types import (
    ColumnNode,
    JoinFact,
    LiteralKind,
    NullabilityFact,
    OutputColumn,
    PredicateFact,
    ProjectedColumn,
    RelationRef,
    SqlAnalysisResult,
    UsageFact,
)
from sqlr.typemap import TYPE_COVER, WIDENING_ORDER, resolve_type_name, widen

__all__ = ["resolve_schema", "widen", "TYPE_COVER", "WIDENING_ORDER"]

# Higher confidence wins when a column has conflicting evidence. `cast` is absent on
# purpose: `cast(x as decimal)` fixes the type of the *result*, not of `x`, which could
# have been anything castable. It types the derived column instead, via `cast_type`.
USAGE_TYPE_WEIGHT: dict[str, int] = {
    "boolean_context": 90,
    "date_function": 70,
    "function_argument": 65,
    "in_list_strings": 60,
    "like": 60,
    "string_function": 60,
    "numeric_function": 60,
    "compared_to_number": 50,
    "compared_to_string": 50,
    "in_list_numbers": 50,
    "arithmetic": 10,
}

NAME_PATTERN_WEIGHT = 5
CAST_WEIGHT = 100
EXPRESSION_WEIGHT = 80
LITERAL_ARGUMENT_WEIGHT = 55

# Type of a column produced by an expression, keyed by sqlglot class name. Only entries
# with a return type the dialect fixes are concrete; arithmetic follows its operands.
FUNCTION_TYPE_MAP: dict[str, ResolvedTypeName] = {
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

# A literal proves its neighbour holds a number, never which kind of number.
LITERAL_KIND_TYPE: dict[LiteralKind, ResolvedTypeName] = {
    "int": "numeric",
    "float": "numeric",
    "string": "string",
}

NULLABILITY_WEIGHT: dict[str, int] = {
    "is_null_predicate": 3,
    "is_not_null_predicate": 3,
    "coalesce_argument": 2,
    "inner_join_key": 1,
    # `outer_join_padded` describes the join result, not the stored column, so it does
    # not decide source nullability. It stays available on the resolution's fact list.
}

NAME_PATTERNS: list[tuple[Literal["prefix", "suffix"], str, ResolvedTypeName]] = [
    ("prefix", "is_", "boolean"),
    ("suffix", "_at", "timestamp"),
    ("suffix", "_timestamp", "timestamp"),
    ("suffix", "_ts", "timestamp"),
    ("suffix", "_date", "date"),
    ("suffix", "_name", "string"),
    ("suffix", "_amount", "numeric"),
]


# ---- evidence collection -------------------------------------------------------------


def _usage_evidence(usage: UsageFact) -> TypeEvidence | None:
    """Type evidence carried by the syntactic context a column appeared in."""
    weight = USAGE_TYPE_WEIGHT.get(usage.kind)
    if weight is None:
        return None

    resolved: ResolvedTypeName | None
    if usage.kind == "boolean_context":
        resolved = "boolean"
    elif usage.kind == "date_function":
        resolved = "date"
    elif usage.kind in (
        "in_list_strings",
        "like",
        "compared_to_string",
        "string_function",
    ):
        resolved = "string"
    elif usage.kind in (
        "compared_to_number",
        "in_list_numbers",
        "arithmetic",
        "numeric_function",
    ):
        resolved = "numeric"
    elif usage.kind == "function_argument":
        resolved = LITERAL_KIND_TYPE.get(usage.detail or "")  # type: ignore[arg-type]
    else:
        resolved = None

    if resolved is None:
        return None

    return TypeEvidence(
        node=usage.node,
        type_name=resolved,
        weight=weight,
        source="usage",
        kind="usage",
        detail=usage.kind,
        span=usage.span,
        context_span=usage.context_span,
    )


def _expression_evidence(
    node: ColumnNode, output: OutputColumn | ProjectedColumn
) -> TypeEvidence | None:
    """Type evidence carried by the expression that produced a derived column.

    The resolver stops at derived columns, so nothing else can type them.
    """
    kind: EvidenceKind
    resolved: ResolvedTypeName | None = None
    detail: str | None = None
    weight = 0

    if output.cast_type:
        resolved = resolve_type_name(output.cast_type)
        if resolved != "unknown":
            kind, weight = "cast", CAST_WEIGHT
            detail = f"cast to {output.cast_type}"
            return TypeEvidence(
                node=node,
                type_name=resolved,
                weight=weight,
                source="expression",
                kind=kind,
                detail=detail,
                span=output.span,
                context_span=output.span,
            )

    from_function = FUNCTION_TYPE_MAP.get(output.function or "")
    if from_function is not None:
        return TypeEvidence(
            node=node,
            type_name=from_function,
            weight=EXPRESSION_WEIGHT,
            source="expression",
            kind="function",
            detail=output.function,
            span=output.span,
            context_span=output.span,
        )

    if output.function in TYPE_TRANSPARENT_FUNCTIONS:
        from_literals = _type_from_literal_kinds(output.literal_kinds)
        if from_literals is not None:
            return TypeEvidence(
                node=node,
                type_name=from_literals,
                weight=LITERAL_ARGUMENT_WEIGHT,
                source="expression",
                kind="literal_argument",
                detail=f"{output.function} literal",
                span=output.span,
                context_span=output.span,
            )

    return None


def _type_from_literal_kinds(kinds: list[LiteralKind]) -> ResolvedTypeName | None:
    resolved: ResolvedTypeName | None = None
    for kind in kinds:
        current = LITERAL_KIND_TYPE.get(kind)
        if current is None:
            return None
        resolved = current if resolved is None else widen(resolved, current)
    return resolved if resolved != "unknown" else None


def _name_evidence(node: ColumnNode, references: list[SourceSpan]) -> TypeEvidence | None:
    """The weakest evidence there is: what the column is called.

    Collected unconditionally rather than only as a fallback. At weight 5 it loses to
    every other kind, so the outcome is unchanged - but a user reading a diagnostic gets
    to see that the name agreed, or did not.
    """
    lower = node.column.lower()
    for position, affix, resolved in NAME_PATTERNS:
        matches = (
            lower.startswith(affix) if position == "prefix" else lower.endswith(affix)
        )
        if not matches:
            continue
        pattern = f"{affix}*" if position == "prefix" else f"*{affix}"
        return TypeEvidence(
            node=node,
            type_name=resolved,
            weight=NAME_PATTERN_WEIGHT,
            source="name_pattern",
            kind="name_pattern",
            detail=pattern,
            span=references[0] if references else None,
        )
    return None


def _collect_evidence(
    node: ColumnNode,
    usages: list[UsageFact],
    derived_outputs: Mapping[ColumnNode, OutputColumn],
    references: list[SourceSpan],
) -> list[TypeEvidence]:
    """Everything that says anything about this column's type. Discards nothing."""
    evidence: list[TypeEvidence] = []

    for usage in usages:
        candidate = _usage_evidence(usage)
        if candidate is not None:
            evidence.append(candidate)

    output = derived_outputs.get(node)
    if output is not None:
        from_expression = _expression_evidence(node, output)
        if from_expression is not None:
            evidence.append(from_expression)

    from_name = _name_evidence(node, references)
    if from_name is not None:
        evidence.append(from_name)

    return evidence


# ---- choosing a winner ---------------------------------------------------------------


def _choose(evidence: list[TypeEvidence]) -> ResolvedType:
    """Heaviest evidence wins; equally heavy evidence that disagrees is widened.

    Widening the top tier can land on `unknown` - `boolean` and `date` have nothing in
    common. Rather than report `unknown` while weaker but usable evidence exists, the
    next tier down is tried.

    The losers travel with the winner: every piece that was weighed ends up on the
    returned `ResolvedType`, whatever it decided.
    """
    collected = _sorted_evidence(evidence)
    usable = [item for item in collected if item.type_name != "unknown"]
    if not usable:
        return ResolvedType(evidence=collected)

    for weight in sorted({item.weight for item in usable}, reverse=True):
        tier = [item for item in usable if item.weight == weight]

        resolved = tier[0].type_name
        for item in tier[1:]:
            resolved = widen(resolved, item.type_name)
        if resolved == "unknown":
            continue

        distinct: list[ResolvedTypeName] = list(
            dict.fromkeys(item.type_name for item in tier)
        )
        # The first entry of the tier is the winner by declaration order, which follows
        # the order the evidence was written in the SQL.
        return ResolvedType(
            type_name=resolved,
            chosen=tier[0],
            evidence=collected,
            widened_from=distinct if len(distinct) > 1 else [],
        )

    return ResolvedType(evidence=collected)


def _sorted_evidence(evidence: list[TypeEvidence]) -> list[TypeEvidence]:
    return sorted(evidence, key=lambda item: -item.weight)


# ---- join groups ---------------------------------------------------------------------


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
        return [sorted(group, key=str) for group in members.values() if len(group) > 1]


def _unify_join_groups(
    groups: list[list[ColumnNode]],
    evidence: dict[ColumnNode, list[TypeEvidence]],
    choices: dict[ColumnNode, ResolvedType],
) -> list[ResolvedTypeName]:
    """Propagate one type across each join group, without erasing member evidence.

    Members of a join group have to share one value domain, so a group holding both an
    `integer` and a `decimal` becomes `number` throughout rather than picking one and
    generating values the other side can never match.

    The unified type arrives as an *additional* piece of evidence on each member, at the
    weight of the member that justified it. Nothing already on the member is removed, so
    a diagnostic can still explain why the group settled where it did.
    """
    unified_types: list[ResolvedTypeName] = []

    for group in groups:
        # Annotated because a tuple built inside a comprehension has no expected type to
        # check against, so the literal union widens back to `str` without one.
        known: list[tuple[ResolvedTypeName, TypeEvidence | None, ColumnNode]] = [
            (choices[node].type_name, choices[node].chosen, node)
            for node in group
            if node in choices and choices[node].type_name != "unknown"
        ]
        if not known:
            unified_types.append("unknown")
            continue

        best_weight = max(
            chosen.weight for _, chosen, _ in known if chosen is not None
        )
        best_node = next(
            node
            for _, chosen, node in known
            if chosen is not None and chosen.weight == best_weight
        )

        unified: ResolvedTypeName = known[0][0]
        for resolved, _, _ in known[1:]:
            unified = widen(unified, resolved)
        unified_types.append(unified)

        for node in group:
            current = choices.get(node)
            if current is not None and current.type_name == unified:
                # Already agrees; leave its own evidence to speak for it.
                continue
            evidence.setdefault(node, []).append(
                TypeEvidence(
                    node=node,
                    type_name=unified,
                    weight=best_weight,
                    source="join_group",
                    kind="join_group",
                    detail=f"joined to {best_node}",
                    via=best_node,
                )
            )
            choices[node] = _choose(evidence[node])

    return unified_types


# ---- entry point ---------------------------------------------------------------------


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

    references_by_node = _references_by_node(analysis)

    derived_outputs = {
        ColumnNode(relation=relation.ref, column=output.name): output
        for relation in analysis.relations
        for output in relation.outputs
        if output.kind == "derived" and output.name
    }

    nodes = _all_nodes(analysis, usages_by_node, predicates_by_node, derived_outputs)

    evidence: dict[ColumnNode, list[TypeEvidence]] = {
        node: _collect_evidence(
            node,
            usages_by_node.get(node, []),
            derived_outputs,
            references_by_node.get(node, []),
        )
        for node in nodes
    }
    choices: dict[ColumnNode, ResolvedType] = {
        node: _choose(items) for node, items in evidence.items()
    }

    join_groups = _JoinGroups()
    for join in analysis.joins:
        join_groups.union(join.left, join.right)
    groups = join_groups.groups()

    unified_types = _unify_join_groups(groups, evidence, choices)

    group_index = {node: index for index, group in enumerate(groups) for node in group}
    join_group_models = _build_join_groups(
        groups, unified_types, analysis.joins, join_groups
    )

    tables = _build_tables(
        analysis,
        choices,
        predicates_by_node,
        nullability_by_node,
        references_by_node,
        group_index,
    )
    projection = _build_projection(analysis, evidence, choices)

    return StatementSchema(
        source=analysis.source,
        tables=tables,
        projection=projection,
        join_groups=join_group_models,
    )


def _references_by_node(
    analysis: SqlAnalysisResult,
) -> dict[ColumnNode, list[SourceSpan]]:
    references: dict[ColumnNode, list[SourceSpan]] = {}
    for source in analysis.sources:
        ref = _table_ref(source.name)
        for column in source.columns:
            references[ColumnNode(relation=ref, column=column.name)] = list(
                column.references
            )
    return references


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
            nodes.add(ColumnNode(relation=_table_ref(source.name), column=column.name))
    for join in analysis.joins:
        nodes.add(join.left)
        nodes.add(join.right)
    for projected in analysis.projection:
        for origin in projected.origins:
            nodes.add(origin.node)
    return nodes


def _table_ref(name: str) -> RelationRef:
    return RelationRef(kind="table", name=name)


def _build_join_groups(
    groups: list[list[ColumnNode]],
    unified_types: list[ResolvedTypeName],
    joins: list[JoinFact],
    union_find: _JoinGroups,
) -> list[JoinGroup]:
    """Attach the joins that linked each group, so the grouping can explain itself."""
    root_to_index = {union_find.find(group[0]): index for index, group in enumerate(groups)}

    facts_by_group: dict[int, list[JoinFact]] = defaultdict(list)
    for join in joins:
        index = root_to_index.get(union_find.find(join.left))
        if index is not None:
            facts_by_group[index].append(join)

    return [
        JoinGroup(
            members=group,
            unified_type=unified_types[index],
            facts=facts_by_group.get(index, []),
        )
        for index, group in enumerate(groups)
    ]


def _build_tables(
    analysis: SqlAnalysisResult,
    choices: dict[ColumnNode, ResolvedType],
    predicates_by_node: dict[ColumnNode, list[PredicateFact]],
    nullability_by_node: dict[ColumnNode, list[NullabilityFact]],
    references_by_node: dict[ColumnNode, list[SourceSpan]],
    group_index: dict[ColumnNode, int],
) -> list[TableSchema]:
    tables: list[TableSchema] = []
    for source in analysis.sources:
        ref = _table_ref(source.name)
        columns: list[ColumnSchema] = []
        for column in source.columns:
            node = ColumnNode(relation=ref, column=column.name)
            columns.append(
                ColumnSchema(
                    name=column.name,
                    resolved_type=choices.get(node) or ResolvedType(),
                    confidence=column.confidence,
                    nullability=_nullability(nullability_by_node.get(node, [])),
                    constraints=_constraints(predicates_by_node.get(node, [])),
                    references=references_by_node.get(node, []),
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


def _nullability(facts: list[NullabilityFact]) -> NullabilityResolution:
    best_weight = 0
    chosen: NullabilityFact | None = None
    for fact in facts:
        weight = NULLABILITY_WEIGHT.get(fact.reason, 0)
        if weight > best_weight:
            best_weight = weight
            chosen = fact
    return NullabilityResolution(
        nullable=None if chosen is None else chosen.nullable,
        chosen=chosen,
        facts=facts,
    )


def _constraints(facts: list[PredicateFact]) -> list[ValueConstraint]:
    # Deduped by position as well as by content: the same predicate written twice in two
    # places is two things to underline, and two hints about the data to generate.
    seen: set[tuple[str, tuple[str, ...], int | None]] = set()
    constraints: list[ValueConstraint] = []
    for fact in facts:
        key = (
            fact.operator,
            tuple(fact.values),
            None if fact.context_span is None else fact.context_span.start,
        )
        if key in seen:
            continue
        seen.add(key)
        constraints.append(
            ValueConstraint(
                operator=fact.operator,
                values=fact.values,
                literal_kind=fact.literal_kind,
                span=fact.span,
                context_span=fact.context_span,
                value_spans=fact.value_spans,
            )
        )
    return constraints


def _build_projection(
    analysis: SqlAnalysisResult,
    evidence: dict[ColumnNode, list[TypeEvidence]],
    choices: dict[ColumnNode, ResolvedType],
) -> list[ProjectionSchema]:
    projection: list[ProjectionSchema] = []
    for projected in analysis.projection:
        node = ColumnNode(
            relation=RelationRef(kind="root", name=""),
            column=projected.name or f"_col_{projected.ordinal}",
        )

        collected: list[TypeEvidence] = []

        # A passthrough column is whatever its origin is. Every origin contributes, so a
        # set-operation column that unions two differently-typed branches shows both.
        #
        # A *derived* column resolves to itself, and that node was already typed from its
        # own expression when the source columns were collected. Deriving the expression
        # evidence again here would produce the same observation a second time - the same
        # `upper(...)` seen from two directions - so the origins are asked first and the
        # expression is only consulted when they had nothing to say.
        for origin in projected.origins:
            collected.extend(evidence.get(origin.node, []))

        if not collected:
            from_expression = _expression_evidence(node, projected)
            if from_expression is not None:
                collected.append(from_expression)

        resolved = _choose(collected)
        if resolved.type_name == "unknown":
            # Fall back to the origins' own resolved types, which include anything a join
            # group propagated onto them after their own evidence was collected. Only the
            # conclusion carries over; the evidence stays listed against the origin, which
            # is where the user has to look to change it.
            for origin in projected.origins:
                inherited = choices.get(origin.node)
                if inherited is not None and inherited.type_name != "unknown":
                    resolved = inherited.model_copy(
                        update={"evidence": resolved.evidence}
                    )
                    break

        projection.append(
            ProjectionSchema(
                name=projected.name,
                ordinal=projected.ordinal,
                resolved_type=resolved,
                origins=[origin.node for origin in projected.origins],
                span=projected.span,
                alias_span=projected.alias_span,
            )
        )
    return projection
