"""Fact extraction: usage, predicates, joins, nullability, cardinality.

Every fact is attached to the `ColumnNode`s a reference resolves to. Because the resolver
terminates at derived columns, evidence about `sum(x)` or `row_number()` can never reach
the base columns feeding them.
"""

from __future__ import annotations

from sqlglot import exp

from sqlrunner.sql_analysis.lineage import Resolver
from sqlrunner.sql_analysis.relations import RelationGraph, RelationInfo
from sqlrunner.sql_analysis.types import (
    CardinalityFact,
    ColumnNode,
    ColumnOrigin,
    JoinFact,
    LiteralKind,
    NullabilityFact,
    PredicateFact,
    PredicateOperator,
    UsageFact,
)

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

COMPARISON_OPERATORS: dict[type[exp.Expr], PredicateOperator] = {
    exp.EQ: "=",
    exp.NEQ: "!=",
    exp.GT: ">",
    exp.LT: "<",
    exp.GTE: ">=",
    exp.LTE: "<=",
}

FLIPPED_OPERATORS: dict[PredicateOperator, PredicateOperator] = {
    "=": "=",
    "!=": "!=",
    ">": "<",
    "<": ">",
    ">=": "<=",
    "<=": ">=",
}


def number_shape(literals: list[exp.Literal]) -> str:
    if any("." in lit.this or "e" in lit.this.lower() for lit in literals):
        return "float"
    return "int"


def _literal_kind(literals: list[exp.Literal]) -> LiteralKind:
    if all(lit.is_string for lit in literals):
        return "string"
    if all(lit.is_number for lit in literals):
        return "int" if number_shape(literals) == "int" else "float"
    return "mixed"


def classify_usage(column: exp.Column) -> list[tuple[str, str | None]]:
    """Type evidence carried by the syntactic context a column appears in."""
    parent = column.parent
    if parent is None:
        return []

    if isinstance(parent, exp.Cast) and parent.this is column:
        return [("cast", parent.to.this.name)]

    if isinstance(parent, COMPARISON_CLASSES):
        other = parent.right if parent.left is column else parent.left
        if isinstance(other, exp.Literal) and other.is_number:
            return [("compared_to_number", number_shape([other]))]
        return []

    if isinstance(parent, exp.In) and parent.this is column:
        literals = parent.expressions
        if literals and all(isinstance(e, exp.Literal) for e in literals):
            if all(e.is_string for e in literals):
                return [("in_list_strings", None)]
            if all(e.is_number for e in literals):
                return [("in_list_numbers", number_shape(literals))]
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


def _predicate(
    column: exp.Column,
) -> tuple[PredicateOperator, list[str], LiteralKind | None] | None:
    """A filter predicate applied to this column, if any."""
    parent = column.parent
    if parent is None:
        return None

    operator: PredicateOperator

    if isinstance(parent, COMPARISON_CLASSES):
        other = parent.right if parent.left is column else parent.left
        if not isinstance(other, exp.Literal):
            return None
        operator = COMPARISON_OPERATORS[type(parent)]
        if parent.right is column:
            operator = FLIPPED_OPERATORS[operator]
        return operator, [str(other.this)], _literal_kind([other])

    if isinstance(parent, exp.In) and parent.this is column:
        literals = [e for e in parent.expressions if isinstance(e, exp.Literal)]
        if not literals or len(literals) != len(parent.expressions):
            return None
        operator = "not_in" if isinstance(parent.parent, exp.Not) else "in"
        return operator, [str(lit.this) for lit in literals], _literal_kind(literals)

    if isinstance(parent, (exp.Like, exp.ILike)) and parent.this is column:
        pattern = parent.expression
        if not isinstance(pattern, exp.Literal):
            return None
        operator = "ilike" if isinstance(parent, exp.ILike) else "like"
        return operator, [str(pattern.this)], "string"

    if isinstance(parent, exp.Between) and parent.this is column:
        bounds = [parent.args.get("low"), parent.args.get("high")]
        literals = [b for b in bounds if isinstance(b, exp.Literal)]
        if len(literals) != 2:
            return None
        return "between", [str(lit.this) for lit in literals], _literal_kind(literals)

    if (
        isinstance(parent, exp.Is)
        and parent.this is column
        and isinstance(parent.expression, exp.Null)
    ):
        if isinstance(parent.parent, exp.Not):
            return "is_not_null", [], None
        return "is_null", [], None

    return None


class FactExtractor:
    def __init__(self, graph: RelationGraph, resolver: Resolver) -> None:
        self.graph = graph
        self.resolver = resolver
        self.usages: list[UsageFact] = []
        self.predicates: list[PredicateFact] = []
        self.joins: list[JoinFact] = []
        self.nullability: list[NullabilityFact] = []
        self.cardinality: list[CardinalityFact] = []
        self.observed: list[ColumnOrigin] = []
        """Every column reference that resolved, with the confidence of its attribution.

        This is what tells the assembler a table has a column even when the column is
        only ever mentioned in a WHERE clause.
        """

    def run(self) -> None:
        for ref in self.graph.order:
            info = self.graph.relations[ref]
            if info.scope is None or info.is_setop:
                continue
            self._columns(info)
            self._joins(info)
            self._cardinality(info)

    # ---- per-column facts -----------------------------------------------------

    def _columns(self, info: RelationInfo) -> None:
        for column in info.own_columns:
            resolution = self.resolver.resolve_column_expr(info, column)
            if not resolution.resolved:
                continue
            nodes = resolution.nodes
            self.observed.extend(resolution.origins)

            for kind, detail in classify_usage(column):
                for node in nodes:
                    self.usages.append(
                        UsageFact(node=node, kind=kind, detail=detail)  # type: ignore[arg-type]
                    )

            predicate = _predicate(column)
            if predicate is not None:
                operator, values, literal_kind = predicate
                for node in nodes:
                    self.predicates.append(
                        PredicateFact(
                            node=node,
                            operator=operator,
                            values=values,
                            literal_kind=literal_kind,
                        )
                    )
                self._nullability_from_predicate(operator, nodes)

            if isinstance(column.parent, exp.Coalesce):
                for node in nodes:
                    self.nullability.append(
                        NullabilityFact(
                            node=node, nullable=True, reason="coalesce_argument"
                        )
                    )

            if column.table:
                binding = info.binding_for(column.table)
                if binding is not None and binding.is_outer_padded:
                    for node in nodes:
                        self.nullability.append(
                            NullabilityFact(
                                node=node, nullable=True, reason="outer_join_padded"
                            )
                        )

    def _nullability_from_predicate(
        self, operator: PredicateOperator, nodes: list[ColumnNode]
    ) -> None:
        if operator == "is_null":
            for node in nodes:
                self.nullability.append(
                    NullabilityFact(node=node, nullable=True, reason="is_null_predicate")
                )
        elif operator == "is_not_null":
            for node in nodes:
                self.nullability.append(
                    NullabilityFact(
                        node=node, nullable=False, reason="is_not_null_predicate"
                    )
                )

    # ---- joins ----------------------------------------------------------------

    def _joins(self, info: RelationInfo) -> None:
        assert info.expression is not None

        for binding in info.bindings:
            join = binding.join
            if join is None:
                continue
            on = join.args.get("on")
            if on is None:
                continue
            join_type = f"{binding.join_side} {binding.join_kind}".strip() or "INNER"
            for equality in on.find_all(exp.EQ):
                self._join_pair(info, equality, join_type)

        where = info.expression.args.get("where")
        if where is not None and len(info.bindings) > 1:
            # Comma joins put the predicate in WHERE rather than ON.
            for equality in where.find_all(exp.EQ):
                self._join_pair(info, equality, "INNER", require_distinct_relations=True)

    def _join_pair(
        self,
        info: RelationInfo,
        equality: exp.EQ,
        join_type: str,
        require_distinct_relations: bool = False,
    ) -> None:
        left, right = equality.left, equality.right
        if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
            return
        if id(left) not in info.scope_column_ids or id(right) not in info.scope_column_ids:
            return

        left_resolution = self.resolver.resolve_column_expr(info, left)
        right_resolution = self.resolver.resolve_column_expr(info, right)
        if not (left_resolution.resolved and right_resolution.resolved):
            return

        self.observed.extend(left_resolution.origins)
        self.observed.extend(right_resolution.origins)

        for left_node in left_resolution.nodes:
            for right_node in right_resolution.nodes:
                if left_node == right_node:
                    continue
                if (
                    require_distinct_relations
                    and left_node.relation == right_node.relation
                ):
                    continue
                self.joins.append(
                    JoinFact(left=left_node, right=right_node, join_type=join_type)
                )
                if join_type == "INNER":
                    for node in (left_node, right_node):
                        self.nullability.append(
                            NullabilityFact(
                                node=node, nullable=False, reason="inner_join_key"
                            )
                        )

    # ---- cardinality ----------------------------------------------------------

    def _cardinality(self, info: RelationInfo) -> None:
        assert info.expression is not None
        expression = info.expression
        nodes: list[ColumnNode]

        group = expression.args.get("group")
        if group is not None:
            nodes = self._resolve_columns(info, group.expressions)
            if nodes:
                self.cardinality.append(CardinalityFact(nodes=nodes, kind="group_by"))

        if expression.args.get("distinct") is not None:
            nodes = []
            for output in info.outputs:
                if output.kind != "passthrough" or output.name is None:
                    continue
                resolution = self.resolver.resolve(info.ref, output.name)
                if resolution.resolved:
                    nodes.extend(resolution.nodes)
            if nodes:
                self.cardinality.append(CardinalityFact(nodes=nodes, kind="distinct"))

        for window in expression.find_all(exp.Window):
            partition: list[exp.Expr] = window.args.get("partition_by") or []
            nodes = self._resolve_columns(info, partition)
            if nodes:
                self.cardinality.append(
                    CardinalityFact(nodes=nodes, kind="partition_by")
                )

            order = window.args.get("order")
            if order is not None:
                ordered = [o.this for o in order.expressions]
                nodes = self._resolve_columns(info, ordered)
                if nodes:
                    self.cardinality.append(
                        CardinalityFact(nodes=nodes, kind="window_order_by")
                    )

    def _resolve_columns(
        self, info: RelationInfo, expressions: list[exp.Expr]
    ) -> list[ColumnNode]:
        nodes: list[ColumnNode] = []
        for expression in expressions:
            if not isinstance(expression, exp.Column):
                continue
            if id(expression) not in info.scope_column_ids:
                continue
            resolution = self.resolver.resolve_column_expr(info, expression)
            if resolution.resolved:
                nodes.extend(resolution.nodes)
        return nodes
