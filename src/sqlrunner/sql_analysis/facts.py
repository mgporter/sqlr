"""Step 4 of the pipeline: extract facts about columns.

    (RelationGraph, Resolver) -> Facts

Usage, predicates, joins, nullability and cardinality. Every fact is attached to the
`ColumnNode`s a reference resolves to. Because the resolver terminates at derived columns,
evidence about `sum(x)` or `row_number()` can never reach the base columns feeding them.

See `__init__.py` for the pipeline as a whole.
"""

from __future__ import annotations

from typing import NamedTuple

from pydantic import BaseModel

from sqlglot import exp

from sqlrunner.source import NO_POSITIONS, Positions, SourceSpan
from sqlrunner.sql_analysis.resolver import Resolver
from sqlrunner.sql_analysis.relations import (
    RelationGraph,
    RelationInfo,
    literal_kind,
    number_shape,
)
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

# Functions that constrain the type of what goes *in*, which is the mirror of
# `FUNCTION_TYPE_MAP` in schema_resolution describing what comes out. `upper(x)` types its
# result string, but it also says something about `x` - and nothing else does, because the
# resolver stops at the derived column the call produces.
#
# Split by arity on purpose. Every argument of `concat` is a string, but only the first
# argument of `substring` is - the others are offsets, and typing them string would be
# worse than saying nothing.
STRING_ARGUMENT_FUNCTIONS = {
    "Concat",
    "DPipe",
    "Upper",
    "Lower",
    "Trim",
    "LTrim",
    "RTrim",
    "Initcap",
    "Length",
}
STRING_FIRST_ARGUMENT_FUNCTIONS = {"Substring", "Left", "Right"}

NUMERIC_ARGUMENT_FUNCTIONS = {
    "Abs",
    "Ceil",
    "Floor",
    "Sqrt",
    "Exp",
    "Ln",
    "Log",
    "Pow",
    "Power",
    "Sign",
}
NUMERIC_FIRST_ARGUMENT_FUNCTIONS = {"Round", "Trunc"}

BOOLEAN_CONTEXT_PARENTS = (exp.Not, exp.Where, exp.And, exp.Or, exp.If)
TYPE_TRANSPARENT_PARENTS = (exp.Coalesce, exp.Nullif, exp.Greatest, exp.Least)
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


def classify_usage(column: exp.Column) -> list[tuple[str, str | None]]:
    """Type evidence carried by the syntactic context a column appears in."""
    parent = column.parent
    if parent is None:
        return []

    if isinstance(parent, exp.Cast) and parent.this is column:
        return [("cast", parent.to.this.name)]

    if isinstance(parent, COMPARISON_CLASSES):
        other = parent.right if parent.left is column else parent.left
        if isinstance(other, exp.Literal):
            if other.is_number:
                return [("compared_to_number", number_shape([other]))]
            if other.is_string:
                # As good as `x in ('a', 'b')`, which has always been string evidence.
                return [("compared_to_string", None)]
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

    if isinstance(parent, TYPE_TRANSPARENT_PARENTS):
        # `coalesce(bonus, 0)` types `bonus`: every argument has to share one domain.
        siblings = [
            argument
            for argument in parent.iter_expressions()
            if isinstance(argument, exp.Literal)
        ]
        if siblings:
            return [("coalesce_default", literal_kind(siblings))]
        return []

    if type(parent).__name__ in DATE_FUNCTION_CLASSES:
        return [("date_function", type(parent).__name__)]

    from_argument = _function_argument_usage(column, parent)
    if from_argument is not None:
        return [from_argument]

    return []


def _function_argument_usage(
    column: exp.Column, parent: exp.Expr
) -> tuple[str, str | None] | None:
    """What the enclosing call requires of this argument."""
    name = type(parent).__name__

    if name in STRING_ARGUMENT_FUNCTIONS or (
        name in STRING_FIRST_ARGUMENT_FUNCTIONS and parent.this is column
    ):
        return ("string_function", name)

    if name in NUMERIC_ARGUMENT_FUNCTIONS or (
        name in NUMERIC_FIRST_ARGUMENT_FUNCTIONS and parent.this is column
    ):
        return ("numeric_function", name)

    return None


class _Predicate(NamedTuple):
    operator: PredicateOperator
    values: list[str]
    literal_kind: LiteralKind | None
    literals: list[exp.Literal]
    """The literal nodes `values` came from, so their positions can be recovered.

    Fixture generation needs both halves: `> 20` says the column is a number *and* that
    the generated data has to straddle 20, and a diagnostic wants to point at the 20.
    """


def _predicate(column: exp.Column) -> _Predicate | None:
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
        return _Predicate(operator, [str(other.this)], literal_kind([other]), [other])

    if isinstance(parent, exp.In) and parent.this is column:
        literals = [e for e in parent.expressions if isinstance(e, exp.Literal)]
        if not literals or len(literals) != len(parent.expressions):
            return None
        operator = "not_in" if isinstance(parent.parent, exp.Not) else "in"
        return _Predicate(
            operator,
            [str(lit.this) for lit in literals],
            literal_kind(literals),
            literals,
        )

    if isinstance(parent, (exp.Like, exp.ILike)) and parent.this is column:
        pattern = parent.expression
        if not isinstance(pattern, exp.Literal):
            return None
        operator = "ilike" if isinstance(parent, exp.ILike) else "like"
        return _Predicate(operator, [str(pattern.this)], "string", [pattern])

    if isinstance(parent, exp.Between) and parent.this is column:
        bounds = [parent.args.get("low"), parent.args.get("high")]
        literals = [b for b in bounds if isinstance(b, exp.Literal)]
        if len(literals) != 2:
            return None
        return _Predicate(
            "between",
            [str(lit.this) for lit in literals],
            literal_kind(literals),
            literals,
        )

    if (
        isinstance(parent, exp.Is)
        and parent.this is column
        and isinstance(parent.expression, exp.Null)
    ):
        if isinstance(parent.parent, exp.Not):
            return _Predicate("is_not_null", [], None, [])
        return _Predicate("is_null", [], None, [])

    return None


# @dataclass
class Facts(BaseModel):
    """Everything step 4 hands to step 5. Flat lists, keyed by `ColumnNode`."""

    usages: list[UsageFact] = []
    predicates: list[PredicateFact] = []
    joins: list[JoinFact] = []
    nullability: list[NullabilityFact] = []
    cardinality: list[CardinalityFact] = []

    observed: list[ColumnOrigin] = []
    """Every column reference that resolved, with the confidence of its attribution.

    This is what tells the assembler a table has a column even when the column is only
    ever mentioned in a WHERE clause.
    """


def extract_facts(
    graph: RelationGraph, resolver: Resolver, positions: Positions = NO_POSITIONS
) -> Facts:
    """Walk every scope in `graph`, resolving references through `resolver`.

    May raise `StarOverJoinAbort` - resolution happens lazily inside the resolver, so the
    error surfaces here rather than when the resolver was constructed.
    """
    return _FactExtractor(graph, resolver, positions).run()


class _FactExtractor:
    def __init__(
        self,
        graph: RelationGraph,
        resolver: Resolver,
        positions: Positions = NO_POSITIONS,
    ) -> None:
        self.graph = graph
        self.resolver = resolver
        self.positions = positions
        self.facts = Facts()

    def run(self) -> Facts:
        for ref in self.graph.order:
            info = self.graph.relations[ref]
            if info.scope is None or info.is_setop:
                continue
            self._columns(info)
            self._joins(info)
            self._cardinality(info)
        return self.facts

    # ---- per-column facts -----------------------------------------------------

    def _columns(self, info: RelationInfo) -> None:
        for column in info.own_columns:
            resolution = self.resolver.resolve_column_expr(info, column)
            if not resolution.resolved:
                continue
            nodes = resolution.nodes
            self.facts.observed.extend(resolution.origins)

            # The reference names the column; its parent is the evidence about it.
            span = self.positions.span_of(column)
            context_span = self.positions.span_of(column.parent) or span

            for kind, detail in classify_usage(column):
                for node in nodes:
                    self.facts.usages.append(
                        UsageFact(
                            node=node,
                            kind=kind,  # type: ignore[arg-type]
                            detail=detail,
                            span=span,
                            context_span=context_span,
                        )
                    )

            predicate = _predicate(column)
            if predicate is not None:
                value_spans = [
                    self.positions.span_of(literal) for literal in predicate.literals
                ]
                for node in nodes:
                    self.facts.predicates.append(
                        PredicateFact(
                            node=node,
                            operator=predicate.operator,
                            values=predicate.values,
                            literal_kind=predicate.literal_kind,
                            value_spans=value_spans,
                            span=span,
                            context_span=context_span,
                        )
                    )
                self._nullability_from_predicate(
                    predicate.operator, nodes, span, context_span
                )

            if isinstance(column.parent, exp.Coalesce):
                for node in nodes:
                    self.facts.nullability.append(
                        NullabilityFact(
                            node=node,
                            nullable=True,
                            reason="coalesce_argument",
                            span=span,
                            context_span=context_span,
                        )
                    )

            if column.table:
                binding = info.binding_for(column.table)
                if binding is not None and binding.is_outer_padded:
                    join_span = self.positions.span_of(binding.join)
                    for node in nodes:
                        self.facts.nullability.append(
                            NullabilityFact(
                                node=node,
                                nullable=True,
                                reason="outer_join_padded",
                                span=span,
                                context_span=join_span or context_span,
                            )
                        )

    def _nullability_from_predicate(
        self,
        operator: PredicateOperator,
        nodes: list[ColumnNode],
        span: SourceSpan | None,
        context_span: SourceSpan | None,
    ) -> None:
        if operator == "is_null":
            for node in nodes:
                self.facts.nullability.append(
                    NullabilityFact(
                        node=node,
                        nullable=True,
                        reason="is_null_predicate",
                        span=span,
                        context_span=context_span,
                    )
                )
        elif operator == "is_not_null":
            for node in nodes:
                self.facts.nullability.append(
                    NullabilityFact(
                        node=node,
                        nullable=False,
                        reason="is_not_null_predicate",
                        span=span,
                        context_span=context_span,
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

        self.facts.observed.extend(left_resolution.origins)
        self.facts.observed.extend(right_resolution.origins)

        left_span = self.positions.span_of(left)
        right_span = self.positions.span_of(right)
        context_span = self.positions.span_of(equality)

        for left_node in left_resolution.nodes:
            for right_node in right_resolution.nodes:
                if left_node == right_node:
                    continue
                if (
                    require_distinct_relations
                    and left_node.relation == right_node.relation
                ):
                    continue
                self.facts.joins.append(
                    JoinFact(
                        left=left_node,
                        right=right_node,
                        join_type=join_type,
                        left_span=left_span,
                        right_span=right_span,
                        span=left_span,
                        context_span=context_span,
                    )
                )
                if join_type == "INNER":
                    for node, node_span in (
                        (left_node, left_span),
                        (right_node, right_span),
                    ):
                        self.facts.nullability.append(
                            NullabilityFact(
                                node=node,
                                nullable=False,
                                reason="inner_join_key",
                                span=node_span,
                                context_span=context_span,
                            )
                        )

    # ---- cardinality ----------------------------------------------------------

    def _cardinality(self, info: RelationInfo) -> None:
        assert info.expression is not None
        expression = info.expression

        group = expression.args.get("group")
        if group is not None:
            self._add_cardinality(
                self._resolve_columns(info, group.expressions),
                "group_by",
                self.positions.span_of(group),
            )

        if expression.args.get("distinct") is not None:
            located: list[tuple[ColumnNode, SourceSpan | None]] = []
            for output in info.outputs:
                if output.kind != "passthrough" or output.name is None:
                    continue
                resolution = self.resolver.resolve(info.ref, output.name)
                if not resolution.resolved:
                    continue
                span = self.positions.span_of(output.expression)
                located.extend((node, span) for node in resolution.nodes)
            self._add_cardinality(located, "distinct", None)

        for window in expression.find_all(exp.Window):
            partition: list[exp.Expr] = window.args.get("partition_by") or []
            self._add_cardinality(
                self._resolve_columns(info, partition),
                "partition_by",
                self.positions.span_of(window),
            )

            order = window.args.get("order")
            if order is not None:
                ordered = [o.this for o in order.expressions]
                self._add_cardinality(
                    self._resolve_columns(info, ordered),
                    "window_order_by",
                    self.positions.span_of(order),
                )

    def _add_cardinality(
        self,
        located: list[tuple[ColumnNode, SourceSpan | None]],
        kind: str,
        context_span: SourceSpan | None,
    ) -> None:
        if not located:
            return
        nodes = [node for node, _ in located]
        spans = [span for _, span in located]
        self.facts.cardinality.append(
            CardinalityFact(
                nodes=nodes,
                kind=kind,  # type: ignore[arg-type]
                spans=spans,
                span=next((span for span in spans if span is not None), None),
                context_span=context_span,
            )
        )

    def _resolve_columns(
        self, info: RelationInfo, expressions: list[exp.Expr]
    ) -> list[tuple[ColumnNode, SourceSpan | None]]:
        """Resolved nodes paired with the reference each came from."""
        located: list[tuple[ColumnNode, SourceSpan | None]] = []
        for expression in expressions:
            if not isinstance(expression, exp.Column):
                continue
            if id(expression) not in info.scope_column_ids:
                continue
            resolution = self.resolver.resolve_column_expr(info, expression)
            if resolution.resolved:
                span = self.positions.span_of(expression)
                located.extend((node, span) for node in resolution.nodes)
        return located
