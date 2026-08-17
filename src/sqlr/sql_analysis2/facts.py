"""Facts - everything the SQL says about the values flowing through it.

One walk over the annotated tree, run after annotation pass 1. Two kinds of type fact:

- a **claim**: *this node must be family F*. `upper(x)` claims STRING of its argument.
- a **link**: *these two nodes share a domain*, with no family named. `a.x = b.y` links
  them, and whichever end has a type gives it to the other.

Both are produced by the same mechanism - reading a call's overloads - so comparisons,
arithmetic and `coalesce` need no code of their own; see `catalog.py`. Alongside them the
walk collects the non-type facts (predicates, joins, nullability, cardinality) that feed
constraint extraction and fixture generation.

**Why after pass 1, not beside `build_graph`.** Before annotation every `.type` is `None`,
so `in_family` cannot tell "undeclared" from "declared and fine" and *every* argument
position generates a claim. That was harmless while claims were internal. They are shown to
the user with a span now, so an over-claim would be a lie in a report.

Facts hold **live node references**, not copies of types, which is why they are frozen
dataclasses rather than pydantic models. The checking pass reads `site.node.type` after the
second annotation pass and sees the widened types with nothing re-extracted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence, cast

from sqlglot import exp
from sqlglot.optimizer.scope import Scope

from sqlr.sql_analysis2.annotate import (
    arguments_of_call,
    candidate_overloads,
    catalog_key,
    is_a_type_variable,
    signatures_for_dialect,
    signatures_of_call,
)
from sqlr.sql_analysis2.catalog import CatalogKey, FamilyName, Sig
from sqlr.sql_analysis2.qualify import (
    ColumnReference,
    QualifiedStatement,
    column_reference_of,
    name_and_kind_of_scope,
)
from sqlr.sql_analysis2.sourcedoc import Positions, SourceSpan
from sqlr.sql_analysis2.types import (
    ArgumentIndex,
    CardinalityKind,
    ColumnName,
    LiteralKind,
    NullabilityReason,
    PredicateOperator,
    TableName,
)

MAXIMUM_LINKED_ARGUMENTS = 8
"""Above this, a type-variable signature stops linking its arguments pairwise. `coalesce`
over four columns is three links per column and worth it; over forty it is noise nobody
reads and quadratic work nobody asked for."""


# ------------------------------------------------------------------- what a fact is about
@dataclass(frozen=True)
class ValueSite:
    """The thing a fact is about: a node, and the column it happens to be.

    The node is the subject. Keying facts on columns instead would mean
    `round(upper(x), 2)` - no column anywhere - is reported by different code from
    `round(x, 2)`, for the same defect. `column` is what a fact *lands on*, and it is what
    inference and fixture generation read.
    """

    node: exp.Expr
    column: ColumnReference | None
    span: SourceSpan | None

    @property
    def source_table_column(self) -> tuple[TableName, ColumnName] | None:
        """The real table and column this site reads, or None when it reads neither.

        A CTE's column is deliberately not one: its type is computed from the CTE's own
        projection, so nothing may be inferred for it and no schema slot exists to widen.
        """
        reference = self.column
        if reference is None or reference.source.kind != "table":
            return None
        table = reference.source.name or reference.source.alias
        return table.lower(), reference.name.lower()

    def describe(self) -> str:
        """What to call this site in a message.

        A column by its name, quoted the way every other finding quotes one; anything else
        as the SQL that produced it, unquoted - a literal already carries its own quotes and
        doubling them reads as a bug.
        """
        if self.column is not None:
            return f"'{self.column.name}'"
        return self.node.sql()


# ------------------------------------------------------------------------- type facts
type ClaimReasonKind = Literal["call-argument", "boolean-context"]
type LinkReasonKind = Literal[
    "call-argument", "between", "case-arm", "in-list", "in-subquery", "set-operation-arm"
]


@dataclass(frozen=True)
class ClaimReason:
    kind: ClaimReasonKind
    call: CatalogKey | None = None
    argument_index: ArgumentIndex | None = None

    def describe(self) -> str:
        if self.kind == "call-argument" and self.call is not None:
            position = (
                "" if self.argument_index is None else f" argument {self.argument_index + 1}"
            )
            return f"{self.call}{position}"
        return "boolean context"


@dataclass(frozen=True)
class TypeClaim:
    """*This value must be one of these families.*

    Read two opposite ways depending on what the value already is, which is the one rule
    the whole pass runs on: contradicted by a known type it is an error, and made about a
    value with no type at all it is the inference.

    A *set* of families, not one, because a position of an overloaded call genuinely
    accepts several: DuckDB's `*` takes `(NUMERIC, NUMERIC)` and `(INTERVAL, NUMERIC)`, so
    `x * 2` says "numeric or interval" and nothing narrower. Both readings respect that -
    a contradiction needs the value to be in *no* family, and an inference needs exactly
    one to choose from - which is what keeps `interval '1 day' * 3` from being reported and
    `'abc' * 3` from being missed.
    """

    site: ValueSite
    families: frozenset[FamilyName]
    because: ClaimReason

    def describe_families(self) -> str:
        """`NUMERIC`, `NUMERIC or INTERVAL` - in a stable order, since a set has none."""
        names = sorted(self.families)
        if len(names) == 1:
            return names[0]
        return f"{', '.join(names[:-1])} or {names[-1]}"


@dataclass(frozen=True)
class TypeLink:
    """*These two values share a domain.* No family named.

    Produced by positions bound to one type variable - `a = b`, `coalesce(a, b)` - and by
    the constructs where two values are the same output column, such as a set operation's
    arms. Types a column from whatever it is compared against, and says nothing at all when
    neither end is known.
    """

    left: ValueSite
    right: ValueSite
    because: LinkReasonKind
    span: SourceSpan | None
    call: CatalogKey | None = None
    """The call whose type variable bound the two ends, when one did. `>` and `COALESCE`
    are both links, and a reader given only "same domain" cannot tell which."""

    def describe(self) -> str:
        """How the link reads in a message: `>`, `COALESCE`, `set operation arm`."""
        if self.because == "call-argument" and self.call is not None:
            return self.call
        return self.because.replace("-", " ")


# --------------------------------------------------------------------- non-type facts
@dataclass(frozen=True)
class PredicateFact:
    """A filter applied to a column. Raw material for constraint extraction."""

    site: ValueSite
    operator: PredicateOperator
    values: list[str]
    literal_kind: LiteralKind | None
    value_spans: list[SourceSpan | None]
    context_span: SourceSpan | None


@dataclass(frozen=True)
class JoinFact:
    """An equi-join between two columns. Raw material for relationship inference.

    Equality only. `!=` constrains the two columns' types exactly as much - and produces a
    `TypeLink` for it - but says nothing about one referencing the other.
    """

    left: ValueSite
    right: ValueSite
    join_type: str
    context_span: SourceSpan | None


@dataclass(frozen=True)
class NullabilityFact:
    site: ValueSite
    nullable: bool
    reason: NullabilityReason
    context_span: SourceSpan | None


@dataclass(frozen=True)
class CardinalityFact:
    sites: list[ValueSite]
    kind: CardinalityKind
    context_span: SourceSpan | None


@dataclass
class Facts:
    """Everything one statement said about its values."""

    type_claims: list[TypeClaim] = field(default_factory=list[TypeClaim])
    type_links: list[TypeLink] = field(default_factory=list[TypeLink])
    predicates: list[PredicateFact] = field(default_factory=list[PredicateFact])
    joins: list[JoinFact] = field(default_factory=list[JoinFact])
    nullability: list[NullabilityFact] = field(default_factory=list[NullabilityFact])
    cardinality: list[CardinalityFact] = field(default_factory=list[CardinalityFact])


# ------------------------------------------------------------------- literal shapes
def shape_of_number_literals(literals: Sequence[exp.Literal]) -> str:
    return (
        "float"
        if any("." in lit.this or "e" in lit.this.lower() for lit in literals)
        else "int"
    )


def kind_of_literals(literals: Sequence[exp.Expr]) -> LiteralKind:
    """The one kind every value shares, or `mixed`.

    Takes `exp.Expr` rather than `exp.Literal` because sqlglot parses `true` and `false`
    into `exp.Boolean`, which is not a literal node at all.
    """
    if not literals:
        return "mixed"
    if all(isinstance(lit, exp.Boolean) for lit in literals):
        return "boolean"

    values = [lit for lit in literals if isinstance(lit, exp.Literal)]
    if len(values) != len(literals):
        return "mixed"
    if all(lit.is_string for lit in values):
        return "string"
    if all(lit.is_number for lit in values):
        return "int" if shape_of_number_literals(values) == "int" else "float"
    return "mixed"


def text_of_literal(expression: exp.Expr) -> str:
    """A literal's value as it would be written in SQL.

    `exp.Boolean` holds a Python bool, so `str` on it would yield `True` rather than the
    `true` a generated fixture has to emit.
    """
    if isinstance(expression, exp.Boolean):
        return "true" if expression.this else "false"
    return str(expression.this)


def expressions_at(node: exp.Expr, key: str) -> list[exp.Expr]:
    """One of sqlglot's list-valued arguments, typed.

    `node.args` is a plain dict of anything, so every read of `joins`, `ifs` or
    `partition_by` would otherwise be untyped at the call site.
    """
    value = node.args.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in cast("list[Any]", value) if isinstance(item, exp.Expr)]


def joins_of(select: exp.Select) -> list[exp.Join]:
    return [join for join in expressions_at(select, "joins") if isinstance(join, exp.Join)]


COMPARISON_CLASSES = (exp.GT, exp.LT, exp.GTE, exp.LTE, exp.EQ, exp.NEQ)

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


@dataclass(frozen=True)
class ParsedPredicate:
    operator: PredicateOperator
    values: list[str]
    literal_kind: LiteralKind | None
    literals: Sequence[exp.Expr]
    """The literal nodes `values` came from, so their positions can be recovered.

    Fixture generation needs both halves: `> 20` says the generated data has to straddle
    20, and a diagnostic wants to point at the 20.
    """


def predicate_applied_to(column: exp.Column) -> ParsedPredicate | None:
    """The filter predicate a column sits under, if any."""
    parent = column.parent
    if parent is None:
        return None

    if isinstance(parent, COMPARISON_CLASSES):
        other = parent.right if parent.left is column else parent.left
        if not isinstance(other, (exp.Literal, exp.Boolean)):
            return None
        operator = COMPARISON_OPERATORS[type(parent)]
        if parent.right is column:
            operator = FLIPPED_OPERATORS[operator]
        return ParsedPredicate(
            operator, [text_of_literal(other)], kind_of_literals([other]), [other]
        )

    if isinstance(parent, exp.In) and parent.this is column:
        values = [
            e for e in parent.expressions if isinstance(e, (exp.Literal, exp.Boolean))
        ]
        if not values or len(values) != len(parent.expressions):
            return None
        return ParsedPredicate(
            "not_in" if isinstance(parent.parent, exp.Not) else "in",
            [text_of_literal(value) for value in values],
            kind_of_literals(values),
            values,
        )

    if isinstance(parent, (exp.Like, exp.ILike)) and parent.this is column:
        pattern = parent.expression
        if not isinstance(pattern, exp.Literal):
            return None
        return ParsedPredicate(
            "ilike" if isinstance(parent, exp.ILike) else "like",
            [str(pattern.this)],
            "string",
            [pattern],
        )

    if isinstance(parent, exp.Between) and parent.this is column:
        bounds = [parent.args.get("low"), parent.args.get("high")]
        literals = [b for b in bounds if isinstance(b, exp.Literal)]
        if len(literals) != 2:
            return None
        return ParsedPredicate(
            "between",
            [str(lit.this) for lit in literals],
            kind_of_literals(literals),
            literals,
        )

    if (
        isinstance(parent, exp.Is)
        and parent.this is column
        and isinstance(parent.expression, exp.Null)
    ):
        if isinstance(parent.parent, exp.Not):
            return ParsedPredicate("is_not_null", [], None, [])
        return ParsedPredicate("is_null", [], None, [])

    return None


# --------------------------------------------------------------------------- the walk
def extract_facts_from_annotated_tree(
    statement: QualifiedStatement, positions: Positions
) -> Facts:
    """Every fact one statement's SQL states about its values.

    Runs on the tree *after* annotation pass 1 - see the module docstring for why that
    ordering is load-bearing rather than incidental.
    """
    return _FactWalk(statement, positions).run()


class _FactWalk:
    def __init__(self, statement: QualifiedStatement, positions: Positions) -> None:
        self.statement = statement
        self.positions = positions
        self.signatures = signatures_for_dialect(statement.dialect_name)
        self.facts = Facts()
        # Which scope owns each column node. Built once: a column's source alias only means
        # something inside the scope that bound it, and every fact needs that resolution.
        self.scope_of_column: dict[int, Scope] = {
            id(column): scope for scope in statement.scopes for column in scope.columns
        }

    # ---- sites ----------------------------------------------------------------
    def site_of(self, node: exp.Expr) -> ValueSite:
        """Describe one node as a fact's subject."""
        inner = node.this if isinstance(node, exp.Alias) else node
        column: ColumnReference | None = None
        if isinstance(inner, exp.Column):
            scope = self.scope_of_column.get(id(inner))
            if scope is not None:
                column = column_reference_of(inner, scope, self.statement)
        return ValueSite(
            node=inner, column=column, span=self.positions.span_of(inner)
        )

    def run(self) -> Facts:
        for node in self.statement.qualified.walk():
            self._type_facts_of_call(node)
            self._type_facts_of_syntax(node)

        for scope in self.statement.scopes:
            self._facts_of_scope(scope)
        return self.facts

    # ---- type facts from the catalog -------------------------------------------
    def _type_facts_of_call(self, node: exp.Expr) -> None:
        """Claims and links for one call or operator, read off its overloads.

        One claim per argument position, naming every family any candidate overload accepts
        there. That union is the whole rule, and it needs no branch for "an overload
        matched" versus "none did": `round('x', 2)` contradicts `{NUMERIC}` and
        `'abc' + 5` contradicts `{NUMERIC, TEMPORAL, INTERVAL}`, both by the same test.

        A position where any candidate accepts anything - `ANY`, or a type variable -
        produces no claim, because a claim that cannot be contradicted is not a claim. The
        type variables produce links instead.
        """
        signatures = signatures_of_call(node, self.signatures)
        if not signatures:
            return
        candidates = candidate_overloads(node, signatures)
        if not candidates:
            # An arity error. A type claim against a signature the call was never going to
            # match would be noise on top of the finding `check.py` already makes.
            return

        arguments = arguments_of_call(node)
        key = catalog_key(node)
        self._claims_accepted_by_candidates(candidates, arguments, key)
        self._links_bound_by_type_variables(candidates, arguments, node, key)

    def _claims_accepted_by_candidates(
        self, candidates: list[Sig], arguments: list[exp.Expr], key: CatalogKey | None
    ) -> None:
        for index, argument in enumerate(arguments):
            families = {candidate.family_at(index) for candidate in candidates}
            if any(
                family == "ANY" or is_a_type_variable(family) for family in families
            ):
                continue  # this position accepts anything, so nothing is claimed of it
            self.facts.type_claims.append(
                TypeClaim(
                    site=self.site_of(argument),
                    families=frozenset(families),
                    because=ClaimReason(
                        kind="call-argument", call=key, argument_index=index
                    ),
                )
            )

    def _links_bound_by_type_variables(
        self,
        candidates: list[Sig],
        arguments: list[exp.Expr],
        node: exp.Expr,
        key: CatalogKey | None,
    ) -> None:
        """Link the positions every candidate binds to one type variable.

        Pairwise rather than a star through argument 0: with one round of inference, a link
        only helps when it directly joins the end that has a type to the end that does not,
        and `coalesce(a, b, c)` may know only `b`.
        """
        if len(arguments) > MAXIMUM_LINKED_ARGUMENTS:
            return
        for left in range(len(arguments)):
            for right in range(left + 1, len(arguments)):
                families = {
                    (candidate.family_at(left), candidate.family_at(right))
                    for candidate in candidates
                }
                if len(families) != 1:
                    continue
                left_family, right_family = families.pop()
                if left_family != right_family or not is_a_type_variable(left_family):
                    continue
                self._add_link(
                    arguments[left], arguments[right], "call-argument", node, key
                )

    # ---- type facts the catalog cannot express ---------------------------------
    def _type_facts_of_syntax(self, node: exp.Expr) -> None:
        """The constructs that constrain types without being calls.

        Each is a shape sqlglot parses into its own node rather than a function, so no
        signature can describe it: a CASE has arms, a BETWEEN has bounds, an IN has a list
        or a subquery, a set operation has two projections that are the same output column.
        """
        if isinstance(node, exp.Where):
            self._claim_boolean_context(node.this)
        elif isinstance(node, exp.Having):
            self._claim_boolean_context(node.this)
        elif isinstance(node, exp.Between):
            self._link_between_bounds(node)
        elif isinstance(node, exp.Case):
            self._link_case_arms(node)
        elif isinstance(node, exp.In):
            self._link_in_values(node)
        elif isinstance(node, exp.SetOperation):
            self._link_set_operation_arms(node)

    def _claim_boolean_context(self, node: exp.Expr | None) -> None:
        if node is None:
            return
        self.facts.type_claims.append(
            TypeClaim(
                site=self.site_of(node),
                families=frozenset({"BOOLEAN"}),
                because=ClaimReason(kind="boolean-context"),
            )
        )

    def _add_link(
        self,
        left: exp.Expr,
        right: exp.Expr,
        because: LinkReasonKind,
        node: exp.Expr,
        call: CatalogKey | None = None,
    ) -> None:
        self.facts.type_links.append(
            TypeLink(
                left=self.site_of(left),
                right=self.site_of(right),
                because=because,
                span=self.positions.span_of(node),
                call=call,
            )
        )

    def _link_between_bounds(self, node: exp.Between) -> None:
        for bound_name in ("low", "high"):
            bound = node.args.get(bound_name)
            if isinstance(bound, exp.Expr) and isinstance(node.this, exp.Expr):
                self._add_link(node.this, bound, "between", node)

    def _link_case_arms(self, node: exp.Case) -> None:
        """Every arm produces the same column, so every arm shares one domain.

        A searched CASE (`case when p then a`) additionally requires each `when` to be a
        boolean; a simple CASE (`case x when 1 then a`) instead compares `x` against each
        `when`, which is a link rather than a claim.
        """
        comparand = node.args.get("this")
        branches = [
            branch for branch in expressions_at(node, "ifs") if isinstance(branch, exp.If)
        ]
        for branch in branches:
            condition = branch.this
            if not isinstance(condition, exp.Expr):
                continue
            if isinstance(comparand, exp.Expr):
                self._add_link(comparand, condition, "case-arm", node)
            else:
                self._claim_boolean_context(condition)

        results = [
            branch.args.get("true")
            for branch in branches
            if isinstance(branch.args.get("true"), exp.Expr)
        ]
        default = node.args.get("default")
        if isinstance(default, exp.Expr):
            results.append(default)
        for index, left in enumerate(results):
            for right in results[index + 1 :]:
                if isinstance(left, exp.Expr) and isinstance(right, exp.Expr):
                    self._add_link(left, right, "case-arm", node)

    def _link_in_values(self, node: exp.In) -> None:
        subject = node.this
        if not isinstance(subject, exp.Expr):
            return
        for value in node.expressions:
            if isinstance(value, exp.Expr):
                self._add_link(subject, value, "in-list", node)

        query = node.args.get("query")
        select = query.this if isinstance(query, exp.Subquery) else query
        if isinstance(select, exp.Select) and len(select.selects) == 1:
            self._add_link(subject, select.selects[0], "in-subquery", node)

    def _link_set_operation_arms(self, node: exp.SetOperation) -> None:
        """Column i of one arm and column i of the other are one output column.

        Nested set operations need no recursion here: each `SetOperation` node is visited
        in its own right, and `selects` on a nested one reports its leftmost arm, so every
        arm ends up linked to the first.
        """
        for left_projection, right_projection in zip(
            node.left.selects, node.right.selects
        ):
            self._add_link(
                left_projection, right_projection, "set-operation-arm", node
            )

    # ---- non-type facts, per scope ---------------------------------------------
    def _facts_of_scope(self, scope: Scope) -> None:
        select = scope.expression
        if not isinstance(select, exp.Select):
            return
        padded = null_padded_source_aliases(select)

        for column in scope.columns:
            site = self.site_of(column)
            context_span = self.positions.span_of(column.parent) or site.span

            predicate = predicate_applied_to(column)
            if predicate is not None:
                self.facts.predicates.append(
                    PredicateFact(
                        site=site,
                        operator=predicate.operator,
                        values=predicate.values,
                        literal_kind=predicate.literal_kind,
                        value_spans=[
                            self.positions.span_of(literal)
                            for literal in predicate.literals
                        ],
                        context_span=context_span,
                    )
                )
                self._nullability_of_predicate(predicate.operator, site, context_span)

            if isinstance(column.parent, exp.Coalesce):
                self._add_nullability(site, True, "coalesce_argument", context_span)

            if column.table and column.table in padded:
                self._add_nullability(site, True, "outer_join_padded", context_span)

        self._joins_of_scope(scope, select)
        self._cardinality_of_scope(scope, select)

    def _add_nullability(
        self,
        site: ValueSite,
        nullable: bool,
        reason: NullabilityReason,
        context_span: SourceSpan | None,
    ) -> None:
        self.facts.nullability.append(
            NullabilityFact(
                site=site, nullable=nullable, reason=reason, context_span=context_span
            )
        )

    def _nullability_of_predicate(
        self,
        operator: PredicateOperator,
        site: ValueSite,
        context_span: SourceSpan | None,
    ) -> None:
        if operator == "is_null":
            self._add_nullability(site, True, "is_null_predicate", context_span)
        elif operator == "is_not_null":
            self._add_nullability(site, False, "is_not_null_predicate", context_span)

    def _joins_of_scope(self, scope: Scope, select: exp.Select) -> None:
        own_columns = {id(column) for column in scope.columns}

        joins = joins_of(select)
        for join in joins:
            on = join.args.get("on")
            if not isinstance(on, exp.Expr):
                continue
            join_type = describe_join_type(join)
            for equality in on.find_all(exp.EQ):
                self._join_pair(equality, join_type, own_columns)

        where = select.args.get("where")
        if isinstance(where, exp.Expr) and joins:
            # Comma joins put the predicate in WHERE rather than ON.
            for equality in where.find_all(exp.EQ):
                self._join_pair(equality, "INNER", own_columns, distinct_sources=True)

    def _join_pair(
        self,
        equality: exp.EQ,
        join_type: str,
        own_columns: set[int],
        distinct_sources: bool = False,
    ) -> None:
        left, right = equality.left, equality.right
        if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
            return
        if id(left) not in own_columns or id(right) not in own_columns:
            return
        if distinct_sources and left.table == right.table:
            return

        left_site, right_site = self.site_of(left), self.site_of(right)
        context_span = self.positions.span_of(equality)
        self.facts.joins.append(
            JoinFact(
                left=left_site,
                right=right_site,
                join_type=join_type,
                context_span=context_span,
            )
        )
        if join_type == "INNER":
            for site in (left_site, right_site):
                self._add_nullability(site, False, "inner_join_key", context_span)

    def _cardinality_of_scope(self, scope: Scope, select: exp.Select) -> None:
        own_columns = {id(column) for column in scope.columns}

        group = select.args.get("group")
        if group is not None:
            self._add_cardinality(
                group.expressions, "group_by", own_columns, self.positions.span_of(group)
            )

        if select.args.get("distinct") is not None:
            self._add_cardinality(
                [
                    projection.this
                    if isinstance(projection, exp.Alias)
                    else projection
                    for projection in select.selects
                ],
                "distinct",
                own_columns,
                self.positions.span_of(select),
            )

        for window in select.find_all(exp.Window):
            self._add_cardinality(
                window.args.get("partition_by") or [],
                "partition_by",
                own_columns,
                self.positions.span_of(window),
            )
            order = window.args.get("order")
            if order is not None:
                self._add_cardinality(
                    [ordered.this for ordered in order.expressions],
                    "window_order_by",
                    own_columns,
                    self.positions.span_of(order),
                )

    def _add_cardinality(
        self,
        expressions: Sequence[exp.Expr],
        kind: CardinalityKind,
        own_columns: set[int],
        context_span: SourceSpan | None,
    ) -> None:
        sites = [
            self.site_of(expression)
            for expression in expressions
            if isinstance(expression, exp.Column) and id(expression) in own_columns
        ]
        if not sites:
            return
        self.facts.cardinality.append(
            CardinalityFact(sites=sites, kind=kind, context_span=context_span)
        )


def describe_join_type(join: exp.Join) -> str:
    """`LEFT OUTER`, `INNER` - what to call this join in a fact."""
    side = str(join.args.get("side") or "").upper()
    kind = str(join.args.get("kind") or "").upper()
    return f"{side} {kind}".strip() or "INNER"


def null_padded_source_aliases(select: exp.Select) -> set[TableName]:
    """Source aliases whose columns can come back NULL because of an outer join.

    A LEFT join pads the side it brings in; a RIGHT join pads everything already joined;
    FULL pads both. Reading a padded column is what makes it nullable regardless of what
    the source declares.
    """
    padded: set[TableName] = set()
    joined_so_far: set[TableName] = set()

    source = select.args.get("from")
    if source is not None and isinstance(source.this, exp.Expr):
        joined_so_far.add(source.this.alias_or_name)

    for join in joins_of(select):
        alias = join.this.alias_or_name if isinstance(join.this, exp.Expr) else ""
        side = str(join.args.get("side") or "").upper()
        if side in ("LEFT", "FULL") and alias:
            padded.add(alias)
        if side in ("RIGHT", "FULL"):
            padded.update(joined_so_far)
        if alias:
            joined_so_far.add(alias)
    return padded


def scope_name_of(scope: Scope) -> str:
    return name_and_kind_of_scope(scope)[0]
