"""Public result model for SQL analysis.

The model is lineage-oriented: every fact is attached to a `ColumnNode`, which is a
(relation, column) pair somewhere in the statement's relation graph. Consumers index the
flat fact lists by node rather than reading nested structures.
"""

from typing import Literal

from pydantic import BaseModel, Field

from sqlr.source import SourceDoc, SourceSpan

RelationKind = Literal["table", "cte", "derived", "subquery", "root"]
"""Where a relation comes from. `table` relations are external and terminal."""

OutputColumnKind = Literal["passthrough", "derived", "star"]

Confidence = Literal["explicit", "inferred", "guessed"]
"""How a column was attributed to its source.

- `explicit`: the column was written out, qualified or unambiguously unqualified.
- `inferred`: resolved through a `*` with a single candidate source, or through a rule
  that had positive evidence (see `star_over_join_behavior`).
- `guessed`: resolved through a `*` over a join with no evidence; leftmost source wins.
"""

CONFIDENCE_RANK: dict[Confidence, int] = {"explicit": 3, "inferred": 2, "guessed": 1}

UsageKind = Literal[
    "compared_to_number",
    "compared_to_string",
    "in_list_strings",
    "in_list_numbers",
    "like",
    "boolean_context",
    "arithmetic",
    "date_function",
    "string_function",
    "numeric_function",
    "cast",
    "function_argument",
]

PredicateOperator = Literal[
    "=",
    "!=",
    ">",
    "<",
    ">=",
    "<=",
    "in",
    "not_in",
    "like",
    "ilike",
    "between",
    "is_null",
    "is_not_null",
]

LiteralKind = Literal["int", "float", "string", "mixed"]

NullabilityReason = Literal[
    "is_null_predicate",
    "is_not_null_predicate",
    "coalesce_argument",
    "outer_join_padded",
    "inner_join_key",
]

CardinalityKind = Literal["group_by", "distinct", "partition_by", "window_order_by"]

AmbiguityReason = Literal[
    "star_over_join",
    "unqualified_multi_source",
    "unknown_alias",
    "unresolved_column",
]

AmbiguityResolution = Literal["attributed", "dropped"]


class RelationRef(BaseModel, frozen=True):
    """Identity of one relation in the statement. Frozen so it can key a dict."""

    kind: RelationKind
    name: str

    def __str__(self) -> str:
        return f"{self.kind}:{self.name}" if self.name else self.kind


class ColumnNode(BaseModel, frozen=True):
    """A column belonging to a specific relation. The unit every fact hangs off."""

    relation: RelationRef
    column: str

    def __str__(self) -> str:
        return f"{self.relation}.{self.column}"


class ColumnOrigin(BaseModel):
    node: ColumnNode
    confidence: Confidence
    span: SourceSpan | None = None
    """Where the reference that produced this origin was written."""


class OutputColumn(BaseModel):
    """A column produced by a relation's SELECT list."""

    name: str | None
    ordinal: int
    kind: OutputColumnKind
    function: str | None = None
    """For `derived` columns, the sqlglot class name of the producing expression."""
    cast_type: str | None = None
    """For `Cast` columns, the target type name, e.g. `DECIMAL`."""
    literal_kinds: list[LiteralKind] = []
    """For `derived` columns, the kinds of the literals written directly as arguments."""
    origins: list[ColumnOrigin] = []
    star_sources: list[RelationRef] = []
    """For `star` columns, the relations the `*` covers."""
    span: SourceSpan | None = None
    """The whole SELECT item, alias included."""
    alias_span: SourceSpan | None = None


class Relation(BaseModel):
    ref: RelationRef
    is_set_operation: bool = False
    depends_on: list[RelationRef] = []
    outputs: list[OutputColumn] = []
    star_sources: list[RelationRef] = []
    span: SourceSpan | None = None
    """The relation's body - for a CTE, everything between its parentheses."""
    name_span: SourceSpan | None = None


class SourceColumn(BaseModel):
    name: str
    confidence: Confidence
    references: list[SourceSpan] = []
    """Every place the statement mentions this column, in first-seen order."""


class SourceTable(BaseModel):
    """An external table plus every column the statement was able to attribute to it."""

    name: str
    columns: list[SourceColumn] = []
    star_expanded: bool = False
    """True when a `*` read this table, so the real column list may be a superset."""


class ProjectedColumn(BaseModel):
    """A column in the statement's final output, in SELECT order."""

    name: str | None
    ordinal: int
    kind: OutputColumnKind
    function: str | None = None
    cast_type: str | None = None
    literal_kinds: list[LiteralKind] = []
    origins: list[ColumnOrigin] = []
    star_of: list[RelationRef] = []
    """Set when the column is an unexpandable `*` over these relations."""
    span: SourceSpan | None = None
    alias_span: SourceSpan | None = None


class Fact(BaseModel):
    """Common shape of everything the analyser observed about a column.

    `span` is the reference itself - what to highlight when naming the column. Both are
    optional because sqlglot only positions tokens it actually parsed; anything it
    synthesised has no location at all.
    """

    span: SourceSpan | None = None
    context_span: SourceSpan | None = None
    """The expression the reference sits in - `amount > 20`, not just `amount`.

    This is what a diagnostic underlines: the comparison is the evidence, the bare column
    name is not.
    """


class UsageFact(Fact):
    node: ColumnNode
    kind: UsageKind
    detail: str | None = None


class PredicateFact(Fact):
    """A filter predicate applied to a column. Raw material for constraint extraction."""

    node: ColumnNode
    operator: PredicateOperator
    values: list[str] = []
    literal_kind: LiteralKind | None = None
    value_spans: list[SourceSpan | None] = []
    """Positions of `values`, parallel to it."""


class JoinFact(Fact):
    """An equi-join between two columns. Raw material for relationship inference."""

    left: ColumnNode
    right: ColumnNode
    join_type: str
    operator: str = "="
    left_span: SourceSpan | None = None
    right_span: SourceSpan | None = None


class NullabilityFact(Fact):
    node: ColumnNode
    nullable: bool
    reason: NullabilityReason


class CardinalityFact(Fact):
    nodes: list[ColumnNode]
    kind: CardinalityKind
    spans: list[SourceSpan | None] = []
    """Positions of `nodes`, parallel to it."""


class Ambiguity(BaseModel):
    relation: RelationRef
    column: str
    candidates: list[RelationRef] = []
    reason: AmbiguityReason
    resolution: AmbiguityResolution
    chosen: RelationRef | None = None
    confidence: Confidence | None = None
    span: SourceSpan | None = None

    @property
    def line(self) -> int | None:
        """1-based line, for messages meant to be read by a human."""
        return None if self.span is None else self.span.start_line + 1


class SqlAnalysisResult(BaseModel):
    source: SourceDoc = Field(default_factory=SourceDoc)
    relations: list[Relation] = []
    sources: list[SourceTable] = []
    projection: list[ProjectedColumn] = []
    usages: list[UsageFact] = []
    predicates: list[PredicateFact] = []
    joins: list[JoinFact] = []
    nullability: list[NullabilityFact] = []
    cardinality: list[CardinalityFact] = []
    ambiguities: list[Ambiguity] = []
    errors: list[str] = []
    warnings: list[str] = []

    @property
    def external_sources(self) -> list[str]:
        return [source.name for source in self.sources]

    @property
    def cte_names(self) -> list[str]:
        return [r.ref.name for r in self.relations if r.ref.kind == "cte"]

    def snippet(self, span: SourceSpan | None) -> str | None:
        """The source text a span covers. Snippets live here, not on the facts."""
        return self.source.slice(span)
