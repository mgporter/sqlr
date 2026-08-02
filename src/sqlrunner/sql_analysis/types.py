"""Public result model for SQL analysis.

The model is lineage-oriented: every fact is attached to a `ColumnNode`, which is a
(relation, column) pair somewhere in the statement's relation graph. Consumers index the
flat fact lists by node rather than reading nested structures.
"""

from typing import Literal

from pydantic import BaseModel

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
    "in_list_strings",
    "in_list_numbers",
    "like",
    "boolean_context",
    "arithmetic",
    "date_function",
    "cast",
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


class OutputColumn(BaseModel):
    """A column produced by a relation's SELECT list."""

    name: str | None
    ordinal: int
    kind: OutputColumnKind
    function: str | None = None
    """For `derived` columns, the sqlglot class name of the producing expression."""
    origins: list[ColumnOrigin] = []
    star_sources: list[RelationRef] = []
    """For `star` columns, the relations the `*` covers."""


class Relation(BaseModel):
    ref: RelationRef
    is_set_operation: bool = False
    depends_on: list[RelationRef] = []
    outputs: list[OutputColumn] = []
    star_sources: list[RelationRef] = []


class SourceColumn(BaseModel):
    name: str
    confidence: Confidence


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
    origins: list[ColumnOrigin] = []
    star_of: list[RelationRef] = []
    """Set when the column is an unexpandable `*` over these relations."""


class UsageFact(BaseModel):
    node: ColumnNode
    kind: UsageKind
    detail: str | None = None


class PredicateFact(BaseModel):
    """A filter predicate applied to a column. Raw material for constraint extraction."""

    node: ColumnNode
    operator: PredicateOperator
    values: list[str] = []
    literal_kind: LiteralKind | None = None


class JoinFact(BaseModel):
    """An equi-join between two columns. Raw material for relationship inference."""

    left: ColumnNode
    right: ColumnNode
    join_type: str
    operator: str = "="


class NullabilityFact(BaseModel):
    node: ColumnNode
    nullable: bool
    reason: NullabilityReason


class CardinalityFact(BaseModel):
    nodes: list[ColumnNode]
    kind: CardinalityKind


class Ambiguity(BaseModel):
    relation: RelationRef
    column: str
    candidates: list[RelationRef] = []
    reason: AmbiguityReason
    resolution: AmbiguityResolution
    chosen: RelationRef | None = None
    confidence: Confidence | None = None
    line: int | None = None


class SqlAnalysisResult(BaseModel):
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
