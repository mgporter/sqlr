"""Public result model for schema resolution.

The guiding rule: **nothing is discarded**. Resolution picks a winner among competing
pieces of type evidence, but the losers stay attached to the column, each with the source
range that produced it.

That is what lets a diagnostic say more than "expected numeric": it can point at the
comparison that proves it, list the two other places that agree, and underline the yml
line that disagrees.
"""

from typing import Literal

from pydantic import BaseModel, Field

from sqlrunner.source import SourceDoc, SourceSpan
from sqlrunner.sql_analysis.types import (
    ColumnNode,
    Confidence,
    JoinFact,
    LiteralKind,
    NullabilityFact,
    PredicateOperator,
)
from sqlrunner.typemap import ResolvedType

__all__ = [
    "ColumnSchema",
    "EvidenceKind",
    "JoinGroup",
    "NullabilityResolution",
    "ProjectedColumnSchema",
    "ResolvedType",
    "StatementSchema",
    "TableSchema",
    "TypeEvidence",
    "TypeSource",
    "ValueConstraint",
]

TypeSource = Literal["usage", "join_group", "expression", "name_pattern", "unknown"]

EvidenceKind = Literal[
    "usage",
    "cast",
    "function",
    "literal_argument",
    "name_pattern",
    "join_group",
]
"""What kind of observation produced a piece of type evidence.

Finer-grained than `TypeSource`, which stays as the coarse bucket the CLI prints.
"""


class TypeEvidence(BaseModel):
    """One observation that says something about a column's type.

    A column usually has several, and they need not agree. `ColumnSchema.chosen` is the
    one that won; the rest are kept because a disagreement with a user's declared type is
    best explained by showing all of them.
    """

    node: ColumnNode
    resolved_type: ResolvedType
    weight: int
    source: TypeSource
    kind: EvidenceKind
    detail: str | None = None
    """What was observed - `compared_to_number`, `cast to DECIMAL`, `*_at`."""
    span: SourceSpan | None = None
    """The column reference."""
    context_span: SourceSpan | None = None
    """The expression that constitutes the evidence - `amount > 20`."""
    via: ColumnNode | None = None
    """For `join_group` evidence, the member whose type was propagated here."""

    @property
    def location(self) -> SourceSpan | None:
        """Where to point when reporting this evidence."""
        return self.context_span or self.span


class ValueConstraint(BaseModel):
    """A predicate a generated value has to satisfy for the query to return rows.

    Carries the literals *and* their positions: `where col > 20` tells a generator to
    straddle 20, and tells a diagnostic where the 20 was written.
    """

    operator: PredicateOperator
    values: list[str] = []
    literal_kind: LiteralKind | None = None
    span: SourceSpan | None = None
    context_span: SourceSpan | None = None
    value_spans: list[SourceSpan | None] = []


class NullabilityResolution(BaseModel):
    nullable: bool | None = None
    chosen: NullabilityFact | None = None
    facts: list[NullabilityFact] = []
    """Every nullability observation, including ones that did not decide the outcome."""


class ColumnSchema(BaseModel):
    name: str
    resolved_type: ResolvedType
    chosen: TypeEvidence | None = None
    """The evidence that decided `resolved_type`. None when nothing typed the column."""
    evidence: list[TypeEvidence] = []
    """All of it, strongest first. Includes `chosen`."""
    widened_from: list[ResolvedType] = []
    """Set when equally-weighted evidence disagreed and was widened rather than picked."""
    confidence: Confidence = "explicit"
    """How confidently the column was attributed to this table, not to its type."""
    nullability: NullabilityResolution = Field(default_factory=NullabilityResolution)
    constraints: list[ValueConstraint] = []
    references: list[SourceSpan] = []
    """Every place the statement mentions this column."""
    join_group: int | None = None
    """Index into `StatementSchema.join_groups`; members share a value domain."""

    @property
    def source(self) -> TypeSource:
        """Coarse bucket of the winning evidence."""
        return self.chosen.source if self.chosen is not None else "unknown"

    @property
    def nullable(self) -> bool | None:
        return self.nullability.nullable

    @property
    def location(self) -> SourceSpan | None:
        """Best single place to point at when reporting about this column."""
        if self.chosen is not None and self.chosen.location is not None:
            return self.chosen.location
        return self.references[0] if self.references else None


class TableSchema(BaseModel):
    name: str
    columns: list[ColumnSchema] = []
    star_expanded: bool = False

    def column(self, name: str) -> ColumnSchema | None:
        lowered = name.lower()
        return next((c for c in self.columns if c.name.lower() == lowered), None)


class ProjectedColumnSchema(BaseModel):
    name: str | None
    ordinal: int
    resolved_type: ResolvedType
    chosen: TypeEvidence | None = None
    evidence: list[TypeEvidence] = []
    widened_from: list[ResolvedType] = []
    origins: list[ColumnNode] = []
    span: SourceSpan | None = None
    alias_span: SourceSpan | None = None

    @property
    def source(self) -> TypeSource:
        return self.chosen.source if self.chosen is not None else "unknown"


class JoinGroup(BaseModel):
    """Columns unified into one value domain by equi-joins.

    Members have to generate overlapping values or the join returns nothing, so their
    types are unified even when the evidence for each differs.
    """

    members: list[ColumnNode] = []
    unified_type: ResolvedType = "unknown"
    facts: list[JoinFact] = []
    """The joins that linked the members, with their source ranges."""

    @property
    def names(self) -> list[str]:
        return [str(node) for node in self.members]


class StatementSchema(BaseModel):
    source: SourceDoc = Field(default_factory=SourceDoc)
    tables: list[TableSchema] = []
    projection: list[ProjectedColumnSchema] = []
    join_groups: list[JoinGroup] = []

    def table(self, name: str) -> TableSchema | None:
        lowered = name.lower()
        return next((t for t in self.tables if t.name.lower() == lowered), None)

    def column(self, table: str, column: str) -> ColumnSchema | None:
        found = self.table(table)
        return found.column(column) if found is not None else None

    def snippet(self, span: SourceSpan | None) -> str | None:
        return self.source.slice(span)
