from typing import Literal

from pydantic import BaseModel

from sqlrunner.sql_analysis.types import Confidence, PredicateOperator

ResolvedType = Literal[
    "integer",
    "decimal",
    "float",
    "number",
    "numeric",
    "string",
    "date",
    "timestamp",
    "boolean",
    "unknown",
]
"""A concrete type, or a family that covers several.

Most evidence pins down that a column holds a number without saying which kind: `x * 12`
and `x > 5` are equally true of an INT, a DECIMAL and a DOUBLE. Rather than pick one and
be wrong, those resolve to a family:

- `number`  = `integer` | `decimal`  (exact)
- `numeric` = `integer` | `decimal` | `float`

Only evidence that names a type - a cast, or a function with a fixed return type - yields
a concrete one.
"""

TypeSource = Literal["usage", "join_group", "expression", "name_pattern", "unknown"]


class ValueConstraint(BaseModel):
    """A predicate a generated value has to satisfy for the query to return rows."""

    operator: PredicateOperator
    values: list[str] = []


class ColumnSchema(BaseModel):
    name: str
    resolved_type: ResolvedType
    source: TypeSource
    evidence: str | None = None
    confidence: Confidence = "explicit"
    """How confidently the column was attributed to this table, not to its type."""
    nullable: bool | None = None
    constraints: list[ValueConstraint] = []
    join_group: int | None = None
    """Index into `StatementSchema.join_groups`; members share a value domain."""


class TableSchema(BaseModel):
    name: str
    columns: list[ColumnSchema] = []
    star_expanded: bool = False


class ProjectedColumnSchema(BaseModel):
    name: str | None
    ordinal: int
    resolved_type: ResolvedType
    source: TypeSource
    origins: list[str] = []


class StatementSchema(BaseModel):
    tables: list[TableSchema] = []
    projection: list[ProjectedColumnSchema] = []
    join_groups: list[list[str]] = []
