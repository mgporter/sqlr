from typing import Literal

from pydantic import BaseModel

ResolvedType = Literal[
    "integer",
    "decimal",
    "string",
    "date",
    "timestamp",
    "boolean",
    "unknown",
]

TypeSource = Literal["usage", "name_pattern", "unknown"]


class ColumnSchema(BaseModel):
    name: str
    resolved_type: ResolvedType
    source: TypeSource
    evidence: str | None = None


class TableSchema(BaseModel):
    name: str
    columns: list[ColumnSchema]
