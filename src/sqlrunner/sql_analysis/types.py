from typing import Literal

from pydantic import BaseModel

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


class Cte(BaseModel):
    name: str
    depends_on: list[str]


class ColumnReference(BaseModel):
    table: str
    column: str


class ColumnUsage(BaseModel):
    table: str
    column: str
    kind: UsageKind
    detail: str | None = None


class SqlAnalysisResult(BaseModel):
    ctes: list[Cte]
    external_sources: list[str]
    column_references: list[ColumnReference]
    column_usages: list[ColumnUsage]
    errors: list[str]
