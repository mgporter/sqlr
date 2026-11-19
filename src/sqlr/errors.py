"""Errors raised when a comparison cannot run.

An error here means the test itself is broken (bad inputs or arguments). That is different from
`TableMismatchError` (see table_compare), which means the SQL under test returned something other than
what the test expects.

Each error carries structured fields so a renderer can build its own message. `str(error)` is a plain
one-line message.
"""

from __future__ import annotations

from typing import Literal, Sequence

import pyarrow as pa

__all__ = [
    "Side",
    "ComparisonError",
    "TableConversionError",
    "SortColumnError",
    "SortOptionsError",
]

Side = Literal["actual", "expected"]


class ComparisonError(Exception):
    """Base class: the comparison could not run."""


class TableConversionError(ComparisonError):
    """One of the inputs could not be converted to an arrow table."""

    def __init__(self, side: Side, cause: Exception) -> None:
        super().__init__(f"Cannot convert {side} to an arrow table: {cause}")
        self.side = side
        self.cause = cause


class SortColumnError(ComparisonError):
    """A `sort_by` column does not exist in the expected result, or its type cannot be sorted."""

    def __init__(
        self,
        column: str,
        reason: Literal["missing", "unsortable"],
        available: Sequence[str],
        type_: pa.DataType | None = None,
    ) -> None:
        if reason == "missing":
            message = f"sort_by column '{column}' does not exist. Available columns: {list(available)}"
        else:
            message = f"sort_by column '{column}' has type {type_}, which cannot be sorted"
        super().__init__(message)
        self.column = column
        self.reason = reason
        self.available = tuple(available)
        self.type = type_


class SortOptionsError(ComparisonError):
    """`sort_by` was given together with `sort_results=False`."""

    def __init__(self, sort_by: Sequence[str]) -> None:
        super().__init__(
            f"sort_by={list(sort_by)} cannot be used with sort_results=False. "
            "Set sort_results=True, or drop sort_by."
        )
        self.sort_by = tuple(sort_by)
