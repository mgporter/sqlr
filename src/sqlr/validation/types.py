"""The result of holding an inferred schema up against a declared one.

One row per column, and the row is the whole finding: what the SQL implied, what the yml
says, which of the two wins, and - when they cannot both be true - enough located evidence
to explain the disagreement without the caller going back to either source.

Nothing here mirrors `ColumnSchema` or `DeclaredColumn`; both are carried whole. The
inferred side keeps its losing evidence for the same reason schema resolution kept it: a
user told their `varchar` is contradicted needs to see every use that contradicts it.

`resolved_type` is deliberately `None` rather than `unknown` for a column the SQL never
mentioned. `unknown` is an answer - "inference looked and found nothing"; absence is the
statement that inference never got to look, which is what a declared-only column is.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, computed_field

from sqlr.declared.types import DeclaredColumn
from sqlr.diagnostics.types import Location
from sqlr.schema_resolution.types import ResolvedType
from sqlr.source import SourceDoc
from sqlr.typemap import ResolvedTypeName

__all__ = [
    "ColumnValidation",
    "StatementValidation",
    "TableValidation",
    "ValidationDetail",
    "ValidationOutcome",
]

ValidationOutcome = Literal["pass", "warning", "error"]

ValidationDetail = Literal[
    "exact match",
    "type narrowed",
    "type widened",
    "declared only",
    "no declaration",
    "unrecognized declaration",
    "inferred type differs from declared type",
]
"""Why an outcome came out the way it did. The pairs that occur:

- `pass` / `exact match` - the two name the same type.
- `pass` / `type narrowed` - the declaration pinned down a family inference left open.
- `warning` / `type widened` - the declaration is looser than what the SQL proved.
- `pass` / `declared only` - inference had nothing, or nothing but the column's name, so
  the declaration stands unopposed.
- `warning` / `no declaration` - inference is all there is.
- `warning` / `unrecognized declaration` - the written type is not one sqlr knows.
- `error` / `inferred type differs from declared type` - no concrete type satisfies both.
"""

OUTCOME_RANK: dict[ValidationOutcome, int] = {"error": 0, "warning": 1, "pass": 2}


class ColumnValidation(BaseModel):
    """One column, judged."""

    name: str
    outcome: ValidationOutcome
    detail: ValidationDetail
    resolved_type: ResolvedTypeName | None = None
    """What to generate for this column. None when nothing can be said."""
    inferred: ResolvedType | None = None
    """None when the SQL never mentioned the column - not the same as `unknown`."""
    declared: DeclaredColumn | None = None
    declaration: Location | None = None
    """Where the `data_type` was written, snippet included.

    Resolved here rather than left as a span so the renderer never needs the yml text.
    """
    ordinal: int | None = None
    """Position in the projection. None for a source-table column."""

    # Serialised, not just exposed: a consumer of `--format json` wants the two type names
    # beside the verdict, without walking into the evidence to reconstruct them.
    @computed_field
    @property
    def inferred_type(self) -> ResolvedTypeName | None:
        return self.inferred.type_name if self.inferred is not None else None

    @computed_field
    @property
    def declared_type(self) -> str | None:
        """The yml's `data_type:` verbatim - `varchar(50)`, not `string`."""
        return self.declared.written_type if self.declared is not None else None


class TableValidation(BaseModel):
    name: str
    kind: Literal["source", "projection"] = "source"
    star_expanded: bool = False
    """A `*` was expanded, so the SQL's column list is a subset of the real one."""
    declared_model: str | None = None
    """The model whose declaration this table was checked against, if any."""
    declared_path: Path | None = None
    columns: list[ColumnValidation] = []

    @property
    def errors(self) -> list[ColumnValidation]:
        return [column for column in self.columns if column.outcome == "error"]


class StatementValidation(BaseModel):
    """Every table of one statement, checked. Carries the SQL it was derived from.

    The text travels with the result because the error report quotes it: the point of the
    report is showing the comparison that forced the inferred type, not just naming it.
    """

    source: SourceDoc = SourceDoc()
    tables: list[TableValidation] = []
    projection: TableValidation = TableValidation(name="projection", kind="projection")

    @property
    def all_tables(self) -> list[TableValidation]:
        return [*self.tables, self.projection]

    @property
    def has_errors(self) -> bool:
        return any(table.errors for table in self.all_tables)

    def errors(self) -> list[tuple[TableValidation, ColumnValidation]]:
        """Every failed column, paired with the table it belongs to."""
        return [
            (table, column) for table in self.all_tables for column in table.errors
        ]

    def counts(self) -> dict[ValidationOutcome, int]:
        counts: dict[ValidationOutcome, int] = {}
        for table in self.all_tables:
            for column in table.columns:
                counts[column.outcome] = counts.get(column.outcome, 0) + 1
        return counts
