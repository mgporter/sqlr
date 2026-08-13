"""What the resolution pass produces, shared by the pipeline and by `reporting`.

Kept apart from `__init__` so a finding can be built from a `ParsedColumn` without the
reporting module importing the pipeline that calls it.
"""

from typing import Literal, NamedTuple

from sqlglot import exp

type TableName = str
type ColumnName = str
type ColumnTypeName = str

type StructuredAccessKind = Literal["dot_field", "bracket_key", "bracket_index"]
"""How a column was read into, at the *first* level only.

- `dot_field`     - `col.field`, a struct or json field read.
- `bracket_key`   - `col['field']`, the same read spelled with a string key.
- `bracket_index` - `col[1]`, or any non-string key: an array, map or (in DuckDB) string
  subscript. Says the column is indexable, not that it is structured.

Deeper levels are not modelled. A declaration can give `mycolumn` a type; it cannot give
one to `mycolumn.b`, so `mycolumn.b.c` says nothing `mycolumn.b` did not already say.
"""

type StructuredRead = tuple[StructuredAccessKind, exp.Column]
"""One read into a column: how it was written, and the node to point a span at."""


class ParsedColumn(NamedTuple):
    """One column name a real table is asked for, and how the SQL read it."""

    name: ColumnName
    structured_access: list[StructuredRead]
    """Every read that went *into* this column, in the order they were written. One name
    can be read several ways in one statement - `x.a` and `x['b']` - and each site needs
    its own span, so the sites are kept rather than reduced to one kind."""

    @property
    def requires_structured_type(self) -> bool:
        """Whether a scalar declaration for this column would contradict the SQL.

        `bracket_index` does not count: DuckDB subscripts strings, so `x[1]` proves only
        that `x` is indexable.
        """
        return any(
            kind in ("dot_field", "bracket_key") for kind, _ in self.structured_access
        )


class ResolvedColumns(NamedTuple):
    """What the probe pass learned: which columns each real table is asked for, how they
    were read, and which columns no source can own."""

    columns_per_table: dict[TableName, dict[ColumnName, ParsedColumn]]
    unresolvable_columns: list[exp.Column]
    columns_read_with_unsupported_dot_notation: list[exp.Column]
    """Dotted names absorbed as struct reads by a dialect that has no such syntax. Still
    harvested into `columns_per_table` - the report is the point, and dropping them only
    makes step 3 fail with a worse message."""
