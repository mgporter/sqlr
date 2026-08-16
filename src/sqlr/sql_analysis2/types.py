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


class AmbiguousColumn(NamedTuple):
    """A bare column that two or more sources are each known to own.

    "Known" is the whole point: a CTE's projections and a declared table's column list are
    both complete answers, so a name appearing in two of them has no correct attribution
    at all. Nothing can break the tie, which is why this stops the statement.
    """

    column: exp.Column
    candidate_sources: list[TableName]
    """Every source known to own the name, sorted - the choices offered to the reader."""


class GuessedColumn(NamedTuple):
    """A bare column attributed by elimination rather than by proof.

    One source is known to own the name (or is the only one left standing), while some
    other source in the scope has an unknown column set and might own it too. The
    attribution is the best available answer and is used, but the reader is told.
    """

    column: exp.Column
    resolved_source: TableName
    """The source the column was credited to."""
    open_sources: list[TableName]
    """Sources whose column set is unknown, sorted - the reason this is a guess."""


class ResolvedColumns(NamedTuple):
    """What the probe pass learned: which columns each real table is asked for, how they
    were read, which columns no source can own, and which were attributed uncertainly."""

    columns_per_table: dict[TableName, dict[ColumnName, ParsedColumn]]
    source_table_names: set[TableName]
    """Every real table the statement reads, whether or not it names a column of it.

    A `select *` names none, so the table would otherwise be absent from the gap-filled
    schema entirely and step 3 would have nothing to expand the star against."""
    unresolvable_columns: list[exp.Column]
    columns_read_with_unsupported_dot_notation: list[exp.Column]
    """Dotted names absorbed as struct reads by a dialect that has no such syntax. Still
    harvested into `columns_per_table` - the report is the point, and dropping them only
    makes step 3 fail with a worse message."""
    ambiguous_columns: list[AmbiguousColumn]
    """Deliberately *not* harvested into `columns_per_table`: recording the probe's pick
    would fabricate a declaration slot on a table that may not own the column, and every
    type inferred from it downstream would inherit the mistake."""
    guessed_columns: list[GuessedColumn]
    """Harvested normally. The guess is the best answer available, and dropping it would
    only cost the column its declared type."""
    offsets_of_columns_written_without_a_source: set[tuple[int, int]]
    """The token hull of every column the user wrote bare, recorded before the probe
    qualified them. Offsets survive qualification, so this is what tells a qualifier the
    user typed apart from one the resolution supplied - the distinction is gone from the
    tree itself, and re-deriving it later would need the pre-probe statement back."""
