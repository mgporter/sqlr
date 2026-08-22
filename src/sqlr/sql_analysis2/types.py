"""What the resolution pass produces, shared by the pipeline and by `reporting`.

Kept apart from `__init__` so a finding can be built from a `ParsedColumn` without the
reporting module importing the pipeline that calls it.
"""

from typing import Literal, NamedTuple

from sqlglot import exp

from sqlr.sql_analysis2.sourcedoc import SourceSpan

type TableName = str
type ColumnName = str
type ColumnTypeName = str

type RelationKey = str
"""A relation's identity: the parts the SQL has to write, lowercased and dotted -
`mydatabase.myschema.raw_address`, or bare `employee` when the yml declares neither part.

The full name and not the bare one, because two sources may each have a `raw_department`
as long as they land in different schemas, and a bare key silently merges their column
sets. The same string `DeclaredSourceTable.key` produces, so both sides of a declaration
lookup are spelled the same way.
"""

type RelationAlias = str
"""The name a column qualifies itself with inside one scope. An alias, a CTE name or a
table name - which of those it is, is `RelationColumnSet.kind`."""

type ScopeKind = Literal["cte", "derived", "final", "branch"]
"""What a scope is, for a reader looking at a report.

- `cte`     - a named `WITH` term.
- `derived` - a subquery in a FROM or JOIN, named by its alias.
- `final`   - the statement's own projection, the one the model is.
- `branch`  - one arm of a set operation, or a scope with no name of its own.
"""

type SourceKind = ScopeKind | Literal["table", "unknown"]
"""What a column's qualifier turned out to name."""

type ArgumentIndex = int
"""A zero-based position in a call's argument list, in sqlglot's node order. Reported to
the user one-based, because that is how a person counts arguments."""

type PredicateOperator = Literal[
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

type LiteralKind = Literal["int", "float", "string", "boolean", "mixed"]
"""The one kind every literal in a predicate shares, or `mixed`. Coarser than a type on
purpose: it describes what was *written*, which is what fixture generation reproduces."""

type NullabilityReason = Literal[
    "is_null_predicate",
    "is_not_null_predicate",
    "coalesce_argument",
    "outer_join_padded",
    "inner_join_key",
]

type CardinalityKind = Literal["group_by", "distinct", "partition_by", "window_order_by"]

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


type AmbiguityKind = Literal["projected_by_several", "star_over_join"]
"""Which of the two dead ends an ambiguous column hit.

- `projected_by_several` - two relations are each *known* to project the name. A CTE's
  projections and a complete declaration are both complete answers, so a name appearing in
  two of them has no correct attribution at all. Qualifying the column fixes it.
- `star_over_join` - the name passes through a star over a join of relations nobody can
  enumerate, and `star_over_join_behavior` is `error`. Qualifying it fixes nothing, because
  no relation in the scope projects the name under its own steam; declaring one of the
  tables does.
"""


class AmbiguousColumn(NamedTuple):
    """A column that two or more relations could each own, with no way to choose.

    Nothing can break the tie, which is why this stops the statement - committing to either
    would hand step 6 an attribution as likely wrong as right, and every type inferred
    downstream would inherit the choice.
    """

    column: exp.Column
    candidate_sources: list[TableName]
    """Every relation that could own the name, sorted - the choices offered to the reader."""
    kind: AmbiguityKind = "projected_by_several"


class GuessedColumn(NamedTuple):
    """A column attributed by elimination rather than by proof.

    One relation is known to project the name (or is the only one left standing), while
    some other relation in the scope cannot enumerate its columns and might project it
    too. The attribution is the best available answer and is used, but the reader is told.
    """

    column: exp.Column
    resolved_source: TableName
    """The relation the column was credited to."""
    open_sources: list[TableName]
    """Relations whose column set is unknown, sorted - the reason this is a guess."""


type UnresolvableReason = Literal[
    "no_such_source", "not_projected", "several_undeclared_sources"
]
"""Why a column belongs to nothing. Three reasons, and no two share a fix.

- `no_such_source`            - its qualifier names no relation in the scope: a mistyped
  alias, and nothing in any yml would change that.
- `not_projected`             - every relation it could read from is closed and none
  projects it: a mistyped column, or a declaration that is complete when it should be
  partial. `UnresolvableRelation.closing_declarations` says which yml entry closed it.
- `several_undeclared_sources` - nothing projects it and *two or more* relations in scope
  leave their columns undeclared, so no one table can be credited with it. Unlike the
  other two this is not a mistake in the SQL: qualifying the column resolves it outright,
  because a single open relation absorbs a name it cannot rule out.
"""


class ClosingDeclaration(NamedTuple):
    """One yml entry whose complete column list is why some relation is closed.

    The entry a reader has to edit, which is not always the relation the SQL named: a CTE
    over `select *` is closed only because the tables under that star are, and telling the
    reader to fix the CTE would send them somewhere they cannot fix anything.
    """

    relation_name: RelationKey
    where: str
    """`project/sources.yml:46` - the file and line the entry was written at."""
    declared_columns: list[ColumnName]
    """What the entry says the relation has, sorted and lowercased."""


class UnresolvableRelation(NamedTuple):
    """One relation a column could have been read from, described for a message."""

    display_name: str
    """The relation as the reader would write it: the full table name when there is one,
    the alias otherwise. What to name when the fix is to edit a yml."""
    kind: SourceKind
    projected_columns: list[ColumnName]
    """What it is known to project, sorted. Empty when it enumerates nothing."""
    closing_declarations: list[ClosingDeclaration]
    """Why it is closed, empty when it is open. Several when a star over a join closed it,
    and transitive - see `ClosingDeclaration`."""


class UnresolvableColumn(NamedTuple):
    """A column no relation can own, and enough context to say why.

    `qualify` reports the same mistake against the tree it rewrote rather than the SQL that
    was written, so its message is strictly harder to act on than one built here: this can
    name the relation that failed to project the column, list what it projects instead, and
    point at the yml entry that made the omission an error rather than an open question.
    """

    column: exp.Column
    reason: UnresolvableReason
    relations: list[UnresolvableRelation]
    """The relations the reason is about, in the order the SQL brought them in.

    `not_projected`: the closed relations that failed to project the name.
    `several_undeclared_sources`: the open relations that could each have owned it.
    `no_such_source`: empty - the qualifier named nothing to describe.
    """


class ProjectionSite(NamedTuple):
    """One place a scope's projection list produces an output column."""

    from_star: bool
    span: SourceSpan | None
    """The text responsible for the projection: the projection itself when it was written
    out, and the `*` it came from when it was not. None when neither can be located."""


class DuplicateProjection(NamedTuple):
    """One output name a scope projects more than once.

    Impossible SQL: a relation has one column per name, so nothing downstream can say
    which of the two a later reference means. sqlglot does not raise on it - it simply
    stops expanding stars over the relation, which turns the mistake into a silently
    empty projection several scopes away.
    """

    scope_name: str
    scope_kind: ScopeKind
    column_name: ColumnName
    sites: list[ProjectionSite]
    """Every projection carrying the name, in projection-list order."""


class ResolvedColumns(NamedTuple):
    """What the probe pass learned: which columns each real table is asked for, how they
    were read, which columns no source can own, and which were attributed uncertainly."""

    columns_per_relation: dict[RelationKey, dict[ColumnName, ParsedColumn]]
    """Keyed by the full relation name, not the bare table name - see `RelationKey`. A
    column reaches a relation here either by being read off it directly or by passing
    through an open relation's star, which is the only way `address.sql` resolves at all."""
    storage_keys: set[RelationKey]
    """Every real table the statement reads, whether or not it names a column of it.

    A `select *` names none, so the table would otherwise be absent from the gap-filled
    schema entirely and step 6 would have nothing to expand the star against."""
    unresolvable_columns: list[UnresolvableColumn]
    columns_read_with_unsupported_dot_notation: list[exp.Column]
    """Dotted names absorbed as struct reads by a dialect that has no such syntax. Still
    harvested into `columns_per_table` - the report is the point, and dropping them only
    makes step 3 fail with a worse message."""
    ambiguous_columns: list[AmbiguousColumn]
    """Deliberately *not* harvested into `columns_per_relation`: recording the probe's pick
    would fabricate a declaration slot on a table that may not own the column, and every
    type inferred from it downstream would inherit the mistake."""
    relations_read_through_an_unexpandable_star: set[RelationKey]
    """Tables whose column set was fabricated from reads rather than declared.

    What the schema ends up saying about these is a *lower bound* - the columns this file
    happens to name, not the columns the table has - so any projection expanded from a star
    over one of them is a lower bound too. `qualify.py` propagates that through the scope
    graph; `validate-schema` needs it to avoid reporting a complete declaration's columns
    as missing from an under-approximated projection."""
    guessed_columns: list[GuessedColumn]
    """Harvested normally. The guess is the best answer available, and dropping it would
    only cost the column its declared type."""
    offsets_of_columns_written_without_a_source: set[tuple[int, int]]
    """The token hull of every column the user wrote bare, recorded before the probe
    qualified them. Offsets survive qualification, so this is what tells a qualifier the
    user typed apart from one the resolution supplied - the distinction is gone from the
    tree itself, and re-deriving it later would need the pre-probe statement back."""
