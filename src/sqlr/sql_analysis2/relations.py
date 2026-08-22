"""Step 3 - work out what every relation in a statement is known to project.

The pipeline needs two different answers about a column and only ever computed one. Which
relation a column *reads from* is lexical, scope-local, and sqlglot answers it during the
probe. Which storage *owns* it - the thing a schema slot can be fabricated on - is
transitive, and is what this module answers.

A relation is **closed** when its column set can be enumerated and **open** when it projects
a star over something that cannot be. An open relation is *transparent*: a name it does not
produce itself passes straight through to whatever its star reads. `address.sql` is the
case that forces this:

    with ranked as (select *, row_number() over (...) as rn from raw_address),
         current_address as (select street, city from ranked ...)

`raw_address` is undeclared, so its star cannot expand, so `ranked` projects `['*', 'rn']`
and `street` belongs to no relation anyone can name. Attributing it to `ranked` fabricates
nothing and step 6 then fails to qualify; attributing it to `raw_address` - through the
star, transitively - is the answer, and is what the closure computed here provides.

Transparency chains: `a` selects `*` from `b`, `b` selects `*` from undeclared `t`, and a
column read off `a` is owned by `t`. That is why this is a walk over the relation graph
rather than a lookup in one scope.
"""

import logging
from typing import NamedTuple

from sqlglot import exp
from sqlglot.optimizer.scope import Scope

from sqlr.declared.types import DeclaredRelation, DeclaredSchemas
from sqlr.sql_analysis2.types import (
    ColumnName,
    ColumnTypeName,
    RelationAlias,
    RelationKey,
    ScopeKind,
    SourceKind,
)

logger = logging.getLogger(__name__)


def name_and_kind_of_scope(scope: Scope) -> tuple[str, ScopeKind]:
    """What to call a scope in a report.

    A scope has no name of its own; what names it is the thing that holds it, so the
    answer comes from the parent node. An unnamed one is a set-operation arm, which is
    reported as a branch rather than given an invented name.
    """
    parent = scope.expression.parent
    if isinstance(parent, exp.CTE):
        return parent.alias, "cte"
    if isinstance(parent, exp.Subquery):
        return parent.alias or "<subquery>", "derived"
    if isinstance(parent, exp.SetOperation):
        return "<branch>", "branch"
    return "<final>", "final"


def relation_key_of(table: exp.Table) -> RelationKey:
    """A table node's identity: the parts the SQL wrote, lowercased and dotted.

    The same string `DeclaredSourceTable.key` produces, so the two sides of a declaration
    lookup are spelled the same way. Bare `employee` stays bare - an omitted part is one
    the SQL does not write, never one filled in from somewhere else, because sqlr has no
    target profile to fill it from.
    """
    parts = (table.catalog, table.db, table.name)
    return ".".join(part.lower() for part in parts if part)


class RelationColumnSet(NamedTuple):
    """What one relation is known to project, and what it is transparent to."""

    alias: RelationAlias
    """The name a column in the scope qualifies itself with."""
    kind: SourceKind
    storage: RelationKey | None
    """The real table behind the alias, when this is a table. None for a CTE or derived
    table, whose columns are computed rather than stored."""
    declaration: DeclaredRelation | None
    """The yml entry describing it, when one does."""
    known: frozenset[ColumnName]
    """Names this relation certainly projects, lowercased."""
    open_origins: tuple[RelationKey, ...]
    """Storage tables reachable through a star this relation could not expand.

    Empty means **closed**: a name not in `known` is a mistake rather than a pass-through,
    and that is the entire payoff for declaring a table's columns. Non-empty means a name
    not in `known` belongs to one of these instead.
    """
    closed_by: tuple[RelationKey, ...] = ()
    """Storage tables whose complete declaration is *why* this relation is closed.

    `open_origins` says a relation is closed; this says who to blame for it, which is what
    a message needs to name a yml entry the reader can edit. Transitive for the same reason
    `open_origins` is: a CTE projecting `select *` over a fully declared table is closed
    only because that table is, so the entry to point at is the table's.

    Empty for a relation closed by its own written projection list - a CTE that names its
    columns is closed because the SQL says so, and no declaration is involved.
    """

    @property
    def is_open(self) -> bool:
        return bool(self.open_origins)


class RelationClosure(NamedTuple):
    """Every relation of one statement, described per scope."""

    per_scope: dict[int, dict[RelationAlias, RelationColumnSet]]
    """Keyed by `id(scope.expression)`. Safe because the caller holds the probe tree alive
    across the whole of steps 3 and 4, and nothing rewrites it in between."""
    declared_types: dict[RelationKey, dict[ColumnName, ColumnTypeName]]
    """Declared types of every table this statement actually reads, keyed the way the SQL
    writes the relation. Only the tables read: a lookup was needed to find each one, and
    doing it once here is what keeps the rest of the pipeline off `DeclaredSchemas`."""
    declarations: dict[RelationKey, DeclaredRelation]
    """The yml entries behind `declared_types`, for messages that need to say where a
    declaration was written."""
    storage_keys: set[RelationKey]
    """Every real table the statement reads, whether or not it names a column of one.

    A `select *` names none, so the table would otherwise be absent from the gap-filled
    schema entirely and step 6 would have nothing to expand the star against.
    """

    def of_scope(self, scope: Scope) -> dict[RelationAlias, RelationColumnSet]:
        return self.per_scope.get(id(scope.expression), {})

    def relation_named(self, alias: RelationAlias, scope: Scope) -> RelationColumnSet | None:
        """The relation an alias names, looking outward until something answers.

        A correlated subquery reads its outer query's aliases -
        `where exists (select 1 from customer c where c.id = o.customer_id)` names `o`,
        which belongs to the enclosing scope and appears nowhere in this one. Stopping at
        the innermost scope reports every such column as a mistyped alias, which is the one
        thing it certainly is not.
        """
        current: Scope | None = scope
        while current is not None:
            found = self.of_scope(current).get(alias)
            if found is not None:
                return found
            current = current.parent
        return None


def _column_set_of_table(
    alias: RelationAlias, table: exp.Table, declared: DeclaredSchemas
) -> RelationColumnSet:
    """One real table, graded by what its declaration says.

    Three states, and the flag chooses between the last two:

    - **undeclared** - nothing is known and everything passes through.
    - **complete** (the default) - the declaration is the whole column set, so the relation
      is closed and a name it omits is an error naming the yml entry.
    - **partial** - `meta.declaration_is_partial: true`. The declared columns keep their
      types and everything else passes through exactly as for an undeclared table.

    A declaration carrying no columns is read as undeclared rather than as an empty
    relation: describing a table without listing its columns says nothing about them, and
    calling that "has no columns" would turn every read of it into an error.
    """
    key = relation_key_of(table)
    declaration = declared.for_relation(key)
    names: frozenset[ColumnName] = (
        frozenset(column.name.lower() for column in declaration.columns)
        if declaration is not None
        else frozenset()
    )
    is_closed = bool(names) and not (
        declaration is not None and declaration.declaration_is_partial
    )
    return RelationColumnSet(
        alias=alias,
        kind="table",
        storage=key,
        declaration=declaration,
        known=names,
        open_origins=() if is_closed else (key,),
        closed_by=(key,) if is_closed else (),
    )


def aliases_a_bare_column_could_read(scope: Scope) -> list[RelationAlias]:
    """The sources of this scope, in the order the SQL brought them in.

    `Scope.sources` also holds every CTE the statement defines, whether or not this scope
    selects from it - so it is the wrong set for both questions that need one. An
    unqualified column can only be reading from what a FROM or JOIN actually brought in
    (`selected_sources`) or what a lateral added, which is the same pair sqlglot's own
    `Resolver` consults; and a `select *` can only be expanding those.

    Order is FROM then JOINs, and it is load-bearing: it decides which origin
    `star_over_join_behavior: guess` credits a column to, and a guess that moves between
    runs is worse than either answer.
    """
    aliases = list(scope.selected_sources)
    already = set(aliases)
    aliases.extend(name for name in scope.lateral_sources if name not in already)
    return aliases


def _aliases_covered_by_a_star(
    select: exp.Select, selected: list[RelationAlias]
) -> list[RelationAlias]:
    """Which of a scope's selected sources its projection list reads through a star.

    `select *` covers every source the FROM and JOINs brought in; `select t.*` covers `t`
    alone; a projection list with no star covers nothing. That distinction is what stops
    `select a.*, b.x from a join b` from making `b` transparent.

    A qualifier is compared as written, never lowercased: `scope.sources` is keyed the way
    the probe normalised identifiers, and that follows the dialect - Snowflake folds them
    up, DuckDB down. Lowercasing one side of the comparison silently matched nothing under
    Snowflake, which turned every `select t.*` there into a relation projecting nothing.
    """
    covered: list[RelationAlias] = []
    for projection in select.expressions:
        if isinstance(projection, exp.Star):
            covered.extend(selected)
        elif isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star):
            qualifier = projection.table
            covered.extend([qualifier] if qualifier else selected)
    return list(dict.fromkeys(covered))


type OwnColumnSet = tuple[frozenset[ColumnName], tuple[RelationKey, ...], tuple[RelationKey, ...]]
"""What a scope projects, what it stays transparent to, and what closed it - the three
fields of `RelationColumnSet` a scope computes for itself, in that order."""


def _own_column_set_of_select(
    scope: Scope, sources: dict[RelationAlias, RelationColumnSet]
) -> OwnColumnSet:
    """What one SELECT projects, what it stays transparent to, and what closed it.

    A star over a *closed* source contributes that source's names, which is the expansion
    step 6 will perform once the schema exists. A star over an open one contributes its
    origins instead, and that is what makes transparency chain.

    A star over a closed source also inherits *why* it was closed, so a CTE that fails to
    project a name can send the reader to the declaration responsible rather than to
    itself. A projection list with no star at all inherits nothing: it is closed because
    the SQL enumerates it, and no yml entry has anything to do with that.
    """
    select = scope.expression
    assert isinstance(select, exp.Select)
    known = {name.lower() for name in select.named_selects if name != "*"}
    open_origins: list[RelationKey] = []
    closed_by: list[RelationKey] = []
    for alias in _aliases_covered_by_a_star(select, aliases_a_bare_column_could_read(scope)):
        source = sources.get(alias)
        if source is None:
            continue
        known.update(source.known)
        open_origins.extend(source.open_origins)
        closed_by.extend(source.closed_by)
    return (
        frozenset(known),
        tuple(dict.fromkeys(open_origins)),
        tuple(dict.fromkeys(closed_by)),
    )


def _own_column_set_of_set_operation(
    scope: Scope, own_sets: dict[int, RelationColumnSet]
) -> OwnColumnSet:
    """What a UNION and friends project.

    A set operation's schema is **positional**: the arms are matched by position and the
    left one supplies the names. So the names are the left arm's, not an intersection - and
    the relation is open if *any* arm is, because a name the left arm cannot enumerate is
    one nothing downstream can place.

    Every arm's `closed_by` is carried, not only the left one's: a name missing from a
    union is missing from each arm that could have supplied it, and a reader fixing it has
    to know about all of them.
    """
    arms = [own_sets.get(id(arm.expression)) for arm in scope.union_scopes]
    present = [arm for arm in arms if arm is not None]
    if not present:
        return frozenset(), (), ()
    open_origins = [origin for arm in present for origin in arm.open_origins]
    closed_by = [key for arm in present for key in arm.closed_by]
    return (
        present[0].known,
        tuple(dict.fromkeys(open_origins)),
        tuple(dict.fromkeys(closed_by)),
    )


def relation_closure(scopes: list[Scope], declared: DeclaredSchemas) -> RelationClosure:
    """Describe every relation of a statement, in dependency order.

    `traverse_scope` yields a scope only after everything it selects from, so each source
    is already closed by the time the scope reading it is reached and one pass suffices.
    """
    per_scope: dict[int, dict[RelationAlias, RelationColumnSet]] = {}
    own_sets: dict[int, RelationColumnSet] = {}
    declared_types: dict[RelationKey, dict[ColumnName, ColumnTypeName]] = {}
    declarations: dict[RelationKey, DeclaredRelation] = {}
    storage_keys: set[RelationKey] = set()

    for scope in scopes:
        sources: dict[RelationAlias, RelationColumnSet] = {}
        for alias, source in scope.sources.items():
            if isinstance(source, exp.Table):
                entry = _column_set_of_table(alias, source, declared)
                assert entry.storage is not None
                storage_keys.add(entry.storage)
                if entry.declaration is not None:
                    declarations[entry.storage] = entry.declaration
                    declared_types[entry.storage] = {
                        column.name.lower(): column.written_type
                        for column in entry.declaration.columns
                    }
            else:
                computed = own_sets.get(id(source.expression))
                kind = name_and_kind_of_scope(source)[1]
                entry = RelationColumnSet(
                    alias=alias,
                    kind=kind,
                    storage=None,
                    declaration=None,
                    known=computed.known if computed else frozenset(),
                    open_origins=computed.open_origins if computed else (),
                    closed_by=computed.closed_by if computed else (),
                )
            sources[alias] = entry
        per_scope[id(scope.expression)] = sources

        expression = scope.expression
        known: frozenset[ColumnName] = frozenset()
        open_origins: tuple[RelationKey, ...] = ()
        closed_by: tuple[RelationKey, ...] = ()
        if isinstance(expression, exp.SetOperation):
            known, open_origins, closed_by = _own_column_set_of_set_operation(
                scope, own_sets
            )
        elif isinstance(expression, exp.Select):
            known, open_origins, closed_by = _own_column_set_of_select(scope, sources)

        name, kind = name_and_kind_of_scope(scope)
        own_sets[id(expression)] = RelationColumnSet(
            alias=name,
            kind=kind,
            storage=None,
            declaration=None,
            known=known,
            open_origins=open_origins,
            closed_by=closed_by,
        )
        logger.debug(
            "%s %s: projects %s%s",
            kind,
            name,
            sorted(known) or "nothing",
            f", transparent to {list(open_origins)}" if open_origins else " (closed)",
        )

    return RelationClosure(
        per_scope=per_scope,
        declared_types=declared_types,
        declarations=declarations,
        storage_keys=storage_keys,
    )
