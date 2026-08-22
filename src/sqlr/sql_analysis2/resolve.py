"""Steps 2 and 4-5 - attribute every column to the relation that owns it.

A probe qualification against an *empty* schema gives every bare column a qualifier, the
closure in `relations.py` says what each relation projects, and one verdict table here puts
the two together. Nothing in this module writes a type: the output is a column *set* per
relation, which is what step 5 turns into a gap-filled schema and step 6 needs to qualify.

The distinction that makes this work is `relations.py`'s: which relation a column *reads
from* is not which storage *owns* it. A column read off a CTE that projects a star nobody
can expand belongs to whatever that star reads, transitively, and attributing it to the CTE
fabricates nothing at all.
"""

import logging
from typing import Literal, NamedTuple, cast

from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope

from sqlr.config.types import StarOverJoinBehavior
from sqlr.declared.types import DeclaredSchemas
from sqlr.sql_analysis2.relations import (
    RelationClosure,
    RelationColumnSet,
    aliases_a_bare_column_could_read,
    relation_closure,
)
from sqlr.sql_analysis2.sourcedoc import token_offsets_of
from sqlr.sql_analysis2.types import (
    AmbiguityKind,
    AmbiguousColumn,
    ClosingDeclaration,
    ColumnName,
    ColumnTypeName,
    GuessedColumn,
    ParsedColumn,
    RelationKey,
    ResolvedColumns,
    StructuredAccessKind,
    UnresolvableColumn,
    UnresolvableReason,
    UnresolvableRelation,
)

logger = logging.getLogger(__name__)

UNKNOWN_TYPE = "UNKNOWN"
"""What a gap-filled slot nobody described holds. A real `sqlglot` DType, not an invented
sentinel, so every later pass already understands it."""


DIALECTS_WITH_DOT_FIELD_ACCESS = frozenset(
    {"duckdb", "spark", "databricks", "bigquery", "hive", "trino", "presto", "athena"}
)
"""Dialects that read `a.b` as a field of the structured column `a` when `a` names no
source. Postgres wants `(a).b` and Snowflake wants `a:b`, so for them a dotted name that
matches no source is a mistake instead."""


def dialect_parses_unresolvable_aliases_as_json_columns(dialect_name: str) -> bool:
    """Whether a qualifier naming no source is legitimately a structured column read.

    Internal for now; a candidate for `GeneralConfig` if a dialect ever needs overriding.
    """
    return dialect_name.lower() in DIALECTS_WITH_DOT_FIELD_ACCESS


def structured_access_of(column: exp.Column) -> StructuredAccessKind | None:
    """The access one column node carries, or None when it is read as a plain value."""
    parent = column.parent
    if parent is None or parent.args.get("this") is not column:
        return None
    if isinstance(parent, exp.Dot):
        return "dot_field"
    if isinstance(parent, exp.Bracket):
        key = parent.expressions[0] if parent.expressions else None
        if isinstance(key, exp.Literal) and key.is_string:
            return "bracket_key"
        return "bracket_index"
    return None


# ------------------------------------------------------------------- step 4: attribution
type AttributionVerdict = Literal["resolved", "ambiguous", "guessed", "unresolvable"]


class ColumnAttribution(NamedTuple):
    """Where one column read belongs, and how sure that is."""

    verdict: AttributionVerdict
    relation: RelationColumnSet | None
    """The relation it reads from. None when nothing could be chosen."""
    storage: RelationKey | None
    """The table to fabricate a schema slot on, when there is one. None for a column of a
    CTE or derived table: its type is computed by annotation, and nobody declares it."""
    candidates: list[str]
    """Ambiguous: every relation that could own it. Guessed: the ones not ruled out.
    Sorted, and named the way the reader would write them."""
    ambiguity: AmbiguityKind | None = None
    """Which dead end an ambiguous column hit - the two have different fixes."""
    reason: UnresolvableReason | None = None


def attribute_a_column(
    column_name: ColumnName,
    candidates: list[RelationColumnSet],
    star_over_join_behavior: StarOverJoinBehavior,
) -> ColumnAttribution:
    """Which relation owns one column name, given everything it could be reading from.

    One table, six rows, and every attribution in the pipeline goes through it:

    | condition                                    | verdict                             |
    |----------------------------------------------|-------------------------------------|
    | two or more relations *project* the name     | ambiguous - nothing can break the tie
    | exactly one does                             | resolved, or guessed when an open   |
    |                                              | relation could have projected it too|
    | none do, no open relation                    | unresolvable - `not_projected`, or  |
    |                                              | `no_such_source` with no candidate  |
    | none do, several open relations              | unresolvable -                      |
    |                                              | `several_undeclared_sources`. A bare|
    |                                              | name over two relations nobody can  |
    |                                              | enumerate has no answer, and        |
    |                                              | inventing one is a coin flip        |
    | none do, one open relation, one origin       | resolved through its star           |
    | none do, one open relation, several origins  | `star_over_join_behavior`           |

    A relation known *not* to project the name is ruled out and counts for nothing, which is
    what keeps a closed relation from turning every column it omits into ambiguity rather
    than the error it is.

    The last two rows are the only place `star_over_join_behavior` applies, and the row above
    them is why: several *relations* that might own a name is a different question from one
    relation whose star reads several *tables*. The first has a fix the reader can apply -
    qualify the column - and the second does not.
    """
    projecting = [relation for relation in candidates if column_name in relation.known]
    open_relations = [relation for relation in candidates if relation.is_open]

    if len(projecting) >= 2:
        return ColumnAttribution(
            verdict="ambiguous",
            relation=None,
            storage=None,
            candidates=sorted(relation.alias for relation in projecting),
            ambiguity="projected_by_several",
        )

    if projecting:
        relation = projecting[0]
        # A relation that projects the name is not itself a reason to doubt, but any *other*
        # relation in the scope that cannot enumerate its columns might project it too.
        others = [other for other in open_relations if other is not relation]
        return ColumnAttribution(
            verdict="guessed" if others else "resolved",
            relation=relation,
            storage=relation.storage if relation.kind == "table" else None,
            candidates=sorted(other.alias for other in others),
        )

    if len(open_relations) != 1:
        # Zero or several open relations, and nothing projects the name. Which of the three
        # reasons it is decides the whole message, because none of them share a fix: no
        # candidate at all is a mistyped qualifier, candidates that are all closed is a
        # mistyped column or an over-complete declaration, and several open ones is a name
        # nobody can place until it is qualified.
        if not candidates:
            reason: UnresolvableReason = "no_such_source"
        elif open_relations:
            reason = "several_undeclared_sources"
        else:
            reason = "not_projected"
        return ColumnAttribution(
            verdict="unresolvable",
            relation=None,
            storage=None,
            candidates=sorted(relation.alias for relation in open_relations),
            reason=reason,
        )

    # Nobody projects it and one relation is transparent, so the name reaches whatever that
    # relation's star reads.
    relation = open_relations[0]
    origins = list(relation.open_origins)
    if len(origins) > 1 and star_over_join_behavior == "error":
        return ColumnAttribution(
            verdict="ambiguous",
            relation=relation,
            storage=None,
            candidates=sorted(origins),
            ambiguity="star_over_join",
        )

    return ColumnAttribution(
        verdict="resolved" if len(origins) == 1 else "guessed",
        relation=relation,
        storage=origins[0],
        candidates=sorted(origins[1:]),
    )


def describe_relation_for_a_failed_read(
    relation: RelationColumnSet, closure: RelationClosure
) -> UnresolvableRelation:
    """One relation, flattened into what a message about it needs.

    The declarations come from `relation.closed_by` rather than from `relation.declaration`,
    and the difference is the point: a CTE has no declaration of its own but is closed
    because the tables under its star are, and those are the entries a reader can edit.

    `storage` is preferred over `alias` for the display name because it is the relation as
    the SQL wrote it - `mydatabase.myschema.raw_address` rather than the bare table name the
    probe left behind. A CTE has no storage, so its alias is lowercased instead: the probe
    normalised it to the dialect's case, which for Snowflake means shouting a name the user
    typed in lower case.
    """
    return UnresolvableRelation(
        display_name=relation.storage or relation.alias.lower(),
        kind=relation.kind,
        projected_columns=sorted(relation.known),
        closing_declarations=[
            ClosingDeclaration(
                relation_name=key,
                where=declaration.where,
                declared_columns=sorted(
                    column.name.lower() for column in declaration.columns
                ),
            )
            for key in relation.closed_by
            if (declaration := closure.declarations.get(key)) is not None
        ],
    )


def _relations_behind_an_unresolvable_column(
    reason: UnresolvableReason,
    candidates: list[RelationColumnSet],
    closure: RelationClosure,
) -> list[UnresolvableRelation]:
    """The relations a failed read's message is about.

    `several_undeclared_sources` is about the *open* ones only: a closed relation in the
    same scope was ruled out for certain and naming it would offer the reader a fix that
    changes nothing.
    """
    relevant = (
        [relation for relation in candidates if relation.is_open]
        if reason == "several_undeclared_sources"
        else candidates
    )
    return [
        describe_relation_for_a_failed_read(relation, closure) for relation in relevant
    ]


def _candidate_relations(
    column: exp.Column, scope: Scope, closure: RelationClosure, written_bare: bool
) -> list[RelationColumnSet]:
    """Everything one column read could be resolving to.

    A column the user qualified has exactly one candidate - the relation its qualifier
    names - or none at all when the qualifier names nothing.

    A column written *bare* has every source the scope selected, and `written_bare` is why
    this cannot read the tree: the probe has already qualified it against the empty schema,
    landing it on whichever source `infer_schema` fell back to. Taking that qualifier at
    face value would turn every fallback into a certainty, and the probe's pick is exactly
    what has to be graded rather than trusted.
    """
    if written_bare:
        relations = closure.of_scope(scope)
        return [
            relation
            for alias in aliases_a_bare_column_could_read(scope)
            if (relation := relations.get(alias)) is not None
        ]
    if column.table:
        # Outward, not just here: a correlated subquery reads the outer query's aliases.
        named = closure.relation_named(column.table, scope)
        return [named] if named is not None else []
    return []


def resolve_columns_to_source_tables(
    statement: exp.Expr,
    dialect_name: str,
    declared: DeclaredSchemas,
    star_over_join_behavior: StarOverJoinBehavior = "guess",
    allow_unresolvable_aliases_as_structured_columns: bool | None = None,
) -> tuple[ResolvedColumns, RelationClosure]:
    """Attribute every column to the relation that owns it.

    **Step 2 - the probe.** Qualify a *throwaway copy* against an **empty** schema. That is
    the whole trick: with no schema every real table reports zero known columns, so it
    becomes the `infer_schema` fallback target, while CTEs and derived tables still report
    their own projections. Bare columns land on the one table that could own them, and only
    a genuinely ambiguous name is left unqualified. Passing the *declared* schema instead
    would defeat it - a partially declared table stops being the fallback target and its
    undeclared columns fail to resolve.

    `Resolver.get_table` also brings join-context disambiguation, USING expansion, set-op
    and lateral derivation and alias shadowing; re-deriving those here would be a worse
    copy. The copy is discarded - only the harvested names are wanted.

    **Step 3 - the closure**, in `relations.py`: what each relation projects, and what it
    stays transparent to.

    **Step 4 - attribution**, `attribute_a_column` above, once per column read.

    sqlglot's `_convert_columns_to_dots` rewrites any qualifier naming no source into a
    struct field read, so `select ghots.id from test` comes back as a column `ghots` of
    `test`. The rewrite is not dialect-gated, so this gates it:
    `allow_unresolvable_aliases_as_structured_columns` false makes such a read a finding
    rather than a column. It only fires when a lone source can absorb the name; with two
    sources the qualifier resolves to nothing and is unresolvable instead.
    """
    if allow_unresolvable_aliases_as_structured_columns is None:
        allow_unresolvable_aliases_as_structured_columns = (
            dialect_parses_unresolvable_aliases_as_json_columns(dialect_name)
        )

    # Which columns were written without a qualifier, recorded before the probe rewrites
    # them. Token offsets survive `copy()` and survive `qualify` - the synthesised table
    # identifier it adds carries none of its own - so the hull of a qualified column is
    # still the hull of the name the user typed, and identifies it across the two trees.
    offsets_of_columns_written_without_a_source = {
        offsets
        for column in statement.find_all(exp.Column)
        if not column.table and (offsets := token_offsets_of(column)) is not None
    }

    probe = qualify(
        statement.copy(),
        dialect=dialect_name,
        schema={},
        infer_schema=True,
        # A column qualified against a table whose columns are unknown is not an error.
        allow_partial_qualification=True,
        # A star cannot expand against an empty schema, and the probe does not need it to.
        expand_stars=False,
        # Unresolvable columns are reported by us, with a span; they must not raise here.
        validate_qualify_columns=False,
        quote_identifiers=False,
    )

    scopes = traverse_scope(probe)
    closure = relation_closure(scopes, declared)

    columns_per_relation: dict[RelationKey, dict[ColumnName, ParsedColumn]] = {}
    unresolvable_columns: list[UnresolvableColumn] = []
    columns_read_with_unsupported_dot_notation: list[exp.Column] = []
    ambiguous_columns: list[AmbiguousColumn] = []
    guessed_columns: list[GuessedColumn] = []
    read_through_a_star: set[RelationKey] = set()
    # `qualify` can clone a projection into GROUP BY or ORDER BY, so one written column can
    # arrive here twice. It is one mistake either way, and deserves one finding.
    offsets_already_judged: set[tuple[int, int]] = set()

    for scope in scopes:
        for column in scope.columns:
            name = column.name.lower()
            offsets = token_offsets_of(column)
            written_bare = offsets in offsets_of_columns_written_without_a_source
            candidates = _candidate_relations(column, scope, closure, written_bare)
            attribution = attribute_a_column(name, candidates, star_over_join_behavior)

            first_time = offsets is None or offsets not in offsets_already_judged
            if offsets is not None:
                offsets_already_judged.add(offsets)

            if attribution.verdict == "unresolvable":
                if first_time:
                    reason = attribution.reason or "no_such_source"
                    unresolvable_columns.append(
                        UnresolvableColumn(
                            column=column,
                            reason=reason,
                            relations=_relations_behind_an_unresolvable_column(
                                reason, candidates, closure
                            ),
                        )
                    )
                continue

            if attribution.verdict == "ambiguous":
                if first_time:
                    # Attributing it at all would fabricate a declaration slot on a table
                    # that may not own the column.
                    ambiguous_columns.append(
                        AmbiguousColumn(
                            column=column,
                            candidate_sources=attribution.candidates,
                            kind=attribution.ambiguity or "projected_by_several",
                        )
                    )
                continue

            relation = attribution.relation
            assert relation is not None
            if attribution.verdict == "guessed" and first_time:
                guessed_columns.append(
                    GuessedColumn(
                        column=column,
                        resolved_source=attribution.storage or relation.alias,
                        open_sources=attribution.candidates,
                    )
                )

            access = structured_access_of(column)
            if access == "dot_field" and not allow_unresolvable_aliases_as_structured_columns:
                columns_read_with_unsupported_dot_notation.append(column)

            if attribution.storage is None:
                # A column of a CTE or derived table. Its type is computed by annotation,
                # so there is no declaration slot to fabricate.
                continue

            if name not in relation.known:
                # It reached the table through a star nobody could expand, so what the
                # schema ends up saying about that table is a lower bound.
                read_through_a_star.add(attribution.storage)

            columns = columns_per_relation.setdefault(attribution.storage, {})
            parsed = columns.setdefault(name, ParsedColumn(name=name, structured_access=[]))
            if access is not None:
                parsed.structured_access.append((access, column))

    resolved = ResolvedColumns(
        columns_per_relation=columns_per_relation,
        storage_keys=closure.storage_keys,
        unresolvable_columns=unresolvable_columns,
        columns_read_with_unsupported_dot_notation=columns_read_with_unsupported_dot_notation,
        ambiguous_columns=ambiguous_columns,
        guessed_columns=guessed_columns,
        relations_read_through_an_unexpandable_star=read_through_a_star,
        offsets_of_columns_written_without_a_source=offsets_of_columns_written_without_a_source,
    )
    return resolved, closure


# --------------------------------------------------------------------- step 5: gap-fill
def type_name_is_recognized(written: ColumnTypeName, dialect_name: str) -> bool:
    """Whether the dialect can parse a written `data_type:` into a real type.

    The discriminator is `udt=False`. With sqlglot's default the name comes back as a
    *user-defined* type instead of raising, and a user-defined type belongs to no family -
    so `in_family` answers `False` for every family and a yml typo arrives as a confident
    `contradicted-type` error about the SQL. `json`, `variant`, `struct(...)` and `map(...)`
    all parse and are not affected; only a name nothing recognises fails here.
    """
    try:
        exp.DataType.build(written, dialect=dialect_name, udt=False)
    except Exception:
        # sqlglot raises ParseError, but a malformed parameter list can surface as others,
        # and every one of them means the same thing: this is not a usable type name.
        return False
    return True


def unrecognized_declared_types(
    declared: dict[RelationKey, dict[ColumnName, ColumnTypeName]], dialect_name: str
) -> dict[RelationKey, set[ColumnName]]:
    """Every declared type name the dialect cannot parse, by relation."""
    out: dict[RelationKey, set[ColumnName]] = {}
    for key, columns in declared.items():
        for column, written in columns.items():
            if is_declared(written) and not type_name_is_recognized(written, dialect_name):
                out.setdefault(key, set()).add(column)
    return out


def get_declared_types_per_relation(
    declared: dict[RelationKey, dict[ColumnName, ColumnTypeName]],
    column_names_per_relation: dict[RelationKey, dict[ColumnName, ParsedColumn]],
    storage_keys: set[RelationKey],
    unusable: dict[RelationKey, set[ColumnName]] | None = None,
) -> dict[RelationKey, dict[ColumnName, ColumnTypeName]]:
    """Complete the schema's *column set* so `qualify` does not raise.

    Declared types where the user wrote them, UNKNOWN everywhere else. Not optional:
    `qualify` validates that every column resolves, and every built-in escape
    (`allow_partial_qualification`, `infer_schema`) still raises.

    The column set is the *union* of the two directions, and needs both. What the SQL names
    but nobody declared has to be here or step 6 raises. What is declared but the SQL never
    names has to be here or `select *` expands to nothing - the star names no column, so the
    harvest alone leaves the table empty and the star survives step 6 silently.

    A table with neither is dropped rather than declared empty: a table read only through a
    star that nobody can expand is a table nobody can enumerate, and saying it has no
    columns would turn every read of it into an error instead of an open question.

    The "typo trap" - a misspelled column silently invented here as a real one - is caught
    upstream by step 4. A name no relation can own comes back in
    `ResolvedColumns.unresolvable_columns`, and against a *closed* relation, which is what
    a complete declaration makes one, that now includes a name the declaration omits.
    """
    unusable = unusable or {}
    out: dict[RelationKey, dict[ColumnName, ColumnTypeName]] = {}
    for key in set(column_names_per_relation) | storage_keys:
        declared_here = declared.get(key, {})
        rejected = unusable.get(key, set())
        columns = set(column_names_per_relation.get(key, {})) | set(declared_here)
        if not columns:
            continue
        out[key] = {
            # A name the dialect cannot parse is worth exactly as much as writing nothing,
            # and is worth strictly less than a user-defined type sqlglot would invent - see
            # `type_name_is_recognized`. So it falls back to UNKNOWN and gets inferred.
            column: UNKNOWN_TYPE
            if column in rejected
            else declared_here.get(column, UNKNOWN_TYPE)
            for column in sorted(columns)
        }
    return out


def nested_schema_for_sqlglot(
    types_per_relation: dict[RelationKey, dict[ColumnName, ColumnTypeName]],
) -> dict[str, object]:
    """A `RelationKey`-keyed schema in the nested shape `MappingSchema` resolves against.

    `{"mydatabase.myschema.raw_address": {...}}` becomes
    `{"mydatabase": {"myschema": {"raw_address": {...}}}}`, which is how sqlglot matches a
    table node's own `catalog`/`db`/`name` parts. Handing it the flat dotted key instead
    would only ever match a table written as one identifier, and merging on the bare name
    would collide two `raw_department`s in different schemas - which is the whole reason
    `RelationKey` carries the parts.
    """
    nested: dict[str, object] = {}
    for key, columns in types_per_relation.items():
        *prefix, table = key.split(".")
        node: dict[str, object] = nested
        for part in prefix:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = cast("dict[str, object]", child)
        node[table] = dict(columns)
    return nested


def needs_inference(schema: dict[RelationKey, dict[ColumnName, ColumnTypeName]]) -> bool:
    """Whether the inference phase has anything to do. O(columns), not O(nodes)."""
    return any(not is_declared(name) for cols in schema.values() for name in cols.values())


def is_declared(type_name: ColumnTypeName | None) -> bool:
    """Whether a slot in a gap-filled schema came from the user rather than from step 5.

    `UNKNOWN` is what `get_declared_types_per_relation` writes into a slot nobody
    described, so it is the one type name that means "not declared".
    """
    return type_name is not None and type_name.upper() != UNKNOWN_TYPE
