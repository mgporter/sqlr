"""Column-to-source resolution for the typing pipeline.

Every fixture here is an inline SQL string, per `tests/README.md`.
"""

from pathlib import Path


import pytest
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.schema import ensure_schema

from declared_helpers import declarations, q

from sqlr.catalog.types import SqlFile
from sqlr.config.types import StarOverJoinBehavior
from sqlr.declared.types import DeclaredSchemas
from sqlr.selection.types import Model
from sqlr.sql_analysis2.reporting import (
    findings_for_ambiguous_columns,
    findings_for_unresolvable_columns,
)
from sqlr.sql_analysis2.sourcedoc import Positions
from sqlr.sql_analysis2.qualify import (
    QualifiedModel,
    columns_per_scope,
    qualify_one_model,
)
from sqlr.sql_analysis2.resolve import (
    get_declared_types_per_relation,
    nested_schema_for_sqlglot,
    resolve_columns_to_source_tables,
)
from sqlr.sql_analysis2.types import (
    ColumnName,
    ColumnTypeName,
    ParsedColumn,
    RelationKey,
    ResolvedColumns,
    StructuredAccessKind,
)
DIALECT = "duckdb"


def resolve(
    sql: str,
    declared: dict[RelationKey, dict[ColumnName, ColumnTypeName]] | None = None,
    partial: frozenset[RelationKey] = frozenset(),
    star_over_join_behavior: StarOverJoinBehavior = "guess",
) -> ResolvedColumns:
    return resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read=DIALECT),
        DIALECT,
        declarations(declared, partial),
        star_over_join_behavior,
    )[0]


def column_names(resolved: ResolvedColumns) -> dict[RelationKey, set[ColumnName]]:
    """Just the names, for the tests that do not care how a column was read."""
    return {
        relation: set(columns)
        for relation, columns in resolved.columns_per_relation.items()
    }


def kinds(
    resolved: ResolvedColumns, relation: RelationKey, column: ColumnName
) -> list[StructuredAccessKind]:
    """How one column was read, one entry per site, in written order."""
    return [
        kind
        for kind, _ in resolved.columns_per_relation[relation][column].structured_access
    ]


def unresolvable_names(resolved: ResolvedColumns) -> list[str]:
    return [entry.column.name for entry in resolved.unresolvable_columns]


def ambiguous_names(resolved: ResolvedColumns) -> list[tuple[str, list[str]]]:
    """Each ambiguous column and the sources it could equally have come from."""
    return [
        (ambiguous.column.name, ambiguous.candidate_sources)
        for ambiguous in resolved.ambiguous_columns
    ]


def guessed_names(
    resolved: ResolvedColumns,
) -> list[tuple[str, str, list[str]]]:
    """Each guessed column, what it was credited to, and what was not ruled out."""
    return [
        (guessed.column.name, guessed.resolved_source, guessed.open_sources)
        for guessed in resolved.guessed_columns
    ]


def parsed(names: list[ColumnName]) -> dict[ColumnName, ParsedColumn]:
    """A `columns_per_relation` entry, for feeding the schema fabricator directly."""
    return {
        name: ParsedColumn(name=name, structured_access=[]) for name in names
    }


# ------------------------------------------------------- columns per real table
def test_qualified_columns_land_on_their_table() -> None:
    resolved = resolve("select t.id, t.name from mydatabase.myschema.test t")
    assert column_names(resolved) == {q("test"): {"id", "name"}}
    assert resolved.unresolvable_columns == []


def test_bare_columns_land_on_the_only_table() -> None:
    resolved = resolve("select id, name from mydatabase.myschema.test")
    assert column_names(resolved) == {q("test"): {"id", "name"}}


def test_bare_columns_resolve_past_a_joined_cte() -> None:
    """The regression this module was rewritten for.

    A scope that joins a real table to a CTE used to drop every bare column, because
    "one source" no longer held. The CTE's column set is knowable without a schema, so
    a name absent from it can only have come from the table.
    """
    resolved = resolve(
        """
        with src as (
          select test_id, name from mydatabase.myschema.table_in_cte
        )
        select
          id,
          test.name,
          date_trunc('day', modified_at) as modified_at,
          list_extract(titles, 1) as first_title
        from mydatabase.myschema.test
        inner join src on test.id = src.test_id
        where modified_at > cast('2023-01-01' as timestamp)
        """
    )
    assert column_names(resolved) == {
        q("test"): {"id", "name", "modified_at", "titles"},
        q("table_in_cte"): {"test_id", "name"},
    }
    assert resolved.unresolvable_columns == []


def test_cte_names_are_not_invented_as_table_columns() -> None:
    """`k` is the CTE's own projection, not a column of any real table."""
    resolved = resolve(
        "with a as (select 1 as k) select k, extra from a, mydatabase.myschema.test"
    )
    assert column_names(resolved) == {q("test"): {"extra"}}


def test_correlated_subquery_columns_reach_the_outer_table() -> None:
    resolved = resolve(
        """
        select id from mydatabase.myschema.test t
        where exists (
          select 1 from mydatabase.myschema.other o where o.fk = t.id and o.flag
        )
        """
    )
    assert column_names(resolved) == {q("test"): {"id"}, q("other"): {"fk", "flag"}}


def test_scalar_subquery_columns_land_on_their_own_table() -> None:
    resolved = resolve(
        """
        select id, (select max(amt) from mydatabase.myschema.other) as m
        from mydatabase.myschema.test
        """
    )
    assert column_names(resolved) == {q("test"): {"id"}, q("other"): {"amt"}}


def test_using_join_credits_both_tables() -> None:
    resolved = resolve(
        """
        select t.id from mydatabase.myschema.test t
        join mydatabase.myschema.other o using (id)
        """
    )
    assert column_names(resolved) == {q("test"): {"id"}, q("other"): {"id"}}


def test_set_operation_inside_a_cte() -> None:
    resolved = resolve(
        """
        with both as (
          select id from mydatabase.myschema.test
          union all
          select id from mydatabase.myschema.other
        )
        select id from both
        """
    )
    assert column_names(resolved) == {q("test"): {"id"}, q("other"): {"id"}}


def test_derived_table_shadows_its_source() -> None:
    resolved = resolve(
        """
        select outer_id
        from (select id as outer_id from mydatabase.myschema.test) sub
        """
    )
    assert column_names(resolved) == {q("test"): {"id"}}


def test_column_names_are_lowercased() -> None:
    resolved = resolve("select ID, Name from MyDatabase.MySchema.Test")
    assert column_names(resolved) == {q("test"): {"id", "name"}}


# ------------------------------------------------------------ unresolvable columns
def test_ambiguous_bare_column_across_two_unknown_tables_is_reported() -> None:
    """Two schema-less tables, so nothing can claim `mystery`. Reported, not invented."""
    resolved = resolve(
        """
        select a.x, mystery
        from mydatabase.myschema.test a
        inner join mydatabase.myschema.other b on a.id = b.id
        """
    )
    assert column_names(resolved) == {q("test"): {"x", "id"}, q("other"): {"id"}}
    assert unresolvable_names(resolved) == ["mystery"]


def test_bare_column_with_only_cte_sources_is_reported() -> None:
    resolved = resolve(
        """
        with src as (select test_id from mydatabase.myschema.table_in_cte)
        select test_id, nonsense from src
        """
    )
    assert column_names(resolved) == {q("table_in_cte"): {"test_id"}}
    assert unresolvable_names(resolved) == ["nonsense"]


def test_unresolvable_column_keeps_its_source_position() -> None:
    """The probe runs on a copy, so the reported node must still carry parse offsets."""
    sql = (
        "select t.id, nonsense\n"
        "from mydatabase.myschema.test t\n"
        "join mydatabase.myschema.other o on t.id = o.id"
    )
    resolved = resolve(sql)
    (unresolvable,) = resolved.unresolvable_columns
    identifier = unresolvable.column.this
    assert isinstance(identifier, exp.Identifier)
    start = identifier.meta["start"]
    end = identifier.meta["end"]
    assert sql[start : end + 1] == "nonsense"


@pytest.mark.parametrize("dialect", ["duckdb", "spark"])
def test_unknown_qualifier_on_a_lone_source_is_read_as_a_struct_field(
    dialect: str,
) -> None:
    """`_convert_columns_to_dots` reinterprets a qualifier that names no source as a
    STRUCT or JSON field lookup, so `ghots.id` comes back as a real column `ghots` of
    `test`. DuckDB and Spark really do read bare dotted field access that way, so this is
    a column and not a finding - one that has to hold a structured value.
    """
    resolved, _ = resolve_columns_to_source_tables(
        sqlglot.parse_one("select ghots.id from mydatabase.myschema.test", read=dialect),
        dialect,
        DeclaredSchemas(),
    )
    assert column_names(resolved) == {q("test"): {"ghots"}}
    assert kinds(resolved, q("test"), "ghots") == ["dot_field"]
    assert resolved.columns_per_relation[q("test")]["ghots"].requires_structured_type
    assert resolved.unresolvable_columns == []
    assert resolved.columns_read_with_unsupported_dot_notation == []


@pytest.mark.parametrize("dialect", ["postgres", "snowflake"])
def test_unknown_qualifier_is_reported_where_the_dialect_has_no_dot_access(
    dialect: str,
) -> None:
    """Postgres needs `(ghots).id` and Snowflake needs `ghots:id`, so a dotted name that
    matches no source is a mistake there. sqlglot rewrites it anyway - the gate is ours.

    The column is still harvested: the finding is already written, and leaving it out of
    the schema only makes step 3 fail with a message about the rewritten tree.
    """
    resolved, _ = resolve_columns_to_source_tables(
        sqlglot.parse_one("select ghots.id from mydatabase.myschema.test", read=dialect),
        dialect,
        DeclaredSchemas(),
    )
    assert [
        column.name.lower()
        for column in resolved.columns_read_with_unsupported_dot_notation
    ] == ["ghots"]
    assert column_names(resolved) == {q("test"): {"ghots"}}
    assert resolved.unresolvable_columns == []


def test_the_dot_access_gate_can_be_set_against_the_dialect_default() -> None:
    """The flag is internal for now, but it is the argument that decides, not the
    dialect - a config knob can hand it something else later."""
    statement = sqlglot.parse_one(
        "select ghots.id from mydatabase.myschema.test", read=DIALECT
    )
    resolved, _ = resolve_columns_to_source_tables(
        statement,
        DIALECT,
        DeclaredSchemas(),
        allow_unresolvable_aliases_as_structured_columns=False,
    )
    assert [
        column.name for column in resolved.columns_read_with_unsupported_dot_notation
    ] == ["ghots"]


# --------------------------------------------------------------- structured access
def test_bracket_access_survives_a_dialect_without_dot_access() -> None:
    """`ghots['id']` is a subscript, not a qualifier, so nothing rewrites it and every
    dialect here reads it as a column of `test`. Only the dotted form is a finding."""
    resolved, _ = resolve_columns_to_source_tables(
        sqlglot.parse_one(
            "select ghots['id'] from mydatabase.myschema.test", read="postgres"
        ),
        "postgres",
        DeclaredSchemas(),
    )
    assert resolved.columns_read_with_unsupported_dot_notation == []
    assert kinds(resolved, q("test"), "ghots") == ["bracket_key"]
    assert resolved.columns_per_relation[q("test")]["ghots"].requires_structured_type


def test_an_integer_subscript_does_not_demand_a_structured_type() -> None:
    """DuckDB subscripts strings, so `titles[1]` proves `titles` is indexable and nothing
    more. A `varchar` declaration for it is not a contradiction."""
    resolved = resolve("select titles[1] from mydatabase.myschema.test")
    column = resolved.columns_per_relation[q("test")]["titles"]
    assert kinds(resolved, q("test"), "titles") == ["bracket_index"]
    assert not column.requires_structured_type


def test_a_column_read_plainly_carries_no_structured_access() -> None:
    resolved = resolve("select id from mydatabase.myschema.test")
    assert resolved.columns_per_relation[q("test")]["id"].structured_access == []


def test_sites_merge_and_the_strongest_claim_wins() -> None:
    """One name, four reads, two shapes: `col.field` and `col['field']`. The merged kind
    is the strongest of them, and every site is kept for its span."""
    resolved = resolve(
        """
        select
          mistyped.col1 as col1,
          mistyped.col2.jsonfield as col2,
          mistyped['col3'] as col3,
          mistyped.col4['jsonfield'] as col4
        from mydatabase.myschema.test
        """
    )
    assert kinds(resolved, q("test"), "mistyped") == [
        "dot_field",
        "dot_field",
        "bracket_key",
        "dot_field",
    ]


def test_only_the_first_level_of_a_nested_read_is_modelled() -> None:
    """`a.b.c` says what `a.b` said: `a` is structured. Nothing can declare a type for
    `a.b`, so nothing needs to be recorded about it."""
    resolved = resolve("select a.b.c.d from mydatabase.myschema.test")
    assert column_names(resolved) == {q("test"): {"a"}}
    assert kinds(resolved, q("test"), "a") == ["dot_field"]


@pytest.mark.parametrize("dialect", ["duckdb", "snowflake", "postgres", "spark"])
def test_unknown_qualifier_with_two_sources_is_reported(dialect: str) -> None:
    """The other half: no lone source absorbs the name, so the typo surfaces."""
    resolved, _ = resolve_columns_to_source_tables(
        sqlglot.parse_one(
            """
            select ghots.id
            from mydatabase.myschema.a x
            join mydatabase.myschema.b y on x.k = y.k
            """,
            read=dialect,
        ),
        dialect,
        DeclaredSchemas(),
    )
    assert column_names(resolved) == {q("a"): {"k"}, q("b"): {"k"}}
    # Case follows the dialect's normalisation - Snowflake upper-cases - so compare
    # case-insensitively. Only `columns_per_relation` is lowercased, because it has to match
    # declarations; a reported column is pointed at its source text instead.
    assert [name.lower() for name in unresolvable_names(resolved)] == ["id"]


def test_column_qualified_against_a_cte_is_neither_harvested_nor_reported() -> None:
    """A CTE is a source, so `src.test_id` is resolved - it just is not a real table."""
    resolved = resolve(
        """
        with src as (select test_id from mydatabase.myschema.table_in_cte)
        select src.test_id, t.id
        from mydatabase.myschema.test t
        join src on t.id = src.test_id
        """
    )
    assert column_names(resolved) == {q("test"): {"id"}, q("table_in_cte"): {"test_id"}}
    assert resolved.unresolvable_columns == []


# --------------------------------------------- columns written without a source
JOINED_TO_A_CTE = """
with src as (select test_id, name from mydatabase.myschema.table_in_cte)
select {selection}
from mydatabase.myschema.test
inner join src on test.id = src.test_id
"""


def test_a_declared_column_set_is_read_as_the_complete_list() -> None:
    """The rule the whole check rests on.

    `name` is projected by the CTE and declared on `test`, so both certainly own it and
    neither can be preferred. Without the declaration sqlglot credits the CTE silently,
    because an undeclared table reports no columns at all.
    """
    resolved = resolve(
        JOINED_TO_A_CTE.format(selection="name"),
        {q("test"): {"id": "varchar(20)", "name": "varchar(20)"}},
    )
    assert ambiguous_names(resolved) == [("name", ["src", "test"])]
    assert guessed_names(resolved) == []


def test_an_ambiguous_column_is_not_credited_to_either_source() -> None:
    """Recording the probe's pick would fabricate a declaration slot on a table that may
    not own the column, and every type inferred from it would inherit the mistake."""
    resolved = resolve(
        JOINED_TO_A_CTE.format(selection="name"),
        {q("test"): {"id": "varchar(20)", "name": "varchar(20)"}},
    )
    assert "name" not in resolved.columns_per_relation[q("test")]
    assert "name" not in column_names(resolved)[q("table_in_cte")] - {"name"}


def test_two_declared_tables_that_both_own_the_name_are_ambiguous() -> None:
    """The probe leaves this one bare - against an empty schema, two tables that both own
    the name look exactly like two that both lack it. Only the declarations separate them,
    so the finding is an ambiguity rather than an unresolvable column."""
    resolved = resolve(
        """
        select name
        from mydatabase.myschema.test
        inner join mydatabase.myschema.other on test.id = other.id
        """,
        {
            q("test"): {"id": "int", "name": "varchar(20)"},
            q("other"): {"id": "int", "name": "varchar(20)"},
        },
    )
    assert ambiguous_names(resolved) == [("name", ["other", "test"])]
    assert unresolvable_names(resolved) == []


def test_an_undeclared_table_leaves_the_attribution_a_guess() -> None:
    resolved = resolve(JOINED_TO_A_CTE.format(selection="name"), {})
    assert guessed_names(resolved) == [("name", "src", ["test"])]
    assert ambiguous_names(resolved) == []


def test_a_declaration_with_no_columns_says_nothing() -> None:
    """An empty column list is an absent answer, not an empty one."""
    resolved = resolve(JOINED_TO_A_CTE.format(selection="name"), {q("test"): {}})
    assert guessed_names(resolved) == [("name", "src", ["test"])]


def test_a_guessed_column_is_still_credited_to_its_source() -> None:
    """Unlike an ambiguous column: the guess is the best answer available, and dropping
    it would only cost the column its declared type."""
    resolved = resolve(JOINED_TO_A_CTE.format(selection="id, name"), {})
    assert guessed_names(resolved) == [("name", "src", ["test"])]
    assert column_names(resolved)[q("table_in_cte")] == {"test_id", "name"}


def test_a_source_ruled_out_by_its_declaration_makes_no_noise() -> None:
    """`test` is declared and does not list `name`, so the CTE is the only candidate
    left and the attribution is forced rather than guessed."""
    resolved = resolve(
        JOINED_TO_A_CTE.format(selection="name"), {q("test"): {"id": "varchar(20)"}}
    )
    assert guessed_names(resolved) == []
    assert ambiguous_names(resolved) == []


def test_a_partly_declared_table_still_absorbs_the_columns_it_omits() -> None:
    """`declaration_is_partial: true`. `modified_at` appears in neither the CTE's
    projections nor `test`'s declaration, and the flag is what says `test` may still own
    it - the declared columns keep their types and the rest is inferred."""
    resolved = resolve(
        JOINED_TO_A_CTE.format(selection="modified_at"),
        {q("test"): {"id": "varchar(20)", "name": "varchar(20)"}},
        partial=frozenset({q("test")}),
    )
    assert guessed_names(resolved) == []
    assert ambiguous_names(resolved) == []
    assert "modified_at" in column_names(resolved)[q("test")]


def test_a_complete_declaration_makes_an_omitted_column_an_error() -> None:
    """The payoff for declaring. The same read against the same declaration, without the
    partial flag: `test` is closed, so `modified_at` is a mistake rather than a column
    invented from the read that nothing can tell apart from a real one."""
    resolved = resolve(
        JOINED_TO_A_CTE.format(selection="modified_at"),
        {q("test"): {"id": "varchar(20)", "name": "varchar(20)"}},
    )
    assert unresolvable_names(resolved) == ["modified_at"]
    assert column_names(resolved).get(q("test"), set()) == {"id"}


def test_a_qualified_column_is_never_judged() -> None:
    resolved = resolve(
        JOINED_TO_A_CTE.format(selection="src.name"),
        {q("test"): {"id": "varchar(20)", "name": "varchar(20)"}},
    )
    assert ambiguous_names(resolved) == []
    assert guessed_names(resolved) == []


def test_a_lone_source_cannot_be_ambiguous() -> None:
    resolved = resolve(
        "select id, name from mydatabase.myschema.test",
        {q("test"): {"id": "varchar(20)", "name": "varchar(20)"}},
    )
    assert ambiguous_names(resolved) == []
    assert guessed_names(resolved) == []


def test_a_projection_cloned_into_group_by_is_judged_once() -> None:
    """`group by 1` is expanded into a copy of the projection, so one written column
    reaches the walk twice. It is one mistake, and gets one finding."""
    resolved = resolve(
        JOINED_TO_A_CTE.format(selection="name, count(*)") + "group by 1",
        {q("test"): {"id": "varchar(20)", "name": "varchar(20)"}},
    )
    assert ambiguous_names(resolved) == [("name", ["src", "test"])]


def test_a_name_written_twice_is_reported_at_both_sites() -> None:
    """The mirror of the case above: `group by name` writes the column a second time,
    and an editor underlining only the first would leave the other unmarked."""
    resolved = resolve(
        JOINED_TO_A_CTE.format(selection="name, count(*)") + "group by name",
        {q("test"): {"id": "varchar(20)", "name": "varchar(20)"}},
    )
    assert ambiguous_names(resolved) == [
        ("name", ["src", "test"]),
        ("name", ["src", "test"]),
    ]


def qualify_one_statement(
    sql: str,
    declared: dict[RelationKey, dict[ColumnName, ColumnTypeName]],
    tmp_path: Path,
    warn_on_column_without_source: bool = True,
    dialect_name: str = DIALECT,
    partial: frozenset[RelationKey] = frozenset(),
    star_over_join_behavior: StarOverJoinBehavior = "guess",
) -> QualifiedModel:
    """Run steps 1-3 over one file.

    `statement is None` is how a test sees that a finding *stopped* the model rather than
    merely being reported alongside it: step 3 is the first thing a stopping finding skips.
    """
    path = tmp_path / "x.sql"
    path.write_text(sql)
    model = Model(
        name="x",
        file=SqlFile(path=path, relative_path="x.sql", mtime=0.0, content_hash=""),
    )
    return qualify_one_model(
        model,
        declarations(declared, partial),
        dialect_name,
        star_over_join_behavior,
        warn_on_column_without_source,
    )


def test_an_ambiguous_column_stops_the_statement(tmp_path: Path) -> None:
    """Committing to either source would hand `qualify` an attribution as likely wrong as
    right, and every type inferred downstream would inherit the choice."""
    result = qualify_one_statement(
        JOINED_TO_A_CTE.format(selection="name"),
        {q("test"): {"id": "varchar(20)", "name": "varchar(20)"}},
        tmp_path,
    )
    assert [finding.code for finding in result.findings] == ["ambiguous-column"]
    assert result.has_errors
    assert result.statement is None


def test_a_guessed_column_does_not_stop_the_statement(tmp_path: Path) -> None:
    result = qualify_one_statement(
        JOINED_TO_A_CTE.format(selection="name"), {}, tmp_path
    )
    assert [finding.severity for finding in result.findings] == ["warning"]
    assert not result.has_errors
    assert result.statement is not None


def test_the_guess_warning_can_be_switched_off(tmp_path: Path) -> None:
    result = qualify_one_statement(
        JOINED_TO_A_CTE.format(selection="name"),
        {},
        tmp_path,
        warn_on_column_without_source=False,
    )
    assert result.findings == []
    assert result.statement is not None


def test_a_file_holding_two_statements_is_reported(tmp_path: Path) -> None:
    """A model is one projection, so two statements have no single answer to what the
    model produces. Checking the last one and calling it the model's schema described the
    wrong thing silently."""
    result = qualify_one_statement(
        "select id from test; select id from test;", {}, tmp_path
    )
    assert result.statement is None
    assert result.has_errors
    assert "2 statements" in result.errors[0]


def test_a_file_that_does_not_parse_is_reported(tmp_path: Path) -> None:
    result = qualify_one_statement("select from from", {}, tmp_path)
    assert result.statement is None
    assert result.has_errors


def test_an_ambiguous_column_keeps_its_source_position() -> None:
    sql = JOINED_TO_A_CTE.format(selection="name")
    resolved = resolve(sql, {q("test"): {"id": "varchar(20)", "name": "varchar(20)"}})
    (ambiguous,) = resolved.ambiguous_columns
    identifier = ambiguous.column.this
    assert isinstance(identifier, exp.Identifier)
    start = identifier.meta["start"]
    end = identifier.meta["end"]
    assert sql[start : end + 1] == "name"


# ------------------------------------------------------------- schema fabrication
def test_declared_types_fill_in_and_gaps_become_unknown() -> None:
    fabricated = get_declared_types_per_relation(
        {q("test"): {"id": "varchar(20)"}},
        {
            q("test"): parsed(["id", "modified_at"]),
            q("other"): parsed(["fk"]),
        },
        {q("test"), q("other")},
    )
    assert fabricated == {
        q("test"): {"id": "varchar(20)", "modified_at": "UNKNOWN"},
        q("other"): {"fk": "UNKNOWN"},
    }


def test_fabricated_schema_covers_declared_columns_the_sql_never_names() -> None:
    """A star names no column, so a schema holding only the harvested ones leaves the
    table empty and `select *` survives step 3 unexpanded."""
    fabricated = get_declared_types_per_relation(
        {q("test"): {"id": "int", "never_selected": "int"}},
        {q("test"): parsed(["id"])},
        {q("test")},
    )
    assert fabricated == {q("test"): {"id": "int", "never_selected": "int"}}


def test_a_table_with_nothing_known_about_it_stays_out_of_the_schema() -> None:
    """Declaring it empty would turn every read of it into an error, where the truth is
    that nobody can enumerate it."""
    fabricated = get_declared_types_per_relation({}, {}, {q("undeclared")})
    assert fabricated == {}


def qualified_projection_names(
    sql: str,
    declared: dict[RelationKey, dict[ColumnName, ColumnTypeName]],
    partial: frozenset[RelationKey] = frozenset(),
) -> list[str]:
    """Steps 2-6 by hand, reporting what the statement ends up projecting."""
    statement = sqlglot.parse_one(sql, read=DIALECT)
    resolved, closure = resolve_columns_to_source_tables(
        statement, DIALECT, declarations(declared, partial)
    )
    fabricated = get_declared_types_per_relation(
        closure.declared_types, resolved.columns_per_relation, resolved.storage_keys
    )
    schema = ensure_schema(
        nested_schema_for_sqlglot(fabricated, DIALECT), dialect=DIALECT
    )

    qualified = qualify(statement, schema=schema, dialect=DIALECT)

    assert isinstance(qualified, exp.Select)
    return [projection.alias_or_name for projection in qualified.selects]


def test_fabricated_schema_lets_qualify_succeed() -> None:
    """End to end: the point of all of the above is that `qualify` stops raising."""
    sql = """
    with src as (select test_id, name from mydatabase.myschema.table_in_cte)
    select id, test.name, modified_at
    from mydatabase.myschema.test
    inner join src on test.id = src.test_id
    """
    assert set(
        qualified_projection_names(
            sql, {q("test"): {"id": "varchar(20)"}}, frozenset({q("test")})
        )
    ) == {
        "id",
        "name",
        "modified_at",
    }


def test_a_star_expands_against_the_declared_columns() -> None:
    """The whole reason the schema carries declared columns the SQL never names."""
    names = qualified_projection_names(
        "select * from test", {"test": {"id": "int", "name": "varchar(20)"}}
    )
    assert names == ["id", "name"]


def test_a_star_over_an_undeclared_table_stays_a_star() -> None:
    """Nothing can enumerate it, so step 3 leaves the star alone rather than inventing a
    projection list. This is the undeclared path the design keeps first-class."""
    assert qualified_projection_names("select * from test", {}) == ["*"]


# --------------------------------------------------- what each scope reads, per bucket
TEST_COLUMNS = {"test": {"id": "int", "name": "varchar(20)", "status": "varchar(20)"}}


def scope_buckets(
    sql: str, tmp_path: Path
) -> dict[str, tuple[list[tuple[str, str]], list[tuple[str, str]]]]:
    """Each scope's output schema and its other reads, as `(name, origin)` pairs."""
    result = qualify_one_statement(sql, TEST_COLUMNS, tmp_path)
    assert result.statement is not None, result.findings
    return {
        scope.name: (
            [(column.name, column.origin) for column in scope.projected],
            [(column.name, column.origin) for column in scope.non_projected],
        )
        for scope in columns_per_scope(result.statement)
    }


def test_a_projected_column_is_never_also_reported_as_non_projected(
    tmp_path: Path,
) -> None:
    """The two lists are disjoint in the result itself, not only in the printed table:
    a downstream pass reading them as two sets must not have to subtract one from the
    other."""
    projected, non_projected = scope_buckets(
        "select test.id from test where test.id > 0", tmp_path
    )["<final>"]
    assert projected == [("id", "written")]
    assert non_projected == []


def test_a_projected_column_keeps_the_origin_it_was_projected_by(tmp_path: Path) -> None:
    """`select * from t where t.id > 0` projects `id` by expanding the star. The written
    read in the WHERE does not change where the projection came from."""
    projected, non_projected = scope_buckets(
        "select * from test where test.id > 0", tmp_path
    )["<final>"]
    assert ("id", "star") in projected
    assert non_projected == []


def test_a_computed_column_is_a_separate_output_from_the_column_it_reads(
    tmp_path: Path,
) -> None:
    """`upper(name) as loud` and the `name` a star also projects are two columns of the
    relation. Collapsing them onto the name they share loses one output entirely."""
    projected, _ = scope_buckets(
        "select test.*, upper(test.name) as loud from test", tmp_path
    )["<final>"]
    assert projected == [
        ("id", "star"),
        ("name", "star"),
        ("status", "star"),
        ("loud", "written"),
    ]


def test_a_column_only_read_in_a_filter_is_non_projected(tmp_path: Path) -> None:
    projected, non_projected = scope_buckets(
        "select test.id from test where status = 'x'", tmp_path
    )["<final>"]
    assert projected == [("id", "written")]
    assert non_projected == [("status", "inferred")]


def test_the_same_name_from_two_sources_stays_two_columns(tmp_path: Path) -> None:
    """Identity is `(name, source)`, so a projected `a.id` does not swallow a filter on
    `b.id`."""
    sql = """
    with src as (select id from other)
    select test.id from test
    inner join src on test.id = src.id
    where src.id > 0
    """
    projected, non_projected = scope_buckets(sql, tmp_path)["<final>"]
    assert [name for name, _ in projected] == ["id"]
    assert [name for name, _ in non_projected] == ["id"]


# ------------------------------------------------------------ duplicate projections
def duplicate_messages(sql: str, tmp_path: Path) -> list[str]:
    result = qualify_one_statement(sql, TEST_COLUMNS, tmp_path)
    return [
        finding.message
        for finding in result.findings
        if finding.code == "duplicate-projected-column"
    ]


def test_a_star_overlapping_a_written_column_is_an_error(tmp_path: Path) -> None:
    """The projection list is impossible, and sqlglot will not say so - it quietly stops
    expanding stars over the broken relation several scopes away instead."""
    result = qualify_one_statement("select id, * from test", TEST_COLUMNS, tmp_path)
    assert result.statement is None
    assert result.has_errors
    (message,) = [
        finding.message
        for finding in result.findings
        if finding.code == "duplicate-projected-column"
    ]
    assert "'id' is projected 2 times" in message
    assert "written at 1:8" in message
    assert "expanded from the '*' at 1:12" in message


def test_a_duplicate_written_without_any_star_is_an_error(tmp_path: Path) -> None:
    (message,) = duplicate_messages("select id, name as id from test", tmp_path)
    assert "written at 1:8" in message
    assert "written at 1:12" in message
    assert "'*'" not in message


def test_two_stars_are_each_blamed_for_their_own_column(tmp_path: Path) -> None:
    """`select a.*, b.*` collides on every name, and the message has to say which star
    produced which half or there is nothing to act on."""
    sql = """
    select a.*, b.*
    from test as a
    inner join test as b on a.id = b.id
    """
    messages = duplicate_messages(sql, tmp_path)
    assert len(messages) == 3
    assert all(
        "expanded from the '*' at 2:12, expanded from the '*' at 2:17" in message
        for message in messages
    )


def test_a_duplicate_inside_a_cte_names_the_cte(tmp_path: Path) -> None:
    sql = """
    with src as (select id, name as id from test)
    select src.id from src
    """
    (message,) = duplicate_messages(sql, tmp_path)
    assert "CTE 'src'" in message


def test_two_unexpanded_stars_are_not_a_duplicate(tmp_path: Path) -> None:
    """Both projections are named `*` because neither table can be enumerated - nothing
    declares them and the SQL names no column of either. They are one unexpanded star
    each, not two columns in collision."""
    sql = """
    select a.*, b.*
    from undeclared as a
    cross join also_undeclared as b
    """
    assert duplicate_messages(sql, tmp_path) == []


def test_distinct_projection_names_are_not_a_duplicate(tmp_path: Path) -> None:
    assert duplicate_messages("select id, name from test", tmp_path) == []


# ------------------------------------------------- output names and output schema
def projected_columns(sql: str, tmp_path: Path, dialect: str = DIALECT) -> list[str]:
    """The final scope's output schema, as the names a downstream reference must quote."""
    result = qualify_one_statement(sql, TEST_COLUMNS, tmp_path, dialect_name=dialect)
    assert result.statement is not None, (result.errors, result.findings)
    scopes = columns_per_scope(result.statement)
    return [column.name for column in scopes[-1].projected]


def test_an_aliased_expression_survives_beside_the_column_it_reads(
    tmp_path: Path,
) -> None:
    """The projected list is the scope's *schema*, so it is keyed by output name. Keying
    it by the underlying column would drop `first_name_upper` on the floor."""
    sql = """
    with src as (select id, name, status from test)
    select *, upper(name) as name_upper from src
    """
    assert projected_columns(sql, tmp_path) == ["id", "name", "status", "name_upper"]


def test_an_unaliased_expression_is_named_the_way_the_engine_names_it(
    tmp_path: Path,
) -> None:
    """sqlglot labels it `_col_1`, which no engine produces. DuckDB echoes the text as
    written, so that is what a downstream reference has to quote."""
    assert projected_columns("select upper(name) from test", tmp_path) == ["upper(name)"]


def test_snowflake_folds_the_derived_name(tmp_path: Path) -> None:
    """The name is case-sensitive, and Snowflake folds it like any unquoted identifier."""
    names = projected_columns("select upper(name) from test", tmp_path, "snowflake")
    assert names == ["UPPER(NAME)"]


def test_postgres_names_a_derived_column_after_its_function(tmp_path: Path) -> None:
    names = projected_columns("select upper(name), id + 1 from test", tmp_path, "postgres")
    assert names == ["upper", "?column?"]


def test_two_postgres_unnamed_columns_are_not_a_duplicate(tmp_path: Path) -> None:
    """`?column?` is Postgres declining to name the projection, not a name. It returns two
    of them side by side and only objects when something references one."""
    result = qualify_one_statement(
        "select id + 1, id + 2 from test", TEST_COLUMNS, tmp_path, dialect_name="postgres"
    )
    assert result.statement is not None
    assert result.findings == []


def test_a_derived_name_still_collides_with_a_written_one(tmp_path: Path) -> None:
    """The duplicate check reads the same names, so an alias colliding with a derived name
    is caught even though sqlglot called one of them `_col_1`."""
    result = qualify_one_statement(
        "select upper(name), id as 'upper(name)' from test", TEST_COLUMNS, tmp_path
    )
    assert result.statement is None
    assert [finding.code for finding in result.findings] == ["duplicate-projected-column"]


def test_a_projection_reading_two_sources_reports_both(tmp_path: Path) -> None:
    sql = """
    select a.id + b.id as total
    from test as a
    inner join other as b on a.id = b.id
    """
    result = qualify_one_statement(sql, TEST_COLUMNS, tmp_path)
    assert result.statement is not None
    (total,) = [
        column
        for column in columns_per_scope(result.statement)[-1].projected
        if column.name == "total"
    ]
    assert sorted(read.source.alias for read in total.reads) == ["a", "b"]
    assert total.column is None


def test_a_projection_reading_no_column_reports_no_source(tmp_path: Path) -> None:
    result = qualify_one_statement("select 1 + 1 from test", TEST_COLUMNS, tmp_path)
    assert result.statement is not None
    (only,) = columns_per_scope(result.statement)[-1].projected
    assert only.reads == []
    assert only.engine_named


def test_a_renamed_column_is_still_that_column(tmp_path: Path) -> None:
    """`select test.id as ident ... where test.id > 0` reads one column and projects it
    renamed, so the filter read adds nothing new."""
    projected, non_projected = scope_buckets(
        "select test.id as ident from test where test.id > 0", tmp_path
    )["<final>"]
    assert projected == [("ident", "written")]
    assert non_projected == []


def test_a_column_read_only_inside_an_expression_stays_non_projected(
    tmp_path: Path,
) -> None:
    """No output column is named `status`, so the filter read is still worth reporting."""
    projected, non_projected = scope_buckets(
        "select upper(test.status) as loud from test where test.status > 'a'", tmp_path
    )["<final>"]
    assert projected == [("loud", "written")]
    assert non_projected == [("status", "written")]


# ------------------------------------------------------------- relation transparency
def test_a_column_passes_through_a_cte_that_could_not_expand_its_star() -> None:
    """The bug the whole closure exists for.

    `ranked` projects `['*', 'rn']` because `raw_address` is undeclared, so `street` is
    read from a relation nobody can enumerate. It belongs to `raw_address`, reachable only
    through that star - and `rn`, which the CTE really does produce, must not follow it.
    """
    resolved = resolve(
        """
        with ranked as (
          select *, row_number() over (partition by person_id) as rn
          from mydatabase.myschema.raw_address
        )
        select person_id, street, city from ranked where rn = 1
        """
    )
    assert column_names(resolved) == {
        q("raw_address"): {"person_id", "street", "city"}
    }
    assert resolved.unresolvable_columns == []
    assert resolved.relations_read_through_an_unexpandable_star == {q("raw_address")}


def test_transparency_chains_through_two_ctes() -> None:
    """`a` selects `*` from `b`, `b` selects `*` from an undeclared table. A column read
    off `a` is owned by that table, two hops away."""
    resolved = resolve(
        """
        with b as (select * from mydatabase.myschema.t),
             a as (select * from b)
        select far from a
        """
    )
    assert column_names(resolved) == {q("t"): {"far"}}
    assert resolved.unresolvable_columns == []


def closing_declarations_of(resolved: ResolvedColumns) -> list[list[RelationKey]]:
    """What each unresolvable column's relations blame, one list per relation."""
    return [
        [entry.relation_name for entry in relation.closing_declarations]
        for unresolvable in resolved.unresolvable_columns
        for relation in unresolvable.relations
    ]


def test_closedness_carries_its_cause_through_a_cte_star() -> None:
    """`ranked` is closed only because `raw_address` is, and the reader can only fix the
    latter. Losing the hop leaves a message pointing at a CTE with no answer in it."""
    resolved = resolve(
        """
        with ranked as (select * from mydatabase.myschema.raw_address)
        select street from ranked
        """,
        {q("raw_address"): {"person_id": "varchar(20)"}},
    )
    assert unresolvable_names(resolved) == ["street"]
    assert closing_declarations_of(resolved) == [[q("raw_address")]]


def test_closedness_carries_its_cause_through_two_ctes() -> None:
    """The same walk `open_origins` does, and for the same reason: transparency chains, so
    the blame has to chain with it."""
    resolved = resolve(
        """
        with b as (select * from mydatabase.myschema.t),
             a as (select * from b)
        select far from a
        """,
        {q("t"): {"near": "int"}},
    )
    assert unresolvable_names(resolved) == ["far"]
    assert closing_declarations_of(resolved) == [[q("t")]]


def test_closedness_carries_every_arm_of_a_set_operation() -> None:
    """A name missing from a union is missing from each arm that could have supplied it,
    and a reader fixing it has to know about all of them - not only the left one, which is
    the only arm that supplies the *names*."""
    resolved = resolve(
        """
        with both as (
          select * from mydatabase.myschema.left_table
          union all
          select * from mydatabase.myschema.right_table
        )
        select mystery from both
        """,
        {
            q("left_table"): {"k": "int"},
            q("right_table"): {"k": "int"},
        },
    )
    assert unresolvable_names(resolved) == ["mystery"]
    assert closing_declarations_of(resolved) == [
        [q("left_table"), q("right_table")]
    ]


def test_a_cte_that_writes_its_own_projection_list_blames_no_declaration() -> None:
    """Closed because the SQL enumerates it, not because any yml says so. Offering a yml
    fix here would send the reader to a file that has nothing to do with the mistake."""
    resolved = resolve(
        """
        with src as (select id from mydatabase.myschema.test)
        select src.nonsense from src
        """,
        {q("test"): {"id": "int"}},
    )
    assert closing_declarations_of(resolved) == [[]]


def test_a_bare_name_over_two_undeclared_tables_says_it_cannot_choose() -> None:
    """Not `no_such_source`: the qualifier is fine, there just isn't one. Its own reason
    because its fix is the only one of the three that is not an edit to the SQL's names."""
    resolved = resolve(
        """
        select mystery
        from mydatabase.myschema.test a
        join mydatabase.myschema.other b on a.id = b.id
        """
    )
    (unresolvable,) = resolved.unresolvable_columns
    assert unresolvable.reason == "several_undeclared_sources"
    assert [relation.display_name for relation in unresolvable.relations] == [
        q("test"),
        q("other"),
    ]


def test_a_closed_relation_beside_an_undeclared_one_is_not_offered_as_a_fix() -> None:
    """`test` declares its columns and does not project `mystery`, so it is ruled out for
    certain. Naming it beside `other` would offer a fix that changes nothing."""
    resolved = resolve(
        """
        select mystery
        from mydatabase.myschema.test a
        join mydatabase.myschema.other b on a.id = b.id
        join mydatabase.myschema.third c on a.id = c.id
        """,
        {q("test"): {"id": "int"}},
    )
    (unresolvable,) = resolved.unresolvable_columns
    assert unresolvable.reason == "several_undeclared_sources"
    assert [relation.display_name for relation in unresolvable.relations] == [
        q("other"),
        q("third"),
    ]


def test_a_closed_relation_names_what_it_does_project() -> None:
    """`qualify` reports this against the tree it rewrote. The message built here can name
    the relation that failed and list what it projects, which is the difference between a
    reader spotting a typo and a reader guessing."""
    resolved = resolve(
        """
        with src as (select id, name from mydatabase.myschema.test)
        select src.nonsense from src
        """
    )
    (unresolvable,) = resolved.unresolvable_columns
    assert unresolvable.reason == "not_projected"
    (relation,) = unresolvable.relations
    assert relation.display_name == "src"
    assert relation.kind == "cte"
    assert relation.projected_columns == ["id", "name"]
    # The CTE writes its own projection list, so nothing in any yml closed it and there is
    # no file to send the reader to - the name is simply a typo.
    assert relation.closing_declarations == []
    (finding,) = findings_for_unresolvable_columns(
        resolved.unresolvable_columns, Positions("")
    )
    assert finding.message == "found undeclared column 'nonsense' in CTE 'src'."


STAR_OVER_A_JOIN = """
with both as (
  select * from mydatabase.myschema.left_table
  join mydatabase.myschema.right_table on left_table.k = right_table.k
)
select mystery from both
"""


def test_a_star_over_a_join_of_undeclared_tables_is_a_guess_by_default() -> None:
    resolved = resolve(STAR_OVER_A_JOIN, star_over_join_behavior="guess")
    assert guessed_names(resolved) == [
        ("mystery", q("left_table"), [q("right_table")])
    ]
    assert ambiguous_names(resolved) == []


def test_a_star_over_a_join_of_undeclared_tables_can_be_an_error() -> None:
    """Qualifying the column fixes nothing here - no relation in the scope projects it
    under its own steam - so the two settings are a real choice."""
    resolved = resolve(STAR_OVER_A_JOIN, star_over_join_behavior="error")
    assert ambiguous_names(resolved) == [
        ("mystery", [q("left_table"), q("right_table")])
    ]
    assert guessed_names(resolved) == []
    (finding,) = findings_for_ambiguous_columns(
        resolved.ambiguous_columns, Positions("")
    )
    assert "arrives through a '*'" in finding.message
    assert "star_over_join_behavior" in finding.message


def test_declaring_one_side_of_the_join_settles_it() -> None:
    """The payoff. One table is closed, so the star over it contributes real names and
    only the other side is still transparent."""
    resolved = resolve(
        STAR_OVER_A_JOIN,
        {q("left_table"): {"k": "int"}},
        star_over_join_behavior="error",
    )
    assert ambiguous_names(resolved) == []
    assert column_names(resolved)[q("right_table")] == {"k", "mystery"}


def test_a_partial_declaration_keeps_its_declared_types_and_stays_open() -> None:
    """Both halves. A partial table that lost its declared types would pass a test that
    only checked the open half."""
    resolved = resolve(
        "select id, undeclared_one from mydatabase.myschema.test",
        {q("test"): {"id": "varchar(20)"}},
        partial=frozenset({q("test")}),
    )
    assert column_names(resolved) == {q("test"): {"id", "undeclared_one"}}
    fabricated = get_declared_types_per_relation(
        {q("test"): {"id": "varchar(20)"}},
        resolved.columns_per_relation,
        resolved.storage_keys,
    )
    assert fabricated == {
        q("test"): {"id": "varchar(20)", "undeclared_one": "UNKNOWN"}
    }


def test_two_relations_with_the_same_table_name_do_not_collide() -> None:
    """What `RelationKey` is for. A bare key would merge these two column sets, and every
    type inferred for one would leak into the other."""
    resolved = resolve(
        """
        select a.x, b.y
        from mydatabase.schema_one.thing a
        join mydatabase.schema_two.thing b on a.k = b.k
        """,
        {
            "mydatabase.schema_one.thing": {"x": "int", "k": "int"},
            "mydatabase.schema_two.thing": {"y": "varchar(10)", "k": "int"},
        },
    )
    assert column_names(resolved) == {
        "mydatabase.schema_one.thing": {"x", "k"},
        "mydatabase.schema_two.thing": {"y", "k"},
    }


def test_a_set_operation_takes_its_names_from_the_left_arm() -> None:
    """Arms are matched by position, not by name, so the left arm supplies the schema -
    and a name only the right arm writes is not a column of the union."""
    resolved = resolve(
        """
        with both as (
          select p from mydatabase.myschema.t1
          union all
          select r from mydatabase.myschema.t2
        )
        select r from both
        """
    )
    assert unresolvable_names(resolved) == ["r"]


def test_a_set_operation_is_open_when_any_arm_is() -> None:
    """The left arm enumerates, the right one cannot. Nothing downstream can place a name
    the left arm does not carry, so the union stays transparent to both."""
    resolved = resolve(
        """
        with both as (
          select p from mydatabase.myschema.t1
          union all
          select * from mydatabase.myschema.t2
        )
        select p from both
        """
    )
    assert resolved.unresolvable_columns == []
    assert "p" in column_names(resolved)[q("t1")]


def test_a_correlated_subquery_reads_its_outer_alias() -> None:
    """`o` belongs to the enclosing scope and appears nowhere in the subquery's sources.
    Stopping at the innermost scope reported it as a mistyped alias, which is the one thing
    it certainly is not."""
    resolved = resolve(
        """
        with all_orders as (select * from mydatabase.myschema.orders)
        select o.order_id
        from all_orders o
        where exists (
          select 1 from mydatabase.myschema.customer c
          where c.customer_id = o.customer_id
        )
        """
    )
    assert resolved.unresolvable_columns == []
    assert column_names(resolved)[q("orders")] == {"order_id", "customer_id"}


def test_a_qualified_star_matches_the_dialect_s_own_identifier_case() -> None:
    """Snowflake folds identifiers up and DuckDB folds them down, and `scope.sources` is
    keyed whichever way the dialect chose. Lowercasing one side of that comparison matched
    nothing under Snowflake, so every `select t.*` there projected an empty relation."""
    resolved, _ = resolve_columns_to_source_tables(
        sqlglot.parse_one(
            """
            with inner_scope as (select * from mydatabase.myschema.raw)
            select o.* from inner_scope o
            """,
            read="snowflake",
        ),
        "snowflake",
        declarations(),
    )
    assert resolved.unresolvable_columns == []


# ------------------------------------------------------- lower-bound projections
def test_a_projection_expanded_over_an_undeclared_table_is_a_lower_bound(
    tmp_path: Path,
) -> None:
    """The star yields the columns this file happens to name, never the columns the table
    has, so nothing may compare it against a complete declaration."""
    result = qualify_one_statement(
        "select * from mydatabase.myschema.undeclared where k > 0", {}, tmp_path
    )
    assert result.statement is not None
    assert [scope.complete for scope in columns_per_scope(result.statement)] == [False]


def test_incompleteness_propagates_through_a_cte(tmp_path: Path) -> None:
    """A star over a lower bound is a lower bound. Deciding this per scope would call the
    outer projection complete because its own source is a CTE, which enumerates fine."""
    result = qualify_one_statement(
        """
        with src as (select * from mydatabase.myschema.undeclared where k > 0)
        select * from src
        """,
        {},
        tmp_path,
    )
    assert result.statement is not None
    assert [scope.complete for scope in columns_per_scope(result.statement)] == [
        False,
        False,
    ]


def test_a_projection_over_a_declared_table_is_complete(tmp_path: Path) -> None:
    result = qualify_one_statement(
        "select * from test", TEST_COLUMNS, tmp_path
    )
    assert result.statement is not None
    assert all(scope.complete for scope in columns_per_scope(result.statement))


def test_enumerating_the_columns_makes_the_projection_complete_again(
    tmp_path: Path,
) -> None:
    """`address.sql`'s shape: the star inside the CTE is a lower bound, but the CTE that
    selects from it writes its columns out, so the model's own projection is exact."""
    result = qualify_one_statement(
        """
        with ranked as (select * from mydatabase.myschema.undeclared),
             named as (select k, v from ranked)
        select * from named
        """,
        {},
        tmp_path,
    )
    assert result.statement is not None
    complete = {scope.name: scope.complete for scope in columns_per_scope(result.statement)}
    assert complete == {"ranked": False, "named": True, "<final>": True}
