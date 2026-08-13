"""Column-to-source resolution for the typing pipeline.

Every fixture here is an inline SQL string, per `tests/README.md`.
"""

import pytest
import sqlglot
from sqlglot import exp

from sqlr.sql_analysis2 import (
    get_declared_types_per_table,
    resolve_columns_to_source_tables,
)
from sqlr.sql_analysis2.types import (
    ColumnName,
    ParsedColumn,
    ResolvedColumns,
    StructuredAccessKind,
    TableName,
)

DIALECT = "duckdb"


def resolve(sql: str) -> ResolvedColumns:
    return resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read=DIALECT), DIALECT
    )


def column_names(resolved: ResolvedColumns) -> dict[TableName, set[ColumnName]]:
    """Just the names, for the tests that do not care how a column was read."""
    return {
        table: set(columns) for table, columns in resolved.columns_per_table.items()
    }


def kinds(
    resolved: ResolvedColumns, table: TableName, column: ColumnName
) -> list[StructuredAccessKind]:
    """How one column was read, one entry per site, in written order."""
    return [
        kind for kind, _ in resolved.columns_per_table[table][column].structured_access
    ]


def unresolvable_names(resolved: ResolvedColumns) -> list[str]:
    return [column.name for column in resolved.unresolvable_columns]


def parsed(names: list[ColumnName]) -> dict[ColumnName, ParsedColumn]:
    """A `columns_per_table` entry, for feeding the schema fabricator directly."""
    return {
        name: ParsedColumn(name=name, structured_access=[]) for name in names
    }


# ------------------------------------------------------- columns per real table
def test_qualified_columns_land_on_their_table() -> None:
    resolved = resolve("select t.id, t.name from mydatabase.myschema.test t")
    assert column_names(resolved) == {"test": {"id", "name"}}
    assert resolved.unresolvable_columns == []


def test_bare_columns_land_on_the_only_table() -> None:
    resolved = resolve("select id, name from mydatabase.myschema.test")
    assert column_names(resolved) == {"test": {"id", "name"}}


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
        "test": {"id", "name", "modified_at", "titles"},
        "table_in_cte": {"test_id", "name"},
    }
    assert resolved.unresolvable_columns == []


def test_cte_names_are_not_invented_as_table_columns() -> None:
    """`k` is the CTE's own projection, not a column of any real table."""
    resolved = resolve(
        "with a as (select 1 as k) select k, extra from a, mydatabase.myschema.test"
    )
    assert column_names(resolved) == {"test": {"extra"}}


def test_correlated_subquery_columns_reach_the_outer_table() -> None:
    resolved = resolve(
        """
        select id from mydatabase.myschema.test t
        where exists (
          select 1 from mydatabase.myschema.other o where o.fk = t.id and o.flag
        )
        """
    )
    assert column_names(resolved) == {"test": {"id"}, "other": {"fk", "flag"}}


def test_scalar_subquery_columns_land_on_their_own_table() -> None:
    resolved = resolve(
        """
        select id, (select max(amt) from mydatabase.myschema.other) as m
        from mydatabase.myschema.test
        """
    )
    assert column_names(resolved) == {"test": {"id"}, "other": {"amt"}}


def test_using_join_credits_both_tables() -> None:
    resolved = resolve(
        """
        select t.id from mydatabase.myschema.test t
        join mydatabase.myschema.other o using (id)
        """
    )
    assert column_names(resolved) == {"test": {"id"}, "other": {"id"}}


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
    assert column_names(resolved) == {"test": {"id"}, "other": {"id"}}


def test_derived_table_shadows_its_source() -> None:
    resolved = resolve(
        """
        select outer_id
        from (select id as outer_id from mydatabase.myschema.test) sub
        """
    )
    assert column_names(resolved) == {"test": {"id"}}


def test_column_names_are_lowercased() -> None:
    resolved = resolve("select ID, Name from MyDatabase.MySchema.Test")
    assert column_names(resolved) == {"test": {"id", "name"}}


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
    assert column_names(resolved) == {"test": {"x", "id"}, "other": {"id"}}
    assert unresolvable_names(resolved) == ["mystery"]


def test_bare_column_with_only_cte_sources_is_reported() -> None:
    resolved = resolve(
        """
        with src as (select test_id from mydatabase.myschema.table_in_cte)
        select test_id, nonsense from src
        """
    )
    assert column_names(resolved) == {"table_in_cte": {"test_id"}}
    assert unresolvable_names(resolved) == ["nonsense"]


def test_unresolvable_column_keeps_its_source_position() -> None:
    """The probe runs on a copy, so the reported node must still carry parse offsets."""
    sql = (
        "select t.id, nonsense\n"
        "from mydatabase.myschema.test t\n"
        "join mydatabase.myschema.other o on t.id = o.id"
    )
    resolved = resolve(sql)
    (column,) = resolved.unresolvable_columns
    identifier = column.this
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
    resolved = resolve_columns_to_source_tables(
        sqlglot.parse_one("select ghots.id from mydatabase.myschema.test", read=dialect),
        dialect,
    )
    assert column_names(resolved) == {"test": {"ghots"}}
    assert kinds(resolved, "test", "ghots") == ["dot_field"]
    assert resolved.columns_per_table["test"]["ghots"].requires_structured_type
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
    resolved = resolve_columns_to_source_tables(
        sqlglot.parse_one("select ghots.id from mydatabase.myschema.test", read=dialect),
        dialect,
    )
    assert [
        column.name.lower()
        for column in resolved.columns_read_with_unsupported_dot_notation
    ] == ["ghots"]
    assert column_names(resolved) == {"test": {"ghots"}}
    assert resolved.unresolvable_columns == []


def test_the_dot_access_gate_can_be_set_against_the_dialect_default() -> None:
    """The flag is internal for now, but it is the argument that decides, not the
    dialect - a config knob can hand it something else later."""
    statement = sqlglot.parse_one(
        "select ghots.id from mydatabase.myschema.test", read=DIALECT
    )
    resolved = resolve_columns_to_source_tables(
        statement, DIALECT, allow_unresolvable_aliases_as_structured_columns=False
    )
    assert [
        column.name for column in resolved.columns_read_with_unsupported_dot_notation
    ] == ["ghots"]


# --------------------------------------------------------------- structured access
def test_bracket_access_survives_a_dialect_without_dot_access() -> None:
    """`ghots['id']` is a subscript, not a qualifier, so nothing rewrites it and every
    dialect here reads it as a column of `test`. Only the dotted form is a finding."""
    resolved = resolve_columns_to_source_tables(
        sqlglot.parse_one(
            "select ghots['id'] from mydatabase.myschema.test", read="postgres"
        ),
        "postgres",
    )
    assert resolved.columns_read_with_unsupported_dot_notation == []
    assert kinds(resolved, "test", "ghots") == ["bracket_key"]
    assert resolved.columns_per_table["test"]["ghots"].requires_structured_type


def test_an_integer_subscript_does_not_demand_a_structured_type() -> None:
    """DuckDB subscripts strings, so `titles[1]` proves `titles` is indexable and nothing
    more. A `varchar` declaration for it is not a contradiction."""
    resolved = resolve("select titles[1] from mydatabase.myschema.test")
    column = resolved.columns_per_table["test"]["titles"]
    assert kinds(resolved, "test", "titles") == ["bracket_index"]
    assert not column.requires_structured_type


def test_a_column_read_plainly_carries_no_structured_access() -> None:
    resolved = resolve("select id from mydatabase.myschema.test")
    assert resolved.columns_per_table["test"]["id"].structured_access == []


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
    assert kinds(resolved, "test", "mistyped") == [
        "dot_field",
        "dot_field",
        "bracket_key",
        "dot_field",
    ]


def test_only_the_first_level_of_a_nested_read_is_modelled() -> None:
    """`a.b.c` says what `a.b` said: `a` is structured. Nothing can declare a type for
    `a.b`, so nothing needs to be recorded about it."""
    resolved = resolve("select a.b.c.d from mydatabase.myschema.test")
    assert column_names(resolved) == {"test": {"a"}}
    assert kinds(resolved, "test", "a") == ["dot_field"]


@pytest.mark.parametrize("dialect", ["duckdb", "snowflake", "postgres", "spark"])
def test_unknown_qualifier_with_two_sources_is_reported(dialect: str) -> None:
    """The other half: no lone source absorbs the name, so the typo surfaces."""
    resolved = resolve_columns_to_source_tables(
        sqlglot.parse_one(
            """
            select ghots.id
            from mydatabase.myschema.a x
            join mydatabase.myschema.b y on x.k = y.k
            """,
            read=dialect,
        ),
        dialect,
    )
    assert column_names(resolved) == {"a": {"k"}, "b": {"k"}}
    # Case follows the dialect's normalisation - Snowflake upper-cases - so compare
    # case-insensitively. Only `columns_per_table` is lowercased, because it has to match
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
    assert column_names(resolved) == {"test": {"id"}, "table_in_cte": {"test_id"}}
    assert resolved.unresolvable_columns == []


# ------------------------------------------------------------- schema fabrication
def test_declared_types_fill_in_and_gaps_become_unknown() -> None:
    declared = {"test": {"id": "varchar(20)", "unused": "int"}}
    fabricated = get_declared_types_per_table(
        declared,
        {
            "test": parsed(["id", "modified_at"]),
            "other": parsed(["fk"]),
        },
    )
    assert fabricated == {
        "test": {"id": "varchar(20)", "modified_at": "UNKNOWN"},
        "other": {"fk": "UNKNOWN"},
    }


def test_fabricated_schema_only_covers_columns_the_sql_reads() -> None:
    """A declared column the SQL never mentions stays out of the schema: the schema
    exists to make `qualify` resolve what is written, not to mirror the declaration."""
    fabricated = get_declared_types_per_table(
        {"test": {"id": "int", "never_selected": "int"}}, {"test": parsed(["id"])}
    )
    assert fabricated == {"test": {"id": "int"}}


def test_fabricated_schema_lets_qualify_succeed() -> None:
    """End to end: the point of all of the above is that `qualify` stops raising."""
    from sqlglot.optimizer.qualify import qualify
    from sqlglot.schema import ensure_schema

    sql = """
    with src as (select test_id, name from mydatabase.myschema.table_in_cte)
    select id, test.name, modified_at
    from mydatabase.myschema.test
    inner join src on test.id = src.test_id
    """
    statement = sqlglot.parse_one(sql, read=DIALECT)
    resolved = resolve_columns_to_source_tables(statement, DIALECT)
    fabricated = get_declared_types_per_table(
        {"test": {"id": "varchar(20)"}}, resolved.columns_per_table
    )
    schema = ensure_schema(fabricated, dialect=DIALECT)

    qualified = qualify(statement, schema=schema, dialect=DIALECT)

    assert {
        projection.alias_or_name for projection in qualified.selects
    } == {"id", "name", "modified_at"}
