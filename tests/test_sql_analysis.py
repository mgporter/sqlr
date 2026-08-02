from pathlib import Path

from sqlrunner.sql_analysis import analyze_file, analyze_sql
from sqlrunner.sql_analysis.types import ColumnUsage


def test_classifies_external_source() -> None:
    result = analyze_sql("select id, name from mydatabase.myschema.person")

    assert result.external_sources == ["mydatabase.myschema.person"]
    assert result.ctes == []
    assert result.errors == []


def test_cte_is_not_external() -> None:
    sql = """
    with base as (
        select id from mydatabase.myschema.person
    )
    select * from base
    """

    result = analyze_sql(sql)

    assert result.external_sources == ["mydatabase.myschema.person"]
    assert [c.name for c in result.ctes] == ["base"]


def test_cte_dependency_graph() -> None:
    sql = """
    with base as (
        select id from mydatabase.myschema.person
    ),
    enriched as (
        select b.id, o.amount
        from base b
        join mydatabase.myschema.orders o on o.person_id = b.id
    )
    select * from enriched
    """

    result = analyze_sql(sql)

    ctes_by_name = {c.name: c for c in result.ctes}
    assert ctes_by_name["base"].depends_on == []
    assert ctes_by_name["enriched"].depends_on == ["base"]
    assert result.external_sources == [
        "mydatabase.myschema.orders",
        "mydatabase.myschema.person",
    ]


def test_dedupes_repeated_external_source() -> None:
    sql = """
    select * from mydatabase.myschema.person a
    join mydatabase.myschema.person b on a.id = b.manager_id
    """

    result = analyze_sql(sql)

    assert result.external_sources == ["mydatabase.myschema.person"]


def test_unqualified_table_name() -> None:
    result = analyze_sql("select * from person")

    assert result.external_sources == ["person"]


def test_syntax_error_is_reported_not_raised() -> None:
    result = analyze_sql("select * from where")

    assert result.external_sources == []
    assert result.ctes == []
    assert len(result.errors) == 1


def test_analyze_file(tmp_path: Path) -> None:
    sql_path = tmp_path / "model.sql"
    sql_path.write_text("select * from mydatabase.myschema.person")

    result = analyze_file(sql_path)

    assert result.external_sources == ["mydatabase.myschema.person"]


def test_dialect_is_respected() -> None:
    result = analyze_sql('select "id" from person', dialect="duckdb")

    assert result.external_sources == ["person"]
    assert result.errors == []


def test_column_usage_numeric_comparison() -> None:
    result = analyze_sql("select * from person where amount > 100")

    assert result.column_usages == [
        ColumnUsage(table="person", column="amount", kind="compared_to_number", detail="int")
    ]


def test_column_usage_numeric_comparison_float_shape() -> None:
    result = analyze_sql("select * from person where amount > 99.5")

    [usage] = result.column_usages
    assert usage.kind == "compared_to_number"
    assert usage.detail == "float"


def test_column_usage_in_list_strings() -> None:
    result = analyze_sql("select * from person where status in ('shipped', 'pending')")

    [usage] = result.column_usages
    assert usage.table == "person"
    assert usage.column == "status"
    assert usage.kind == "in_list_strings"


def test_column_usage_date_function() -> None:
    result = analyze_sql("select date_trunc('day', created_at) from person")

    [usage] = result.column_usages
    assert usage.column == "created_at"
    assert usage.kind == "date_function"
    assert usage.detail == "DateTrunc"


def test_column_usage_cast() -> None:
    result = analyze_sql("select * from person where cast(x as date) = current_date")

    [usage] = result.column_usages
    assert usage.column == "x"
    assert usage.kind == "cast"
    assert usage.detail == "DATE"


def test_column_usage_boolean_context() -> None:
    result = analyze_sql("select * from person where is_active")

    [usage] = result.column_usages
    assert usage.column == "is_active"
    assert usage.kind == "boolean_context"


def test_column_usage_like() -> None:
    result = analyze_sql("select * from person where name like 'A%'")

    [usage] = result.column_usages
    assert usage.kind == "like"


def test_column_usage_arithmetic() -> None:
    result = analyze_sql("select amount + tax from person")

    kinds = {u.kind for u in result.column_usages}
    assert kinds == {"arithmetic"}


def test_column_usage_skips_ambiguous_unqualified_column() -> None:
    sql = """
    select amount from person p
    join orders o on o.person_id = p.id
    where amount > 100
    """

    result = analyze_sql(sql)

    assert result.column_usages == []


def test_column_usage_qualified_column_in_join() -> None:
    sql = """
    select * from person p
    join orders o on o.person_id = p.id
    where o.amount > 100
    """

    result = analyze_sql(sql)

    [usage] = result.column_usages
    assert usage.table == "orders"
    assert usage.column == "amount"
    assert usage.kind == "compared_to_number"


def test_column_references_plain_select() -> None:
    result = analyze_sql("select id, name, email from person")

    assert {(r.table, r.column) for r in result.column_references} == {
        ("person", "id"),
        ("person", "name"),
        ("person", "email"),
    }
    assert result.column_usages == []


def test_column_references_include_columns_with_usage_evidence() -> None:
    result = analyze_sql("select id from person where amount > 100")

    assert {(r.table, r.column) for r in result.column_references} == {
        ("person", "id"),
        ("person", "amount"),
    }


def test_column_usage_not_attributed_to_cte() -> None:
    sql = """
    with base as (
        select id from person where id > 0
    )
    select * from base where id > 0
    """

    result = analyze_sql(sql)

    assert len(result.column_usages) == 1
    assert result.column_usages[0].table == "person"
