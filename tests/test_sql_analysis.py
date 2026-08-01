from pathlib import Path

from sqlrunner.sql_analysis import analyze_file, analyze_sql


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
