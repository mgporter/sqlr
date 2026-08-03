from pathlib import Path

import pytest

from sqlrunner.sql_analysis import analyze_file, analyze_sql
from sqlrunner.sql_analysis.types import SqlAnalysisResult

FIXTURES = Path(__file__).parent / "fixtures"
PERSON_SQL = FIXTURES / "person.sql"


def source_columns(result: SqlAnalysisResult, table: str) -> set[str]:
    for source in result.sources:
        if source.name == table:
            return {column.name for column in source.columns}
    return set()


def confidences(result: SqlAnalysisResult, table: str) -> dict[str, str]:
    for source in result.sources:
        if source.name == table:
            return {column.name: column.confidence for column in source.columns}
    return {}


def usage_tuples(result: SqlAnalysisResult) -> set[tuple[str, str, str, str | None]]:
    return {
        (usage.node.relation.name, usage.node.column, usage.kind, usage.detail)
        for usage in result.usages
    }


def projection_names(result: SqlAnalysisResult) -> list[str | None]:
    return [column.name for column in result.projection]


def origin_names(result: SqlAnalysisResult, name: str) -> set[str]:
    for column in result.projection:
        if column.name == name:
            return {
                f"{origin.node.relation.name}.{origin.node.column}"
                for origin in column.origins
            }
    return set()


# ---- basics ------------------------------------------------------------------------


def test_classifies_external_source() -> None:
    result = analyze_sql("select id, name from mydatabase.myschema.person")

    assert result.external_sources == ["mydatabase.myschema.person"]
    assert result.cte_names == []
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
    assert result.cte_names == ["base"]


def test_relation_dependency_graph() -> None:
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

    depends = {
        relation.ref.name: [ref.name for ref in relation.depends_on]
        for relation in result.relations
    }
    assert depends["base"] == ["mydatabase.myschema.person"]
    assert depends["enriched"] == ["base", "mydatabase.myschema.orders"]
    assert result.external_sources == [
        "mydatabase.myschema.orders",
        "mydatabase.myschema.person",
    ]


def test_dedupes_repeated_external_source() -> None:
    sql = """
    select a.id, b.manager_id from mydatabase.myschema.person a
    join mydatabase.myschema.person b on a.id = b.manager_id
    """

    result = analyze_sql(sql)

    assert result.external_sources == ["mydatabase.myschema.person"]
    assert source_columns(result, "mydatabase.myschema.person") == {"id", "manager_id"}


def test_unqualified_table_name() -> None:
    result = analyze_sql("select id from person")

    assert result.external_sources == ["person"]


def test_syntax_error_is_reported_not_raised() -> None:
    result = analyze_sql("select * from where")

    assert result.sources == []
    assert result.relations == []
    assert len(result.errors) == 1


def test_analyze_file(tmp_path: Path) -> None:
    sql_path = tmp_path / "model.sql"
    sql_path.write_text("select id from mydatabase.myschema.person")

    result = analyze_file(sql_path)

    assert result.external_sources == ["mydatabase.myschema.person"]


def test_dialect_is_respected() -> None:
    result = analyze_sql('select "id" from person', dialect="duckdb")

    assert result.external_sources == ["person"]
    assert result.errors == []


# ---- usage evidence ----------------------------------------------------------------


def test_usage_numeric_comparison() -> None:
    result = analyze_sql("select id from person where amount > 100")

    assert ("person", "amount", "compared_to_number", "int") in usage_tuples(result)


def test_usage_numeric_comparison_float_shape() -> None:
    result = analyze_sql("select id from person where amount > 99.5")

    assert ("person", "amount", "compared_to_number", "float") in usage_tuples(result)


def test_usage_in_list_strings() -> None:
    result = analyze_sql("select id from person where status in ('shipped', 'pending')")

    assert ("person", "status", "in_list_strings", None) in usage_tuples(result)


def test_usage_date_function() -> None:
    result = analyze_sql("select date_trunc('day', created_at) from person")

    assert ("person", "created_at", "date_function", "DateTrunc") in usage_tuples(result)


def test_usage_cast() -> None:
    result = analyze_sql("select id from person where cast(x as date) = current_date")

    assert ("person", "x", "cast", "DATE") in usage_tuples(result)


def test_usage_boolean_context() -> None:
    result = analyze_sql("select id from person where is_active")

    assert ("person", "is_active", "boolean_context", None) in usage_tuples(result)


def test_usage_like() -> None:
    result = analyze_sql("select id from person where name like 'A%'")

    assert ("person", "name", "like", None) in usage_tuples(result)


def test_usage_arithmetic() -> None:
    result = analyze_sql("select amount + tax from person")

    assert {u.kind for u in result.usages} == {"arithmetic"}


def test_usage_coalesce_default() -> None:
    result = analyze_sql("select coalesce(bonus, 0) from person")

    assert ("person", "bonus", "coalesce_default", "int") in usage_tuples(result)


def test_coalesce_of_two_columns_carries_no_literal_evidence() -> None:
    result = analyze_sql("select coalesce(bonus, fallback) from person")

    assert not [u for u in result.usages if u.kind == "coalesce_default"]


def test_date_part_keyword_is_not_a_column() -> None:
    # Without a dialect sqlglot parses the `year` unit as a column reference, which would
    # otherwise invent a `year` column on `person`.
    result = analyze_sql("select datediff(year, hire_date, current_date()) from person")

    assert source_columns(result, "person") == {"hire_date"}


def test_a_qualified_date_part_name_is_still_a_column() -> None:
    result = analyze_sql("select datediff(p.year, p.hire_date, current_date()) from person p")

    assert source_columns(result, "person") == {"year", "hire_date"}


def test_ambiguous_unqualified_column_is_dropped_and_reported() -> None:
    sql = """
    select p.id from person p
    join orders o on o.person_id = p.id
    where amount > 100
    """

    result = analyze_sql(sql)

    assert "amount" not in source_columns(result, "person")
    assert "amount" not in source_columns(result, "orders")
    assert [a.reason for a in result.ambiguities] == ["unqualified_multi_source"]


def test_qualified_column_in_join_is_attributed() -> None:
    sql = """
    select p.id from person p
    join orders o on o.person_id = p.id
    where o.amount > 100
    """

    result = analyze_sql(sql)

    assert ("orders", "amount", "compared_to_number", "int") in usage_tuples(result)


# ---- lineage through CTEs ----------------------------------------------------------


def test_columns_flow_through_a_star_cte() -> None:
    result = analyze_file(PERSON_SQL)

    assert result.errors == []
    assert source_columns(result, "mydatabase.myschema.person") == {
        "id",
        "name",
        "age",
        "email",
        "modified_at",
    }
    assert source_columns(result, "mydatabase.myschema.address") == {
        "person_id",
        "street",
        "city",
    }


def test_star_read_marks_the_source_as_possibly_incomplete() -> None:
    result = analyze_file(PERSON_SQL)

    by_name = {source.name: source for source in result.sources}
    assert by_name["mydatabase.myschema.person"].star_expanded is True
    assert by_name["mydatabase.myschema.address"].star_expanded is False


def test_columns_resolved_through_a_star_are_inferred_not_explicit() -> None:
    result = analyze_file(PERSON_SQL)

    person = confidences(result, "mydatabase.myschema.person")
    assert person["name"] == "inferred"
    assert person["email"] == "inferred"
    # `age` is also written out directly, inside the CTE that owns the star.
    assert person["age"] == "explicit"


def test_final_projection_is_expanded_through_ctes() -> None:
    result = analyze_file(PERSON_SQL)

    assert projection_names(result) == [
        "id",
        "name",
        "age",
        "email",
        "modified_at",
        "street",
        "city",
    ]
    assert origin_names(result, "street") == {"mydatabase.myschema.address.street"}


def test_usage_on_a_derived_column_does_not_reach_its_inputs() -> None:
    result = analyze_file(PERSON_SQL)

    # `where b.rn = 1` is evidence about row_number(), not about any person column.
    assert ("dedupped", "rn", "compared_to_number", "int") in usage_tuples(result)
    assert not any(
        usage.node.relation.kind == "table" and usage.node.column == "rn"
        for usage in result.usages
    )


def test_aggregate_evidence_does_not_reach_base_columns() -> None:
    sql = """
    with totals as (
        select person_id, sum(amount) as total from orders group by person_id
    )
    select person_id from totals where total > 1000
    """

    result = analyze_sql(sql)

    kinds = {(u.node.relation.name, u.node.column): u.kind for u in result.usages}
    assert kinds[("totals", "total")] == "compared_to_number"
    assert ("orders", "amount") not in kinds


def test_join_predicate_is_extracted() -> None:
    result = analyze_file(PERSON_SQL)

    assert [
        (f"{j.left.relation.name}.{j.left.column}", f"{j.right.relation.name}.{j.right.column}", j.join_type)
        for j in result.joins
    ] == [
        (
            "mydatabase.myschema.person.id",
            "mydatabase.myschema.address.person_id",
            "LEFT",
        )
    ]


def test_filter_predicate_is_extracted() -> None:
    result = analyze_sql("select id from orders where status in ('a', 'b') and amount >= 5")

    predicates = {
        (p.node.column, p.operator, tuple(p.values), p.literal_kind)
        for p in result.predicates
    }
    assert ("status", "in", ("a", "b"), "string") in predicates
    assert ("amount", ">=", ("5",), "int") in predicates


def test_reversed_comparison_operator_is_flipped() -> None:
    result = analyze_sql("select id from orders where 100 < amount")

    assert [(p.node.column, p.operator, p.values) for p in result.predicates] == [
        ("amount", ">", ["100"])
    ]


def test_window_partition_is_reported_as_cardinality() -> None:
    result = analyze_file(PERSON_SQL)

    facts = {
        (fact.kind, tuple(node.column for node in fact.nodes))
        for fact in result.cardinality
    }
    assert ("partition_by", ("id",)) in facts
    assert ("window_order_by", ("modified_at",)) in facts


def test_outer_join_marks_the_padded_side_nullable() -> None:
    result = analyze_file(PERSON_SQL)

    padded = {
        fact.node.column
        for fact in result.nullability
        if fact.reason == "outer_join_padded"
    }
    assert {"street", "city"} <= padded


# ---- other constructs --------------------------------------------------------------


def test_set_operation_matches_columns_positionally() -> None:
    sql = """
    with u as (
        select id, name from x
        union all
        select pid, pname from y
    )
    select * from u
    """

    result = analyze_sql(sql)

    assert projection_names(result) == ["id", "name"]
    assert origin_names(result, "id") == {"x.id", "y.pid"}
    assert origin_names(result, "name") == {"x.name", "y.pname"}


def test_derived_table_is_resolved() -> None:
    sql = """
    select d.id, d.total
    from (select id, sum(amt) as total from orders group by id) d
    where d.total > 5
    """

    result = analyze_sql(sql)

    assert source_columns(result, "orders") == {"id", "amt"}
    assert origin_names(result, "id") == {"orders.id"}
    assert origin_names(result, "total") == {"d.total"}


def test_subquery_columns_are_not_attributed_to_the_outer_table() -> None:
    sql = "select id from person where id in (select person_id from orders where amt > 5)"

    result = analyze_sql(sql)

    assert source_columns(result, "person") == {"id"}
    assert source_columns(result, "orders") == {"person_id", "amt"}


def test_correlated_subquery_resolves_the_outer_alias() -> None:
    sql = """
    select p.id from person p
    where exists (select 1 from orders o where o.person_id = p.id)
    """

    result = analyze_sql(sql)

    assert source_columns(result, "person") == {"id"}
    assert source_columns(result, "orders") == {"person_id"}


def test_unexpandable_star_is_flagged_not_silently_dropped() -> None:
    result = analyze_sql("select * from person")

    assert [column.kind for column in result.projection] == ["star"]
    assert [ref.name for ref in result.projection[0].star_of] == ["person"]
    assert result.warnings


def test_recursive_cte_does_not_hang() -> None:
    sql = """
    with recursive r as (
        select id, parent_id from nodes
        union all
        select n.id, n.parent_id from nodes n join r on n.parent_id = r.id
    )
    select id from r
    """

    result = analyze_sql(sql)

    assert result.errors == []
    assert "nodes" in result.external_sources


# ---- star over join ----------------------------------------------------------------

STAR_JOIN_SQL = """
with j as (
    select * from a join b on a.id = b.person_id
)
select id, person_id, city from j
"""


def test_star_over_join_guess_attributes_by_declared_column() -> None:
    sql = """
    with inner_cte as (select label from labels),
    j as (select * from inner_cte join b on b.id = 1)
    select label from j
    """

    result = analyze_sql(sql)

    # `label` is declared by inner_cte, so it is not guessed onto the leftmost source.
    assert source_columns(result, "labels") == {"label"}
    assert "label" not in source_columns(result, "b")


def test_star_over_join_guess_attributes_by_qualified_reference() -> None:
    result = analyze_sql(STAR_JOIN_SQL)

    # `on a.id = b.person_id` proves where id and person_id live.
    assert "id" in source_columns(result, "a")
    assert "person_id" in source_columns(result, "b")

    by_column = {a.column: a for a in result.ambiguities if a.reason == "star_over_join"}
    assert by_column["person_id"].confidence == "inferred"
    assert by_column["person_id"].chosen is not None
    assert by_column["person_id"].chosen.name == "b"


def test_star_over_join_guess_falls_back_to_the_leftmost_source() -> None:
    result = analyze_sql(STAR_JOIN_SQL)

    assert "city" in source_columns(result, "a")
    assert "city" not in source_columns(result, "b")
    assert confidences(result, "a")["city"] == "guessed"


def test_star_over_join_guess_records_the_ambiguity() -> None:
    result = analyze_sql(STAR_JOIN_SQL)

    guessed = [
        a for a in result.ambiguities if a.reason == "star_over_join" and a.column == "city"
    ]
    assert len(guessed) == 1
    assert guessed[0].resolution == "attributed"
    assert guessed[0].chosen is not None and guessed[0].chosen.name == "a"
    assert guessed[0].confidence == "guessed"
    assert [ref.name for ref in guessed[0].candidates] == ["a", "b"]


def test_explicit_stars_do_not_change_attribution() -> None:
    sql = """
    with j as (select a.*, b.* from a join b)
    select city from j
    """

    result = analyze_sql(sql)

    assert "city" in source_columns(result, "a")
    assert "city" not in source_columns(result, "b")


def test_star_over_a_single_source_is_inferred_not_guessed() -> None:
    sql = "with j as (select * from a) select city from j"

    result = analyze_sql(sql)

    assert confidences(result, "a")["city"] == "inferred"
    assert result.ambiguities == []


def test_self_join_star_is_not_ambiguous() -> None:
    sql = "with j as (select * from a x join a y on x.id = y.parent_id) select city from j"

    result = analyze_sql(sql)

    assert "city" in source_columns(result, "a")
    assert not [a for a in result.ambiguities if a.reason == "star_over_join"]


def test_star_over_join_error_mode_reports_a_snippet() -> None:
    result = analyze_sql(STAR_JOIN_SQL, star_over_join_behavior="error")

    assert len(result.errors) == 1
    error = result.errors[0]
    assert "star over join is ambiguous" in error
    # `city` is the column with no evidence; id and person_id are provable from the ON.
    assert '"city"' in error
    assert "JOIN b" in error
    assert result.sources == []


def test_star_over_join_error_mode_accepts_evidence_backed_attribution() -> None:
    sql = """
    with j as (
        select * from a join b on a.id = b.person_id
    )
    select id, person_id from j
    """

    result = analyze_sql(sql, star_over_join_behavior="error")

    assert result.errors == []
    assert "id" in source_columns(result, "a")
    assert "person_id" in source_columns(result, "b")


def test_star_over_join_error_mode_accepts_a_declared_column() -> None:
    sql = """
    with inner_cte as (select label from labels),
    j as (select * from inner_cte join b on b.id = 1)
    select label from j
    """

    result = analyze_sql(sql, star_over_join_behavior="error")

    assert result.errors == []
    assert source_columns(result, "labels") == {"label"}


def test_star_over_join_error_mode_leaves_unambiguous_sql_alone() -> None:
    result = analyze_sql(
        "with j as (select * from a) select city from j",
        star_over_join_behavior="error",
    )

    assert result.errors == []
    assert "city" in source_columns(result, "a")


@pytest.mark.parametrize("behavior", ["guess", "error"])
def test_person_example_is_clean_under_both_behaviors(behavior: str) -> None:
    result = analyze_file(PERSON_SQL, star_over_join_behavior=behavior)  # type: ignore[arg-type]

    assert result.errors == []
    assert len(result.projection) == 7
