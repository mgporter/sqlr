from pathlib import Path

import pytest

from sqlrunner.schema_resolution import resolve_schema, widen
from sqlrunner.schema_resolution.types import ColumnSchema, ResolvedType, StatementSchema
from sqlrunner.sql_analysis import analyze_file, analyze_sql

FIXTURES = Path(__file__).parent / "fixtures"
PERSON_SQL = FIXTURES / "person.sql"


def schema_for(sql: str) -> StatementSchema:
    return resolve_schema(analyze_sql(sql))


def columns_of(schema: StatementSchema, table: str) -> dict[str, ColumnSchema]:
    for candidate in schema.tables:
        if candidate.name == table:
            return {column.name: column for column in candidate.columns}
    return {}


def _columns_by_name(sql: str) -> dict[str, tuple[str, str]]:
    schema = schema_for(sql)
    [table] = schema.tables
    return {c.name: (c.resolved_type, c.source) for c in table.columns}


# ---- type inference ----------------------------------------------------------------


def test_usage_beats_name_pattern() -> None:
    cols = _columns_by_name("select order_id from orders where order_id in ('a', 'b')")

    assert cols["order_id"] == ("string", "usage")


def test_int_literal_comparison_infers_numeric() -> None:
    # `quantity > 5` is equally true of an INT, a DECIMAL and a DOUBLE column.
    cols = _columns_by_name("select id from orders where quantity > 5")

    assert cols["quantity"] == ("numeric", "usage")


def test_float_literal_comparison_infers_numeric() -> None:
    cols = _columns_by_name("select id from orders where amount > 5.5")

    assert cols["amount"] == ("numeric", "usage")


def test_arithmetic_infers_numeric() -> None:
    cols = _columns_by_name("select salary * 12 as annual from orders")

    assert cols["salary"] == ("numeric", "usage")


def test_in_list_strings_infers_string() -> None:
    cols = _columns_by_name("select id from orders where status in ('shipped', 'pending')")

    assert cols["status"] == ("string", "usage")


def test_date_function_infers_date() -> None:
    cols = _columns_by_name("select date_trunc('day', created_at) from orders")

    assert cols["created_at"] == ("date", "usage")


def test_cast_does_not_type_the_column_it_reads() -> None:
    # A cast fixes the type of its *result*. `shipped_on` only has to be castable to a
    # timestamp, which a string column is too.
    cols = _columns_by_name(
        "select id from orders where cast(shipped_on as timestamp) = current_timestamp"
    )

    assert cols["shipped_on"] == ("unknown", "unknown")


def test_coalesce_literal_types_the_column_beside_it() -> None:
    cols = _columns_by_name("select coalesce(bonus, 0) as bonus from orders")

    assert cols["bonus"] == ("numeric", "usage")


def test_coalesce_string_literal_types_the_column_beside_it() -> None:
    cols = _columns_by_name("select coalesce(nickname, 'n/a') as nickname from orders")

    assert cols["nickname"] == ("string", "usage")


def test_boolean_context_infers_boolean() -> None:
    cols = _columns_by_name("select id from orders where active")

    assert cols["active"] == ("boolean", "usage")


def test_id_suffix_is_not_a_name_pattern() -> None:
    # An `*_id` is as likely to be a uuid string as an integer.
    cols = _columns_by_name("select customer_id from orders")

    assert cols["customer_id"] == ("unknown", "unknown")


def test_name_pattern_fallback_at() -> None:
    cols = _columns_by_name("select updated_at from orders")

    assert cols["updated_at"] == ("timestamp", "name_pattern")


def test_name_pattern_fallback_is_prefix() -> None:
    cols = _columns_by_name("select is_deleted from orders")

    assert cols["is_deleted"] == ("boolean", "name_pattern")


def test_unresolvable_column_is_unknown() -> None:
    cols = _columns_by_name("select notes from orders")

    assert cols["notes"] == ("unknown", "unknown")


def test_cast_to_unknown_target_falls_back_to_name_pattern() -> None:
    cols = _columns_by_name("select cast(created_at as variant) from orders")

    assert cols["created_at"] == ("timestamp", "name_pattern")


def test_conflicting_usage_prefers_higher_confidence() -> None:
    cols = _columns_by_name(
        "select id from orders where code in ('a', 'b') and code > 1"
    )

    assert cols["code"] == ("string", "usage")


def test_equally_weighted_conflicting_usage_is_widened() -> None:
    # Two coalesce defaults of the same weight disagree; nothing covers both.
    cols = _columns_by_name(
        "select coalesce(code, 0), coalesce(code, 'x') from orders"
    )

    assert cols["code"] == ("unknown", "unknown")


def test_multiple_tables_resolved_independently() -> None:
    sql = """
    select o.amount, p.nickname
    from orders o
    join person p on p.id = o.person_id
    where o.amount > 100 and p.nickname like 'a%'
    """
    schema = schema_for(sql)

    assert columns_of(schema, "orders")["amount"].resolved_type == "numeric"
    assert columns_of(schema, "person")["nickname"].resolved_type == "string"


# ---- type families -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("integer", "integer", "integer"),
        ("integer", "decimal", "number"),
        ("integer", "float", "numeric"),
        ("number", "float", "numeric"),
        ("integer", "number", "number"),
        ("numeric", "decimal", "numeric"),
        ("integer", "string", "unknown"),
        ("date", "timestamp", "unknown"),
        ("integer", "unknown", "unknown"),
    ],
)
def test_widen_returns_the_narrowest_covering_type(
    left: ResolvedType, right: ResolvedType, expected: ResolvedType
) -> None:
    assert widen(left, right) == expected
    assert widen(right, left) == expected


# ---- join groups -------------------------------------------------------------------


def test_join_group_propagates_a_type_to_the_untyped_side() -> None:
    sql = """
    select o.id
    from orders o
    join person p on p.id = o.person_id
    where p.id in ('x', 'y')
    """
    schema = schema_for(sql)

    person_id = columns_of(schema, "orders")["person_id"]
    assert person_id.resolved_type == "string"
    assert person_id.source == "join_group"
    assert person_id.evidence is not None and "joined to" in person_id.evidence


def test_join_group_members_share_an_index() -> None:
    schema = resolve_schema(analyze_file(PERSON_SQL))

    person = columns_of(schema, "mydatabase.myschema.person")
    address = columns_of(schema, "mydatabase.myschema.address")

    assert person["id"].join_group is not None
    assert person["id"].join_group == address["person_id"].join_group
    assert len(schema.join_groups) == 1


def test_stronger_evidence_survives_join_group_unification() -> None:
    sql = """
    select o.id
    from orders o
    join person p on p.id = o.person_id
    where o.person_id in ('a', 'b')
    """
    schema = schema_for(sql)

    person_id = columns_of(schema, "orders")["person_id"]
    assert person_id.resolved_type == "string"
    assert person_id.source == "usage"


def test_join_group_with_conflicting_member_types_is_widened() -> None:
    sql = """
    with a as (select cast(k as int) as k from t1),
         b as (select cast(k as decimal) as k from t2)
    select a.k from a join b on a.k = b.k
    """
    schema = schema_for(sql)

    assert [group for group in schema.join_groups] == [["cte:a.k", "cte:b.k"]]
    projection = {column.name: column for column in schema.projection}
    assert projection["k"].resolved_type == "number"


# ---- constraints, nullability, confidence ------------------------------------------


def test_predicate_literals_become_constraints() -> None:
    schema = schema_for("select id from orders where status in ('a', 'b') and qty > 3")

    columns = columns_of(schema, "orders")
    assert [(c.operator, c.values) for c in columns["status"].constraints] == [
        ("in", ["a", "b"])
    ]
    assert [(c.operator, c.values) for c in columns["qty"].constraints] == [(">", ["3"])]


def test_is_not_null_predicate_sets_nullable_false() -> None:
    schema = schema_for("select id from orders where shipped_on is not null")

    assert columns_of(schema, "orders")["shipped_on"].nullable is False


def test_is_null_predicate_sets_nullable_true() -> None:
    schema = schema_for("select id from orders where shipped_on is null")

    assert columns_of(schema, "orders")["shipped_on"].nullable is True


def test_outer_join_padding_does_not_decide_source_nullability() -> None:
    schema = resolve_schema(analyze_file(PERSON_SQL))

    assert columns_of(schema, "mydatabase.myschema.address")["street"].nullable is None


def test_attribution_confidence_is_carried_through() -> None:
    schema = resolve_schema(analyze_file(PERSON_SQL))

    person = columns_of(schema, "mydatabase.myschema.person")
    assert person["name"].confidence == "inferred"
    assert person["age"].confidence == "explicit"


def test_star_expanded_flag_is_carried_through() -> None:
    schema = resolve_schema(analyze_file(PERSON_SQL))

    by_name = {table.name: table for table in schema.tables}
    assert by_name["mydatabase.myschema.person"].star_expanded is True
    assert by_name["mydatabase.myschema.address"].star_expanded is False


# ---- projection --------------------------------------------------------------------


def test_projection_types_follow_lineage_to_the_source() -> None:
    schema = resolve_schema(analyze_file(PERSON_SQL))

    projection = {column.name: column for column in schema.projection}
    assert projection["age"].resolved_type == "numeric"
    assert projection["modified_at"].resolved_type == "timestamp"
    assert projection["age"].origins == ["table:mydatabase.myschema.person.age"]


def test_derived_projection_type_comes_from_the_expression() -> None:
    schema = schema_for("select row_number() over (order by id) as rn from orders")

    projection = {column.name: column for column in schema.projection}
    assert projection["rn"].resolved_type == "integer"
    assert projection["rn"].source == "expression"


def test_aggregate_projection_type_comes_from_the_expression() -> None:
    schema = schema_for("select person_id, sum(amount) as total from orders group by person_id")

    projection = {column.name: column for column in schema.projection}
    assert projection["total"].resolved_type == "numeric"
    assert projection["total"].source == "expression"


def test_cast_types_the_column_it_produces() -> None:
    schema = schema_for("select cast(salary as decimal(10, 2)) as salary from orders")

    projection = {column.name: column for column in schema.projection}
    assert projection["salary"].resolved_type == "decimal"
    assert projection["salary"].source == "expression"


def test_cast_to_a_float_type_is_not_a_decimal() -> None:
    schema = schema_for("select cast(ratio as double) as ratio from orders")

    projection = {column.name: column for column in schema.projection}
    assert projection["ratio"].resolved_type == "float"


def test_cast_through_a_cte_reaches_the_final_projection() -> None:
    sql = """
    with c as (select cast(salary as decimal(10, 2)) as salary from orders)
    select * from c
    """
    schema = schema_for(sql)

    projection = {column.name: column for column in schema.projection}
    assert projection["salary"].resolved_type == "decimal"


def test_coalesce_literal_types_the_column_it_produces() -> None:
    schema = schema_for("select coalesce(bonus, 0) as bonus from orders")

    projection = {column.name: column for column in schema.projection}
    assert projection["bonus"].resolved_type == "numeric"
    assert projection["bonus"].source == "expression"


def test_coalesce_of_two_columns_stays_unknown() -> None:
    schema = schema_for("select coalesce(bonus, fallback) as bonus from orders")

    projection = {column.name: column for column in schema.projection}
    assert projection["bonus"].resolved_type == "unknown"


def test_datediff_projection_is_an_integer() -> None:
    schema = schema_for(
        "select datediff(year, hire_date, current_date()) as tenure from employee"
    )

    projection = {column.name: column for column in schema.projection}
    assert projection["tenure"].resolved_type == "integer"
    assert projection["tenure"].source == "expression"
