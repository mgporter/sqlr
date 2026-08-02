from pathlib import Path

from sqlrunner.schema_resolution import resolve_schema
from sqlrunner.schema_resolution.types import ColumnSchema, StatementSchema
from sqlrunner.sql_analysis import analyze_file, analyze_sql

PERSON_SQL = Path("examples/bunch_of_sql_files/project/person.sql")


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


def test_numeric_comparison_infers_integer() -> None:
    cols = _columns_by_name("select id from orders where quantity > 5")

    assert cols["quantity"] == ("integer", "usage")


def test_numeric_comparison_infers_decimal() -> None:
    cols = _columns_by_name("select id from orders where amount > 5.5")

    assert cols["amount"] == ("decimal", "usage")


def test_in_list_strings_infers_string() -> None:
    cols = _columns_by_name("select id from orders where status in ('shipped', 'pending')")

    assert cols["status"] == ("string", "usage")


def test_date_function_infers_date() -> None:
    cols = _columns_by_name("select date_trunc('day', created_at) from orders")

    assert cols["created_at"] == ("date", "usage")


def test_cast_infers_explicit_type() -> None:
    cols = _columns_by_name(
        "select id from orders where cast(shipped_on as timestamp) = current_timestamp"
    )

    assert cols["shipped_on"] == ("timestamp", "usage")


def test_boolean_context_infers_boolean() -> None:
    cols = _columns_by_name("select id from orders where active")

    assert cols["active"] == ("boolean", "usage")


def test_name_pattern_fallback_id() -> None:
    cols = _columns_by_name("select customer_id from orders")

    assert cols["customer_id"] == ("integer", "name_pattern")


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
        "select id from orders where cast(code as varchar) = '5' and code > 1"
    )

    assert cols["code"] == ("string", "usage")


def test_multiple_tables_resolved_independently() -> None:
    sql = """
    select o.amount, p.customer_id
    from orders o
    join person p on p.id = o.person_id
    where o.amount > 100
    """
    schema = schema_for(sql)

    assert columns_of(schema, "orders")["amount"].resolved_type == "integer"
    assert columns_of(schema, "person")["customer_id"].resolved_type == "integer"


# ---- join groups -------------------------------------------------------------------


def test_join_group_propagates_a_type_to_the_untyped_side() -> None:
    sql = """
    select o.id
    from orders o
    join person p on p.id = o.person_id
    where cast(p.id as varchar) = 'x'
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
    assert projection["age"].resolved_type == "integer"
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
    assert projection["total"].resolved_type == "decimal"
    assert projection["total"].source == "expression"
