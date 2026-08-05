from pathlib import Path

import pytest

from sqlrunner.schema_resolution import resolve_schema, widen
from sqlrunner.schema_resolution.types import (
    ColumnSchema,
    ResolvedTypeName,
    StatementSchema,
)
from sqlrunner.sql_analysis import analyze_file, analyze_sql
from sqlrunner.sql_analysis.types import ColumnNode, RelationRef

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
    return {c.name: (c.type.type_name, c.type.source) for c in table.columns}


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

    assert columns_of(schema, "orders")["amount"].type.type_name == "numeric"
    assert columns_of(schema, "person")["nickname"].type.type_name == "string"


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
    left: ResolvedTypeName, right: ResolvedTypeName, expected: ResolvedTypeName
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
    assert person_id.type.type_name == "string"
    assert person_id.type.source == "join_group"
    assert person_id.type.chosen is not None
    assert person_id.type.chosen.detail is not None
    assert "joined to" in person_id.type.chosen.detail
    assert person_id.type.chosen.via == ColumnNode(
        relation=RelationRef(kind="table", name="person"), column="id"
    )


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
    assert person_id.type.type_name == "string"
    assert person_id.type.source == "usage"


def test_join_group_with_conflicting_member_types_is_widened() -> None:
    sql = """
    with a as (select cast(k as int) as k from t1),
         b as (select cast(k as decimal) as k from t2)
    select a.k from a join b on a.k = b.k
    """
    schema = schema_for(sql)

    assert [group.names for group in schema.join_groups] == [["cte:a.k", "cte:b.k"]]
    assert schema.join_groups[0].unified_type == "number"
    projection = {column.name: column for column in schema.projection}
    assert projection["k"].type.type_name == "number"


def test_a_widened_type_is_carried_by_no_single_piece_of_evidence() -> None:
    # Why `ResolvedType.type_name` is a field rather than a read of `chosen.type_name`:
    # the two casts weigh the same and disagree, so the tier is widened and the result is
    # a family neither cast names. `chosen` still points at the one that led.
    sql = """
    select cast(k as int) as k from t1
    union all
    select cast(k as decimal) as k from t2
    """
    [column] = schema_for(sql).projection

    assert column.type.type_name == "number"
    assert column.type.widened_from == ["integer", "decimal"]
    assert column.type.chosen is not None
    assert column.type.chosen.type_name == "integer"


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
    assert projection["age"].type.type_name == "numeric"
    assert projection["modified_at"].type.type_name == "timestamp"
    assert [str(origin) for origin in projection["age"].origins] == [
        "table:mydatabase.myschema.person.age"
    ]


def test_derived_projection_type_comes_from_the_expression() -> None:
    schema = schema_for("select row_number() over (order by id) as rn from orders")

    projection = {column.name: column for column in schema.projection}
    assert projection["rn"].type.type_name == "integer"
    assert projection["rn"].type.source == "expression"


def test_aggregate_projection_type_comes_from_the_expression() -> None:
    schema = schema_for("select person_id, sum(amount) as total from orders group by person_id")

    projection = {column.name: column for column in schema.projection}
    assert projection["total"].type.type_name == "numeric"
    assert projection["total"].type.source == "expression"


def test_cast_types_the_column_it_produces() -> None:
    schema = schema_for("select cast(salary as decimal(10, 2)) as salary from orders")

    projection = {column.name: column for column in schema.projection}
    assert projection["salary"].type.type_name == "decimal"
    assert projection["salary"].type.source == "expression"


def test_cast_to_a_float_type_is_not_a_decimal() -> None:
    schema = schema_for("select cast(ratio as double) as ratio from orders")

    projection = {column.name: column for column in schema.projection}
    assert projection["ratio"].type.type_name == "float"


def test_cast_through_a_cte_reaches_the_final_projection() -> None:
    sql = """
    with c as (select cast(salary as decimal(10, 2)) as salary from orders)
    select * from c
    """
    schema = schema_for(sql)

    projection = {column.name: column for column in schema.projection}
    assert projection["salary"].type.type_name == "decimal"


def test_coalesce_literal_types_the_column_it_produces() -> None:
    schema = schema_for("select coalesce(bonus, 0) as bonus from orders")

    projection = {column.name: column for column in schema.projection}
    assert projection["bonus"].type.type_name == "numeric"
    assert projection["bonus"].type.source == "expression"


def test_coalesce_of_two_columns_stays_unknown() -> None:
    schema = schema_for("select coalesce(bonus, fallback) as bonus from orders")

    projection = {column.name: column for column in schema.projection}
    assert projection["bonus"].type.type_name == "unknown"


def test_datediff_projection_is_an_integer() -> None:
    schema = schema_for(
        "select datediff(year, hire_date, current_date()) as tenure from employee"
    )

    projection = {column.name: column for column in schema.projection}
    assert projection["tenure"].type.type_name == "integer"
    assert projection["tenure"].type.source == "expression"


# ---- evidence is kept, not just the winner -----------------------------------------
#
# The point of the refactor: inference still picks one type, but everything that argued
# for a different one survives, located, so a divergence can be explained rather than
# merely asserted.


def test_losing_evidence_is_retained_and_ordered() -> None:
    schema = schema_for(
        "select id from orders\nwhere revenue > 1000 and revenue in ('a', 'b')"
    )

    revenue = columns_of(schema, "orders")["revenue"]
    assert revenue.type.type_name == "string"
    assert [(e.type_name, e.weight) for e in revenue.type.evidence] == [
        ("string", 60),
        ("numeric", 50),
    ]
    assert revenue.type.chosen is not None and revenue.type.chosen.type_name == "string"


def test_evidence_carries_the_source_range_that_produced_it() -> None:
    schema = schema_for("select id from orders\nwhere revenue > 1000")

    revenue = columns_of(schema, "orders")["revenue"]
    assert revenue.type.chosen is not None
    assert schema.snippet(revenue.type.chosen.context_span) == "revenue > 1000"
    assert schema.snippet(revenue.type.chosen.span) == "revenue"


def test_name_pattern_is_collected_even_when_it_loses() -> None:
    schema = schema_for("select id from orders where total_amount in ('a', 'b')")

    column = columns_of(schema, "orders")["total_amount"]
    assert column.type.type_name == "string"
    kinds = {(e.kind, e.type_name) for e in column.type.evidence}
    assert ("name_pattern", "numeric") in kinds
    assert ("usage", "string") in kinds


def test_join_group_unification_does_not_erase_member_evidence() -> None:
    # The regression this refactor exists to prevent: unification used to overwrite the
    # member's own evidence, so nothing could explain why the group settled where it did.
    sql = """
    select o.id
    from orders o
    join person p on p.id = o.person_id
    where p.id in ('x', 'y')
    """
    schema = schema_for(sql)

    person_id = columns_of(schema, "person")["id"]
    assert person_id.type.type_name == "string"
    assert any(e.kind == "usage" for e in person_id.type.evidence)

    propagated = columns_of(schema, "orders")["person_id"]
    assert propagated.type.type_name == "string"
    assert any(e.kind == "join_group" for e in propagated.type.evidence)


def test_equal_weight_disagreement_records_what_was_widened() -> None:
    schema = schema_for(
        "with a as (select cast(k as int) as k from t1),\n"
        "     b as (select cast(k as decimal) as k from t2)\n"
        "select a.k from a join b on a.k = b.k"
    )

    assert schema.join_groups[0].unified_type == "number"
    assert schema.join_groups[0].facts, "the joins that linked the group are kept"


def test_join_group_facts_are_located() -> None:
    schema = schema_for(
        "select o.id from orders o join person p on p.id = o.person_id"
    )

    [group] = schema.join_groups
    [fact] = group.facts
    assert schema.snippet(fact.context_span) == "p.id = o.person_id"


# ---- constraints keep what a generator needs ---------------------------------------


def test_constraints_keep_the_literal_kind() -> None:
    schema = schema_for("select id from orders where qty > 20")

    [constraint] = columns_of(schema, "orders")["qty"].constraints
    assert (constraint.operator, constraint.values) == (">", ["20"])
    assert constraint.literal_kind == "int"


def test_constraints_are_located_for_every_literal() -> None:
    schema = schema_for("select id from orders where status in ('a', 'b')")

    [constraint] = columns_of(schema, "orders")["status"].constraints
    assert schema.snippet(constraint.context_span) == "status in ('a', 'b')"
    assert [schema.snippet(s) for s in constraint.value_spans] == ["'a'", "'b'"]


def test_the_same_predicate_written_twice_is_kept_twice() -> None:
    # Two places to underline, and two hints about the data to generate.
    schema = schema_for("select id from orders where qty > 20 or qty > 20")

    constraints = columns_of(schema, "orders")["qty"].constraints
    assert len(constraints) == 2
    assert constraints[0].context_span != constraints[1].context_span


def test_an_identical_predicate_at_one_place_is_kept_once() -> None:
    schema = schema_for("select id from orders where qty > 20")

    assert len(columns_of(schema, "orders")["qty"].constraints) == 1


# ---- nullability keeps its losing facts ---------------------------------------------


def test_nullability_keeps_every_fact_including_the_ones_that_lost() -> None:
    schema = resolve_schema(analyze_file(PERSON_SQL))

    street = columns_of(schema, "mydatabase.myschema.address")["street"]
    assert street.nullable is None
    # `outer_join_padded` describes the join result, not the stored column, so it does
    # not decide - but it is still available to anything that wants it.
    assert [f.reason for f in street.nullability.facts] == ["outer_join_padded"]


def test_the_deciding_nullability_fact_is_identified() -> None:
    schema = schema_for("select id from orders where shipped_on is not null")

    resolution = columns_of(schema, "orders")["shipped_on"].nullability
    assert resolution.nullable is False
    assert resolution.chosen is not None
    assert resolution.chosen.reason == "is_not_null_predicate"


# ---- column references and lookups --------------------------------------------------


def test_columns_carry_every_reference() -> None:
    schema = schema_for("select amount\nfrom orders\nwhere amount > 20")

    amount = columns_of(schema, "orders")["amount"]
    assert [schema.snippet(span) for span in amount.references] == ["amount", "amount"]


def test_schema_lookup_helpers() -> None:
    schema = schema_for("select id from orders where qty > 1")

    assert schema.table("ORDERS") is not None
    assert schema.column("orders", "QTY") is not None
    assert schema.column("orders", "nope") is None


def test_the_schema_carries_its_source() -> None:
    schema = resolve_schema(analyze_file(PERSON_SQL))

    assert schema.source.path == PERSON_SQL


# ---- inferring input column types --------------------------------------------------
#
# These columns feed fixture generation, so leaving them `unknown` means there is nothing
# to generate from. Each case below was previously untyped.


def test_equality_with_a_string_literal_types_the_column() -> None:
    cols = _columns_by_name("select id from orders where country = 'US'")

    assert cols["country"] == ("string", "usage")


def test_string_concatenation_types_its_operands() -> None:
    schema = schema_for("select street || ', ' || city as full_address from address")

    columns = columns_of(schema, "address")
    assert columns["street"].type.type_name == "string"
    assert columns["city"].type.type_name == "string"


def test_a_string_function_types_its_input_column() -> None:
    cols = _columns_by_name("select upper(code) as code from orders")

    assert cols["code"] == ("string", "usage")


def test_a_numeric_function_types_its_input_column() -> None:
    cols = _columns_by_name("select round(rate, 2) as rate from orders")

    assert cols["rate"] == ("numeric", "usage")


def test_a_string_function_and_a_name_pattern_are_both_kept() -> None:
    # The user should see that both agreed, not just that the answer was `string`.
    schema = schema_for("select upper(department_name) as department_name from d")

    column = columns_of(schema, "d")["department_name"]
    assert column.type.type_name == "string"
    assert [(e.kind, e.detail) for e in column.type.evidence] == [
        ("usage", "string_function"),
        ("name_pattern", "*_name"),
    ]


def test_usage_evidence_outranks_the_name_pattern_that_agrees_with_it() -> None:
    schema = schema_for("select upper(department_name) as department_name from d")

    column = columns_of(schema, "d")["department_name"]
    assert column.type.chosen is not None
    assert column.type.chosen.kind == "usage"


# ---- no duplicate evidence ----------------------------------------------------------


def test_a_derived_projection_carries_its_expression_evidence_once() -> None:
    # The derived column is typed once as its relation's output; the projection resolves
    # to that same node. Deriving the expression evidence a second time here would report
    # one `upper(...)` as two independent observations.
    schema = schema_for("select upper(name) as name from t")

    [projected] = [p for p in schema.projection if p.name == "name"]
    functions = [e for e in projected.type.evidence if e.kind == "function"]
    assert len(functions) == 1
    assert functions[0].detail == "Upper"


def test_a_cast_projection_carries_its_evidence_once() -> None:
    schema = schema_for("select cast(salary as decimal(10, 2)) as salary from t")

    [projected] = [p for p in schema.projection if p.name == "salary"]
    assert len([e for e in projected.type.evidence if e.kind == "cast"]) == 1
    assert projected.type.type_name == "decimal"


def test_two_separate_uses_of_one_kind_are_both_kept() -> None:
    # Deduplication must not collapse genuinely distinct observations.
    schema = schema_for("select id from orders\nwhere qty > 1 and qty > 5")

    column = columns_of(schema, "orders")["qty"]
    numeric = [e for e in column.type.evidence if e.detail == "compared_to_number"]
    assert len(numeric) == 2
    assert numeric[0].span != numeric[1].span
