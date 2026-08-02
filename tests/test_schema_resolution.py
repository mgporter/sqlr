from sqlrunner.schema_resolution import resolve_schema
from sqlrunner.sql_analysis import analyze_sql


def _columns_by_name(sql: str) -> dict[str, tuple[str, str]]:
    analysis = analyze_sql(sql)
    [table] = resolve_schema(analysis)
    return {c.name: (c.resolved_type, c.source) for c in table.columns}


def test_usage_beats_name_pattern() -> None:
    # order_id would name-pattern to integer, but usage says it's compared to a string list.
    cols = _columns_by_name("select order_id from orders where order_id in ('a', 'b')")

    assert cols["order_id"] == ("string", "usage")


def test_numeric_comparison_infers_integer() -> None:
    cols = _columns_by_name("select * from orders where quantity > 5")

    assert cols["quantity"] == ("integer", "usage")


def test_numeric_comparison_infers_decimal() -> None:
    cols = _columns_by_name("select * from orders where amount > 5.5")

    assert cols["amount"] == ("decimal", "usage")


def test_in_list_strings_infers_string() -> None:
    cols = _columns_by_name("select * from orders where status in ('shipped', 'pending')")

    assert cols["status"] == ("string", "usage")


def test_date_function_infers_date() -> None:
    cols = _columns_by_name("select date_trunc('day', created_at) from orders")

    assert cols["created_at"] == ("date", "usage")


def test_cast_infers_explicit_type() -> None:
    cols = _columns_by_name("select * from orders where cast(shipped_on as timestamp) = current_timestamp")

    assert cols["shipped_on"] == ("timestamp", "usage")


def test_boolean_context_infers_boolean() -> None:
    cols = _columns_by_name("select * from orders where active")

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


def test_multiple_tables_resolved_independently() -> None:
    sql = """
    select o.amount, p.customer_id
    from orders o
    join person p on p.id = o.person_id
    where o.amount > 100
    """
    analysis = analyze_sql(sql)
    tables = {t.name: {c.name: c.resolved_type for c in t.columns} for t in resolve_schema(analysis)}

    assert tables["orders"]["amount"] == "integer"
    assert tables["person"]["customer_id"] == "integer"


def test_cast_to_unknown_target_falls_back_to_name_pattern() -> None:
    cols = _columns_by_name("select cast(created_at as variant) from orders")

    assert cols["created_at"] == ("timestamp", "name_pattern")


def test_conflicting_usage_prefers_higher_confidence() -> None:
    # cast is more authoritative than a bare numeric comparison.
    sql = "select * from orders where cast(code as varchar) = '5' and code > 1"
    cols = _columns_by_name(sql)

    assert cols["code"] == ("string", "usage")
