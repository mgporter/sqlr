"""Steps 4-7: annotate, extract facts, infer, re-annotate, check.

Every fixture here is an inline SQL string, per `tests/README.md`.

The one rule these tests exist to pin down: a fact is a claim about a value, a claim
contradicted by that value's actual type is an error, and a claim about a value with no type
at all is the inference. Most of the tests below are one half or the other of that sentence.
"""

from pathlib import Path

import pytest
from sqlglot import exp

from sqlr.catalog.types import SqlFile
from sqlr.selection.types import Model
from sqlr.sql_analysis2.annotate import (
    arguments_of_call,
    catalog_key,
    expression_metadata,
    in_family,
)
from sqlr.sql_analysis2.annotate_types import AnnotatedModel, annotate_one_model
from sqlr.sql_analysis2.catalog import CATALOG, Sig
from sqlr.sql_analysis2.infer import widen_schema_with_inferred_types
from sqlr.sql_analysis2.qualify import qualify_one_model
from sqlr.sql_analysis2.types import ColumnName, ColumnTypeName, TableName

DIALECT = "duckdb"
METADATA = expression_metadata(DIALECT)


def annotate(
    sql: str,
    declared: dict[TableName, dict[ColumnName, ColumnTypeName]],
    tmp_path: Path,
) -> AnnotatedModel:
    """Steps 1-7 over one file, the way `validate-schema` runs them."""
    path = tmp_path / "x.sql"
    path.write_text(sql)
    model = Model(
        name="x",
        file=SqlFile(path=path, relative_path="x.sql", mtime=0.0, content_hash=""),
    )
    return annotate_one_model(qualify_one_model(model, declared, DIALECT), METADATA)


def codes(result: AnnotatedModel) -> list[str]:
    return [finding.code for finding in result.findings]


def inferred_types(result: AnnotatedModel) -> dict[tuple[TableName, ColumnName], str]:
    return {
        (entry.table, entry.column): entry.type_name
        for entry in result.inference.inferred
    }


def projected_types(result: AnnotatedModel, scope: str = "<final>") -> dict[str, str]:
    return {
        column.name: column.type_name
        for entry in result.scopes
        if entry.name == scope
        for column in entry.columns
    }


ORDERS = {
    "orders": {
        "order_id": "bigint",
        "status": "varchar",
        "order_ts": "timestamp",
        "amount": "UNKNOWN",
    }
}


# ---------------------------------------------------------------- contradicted claims
def test_a_declared_column_used_wrongly_is_an_error(tmp_path: Path) -> None:
    """The declaration is the type, so the SQL is what is wrong. One finding, at the
    argument, naming the declaration that it contradicts."""
    result = annotate("select round(status, 2) as r from orders", ORDERS, tmp_path)

    assert codes(result) == ["contradicted-type"]
    finding = result.findings[0]
    assert finding.function == "ROUND"
    assert finding.argument_index == 0
    assert "expects NUMERIC" in finding.message
    assert "declared varchar" in finding.message
    assert result.has_errors


def test_a_computed_cte_column_used_wrongly_is_an_error(tmp_path: Path) -> None:
    """Nothing is declared about `status_up`; sqlglot computed it. The check does not care
    where a type came from, which is what makes "declared is the truth" a property of the
    data rather than a branch in the code."""
    result = annotate(
        """
        with enriched as (select upper(status) as status_up from orders)
        select round(status_up, 2) as bogus from enriched
        """,
        ORDERS,
        tmp_path,
    )
    assert codes(result) == ["contradicted-type"]


def test_a_column_with_no_type_produces_no_finding(tmp_path: Path) -> None:
    """The false-positive regression test. `in_family` answers None, not False, and None is
    never reported."""
    result = annotate("select round(amount, 2) as r from orders", ORDERS, tmp_path)
    assert codes(result) == []


def test_a_string_in_arithmetic_is_an_error(tmp_path: Path) -> None:
    """Operators are catalog entries, so `'abc' + 5` is caught by the same rule as a bad
    function argument - no arithmetic-specific code anywhere."""
    result = annotate("select 'abc' + 5 as x from orders", ORDERS, tmp_path)

    assert codes(result) == ["contradicted-type"]
    assert "expects" in result.findings[0].message


def test_date_arithmetic_is_not_an_error(tmp_path: Path) -> None:
    """`date + integer` is a real DuckDB overload, not an autocast. Claiming NUMERIC of a
    temporal operand is the mistake the catalog exists to prevent."""
    result = annotate("select order_ts + 7 as shifted from orders", ORDERS, tmp_path)
    assert codes(result) == []


def test_a_where_clause_requires_a_boolean(tmp_path: Path) -> None:
    result = annotate("select order_id from orders where status", ORDERS, tmp_path)

    assert codes(result) == ["contradicted-type"]
    assert "BOOLEAN" in result.findings[0].message


# ------------------------------------------------------------------------- inference
def test_a_claim_types_an_undeclared_column(tmp_path: Path) -> None:
    result = annotate("select upper(amount) as a from orders", ORDERS, tmp_path)

    assert inferred_types(result) == {("orders", "amount"): "VARCHAR"}
    assert codes(result) == []


def test_a_link_carries_the_other_end_s_exact_type(tmp_path: Path) -> None:
    """A link is more precise than a claim: comparing against a `bigint` column infers
    `bigint`, not "some numeric type"."""
    result = annotate(
        """
        select o.amount as a
        from orders o
        join customers c on o.amount = c.id
        """,
        {**ORDERS, "customers": {"id": "bigint"}},
        tmp_path,
    )
    assert inferred_types(result) == {("orders", "amount"): "BIGINT"}


def test_a_link_crosses_a_set_operation(tmp_path: Path) -> None:
    """Column i of one arm and column i of the other are one output column."""
    result = annotate(
        "select id from arm_a union all select id from arm_b",
        {"arm_a": {"id": "date"}, "arm_b": {"id": "UNKNOWN"}},
        tmp_path,
    )
    assert inferred_types(result) == {("arm_b", "id"): "DATE"}


def test_an_inferred_type_reaches_the_projection(tmp_path: Path) -> None:
    """What step 6 is for: after the second pass, declared and inferred are
    indistinguishable to everything downstream."""
    result = annotate("select upper(amount) as a from orders", ORDERS, tmp_path)
    assert projected_types(result) == {"a": "VARCHAR"}


def test_an_ambiguous_operator_position_infers_nothing(tmp_path: Path) -> None:
    """`+` accepts numeric, temporal and interval in its first position, so an unknown
    operand there is a disjunction. A disjunction is not something to infer from."""
    result = annotate("select amount + 7 as x from orders", ORDERS, tmp_path)

    assert inferred_types(result) == {}
    assert codes(result) == []


def test_conflicting_facts_report_every_site_and_infer_nothing(tmp_path: Path) -> None:
    """Engine autocasting is not something sqlr relies on, so this is an error - and the
    column stays UNKNOWN, so nothing downstream inherits a guess."""
    result = annotate(
        "select upper(amount) as a, amount / 2 as b from orders", ORDERS, tmp_path
    )

    assert codes(result) == ["conflicting-usage", "conflicting-usage"]
    assert inferred_types(result) == {}
    assert projected_types(result)["b"] == "unknown"
    assert {finding.span.start_line for finding in result.findings if finding.span} == {0}


def test_a_declared_type_is_never_overwritten(tmp_path: Path) -> None:
    """The monotonicity invariant, stated as the thing it protects: widening only ever
    fills UNKNOWN slots, which is what makes the steps 5+6 skip sound."""
    result = annotate("select upper(status) as s from orders", ORDERS, tmp_path)
    statement = result.qualified.statement
    assert statement is not None

    widened = widen_schema_with_inferred_types(
        statement.declared_types_per_table, result.inference
    )
    assert widened["orders"]["status"] == "varchar"


def test_a_fully_declared_schema_infers_nothing(tmp_path: Path) -> None:
    """Steps 5 and 6 are skipped, and the skip is a no-op by construction: no fact can be
    about an unknown slot when there are none."""
    declared = {"orders": {"order_id": "bigint", "status": "varchar"}}
    result = annotate(
        "select upper(status) as s, order_id from orders", declared, tmp_path
    )

    assert result.inference.inferred == []
    assert result.inference.conflicts == []
    assert projected_types(result) == {"s": "VARCHAR", "order_id": "BIGINT"}


# ------------------------------------------------------------- structural findings
def test_an_unknown_function_is_reported(tmp_path: Path) -> None:
    result = annotate("select zeroifnull(order_id) as z from orders", ORDERS, tmp_path)

    assert codes(result) == ["unknown-function"]
    finding = result.findings[0]
    assert "zeroifnull" in finding.message
    # The name, not the hull: sqlglot puts no position on the closing paren, so blaming the
    # call would underline `zeroifnull(order_id` and read as a bug in the tool.
    assert finding.span is not None
    assert result.qualified.source.text[finding.span.start : finding.span.end] == (
        "zeroifnull"
    )


def test_an_uncatalogued_function_with_a_class_is_not_reported(tmp_path: Path) -> None:
    """A function sqlglot parsed into a class of its own exists somewhere. Its absence from
    the catalog says our catalog is thin, not that the SQL is wrong."""
    result = annotate("select md5(status) as h from orders", ORDERS, tmp_path)
    assert codes(result) == []


def test_wrong_arity_is_reported_instead_of_a_type_error(tmp_path: Path) -> None:
    """Reported instead of, never beside: a one-argument call must not also be told its
    argument contradicts a two-argument signature it was never going to match."""
    result = annotate("select list_extract(status) as x from orders", ORDERS, tmp_path)

    assert codes(result) == ["function-arity"]
    assert "takes 2 arguments, 1 given" in result.findings[0].message


# ------------------------------------------------------------------ the catalog itself
@pytest.mark.parametrize("dialect_name", sorted(CATALOG))
def test_every_catalogued_function_can_be_called_with_its_signature_arity(
    dialect_name: str,
) -> None:
    """Catches the whole class of bug where a signature is written against the SQL's
    argument order rather than sqlglot's node order.

    A class's `arg_types` bounds how many positional arguments `arguments_of_call` can ever
    produce, so a signature wanting more than that can never match anything.
    """
    for key, signatures in CATALOG[dialect_name].items():
        cls = exp.FUNCTION_BY_NAME.get(key)
        if cls is None or issubclass(cls, exp.Anonymous):
            continue  # an operator or an Anonymous-only call; no arg_types to check
        for signature in signatures:
            assert len(signature.params) <= len(cls.arg_types), (
                f"{key} signature {signature.params} exceeds {cls.__name__}.arg_types"
            )


def test_a_variadic_signature_accepts_any_arity_above_its_minimum() -> None:
    signature = Sig(params=("@T",), returns="@arg0", variadic=True)
    assert signature.accepts_arity(1)
    assert signature.accepts_arity(4)
    assert not signature.accepts_arity(0)
    assert signature.family_at(3) == "@T"


def test_in_family_is_three_valued() -> None:
    """`False` means "definitely wrong, report it"; `None` means "cannot say, stay quiet".
    Conflating them is how a checker gets a reputation for lying."""
    assert in_family(exp.DataType.build("VARCHAR"), "STRING") is True
    assert in_family(exp.DataType.build("VARCHAR"), "NUMERIC") is False
    assert in_family(exp.DataType.build("UNKNOWN"), "NUMERIC") is None
    assert in_family(None, "NUMERIC") is None
    assert in_family(None, "ANY") is True


def test_operators_are_looked_up_by_symbol() -> None:
    """Operators have no `sql_name()` - they are `exp.Binary`, not `exp.Func` - which is
    why `OPERATOR_SQL_NAMES` is the one place the catalog names a class."""
    import sqlglot

    tree = sqlglot.parse_one("select a + b, a = b from t", read=DIALECT)
    keys = {catalog_key(node) for node in tree.walk() if catalog_key(node)}
    assert {"+", "="} <= keys


def test_arguments_of_call_follows_sqlglot_node_order() -> None:
    """`date_trunc('day', ts)` parses to `TimestampTrunc(this=ts, unit='day')`. A signature
    written the way the SQL reads produces a confident error on valid SQL."""
    import sqlglot

    tree = sqlglot.parse_one("select date_trunc('day', ts) from t", read=DIALECT)
    call = tree.find(exp.TimestampTrunc)
    assert call is not None
    assert [argument.sql() for argument in arguments_of_call(call)] == ["ts", "DAY"]
