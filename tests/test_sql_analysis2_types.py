"""Steps 4-7: annotate, extract facts, infer, re-annotate, check.

Every fixture here is an inline SQL string, per `tests/README.md`.

The one rule these tests exist to pin down: a fact is a claim about a value, a claim
contradicted by that value's actual type is an error, and a claim about a value with no type
at all is the inference. Most of the tests below are one half or the other of that sentence.
"""

from pathlib import Path

import pytest
from sqlglot import exp

from declared_helpers import declarations, q

from sqlr.catalog.types import SqlFile
from sqlr.selection.types import Model
from sqlr.sql_analysis2.annotate import (
    arguments_of_call,
    catalog_key,
    expression_metadata,
)
from sqlr.sql_analysis2.annotate_types import AnnotatedModel, annotate_one_model
from sqlr.sql_analysis2.catalog import (
    CATALOG_BY_DIALECT,
    CATALOG_IS_COMPLETE,
    Sig,
    signatures_for_dialect,
)
from sqlr.sql_analysis2.families import in_family
from sqlr.sql_analysis2.infer import widen_schema_with_inferred_types
from sqlr.sql_analysis2.qualify import qualify_one_model
from sqlr.sql_analysis2.types import ColumnName, ColumnTypeName, RelationKey

DIALECT = "duckdb"
METADATA = expression_metadata(DIALECT)


def annotate(
    sql: str,
    declared: dict[RelationKey, dict[ColumnName, ColumnTypeName]],
    tmp_path: Path,
    partial: frozenset[RelationKey] = frozenset(),
    built_by: dict[RelationKey, str] | None = None,
) -> AnnotatedModel:
    """Steps 1-7 over one file, the way `validate-schema` runs them.

    The file is always `x.sql`, so `built_by={key: "x"}` is what makes one of the
    declarations a description of *this file's output* rather than of a table it reads.
    """
    path = tmp_path / "x.sql"
    path.write_text(sql)
    model = Model(
        name="x",
        file=SqlFile(path=path, relative_path="x.sql", mtime=0.0, content_hash=""),
    )
    return annotate_one_model(
        qualify_one_model(
            model, declarations(declared, partial, built_by), DIALECT
        ),
        METADATA,
    )


def codes(result: AnnotatedModel) -> list[str]:
    return [finding.code for finding in result.findings]


def inferred_types(result: AnnotatedModel) -> dict[tuple[RelationKey, ColumnName], str]:
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
        statement.declared_types_per_relation, result.inference
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
def test_an_unknown_function_is_dormant_while_the_catalog_is_incomplete(
    tmp_path: Path,
) -> None:
    """"This function does not exist" is only truthful from an exhaustive list.

    Every catalog sqlr ships is a hand-written gap-filler, so the finding stays off until a
    dialect is marked complete - otherwise every real function sqlglot happens to parse as
    `Anonymous` becomes a user-facing error.
    """
    result = annotate("select frobnicate(order_id) as z from orders", ORDERS, tmp_path)

    assert codes(result) == []


def test_an_unknown_function_is_reported_when_the_catalog_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(CATALOG_IS_COMPLETE, DIALECT, True)
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
@pytest.mark.parametrize("dialect_name", sorted(CATALOG_BY_DIALECT))
def test_every_catalogued_function_can_be_called_with_its_signature_arity(
    dialect_name: str,
) -> None:
    """Catches the whole class of bug where a signature is written against the SQL's
    argument order rather than sqlglot's node order.

    A class's `arg_types` bounds how many positional arguments `arguments_of_call` can ever
    produce, so a signature wanting more than that can never match anything.
    """
    for key, signatures in signatures_for_dialect(dialect_name).items():
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


# ============================================================== strength and the lattice
def annotate_in(
    sql: str,
    declared: dict[RelationKey, dict[ColumnName, ColumnTypeName]],
    tmp_path: Path,
    dialect_name: str,
) -> AnnotatedModel:
    """`annotate`, for the tests that are about a dialect rather than about SQL."""
    path = tmp_path / "x.sql"
    path.write_text(sql)
    model = Model(
        name="x",
        file=SqlFile(path=path, relative_path="x.sql", mtime=0.0, content_hash=""),
    )
    return annotate_one_model(
        qualify_one_model(model, declarations(declared), dialect_name),
        expression_metadata(dialect_name),
    )


def severities(result: AnnotatedModel) -> list[str]:
    return [finding.severity for finding in result.findings]


# ---- a literal is a stated type ---------------------------------------------------
def test_a_string_literal_comparison_states_the_column_is_a_string(
    tmp_path: Path,
) -> None:
    """sqlr does not model engine autocasting. If the column is really a date, the SQL
    should say `date '2024-01-01'` or the yml should declare it - so the honest reading of
    this file is that `d` is a string, and the tool says so rather than guessing."""
    result = annotate(
        "select d from events where d >= '2024-01-01'", {"events": {"d": "UNKNOWN"}}, tmp_path
    )

    assert inferred_types(result) == {("events", "d"): "VARCHAR"}
    assert result.inference.inferred[0].strength == "stated"
    assert codes(result) == []


def test_a_stated_string_used_as_a_date_is_an_error(tmp_path: Path) -> None:
    """The headline behaviour. `d` is stated STRING by the literal it is compared against,
    so a temporal use of it is a contradiction and not a resolution failure."""
    result = annotate(
        """
        select date_trunc('day', d) as day
        from events
        where d >= '2024-01-01'
        """,
        {"events": {"d": "UNKNOWN"}},
        tmp_path,
    )

    assert codes(result) == ["contradicted-type"]
    assert severities(result) == ["error"]
    assert "TEMPORAL" in result.findings[0].message


def test_a_numeric_literal_states_the_family_and_not_the_width(tmp_path: Path) -> None:
    """`> 0` proves the column is numeric and proves nothing about its width. Inferring
    `INT` from it would falsely contradict `amount * 1.5` three CTEs later."""
    result = annotate(
        "select amount from orders where amount > 0", ORDERS, tmp_path
    )

    entry = result.inference.inferred[0]
    assert (entry.family, entry.type_name) == ("NUMERIC", "NUMERIC")
    # The schema sqlglot re-annotates with needs something it can parse, and that stand-in
    # is the one place a concrete type is allowed to appear.
    assert entry.schema_type_name == "DECIMAL"


def test_a_family_with_no_parameter_free_type_of_its_own_still_annotates(
    tmp_path: Path,
) -> None:
    """`TEMPORAL` is what a report should say and is not a type any dialect parses, so the
    schema handed to pass 2 gets the stand-in. Reporting the stand-in instead would claim
    the column is a timestamp when a date satisfies the evidence just as well."""
    result = annotate(
        "select date_trunc('day', ts) as d from events",
        {"events": {"ts": "UNKNOWN"}},
        tmp_path,
    )

    entry = result.inference.inferred[0]
    assert (entry.family, entry.type_name, entry.schema_type_name) == (
        "TEMPORAL",
        "TEMPORAL",
        "TIMESTAMP",
    )
    # Pass 2 parsed the stand-in and typed the projection from it - `unknown` here would
    # mean the schema never took the inferred type at all.
    assert projected_types(result) == {"d": "TIMESTAMPNTZ"}


def test_a_type_carrying_no_parameters_is_reported_without_any(tmp_path: Path) -> None:
    """sqlglot's generator fills defaults in: a `DataType` built from `DECIMAL` holds no
    parameters and still renders as `DECIMAL(38, 0)`. Printing that would put a precision in
    the report that the yml never wrote."""
    result = annotate(
        "select amount as a from orders",
        {"orders": {"amount": "decimal"}},
        tmp_path,
    )

    assert projected_types(result) == {"a": "DECIMAL"}


def test_a_passthrough_of_an_inferred_column_is_reported_as_that_column(
    tmp_path: Path,
) -> None:
    """A scope that does nothing but read an inferred column reports what the source column
    reports. sqlglot only ever saw the stand-in, so asking it would print `DECIMAL` one line
    under the `NUMERIC` the source table is listed as. Anything *computed* keeps sqlglot's
    answer - `amount * 2` really is a decimal expression."""
    result = annotate(
        """
        with a as (select amount, amount * 2 as doubled from orders where amount > 0)
        select amount, doubled from a
        """,
        ORDERS,
        tmp_path,
    )

    assert projected_types(result, scope="a") == {"amount": "NUMERIC", "doubled": "DECIMAL"}
    # And a width appears nowhere down the chain: the stand-in in the schema carries none,
    # so neither does anything sqlglot computed from it.
    assert projected_types(result) == {"amount": "DECIMAL", "doubled": "DECIMAL"}


def test_a_link_to_a_declared_column_keeps_its_exact_type(tmp_path: Path) -> None:
    """The other half of the rule: a *column* anchor is exact, only a literal is widened."""
    result = annotate(
        "select o.amount from orders o join t2 on o.amount = t2.n",
        {**ORDERS, "t2": {"n": "decimal(10,2)"}},
        tmp_path,
    )
    assert inferred_types(result)[("orders", "amount")] == "DECIMAL(10, 2)"


# ---- the lattice ------------------------------------------------------------------
def test_an_integer_compared_with_a_decimal_is_not_a_conflict(tmp_path: Path) -> None:
    """Compatible iff the nearest common family is not ANY. `int_col = decimal_col` is
    ordinary SQL, and reporting it is how a checker gets switched off."""
    result = annotate(
        "select a.n from a join b on a.n = b.n",
        {"a": {"n": "bigint"}, "b": {"n": "decimal(10,2)"}},
        tmp_path,
    )
    assert codes(result) == []


def test_a_date_compared_with_a_timestamp_is_not_a_conflict(tmp_path: Path) -> None:
    result = annotate(
        "select a.d from a join b on a.d = b.ts",
        {"a": {"d": "date"}, "b": {"ts": "timestamp"}},
        tmp_path,
    )
    assert codes(result) == []


def test_two_decimals_of_different_precision_are_not_a_conflict(tmp_path: Path) -> None:
    """Precision and scale are recorded, printed, and never compared."""
    result = annotate(
        "select a.n from a join b on a.n = b.n",
        {"a": {"n": "decimal(10,2)"}, "b": {"n": "decimal(38,9)"}},
        tmp_path,
    )
    assert codes(result) == []


def test_a_string_compared_with_a_number_is_an_error_at_every_site(
    tmp_path: Path,
) -> None:
    """Two stated types in one component: two things were written down and one is wrong.
    Every site gets a finding, because every site is a place the user has to look."""
    result = annotate(
        "select a.s from a join b on a.s = b.n",
        {"a": {"s": "varchar"}, "b": {"n": "bigint"}},
        tmp_path,
    )

    assert codes(result) == ["conflicting-usage", "conflicting-usage"]
    assert severities(result) == ["error", "error"]
    assert "does not assume the engine will cast" in result.findings[0].message


def test_conflicting_claims_are_a_warning_rather_than_an_error(tmp_path: Path) -> None:
    """Inferred-versus-inferred. sqlr guessed twice and the guesses fought; the SQL may be
    fine and the honest answer is "I could not tell"."""
    result = annotate(
        "select upper(amount) as a, amount / 2 as b from orders", ORDERS, tmp_path
    )

    assert severities(result) == ["warning", "warning"]
    assert not result.has_errors
    assert inferred_types(result) == {}


# ---- components -------------------------------------------------------------------
def test_a_type_crosses_a_chain_of_links(tmp_path: Path) -> None:
    """Transitivity, and strength crossing it intact: `c.z` is declared, so `a.x` is
    stated - not "stated, then weaker, then weaker still"."""
    result = annotate(
        """
        select a.x
        from a
        join b on a.x = b.y
        join c on b.y = c.z
        """,
        {"a": {"x": "UNKNOWN"}, "b": {"y": "UNKNOWN"}, "c": {"z": "date"}},
        tmp_path,
    )

    assert inferred_types(result) == {("a", "x"): "DATE", ("b", "y"): "DATE"}
    assert {entry.strength for entry in result.inference.inferred} == {"stated"}


def test_two_reads_of_one_column_are_one_component(tmp_path: Path) -> None:
    """`orders.amount` at one line and at another are two `exp.Column` nodes and one
    column. Without the union by schema slot the claim and the link land in different
    components and neither ever sees the other - so this would infer VARCHAR in silence."""
    result = annotate(
        "select upper(amount) as a from orders where amount = 5", ORDERS, tmp_path
    )

    assert inferred_types(result)[("orders", "amount")] == "NUMERIC"
    assert codes(result) == ["contradicted-type"]


def test_a_claim_reaches_a_storage_column_through_three_ctes(tmp_path: Path) -> None:
    """Projection-passthrough. Without it transitivity stops at the first CTE boundary and
    a claim made downstream never reaches the table it is really about."""
    result = annotate(
        """
        with a as (select amount from orders),
             b as (select amount from a),
             c as (select amount from b)
        select upper(amount) as x from c
        """,
        ORDERS,
        tmp_path,
    )
    assert inferred_types(result) == {("orders", "amount"): "VARCHAR"}


def test_arithmetic_carries_a_claim_back_to_its_operand(tmp_path: Path) -> None:
    """`min(x)` returns `@arg0`, which says the result *is* `x`'s type - a link written in
    the catalog. `min` claims nothing about its argument (its parameter is ANY), so this
    type can only have arrived backwards through the return marker."""
    result = annotate(
        """
        with a as (select min(amount) as m from orders)
        select upper(m) as x from a
        """,
        ORDERS,
        tmp_path,
    )
    assert inferred_types(result) == {("orders", "amount"): "VARCHAR"}


def test_a_family_return_marker_carries_the_family_and_not_the_width(
    tmp_path: Path,
) -> None:
    """`SUM(INT)` is HUGEINT in DuckDB and NUMBER(38,0) in Snowflake, so `sum` is not
    `@arg0`. The component keeps the family and forfeits the concrete type - `decimal(10,2)`
    is at the far end of the family link, and its width does not cross."""
    result = annotate(
        """
        with a as (select employee_id, sum(amount) as total from orders group by employee_id)
        select a.total from a join t2 on a.total = t2.n
        """,
        {**ORDERS, "orders": {**ORDERS["orders"], "employee_id": "bigint"},
         "t2": {"n": "decimal(10,2)"}},
        tmp_path,
    )
    assert inferred_types(result)[("orders", "amount")] == "DECIMAL"


# ---- the catalog ------------------------------------------------------------------
@pytest.mark.parametrize("dialect_name", ["duckdb", "snowflake", "spark"])
def test_upper_of_a_number_is_an_error_in_every_dialect(
    tmp_path: Path, dialect_name: str
) -> None:
    """The common catalog's whole reason for existing: this signature belongs to no engine
    in particular, so it must not have to be written down once per engine."""
    result = annotate_in(
        "select upper(n) as u from t", {"t": {"n": "bigint"}}, tmp_path, dialect_name
    )
    assert codes(result) == ["contradicted-type"]


def test_a_dialect_entry_replaces_the_common_one(tmp_path: Path) -> None:
    """Replacement, never merged: a dialect layer exists to say something *different*, and
    appending overloads could only ever widen what a key accepts."""
    common = signatures_for_dialect("duckdb")["CONCAT"]
    spark = signatures_for_dialect("spark")["CONCAT"]

    assert len(common) == 1
    assert len(spark) == 2
    assert {sig.params[0] for sig in spark} == {"STRING", "ARRAY"}


# ---- the declared-type boundary ---------------------------------------------------
def test_an_unparseable_declared_type_is_reported_and_not_believed(
    tmp_path: Path,
) -> None:
    """Without the boundary check the name reaches sqlglot as a *user-defined* type, which
    belongs to no family, which makes every family test answer False - so a typo in the yml
    would come back as a confident `contradicted-type` error about the SQL."""
    result = annotate("select upper(s) as u from t", {"t": {"s": "frobnicate"}}, tmp_path)

    qualified = result.qualified
    assert [finding.code for finding in qualified.findings] == [
        "unrecognized-declared-type"
    ]
    assert qualified.findings[0].severity == "warning"
    assert "frobnicate" in qualified.findings[0].message
    # ...and the column falls back to undeclared, so the SQL is checked, not blamed.
    assert codes(result) == []
    assert inferred_types(result) == {("t", "s"): "VARCHAR"}


def test_a_structured_type_name_still_parses(tmp_path: Path) -> None:
    """`json`, `variant`, `struct(...)` and `map(...)` are real type names. Only a name
    nothing recognises is rejected."""
    for written in ("json", "variant", "struct(a int)", "map(varchar, int)"):
        result = annotate("select s from t", {"t": {"s": written}}, tmp_path)
        assert result.qualified.findings == [], written


# ---- provenance -------------------------------------------------------------------
def test_a_declared_column_reports_provenance_declared(tmp_path: Path) -> None:
    """The schema is keyed on the full `RelationKey`, so a bare-name lookup misses every
    time - which reported every declared column as `unknown`."""
    result = annotate("select status from orders", ORDERS, tmp_path)

    provenance = {
        column.name: column.provenance
        for entry in result.scopes
        for column in entry.columns
    }
    assert provenance == {"status": "declared"}


# ---- the file's own output against its own declaration ----------------------------
#
# The one check that reads a declaration of the *model* rather than of a table the file
# reads. Its whole effect is on findings: a model's declaration describes the output, so
# nothing in the file may be typed from it - which is what the isolation test below pins.

BUILT = {q("x"): "x"}
"""`meta.source_file: x` - the declaration of `x.sql`'s own output."""

ORDERS_TYPED = {q("orders"): {"order_id": "bigint", "amount": "decimal(10,2)"}}


def declaration_findings(result: AnnotatedModel) -> list[tuple[str, str, str]]:
    """Each output finding as `(code, severity, column)`, in the order reported."""
    return [
        (finding.code, finding.severity, finding.column_name or "")
        for finding in result.findings
        if finding.code
        in (
            "contradicted-declaration",
            "projected-column-not-declared",
            "declared-column-not-projected",
        )
    ]


def test_a_file_with_no_declaration_of_its_own_is_not_held_against_one(
    tmp_path: Path,
) -> None:
    """The regression that matters most here: every model without `meta.source_file` has to
    go on being checked exactly as it was."""
    result = annotate(
        f"select order_id, amount from {q('orders')}", ORDERS_TYPED, tmp_path
    )
    assert declaration_findings(result) == []


def test_a_projected_column_the_declaration_omits_is_an_error(tmp_path: Path) -> None:
    """The same rule reading a column off a closed table follows. A declaration without
    `declaration_is_partial` is the complete column list, so a file producing a column it
    omits contradicts it - and any file *reading* that column already errors."""
    result = annotate(
        f"select order_id, amount from {q('orders')}",
        {**ORDERS_TYPED, q("x"): {"order_id": "bigint"}},
        tmp_path,
        built_by=BUILT,
    )

    assert declaration_findings(result) == [
        ("projected-column-not-declared", "error", "amount")
    ]
    assert "does not declare it" in result.findings[0].message
    assert result.has_errors


def test_a_partial_declaration_allows_columns_it_does_not_list(tmp_path: Path) -> None:
    """`declaration_is_partial` means the same thing on a model as on a source table: the
    listed columns are described and the rest are not claimed either way."""
    result = annotate(
        f"select order_id, amount from {q('orders')}",
        {**ORDERS_TYPED, q("x"): {"order_id": "bigint"}},
        tmp_path,
        partial=frozenset({q("x")}),
        built_by=BUILT,
    )
    assert declaration_findings(result) == []


def test_a_declared_column_the_file_does_not_produce_is_an_error(
    tmp_path: Path,
) -> None:
    """The projection list is the whole answer here - no star expanded over anything - so
    the column really is not produced."""
    result = annotate(
        f"select order_id from {q('orders')}",
        {**ORDERS_TYPED, q("x"): {"order_id": "bigint", "absent": "varchar"}},
        tmp_path,
        built_by=BUILT,
    )

    assert declaration_findings(result) == [
        ("declared-column-not-projected", "error", "absent")
    ]


def test_a_declared_column_missing_from_a_lower_bound_projection_is_a_warning(
    tmp_path: Path,
) -> None:
    """A `*` over a table nobody declares expands to the columns this file happens to name,
    not to the columns the table has, so the projection is a lower bound and a column
    missing from it may still be produced."""
    result = annotate(
        f"""
        with ranked as (select * from {q('orders')} where status = 'new')
        select * from ranked
        """,
        {q("x"): {"status": "varchar", "amount": "decimal(10,2)"}},
        tmp_path,
        built_by=BUILT,
    )

    assert declaration_findings(result) == [
        ("declared-column-not-projected", "warning", "amount")
    ]
    assert "lower bound" in result.findings[0].message
    assert not result.has_errors


def test_a_computed_type_that_contradicts_the_declaration_is_an_error(
    tmp_path: Path,
) -> None:
    """Two written-down things disagree and one of them is wrong."""
    result = annotate(
        f"select order_id * 2 as order_id from {q('orders')}",
        {**ORDERS_TYPED, q("x"): {"order_id": "varchar(10)"}},
        tmp_path,
        built_by=BUILT,
    )

    assert declaration_findings(result) == [
        ("contradicted-declaration", "error", "order_id")
    ]
    assert "declared varchar(10)" in result.findings[0].message


def test_an_inferred_type_that_contradicts_the_declaration_is_a_warning(
    tmp_path: Path,
) -> None:
    """sqlr read a family off how an undeclared column is used, and the guess fought the
    declaration. An admission about sqlr's confidence, not an accusation about the SQL -
    the same split `conflicting-usage` makes."""
    result = annotate(
        f"select amount as amount from {q('orders')} where amount > 0",
        {q("orders"): {"amount": "UNKNOWN"}, q("x"): {"amount": "varchar(10)"}},
        tmp_path,
        built_by=BUILT,
    )

    assert declaration_findings(result) == [
        ("contradicted-declaration", "warning", "amount")
    ]
    assert "infers" in result.findings[0].message
    assert not result.has_errors


def test_a_type_the_declaration_merely_narrows_is_not_a_finding(
    tmp_path: Path,
) -> None:
    """Precision and scale are recorded, never checked - two types conflict only when their
    nearest common family is ANY."""
    result = annotate(
        f"select amount * 2 as amount from {q('orders')}",
        {**ORDERS_TYPED, q("x"): {"amount": "decimal(10,2)"}},
        tmp_path,
        built_by=BUILT,
    )
    assert declaration_findings(result) == []


def test_a_column_with_no_type_is_not_held_against_the_declaration(
    tmp_path: Path,
) -> None:
    """Three-valued to the end: nothing could say what this column is, so nothing is said
    about the declaration either."""
    result = annotate(
        f"select amount from {q('orders')}",
        {q("orders"): {"amount": "UNKNOWN"}, q("x"): {"amount": "varchar(10)"}},
        tmp_path,
        built_by=BUILT,
    )
    assert declaration_findings(result) == []


def test_a_declaration_with_no_columns_says_nothing(tmp_path: Path) -> None:
    """Naming a relation without listing its columns says nothing about them - the same
    reading a source table's empty declaration gets."""
    result = annotate(
        f"select order_id from {q('orders')}",
        {**ORDERS_TYPED, q("x"): {}},
        tmp_path,
        built_by=BUILT,
    )
    assert declaration_findings(result) == []


def test_a_model_declaration_never_types_the_columns_its_own_file_reads(
    tmp_path: Path,
) -> None:
    """The isolation rule. A declaration of what a file *produces* cannot describe the
    columns feeding it, so it stays out of the schema steps 4-6 work from: `amount` is
    inferred from usage here, and would have been `declared` had the declaration leaked in.
    """
    result = annotate(
        f"select amount from {q('orders')} where amount > 0",
        {q("orders"): {"amount": "UNKNOWN"}, q("x"): {"amount": "decimal(10,2)"}},
        tmp_path,
        built_by=BUILT,
    )

    statement = result.qualified.statement
    assert statement is not None
    assert statement.declared_types_per_relation[q("orders")]["amount"] == "UNKNOWN"
    assert inferred_types(result) == {(q("orders"), "amount"): "NUMERIC"}
    assert [column.provenance for column in result.scopes[-1].columns] == ["inferred"]


def test_a_file_that_reads_its_own_relation_reads_it_as_a_source(
    tmp_path: Path,
) -> None:
    """The incremental-model shape. Where the file reads the relation it builds, that
    declaration is a source declaration like any other and types the read."""
    result = annotate(
        f"select amount from {q('x')}",
        {q("x"): {"amount": "decimal(10,2)"}},
        tmp_path,
        built_by=BUILT,
    )

    assert declaration_findings(result) == []
    assert [column.provenance for column in result.scopes[-1].columns] == ["declared"]


def test_a_set_operation_is_held_against_the_declaration_by_its_left_arm(
    tmp_path: Path,
) -> None:
    """A union has no projection list of its own: the arms are matched by position and the
    left one supplies the names, so that is the arm the declaration is compared against."""
    result = annotate(
        f"""
        select id, 'a' as tag from {q('arm_a')}
        union all
        select id, 'b' as label from {q('arm_b')}
        """,
        {
            q("arm_a"): {"id": "bigint"},
            q("arm_b"): {"id": "bigint"},
            q("x"): {"id": "bigint", "tag": "varchar"},
        },
        tmp_path,
        built_by=BUILT,
    )
    assert declaration_findings(result) == []
