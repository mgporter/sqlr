"""Findings, and the spans that point at the SQL they came from.

Every fixture here is an inline SQL string, per `tests/README.md`.
"""

import pytest
import sqlglot

from declared_helpers import declarations, q

from sqlr.sql_analysis2.resolve import resolve_columns_to_source_tables
from sqlr.sql_analysis2.reporting import (
    ColumnFinding,
    findings_for_ambiguous_columns,
    findings_for_columns_declared_as_scalar_but_read_as_structured,
    findings_for_columns_read_with_unsupported_dot_notation,
    findings_for_columns_without_a_source,
    findings_for_unresolvable_columns,
    print_findings,
)
from sqlr.sql_analysis2.sourcedoc import Positions, SourceSpan

DIALECT = "duckdb"


def text_at(sql: str, span: SourceSpan | None) -> str:
    """The SQL a finding points at. A finding without a span is a test failure here."""
    assert span is not None
    return sql[span.start : span.end]


# ------------------------------------------------------------ unresolvable columns
def test_an_unresolvable_column_points_at_its_own_name() -> None:
    sql = (
        "select t.id, nonsense\n"
        "from mydatabase.myschema.test t\n"
        "join mydatabase.myschema.other o on t.id = o.id"
    )
    resolved, _ = resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read=DIALECT), DIALECT, declarations()
    )
    (finding,) = findings_for_unresolvable_columns(
        resolved.unresolvable_columns, Positions(sql)
    )
    assert finding.code == "unresolvable-column"
    assert text_at(sql, finding.span) == "nonsense"
    assert finding.span is not None and str(finding.span) == "1:14"


def unresolvable_findings(
    sql: str,
    declared: dict[str, dict[str, str]] | None = None,
    dialect: str = DIALECT,
) -> list[ColumnFinding]:
    resolved, _ = resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read=dialect), dialect, declarations(declared)
    )
    return findings_for_unresolvable_columns(
        resolved.unresolvable_columns, Positions(sql)
    )


def test_a_complete_declaration_names_the_yml_entry_and_the_two_ways_to_fix_it() -> None:
    """The case the fix sentence exists for.

    A table declared without one of the columns the SQL reads is not a typo the reader can
    spot in the SQL - nothing there is wrong. The error only makes sense next to the yml
    entry that made the omission binding, so the message has to name it.
    """
    (finding,) = unresolvable_findings(
        "select person_id, updated_at from mydatabase.myschema.raw_address",
        {q("raw_address"): {"person_id": "varchar(20)"}},
    )
    assert finding.message == (
        "found undeclared column 'updated_at' in table "
        "'mydatabase.myschema.raw_address'. Declare it on "
        "'mydatabase.myschema.raw_address' at schema.yml, or set "
        "'declaration_is_partial' to 'true' there."
    )


def test_a_cte_over_a_declared_table_points_at_the_table_not_the_cte() -> None:
    """`street` fails against `ranked`, but `ranked` is not the thing anyone can fix.

    The CTE is closed only because the table under its `*` is, and a reader told to fix the
    CTE has been sent somewhere with no answer in it. The fix has to name the table.
    """
    sql = (
        "with ranked as (select * from mydatabase.myschema.raw_address)\n"
        "select street from ranked"
    )
    (finding,) = unresolvable_findings(
        sql, {q("raw_address"): {"person_id": "varchar(20)"}}
    )
    assert finding.message == (
        "found undeclared column 'street' in CTE 'ranked'. Declare it on "
        "'mydatabase.myschema.raw_address' at schema.yml, or set "
        "'declaration_is_partial' to 'true' there."
    )


def test_a_cte_that_writes_its_own_columns_is_offered_no_yml_fix() -> None:
    """Closed because the SQL enumerates it, so no declaration is involved and the name is
    simply a typo. A `declaration_is_partial` hint here points at nothing."""
    sql = (
        "with src as (select id, name from mydatabase.myschema.test)\n"
        "select src.nonsense from src"
    )
    (finding,) = unresolvable_findings(sql)
    assert finding.message == "found undeclared column 'nonsense' in CTE 'src'."


def test_a_name_no_undeclared_table_can_be_credited_with_says_so() -> None:
    """Neither table declares anything, so nothing places `mystery` on one of them.

    Distinct from every other unresolvable column in that the SQL is not wrong, so the fix
    is a qualifier rather than a correction - and there is no yml entry to point at.
    """
    sql = (
        "select a.x, mystery\n"
        "from mydatabase.myschema.test a\n"
        "join mydatabase.myschema.other b on a.id = b.id"
    )
    (finding,) = unresolvable_findings(sql)
    assert finding.message == (
        "column 'mystery' has no source alias and could come from "
        "'mydatabase.myschema.test' and 'mydatabase.myschema.other'. Qualify it, or "
        "declare the columns of the table that owns it."
    )
    assert "declaration_is_partial" not in finding.message


def test_a_qualifier_naming_nothing_is_reported_as_a_bad_alias() -> None:
    """No yml would change this, so no fix is offered."""
    sql = (
        "select bogus.id from mydatabase.myschema.test t "
        "join mydatabase.myschema.other o on t.id = o.id"
    )
    (finding,) = unresolvable_findings(sql)
    assert finding.message == (
        "column 'id' is qualified with 'bogus', which matches no relation in this statement"
    )


def test_a_name_every_relation_rules_out_lists_none_of_their_columns() -> None:
    """Several closed relations could each have been meant, and naming what each one
    declares means printing most of the schema to say one thing. The fix is generic for the
    same reason: there is no single entry to send the reader to."""
    (finding,) = unresolvable_findings(
        "select typo from mydatabase.myschema.test t "
        "join mydatabase.myschema.other o on t.id = o.id",
        {q("test"): {"id": "int"}, q("other"): {"id": "int"}},
    )
    assert finding.message == (
        "column 'typo' is projected by none of the relations in scope. Declare it on a "
        "source, or set 'declaration_is_partial' to 'true' for one of the relations."
    )
    assert "'id'" not in finding.message


def test_a_message_quotes_the_name_the_user_typed_not_the_normalised_one() -> None:
    """Snowflake folds unquoted identifiers up, so the tree holds `UPDATED_AT`.

    A message built from the tree would put that beside the lower-case names the yml uses,
    and neither spelling would be the one the reader can search their file for.
    """
    (finding,) = unresolvable_findings(
        "select updated_at from mydatabase.myschema.raw_address",
        {q("raw_address"): {"person_id": "varchar(20)"}},
        dialect="snowflake",
    )
    assert "'updated_at'" in finding.message
    assert "UPDATED_AT" not in finding.message


# ------------------------------------------------------------------- dot access
def test_a_dotted_read_reports_the_name_and_the_whole_access() -> None:
    """Both spans, because an editor underlines the name but a message quotes the read.

    The offsets survive `qualify` even though it rebuilds the column: the identifier
    nodes it reuses are the ones the parser positioned.
    """
    sql = "select mistyped.col1 as col1\nfrom mydatabase.myschema.test"
    resolved, _ = resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read="postgres"), "postgres", declarations()
    )
    (finding,) = findings_for_columns_read_with_unsupported_dot_notation(
        resolved.columns_read_with_unsupported_dot_notation, Positions(sql), "postgres"
    )
    assert finding.code == "dot-access-unsupported"
    assert text_at(sql, finding.span) == "mistyped"
    assert text_at(sql, finding.access_span) == "mistyped.col1"
    assert "postgres" in finding.message


def test_a_nested_read_spans_only_its_first_level() -> None:
    """`a.b.c` is recorded as the read `a.b`, matching what `structured_access` models."""
    sql = "select mistyped.col2.jsonfield from mydatabase.myschema.test"
    resolved, _ = resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read="postgres"), "postgres", declarations()
    )
    (finding,) = findings_for_columns_read_with_unsupported_dot_notation(
        resolved.columns_read_with_unsupported_dot_notation, Positions(sql), "postgres"
    )
    assert text_at(sql, finding.access_span) == "mistyped.col2"


def test_one_finding_per_read_site() -> None:
    sql = (
        "select\n"
        "  mistyped.col1 as col1,\n"
        "  mistyped['col3'] as col3,\n"
        "  mistyped.col4 as col4\n"
        "from mydatabase.myschema.test"
    )
    resolved, _ = resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read="postgres"), "postgres", declarations()
    )
    findings = findings_for_columns_read_with_unsupported_dot_notation(
        resolved.columns_read_with_unsupported_dot_notation, Positions(sql), "postgres"
    )
    # The bracket read is legal in Postgres, so only the two dotted ones are findings.
    assert [str(finding.span) for finding in findings] == ["2:3", "4:3"]


# --------------------------------------------- columns written without a source
JOINED_TO_A_CTE = (
    "with src as (select test_id, name from mydatabase.myschema.other)\n"
    "select {selection}\n"
    "from mydatabase.myschema.test\n"
    "inner join src on test.id = src.test_id"
)


def test_an_ambiguous_column_names_every_candidate() -> None:
    sql = JOINED_TO_A_CTE.format(selection="name")
    resolved, _ = resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read=DIALECT),
        DIALECT,
        declarations({q("test"): {"id": "int", "name": "varchar(20)"}}),
    )
    (finding,) = findings_for_ambiguous_columns(
        resolved.ambiguous_columns, Positions(sql)
    )
    assert finding.code == "ambiguous-column"
    assert finding.severity == "error"
    assert text_at(sql, finding.span) == "name"
    assert finding.message == (
        "column 'name' is ambiguous: 'src' and 'test' both project it; "
        "qualify it with a source alias"
    )


def test_a_guessed_column_names_what_it_was_read_from() -> None:
    """Naming the source that was *not* ruled out is what makes the warning actionable:
    it is the difference between "add a qualifier" and "declare that table"."""
    sql = JOINED_TO_A_CTE.format(selection="name")
    resolved, _ = resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read=DIALECT), DIALECT, declarations({})
    )
    (finding,) = findings_for_columns_without_a_source(
        resolved.guessed_columns, Positions(sql)
    )
    assert finding.code == "column-without-source"
    assert finding.severity == "warning"
    assert text_at(sql, finding.span) == "name"
    assert finding.message == (
        "column 'name' has no source alias and 'test' does not declare its columns; "
        "reading it from 'src'"
    )


def test_the_printer_takes_its_prefix_from_the_severity(
    capsys: pytest.CaptureFixture[str],
) -> None:
    findings = [
        ColumnFinding(code="unresolvable-column", column_name="a", message="gone"),
        ColumnFinding(
            code="column-without-source",
            column_name="b",
            message="guessed",
            severity="warning",
        ),
    ]
    print_findings(findings, "models/x.sql")
    captured = capsys.readouterr()
    assert captured.out == (
        "error: models/x.sql: gone\nwarning: models/x.sql: guessed\n"
    )


# --------------------------------------------------- structured column declarations
def test_a_scalar_declaration_contradicts_a_structured_read() -> None:
    sql = "select mistyped.col1, mistyped['col3'] from mydatabase.myschema.test"
    resolved, _ = resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read=DIALECT), DIALECT, declarations()
    )
    findings = findings_for_columns_declared_as_scalar_but_read_as_structured(
        {q("test"): {"mistyped": "varchar(50)"}}, resolved.columns_per_relation, Positions(sql)
    )
    assert [finding.code for finding in findings] == [
        "structured-column-declared-scalar"
    ] * 2
    assert [text_at(sql, finding.access_span) for finding in findings] == [
        "mistyped.col1",
        "mistyped['col3']",
    ]
    assert "varchar(50)" in findings[0].message


def test_a_structured_declaration_is_not_contradicted() -> None:
    """`json`, `struct(...)`, `map` and `variant` all sit off the scalar lattice."""
    sql = "select mistyped.col1 from mydatabase.myschema.test"
    resolved, _ = resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read=DIALECT), DIALECT, declarations()
    )
    for written in ("json", "struct(col1 int)", "map(varchar, int)", "variant"):
        assert (
            findings_for_columns_declared_as_scalar_but_read_as_structured(
                {q("test"): {"mistyped": written}}, resolved.columns_per_relation, Positions(sql)
            )
            == []
        )


def test_an_undeclared_column_is_not_contradicted() -> None:
    sql = "select mistyped.col1 from mydatabase.myschema.test"
    resolved, _ = resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read=DIALECT), DIALECT, declarations()
    )
    assert (
        findings_for_columns_declared_as_scalar_but_read_as_structured(
            {q("test"): {"id": "int"}}, resolved.columns_per_relation, Positions(sql)
        )
        == []
    )


def test_an_integer_subscript_does_not_contradict_a_string_declaration() -> None:
    """DuckDB subscripts strings, so `titles[1]` says nothing about `varchar`."""
    sql = "select titles[1] from mydatabase.myschema.test"
    resolved, _ = resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read=DIALECT), DIALECT, declarations()
    )
    assert (
        findings_for_columns_declared_as_scalar_but_read_as_structured(
            {q("test"): {"titles": "varchar"}}, resolved.columns_per_relation, Positions(sql)
        )
        == []
    )
