"""Findings, and the spans that point at the SQL they came from.

Every fixture here is an inline SQL string, per `tests/README.md`.
"""

import sqlglot

from sqlr.sql_analysis2 import resolve_columns_to_source_tables
from sqlr.sql_analysis2.reporting import (
    findings_for_columns_declared_as_scalar_but_read_as_structured,
    findings_for_columns_read_with_unsupported_dot_notation,
    findings_for_unresolvable_columns,
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
    resolved = resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read=DIALECT), DIALECT
    )
    (finding,) = findings_for_unresolvable_columns(
        resolved.unresolvable_columns, Positions(sql)
    )
    assert finding.code == "unresolvable-column"
    assert text_at(sql, finding.span) == "nonsense"
    assert finding.span is not None and str(finding.span) == "1:14"


# ------------------------------------------------------------------- dot access
def test_a_dotted_read_reports_the_name_and_the_whole_access() -> None:
    """Both spans, because an editor underlines the name but a message quotes the read.

    The offsets survive `qualify` even though it rebuilds the column: the identifier
    nodes it reuses are the ones the parser positioned.
    """
    sql = "select mistyped.col1 as col1\nfrom mydatabase.myschema.test"
    resolved = resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read="postgres"), "postgres"
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
    resolved = resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read="postgres"), "postgres"
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
    resolved = resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read="postgres"), "postgres"
    )
    findings = findings_for_columns_read_with_unsupported_dot_notation(
        resolved.columns_read_with_unsupported_dot_notation, Positions(sql), "postgres"
    )
    # The bracket read is legal in Postgres, so only the two dotted ones are findings.
    assert [str(finding.span) for finding in findings] == ["2:3", "4:3"]


# --------------------------------------------------- structured column declarations
def test_a_scalar_declaration_contradicts_a_structured_read() -> None:
    sql = "select mistyped.col1, mistyped['col3'] from mydatabase.myschema.test"
    resolved = resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read=DIALECT), DIALECT
    )
    findings = findings_for_columns_declared_as_scalar_but_read_as_structured(
        {"test": {"mistyped": "varchar(50)"}}, resolved.columns_per_table, Positions(sql)
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
    resolved = resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read=DIALECT), DIALECT
    )
    for written in ("json", "struct(col1 int)", "map(varchar, int)", "variant"):
        assert (
            findings_for_columns_declared_as_scalar_but_read_as_structured(
                {"test": {"mistyped": written}}, resolved.columns_per_table, Positions(sql)
            )
            == []
        )


def test_an_undeclared_column_is_not_contradicted() -> None:
    sql = "select mistyped.col1 from mydatabase.myschema.test"
    resolved = resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read=DIALECT), DIALECT
    )
    assert (
        findings_for_columns_declared_as_scalar_but_read_as_structured(
            {"test": {"id": "int"}}, resolved.columns_per_table, Positions(sql)
        )
        == []
    )


def test_an_integer_subscript_does_not_contradict_a_string_declaration() -> None:
    """DuckDB subscripts strings, so `titles[1]` says nothing about `varchar`."""
    sql = "select titles[1] from mydatabase.myschema.test"
    resolved = resolve_columns_to_source_tables(
        sqlglot.parse_one(sql, read=DIALECT), DIALECT
    )
    assert (
        findings_for_columns_declared_as_scalar_but_read_as_structured(
            {"test": {"titles": "varchar"}}, resolved.columns_per_table, Positions(sql)
        )
        == []
    )
