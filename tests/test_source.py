"""Spans are the foundation everything else points with, so they are tested by
round-tripping: a span is right when slicing the source with it returns the text the
span was supposed to cover.
"""

import sqlglot
from sqlglot import exp

from sqlr.source import NO_POSITIONS, Positions, SourceDoc


def _column(sql: str, name: str) -> exp.Column:
    parsed = sqlglot.parse_one(sql)
    return next(c for c in parsed.find_all(exp.Column) if c.name == name)


# ---- offsets to line/column ---------------------------------------------------------


def test_line_col_is_zero_based() -> None:
    positions = Positions("select a\nfrom t")

    assert positions.line_col(0) == (0, 0)
    assert positions.line_col(7) == (0, 7)
    assert positions.line_col(9) == (1, 0)


def test_line_col_clamps_out_of_range_offsets() -> None:
    positions = Positions("abc")

    assert positions.line_col(-5) == (0, 0)
    assert positions.line_col(999) == (0, 3)


def test_empty_text_has_one_line() -> None:
    assert Positions("").line_col(0) == (0, 0)


# ---- spans over expressions ---------------------------------------------------------


def test_span_of_a_column_slices_back_to_the_reference() -> None:
    sql = "select a\nfrom t\nwhere amount > 20"
    positions = Positions(sql)
    doc = SourceDoc(text=sql)

    span = positions.span_of(_column(sql, "amount"))

    assert doc.slice(span) == "amount"
    assert span is not None and (span.start_line, span.start_col) == (2, 6)


def test_span_of_a_parent_covers_the_whole_predicate() -> None:
    sql = "select a from t where amount > 20"
    positions = Positions(sql)
    doc = SourceDoc(text=sql)

    column = _column(sql, "amount")
    assert doc.slice(positions.span_of(column.parent)) == "amount > 20"


def test_span_end_is_exclusive() -> None:
    sql = "select id from t"
    positions = Positions(sql)

    span = positions.span_of(_column(sql, "id"))

    assert span is not None
    assert (span.start, span.end) == (7, 9)
    assert sql[span.start : span.end] == "id"


def test_span_covering_takes_the_hull() -> None:
    sql = "select a, b from t"
    positions = Positions(sql)
    doc = SourceDoc(text=sql)
    parsed = sqlglot.parse_one(sql)
    columns = list(parsed.find_all(exp.Column))

    assert doc.slice(positions.span_covering(columns)) == "a, b"


def test_span_is_none_when_nothing_carries_a_position() -> None:
    # sqlglot positions the tokens it parsed; anything built by hand has no location,
    # and callers have to cope rather than crash.
    assert Positions("select 1").span_of(exp.column("invented")) is None


def test_span_of_none_is_none() -> None:
    assert Positions("select 1").span_of(None) is None


def test_no_positions_yields_nothing() -> None:
    assert NO_POSITIONS.span_of(_column("select a from t", "a")) is None


# ---- bracket balancing ---------------------------------------------------------------


def test_hull_is_extended_over_a_trailing_close_paren() -> None:
    # The `)` is punctuation and carries no position, so the raw hull would stop at the
    # final quote and the underline would look broken.
    sql = "select a from t where s in ('x', 'y')"
    positions = Positions(sql)
    doc = SourceDoc(text=sql)

    column = _column(sql, "s")
    assert doc.slice(positions.span_of(column.parent)) == "s in ('x', 'y')"


def test_a_paren_inside_a_string_literal_does_not_swallow_the_line() -> None:
    sql = "select a from t where c like '%(x%' and d > 1"
    positions = Positions(sql)
    doc = SourceDoc(text=sql)

    column = _column(sql, "c")
    assert doc.slice(positions.span_of(column.parent)) == "c like '%(x%'"


# ---- documents -----------------------------------------------------------------------


def test_slice_of_none_is_none() -> None:
    assert SourceDoc(text="abc").slice(None) is None


def test_lines_of_returns_every_touched_line() -> None:
    sql = "select\n  a,\n  b\nfrom t"
    positions = Positions(sql)
    doc = SourceDoc(text=sql)
    parsed = sqlglot.parse_one(sql)

    span = positions.span_covering(list(parsed.find_all(exp.Column)))

    assert doc.lines_of(span) == ["  a,", "  b"]
