import datetime as dt
from typing import Any

import pyarrow as pa
import pytest

from sqlr.app import FixtureColumn, FixtureData, SqlType
from sqlr.errors import (
    ComparisonError,
    SortColumnError,
    SortOptionsError,
    TableConversionError,
)
from sqlr.schema_compare import Policy, SchemaCastError, SchemaPolicyError
from sqlr.table_compare import (
    RowCountMismatch,
    RowDiff,
    TableMismatchError,
    assert_equals,
    compare_tables,
)


def table(**columns: Any) -> pa.Table:
    return pa.table(columns)


def failure(*args: Any, **kwargs: Any) -> TableMismatchError:
    with pytest.raises(TableMismatchError) as excinfo:
        assert_equals(*args, **kwargs)
    return excinfo.value


# --- equal --------------------------------------------------------------------------------------


def test_equal_tables_pass() -> None:
    t = table(a=[1, 2], b=["x", "y"])
    assert_equals(t, t)

    comparison = compare_tables(t, t)
    assert comparison.ok
    assert comparison.row_count is None
    assert comparison.row_diffs == ()


def test_equal_empty_tables_pass() -> None:
    t = pa.table({"a": pa.array([], pa.int32())})
    assert compare_tables(t, t).ok


def test_accepts_fixture_data() -> None:
    fixture = FixtureData(data=[(1, "x")], columns=[FixtureColumn("a", SqlType.INT), FixtureColumn("b", SqlType.STRING)])
    assert_equals(fixture, fixture)
    assert_equals(fixture.to_arrow(), fixture)


def test_fixture_can_be_used_more_than_once() -> None:
    fixture = FixtureData(data=[(1,), (2,)], columns=[FixtureColumn("a", SqlType.INT)])
    assert fixture.to_arrow().num_rows == 2
    assert list(fixture.data) == [(1,), (2,)]
    assert_equals(fixture, fixture)


def test_fixture_that_failed_to_convert_fails_again_the_same_way() -> None:
    fixture = unconvertible_fixture()
    for _ in range(2):
        with pytest.raises(TableConversionError):
            compare_tables(fixture, fixture)


def test_types_are_promoted_before_comparing() -> None:
    actual = table(a=pa.array([1, 2], pa.int16()))
    expected = table(a=pa.array([1, 2], pa.int64()))
    assert_equals(actual, expected)


def test_extra_actual_columns_are_dropped_when_the_policy_allows() -> None:
    actual = table(a=[1], extra=["x"])
    expected = table(a=[1])
    comparison = compare_tables(actual, expected, policy=Policy(allow_extra_columns_in_result=True))
    assert comparison.ok
    assert comparison.actual is not None
    assert comparison.actual.column_names == ["a"]


# --- sorting ------------------------------------------------------------------------------------


def test_rows_in_a_different_order_pass_by_default() -> None:
    assert_equals(table(a=[2, 1, 3]), table(a=[1, 2, 3]))


def test_sort_by_ties_are_broken_by_the_remaining_columns_left_to_right() -> None:
    actual = table(num=[5, 1, 5], name=["five??", "one", "five"])
    expected = table(num=[1, 5, 5], name=["one", "five", "five??"])
    assert_equals(actual, expected, sort_by=["num"])


def test_sort_by_leads_and_the_other_columns_follow_in_table_order() -> None:
    # Sorted by b first, then a. Without sort_by it would be a first, then b.
    actual = table(a=[1, 2, 3], b=[9, 9, 0])
    comparison = compare_tables(actual, actual, sort_by=["b"])
    assert comparison.actual is not None
    assert comparison.actual.column("a").to_pylist() == [3, 1, 2]

    comparison = compare_tables(actual, actual)
    assert comparison.actual is not None
    assert comparison.actual.column("a").to_pylist() == [1, 2, 3]


def test_sort_by_matches_column_names_case_insensitively_by_default() -> None:
    t = table(num=[2, 1])
    comparison = compare_tables(t, t, sort_by=["NUM"])
    assert comparison.ok
    assert comparison.actual is not None
    assert comparison.actual.column("num").to_pylist() == [1, 2]


def test_list_columns_are_skipped_as_tiebreakers() -> None:
    t = table(a=[2, 1], items=[[1], [2]])
    comparison = compare_tables(t, t)
    assert comparison.ok
    assert comparison.actual is not None
    assert comparison.actual.column("a").to_pylist() == [1, 2]


def test_sort_results_false_compares_in_the_given_order() -> None:
    actual = table(a=[2, 1])
    expected = table(a=[1, 2])
    assert_equals(actual, expected)  # sorted by default
    comparison = compare_tables(actual, expected, sort_results=False)
    assert not comparison.ok
    assert comparison.row_diffs == (RowDiff(0, ("a",)), RowDiff(1, ("a",)))


def test_sort_results_false_passes_when_the_order_matches() -> None:
    t = table(a=[2, 1])
    assert_equals(t, t, sort_results=False)


# --- errors -------------------------------------------------------------------------------------


def test_sort_by_with_sort_results_false_is_an_error() -> None:
    t = table(a=[1])
    with pytest.raises(SortOptionsError) as excinfo:
        compare_tables(t, t, sort_by=["a"], sort_results=False)
    assert excinfo.value.sort_by == ("a",)


def test_empty_sort_by_with_sort_results_false_is_still_an_error() -> None:
    t = table(a=[1])
    with pytest.raises(SortOptionsError):
        compare_tables(t, t, sort_by=[], sort_results=False)


def test_sort_options_are_checked_before_anything_else() -> None:
    with pytest.raises(SortOptionsError):
        compare_tables(object(), object(), sort_by=["a"], sort_results=False)  # pyright: ignore[reportArgumentType]


def unconvertible_fixture() -> FixtureData:
    return FixtureData(data=[("not a number",)], columns=[FixtureColumn("a", SqlType.INT)])


def test_fixture_that_cannot_convert_is_an_error() -> None:
    good = table(a=pa.array([1], pa.int32()))

    with pytest.raises(TableConversionError) as excinfo:
        compare_tables(good, unconvertible_fixture())
    assert excinfo.value.side == "expected"
    assert isinstance(excinfo.value.cause, Exception)

    with pytest.raises(TableConversionError) as excinfo:
        compare_tables(unconvertible_fixture(), good)
    assert excinfo.value.side == "actual"


def test_fixture_with_the_wrong_number_of_columns_is_an_error() -> None:
    bad = FixtureData(data=[(1, 2)], columns=[FixtureColumn("a", SqlType.INT)])
    with pytest.raises(TableConversionError):
        compare_tables(bad, bad)


def test_input_that_is_not_a_table_is_an_error() -> None:
    with pytest.raises(TableConversionError) as excinfo:
        compare_tables([1, 2], table(a=[1]))  # pyright: ignore[reportArgumentType]
    assert excinfo.value.side == "actual"
    assert isinstance(excinfo.value.cause, TypeError)


def test_missing_sort_by_column_is_an_error() -> None:
    t = table(a=[1], b=[2])
    with pytest.raises(SortColumnError) as excinfo:
        compare_tables(t, t, sort_by=["nope"])
    assert excinfo.value.column == "nope"
    assert excinfo.value.reason == "missing"
    assert excinfo.value.available == ("a", "b")


def test_sort_by_column_case_matters_when_the_policy_is_case_sensitive() -> None:
    t = table(num=[1])
    with pytest.raises(SortColumnError) as excinfo:
        compare_tables(t, t, sort_by=["NUM"], policy=Policy(case_sensitive_column_names=True))
    assert excinfo.value.reason == "missing"


def test_unsortable_sort_by_column_is_an_error() -> None:
    t = table(items=[[1], [2]])
    with pytest.raises(SortColumnError) as excinfo:
        compare_tables(t, t, sort_by=["items"])
    assert excinfo.value.reason == "unsortable"
    assert excinfo.value.type == pa.list_(pa.int64())


def test_sort_by_error_comes_before_a_schema_failure() -> None:
    with pytest.raises(SortColumnError):
        compare_tables(table(a=[1]), table(b=[1]), sort_by=["nope"])


def test_a_cast_that_does_not_fit_is_an_error() -> None:
    far_future = dt.datetime(3000, 1, 1)
    actual = table(when=pa.array([far_future], pa.timestamp("us")))
    expected = table(when=pa.array([dt.datetime(2000, 1, 1)], pa.timestamp("ns")))
    with pytest.raises(SchemaCastError) as excinfo:
        compare_tables(actual, expected)
    assert excinfo.value.column == "when"
    assert excinfo.value.side == "actual"


def test_all_errors_share_one_base_class() -> None:
    for error in (SortOptionsError, TableConversionError, SortColumnError, SchemaCastError, SchemaPolicyError):
        assert issubclass(error, ComparisonError)
    assert not issubclass(TableMismatchError, ComparisonError)
    assert issubclass(TableMismatchError, AssertionError)


# --- failure: schema ----------------------------------------------------------------------------


def test_missing_column_is_a_schema_failure() -> None:
    actual = table(a=[1])
    expected = table(a=[1], b=[2])
    error = failure(actual, expected)
    comparison = error.comparison
    assert not comparison.schema.ok
    assert comparison.actual is None and comparison.expected is None
    assert comparison.row_count is None and comparison.row_diffs is None
    assert not comparison.ok
    assert "'b'" in str(error) and "missing column" in str(error)


def test_incompatible_types_are_a_schema_failure() -> None:
    actual = table(a=pa.array([1], pa.int32()))
    expected = table(a=["1"])
    comparison = failure(actual, expected).comparison
    assert not comparison.schema.ok
    assert comparison.schema.columns[0].problem is not None
    assert comparison.row_count is None and comparison.row_diffs is None


def test_compare_tables_returns_failures_instead_of_raising() -> None:
    comparison = compare_tables(table(a=[1]), table(b=[1]))
    assert not comparison.ok


# --- failure: row count -------------------------------------------------------------------------


def test_row_count_mismatch_carries_both_counts_and_both_tables() -> None:
    actual = table(a=[1, 2, 3], b=["x", "y", "z"])
    expected = table(a=[1, 2], b=["x", "y"])
    error = failure(actual, expected)
    comparison = error.comparison

    assert comparison.row_count == RowCountMismatch(actual=3, expected=2)
    assert comparison.row_diffs is None
    assert comparison.actual is not None and comparison.expected is not None
    assert comparison.actual.num_rows == 3 and comparison.expected.num_rows == 2
    assert comparison.actual.column_names == comparison.expected.column_names == ["a", "b"]
    assert "expected 2 rows, actual 3 rows" in str(error)


def test_empty_actual_against_rows_is_a_row_count_failure() -> None:
    actual = pa.table({"a": pa.array([], pa.int64())})
    comparison = failure(actual, table(a=[1])).comparison
    assert comparison.row_count == RowCountMismatch(actual=0, expected=1)


# --- failure: values ----------------------------------------------------------------------------


def test_value_mismatch_names_only_the_differing_columns() -> None:
    actual = table(c1=[123, 7], c2=["bye", "same"], c3=[1004, 8])
    expected = table(c1=[123, 7], c2=["hello", "same"], c3=[1004, 8])
    error = failure(actual, expected)
    comparison = error.comparison

    assert comparison.row_count is None
    # Sorted by c1, so the (123, ...) row is second.
    assert comparison.row_diffs == (RowDiff(row=1, columns=("c2",)),)
    assert comparison.actual is not None and comparison.expected is not None
    assert comparison.actual.slice(1, 1).to_pylist() == [{"c1": 123, "c2": "bye", "c3": 1004}]
    assert comparison.expected.slice(1, 1).to_pylist() == [{"c1": 123, "c2": "hello", "c3": 1004}]
    assert "row 1, column 'c2': expected 'hello', actual 'bye'" in str(error)


def test_value_mismatch_reports_every_differing_row_and_column() -> None:
    actual = table(a=[1, 2, 3], b=["x", "y", "z"])
    expected = table(a=[1, 9, 3], b=["x", "Y", "Z"])
    comparison = failure(actual, expected, sort_results=False).comparison
    assert comparison.row_diffs == (RowDiff(1, ("a", "b")), RowDiff(2, ("b",)))


def test_row_position_is_the_position_after_sorting() -> None:
    actual = table(a=[3, 1, 2], b=["c", "a", "WRONG"])
    expected = table(a=[1, 2, 3], b=["a", "b", "c"])
    comparison = failure(actual, expected).comparison
    assert comparison.row_diffs == (RowDiff(1, ("b",)),)


def test_null_differs_from_a_value_but_equals_null() -> None:
    assert_equals(table(a=[1, None]), table(a=[1, None]))
    comparison = failure(table(a=[1, None]), table(a=[1, 2]), sort_results=False).comparison
    assert comparison.row_diffs == (RowDiff(1, ("a",)),)
    comparison = failure(table(a=[1, 2]), table(a=[1, None]), sort_results=False).comparison
    assert comparison.row_diffs == (RowDiff(1, ("a",)),)


def test_nan_equals_nan() -> None:
    nan = float("nan")
    assert_equals(table(a=[nan, 1.0]), table(a=[nan, 1.0]))
    comparison = failure(table(a=[nan, 1.0]), table(a=[1.0, 1.0]), sort_results=False).comparison
    assert comparison.row_diffs == (RowDiff(0, ("a",)),)


def test_nan_in_one_column_does_not_hide_a_difference_in_another() -> None:
    nan = float("nan")
    comparison = failure(table(a=[nan], b=[1]), table(a=[nan], b=[2])).comparison
    assert comparison.row_diffs == (RowDiff(0, ("b",)),)


def test_nested_columns_are_compared() -> None:
    same = table(a=[1, 2], items=[[1, 2], [3]], info=[{"k": 1}, {"k": 2}])
    assert_equals(same, same)

    other = table(a=[1, 2], items=[[1, 2], [4]], info=[{"k": 1}, {"k": 3}])
    comparison = failure(same, other).comparison
    assert comparison.row_diffs == (RowDiff(1, ("items", "info")),)


def test_null_typed_columns_are_compared() -> None:
    t = table(a=pa.array([None, None], pa.null()))
    assert_equals(t, t)


def test_summary_limits_the_number_of_rows_listed() -> None:
    actual = table(a=list(range(15)))
    expected = table(a=list(range(100, 115)))
    message = str(failure(actual, expected))
    assert message.startswith("15 of 15 rows differ:")
    assert "... and 5 more rows" in message
