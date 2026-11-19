"""Compare the data of an actual query result against an expected one.

`compare_tables` runs the comparison. There are two kinds of bad outcome:

* Error: the comparison could not run (see `sqlr.errors`, plus `SchemaCastError` from schema_compare).
  `compare_tables` raises these.
* Failure: the comparison ran and the tables differ. `compare_tables` returns these as a `TableComparison`
  holding everything a renderer needs. `assert_equals` raises them as `TableMismatchError`, which carries
  the same `TableComparison`.

Stages, each stopping the comparison if it fails:

    1. convert both inputs to arrow tables                      error: TableConversionError
    2. check the sort_by columns                                error: SortOptionsError, SortColumnError
    3. compare the schemas                                      failure: schema
    4. cast both tables to the common schema                    error: SchemaCastError
    5. sort both tables (sort_results=True)
    6. compare row counts                                       failure: row_count
    7. compare values row by row                                failure: row_diffs

NaN equals NaN and null equals null.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Any, Protocol, Sequence, cast, runtime_checkable

import pyarrow as pa
import pyarrow.compute as pc

from sqlr.errors import Side, SortColumnError, SortOptionsError, TableConversionError
from sqlr.schema_compare import Policy, SchemaComparison, cast_to_common, compare_schemas

__all__ = [
    "ArrowConvertible",
    "RowCountMismatch",
    "RowDiff",
    "TableComparison",
    "TableMismatchError",
    "compare_tables",
    "assert_equals",
]


@runtime_checkable
class ArrowConvertible(Protocol):
    """Anything with a `to_arrow()` method, such as `FixtureData`."""

    def to_arrow(self) -> pa.Table: ...


@dataclass(frozen=True)
class RowCountMismatch:
    actual: int
    expected: int


@dataclass(frozen=True)
class RowDiff:
    row: int  # 0-based position in the compared tables, which are sorted when sort_results=True
    columns: tuple[str, ...]  # the columns whose values differ in this row (the ones to highlight)


@dataclass(frozen=True)
class TableComparison:
    """The outcome of a comparison that ran to completion or stopped at a failure.

    A field that is None was not reached because an earlier stage failed.
    """

    schema: SchemaComparison
    actual: pa.Table | None  # cast to the common schema and sorted; None if the schemas did not match
    expected: pa.Table | None
    row_count: RowCountMismatch | None  # set only when the row counts differ
    row_diffs: tuple[RowDiff, ...] | None  # () when every value matches; None when not compared

    @property
    def ok(self) -> bool:
        return self.schema.ok and self.row_count is None and not self.row_diffs


class TableMismatchError(AssertionError):
    """The comparison ran and the tables differ. `comparison` holds the details."""

    def __init__(self, comparison: TableComparison) -> None:
        super().__init__(_summary(comparison))
        self.comparison = comparison


def compare_tables(
    actual: pa.Table | ArrowConvertible,
    expected: pa.Table | ArrowConvertible,
    sort_by: Sequence[str] | None = None,
    policy: Policy = Policy(),
    sort_results: bool = True,
) -> TableComparison:
    """Compare actual against expected. Raises a ComparisonError if the comparison cannot run.

    sort_results=True sorts both tables before comparing: first by the sort_by columns (ascending), then by
    every remaining column left to right. Columns arrow cannot sort (lists, maps) are skipped as tiebreakers.
    sort_results=False compares rows in the order given, which tests the query's own ordering; it cannot be
    combined with sort_by.
    """
    if not sort_results and sort_by is not None:
        raise SortOptionsError(sort_by)

    actual_table = _to_table(actual, "actual")
    expected_table = _to_table(expected, "expected")

    if sort_by is not None:
        _resolve_sort_by(sort_by, expected_table.schema, policy)

    schema = compare_schemas(actual_table.schema, expected_table.schema, policy)
    if not schema.ok:
        return TableComparison(schema, None, None, None, None)

    actual_table, expected_table = cast_to_common(actual_table, expected_table, schema)

    if sort_results:
        assert schema.common_schema is not None  # schema.ok
        keys = _sort_keys(sort_by or (), schema.common_schema, policy)
        actual_table = _sort(actual_table, keys)
        expected_table = _sort(expected_table, keys)

    if actual_table.num_rows != expected_table.num_rows:
        return TableComparison(
            schema, actual_table, expected_table, RowCountMismatch(actual_table.num_rows, expected_table.num_rows), None
        )

    return TableComparison(schema, actual_table, expected_table, None, _row_diffs(actual_table, expected_table))


def assert_equals(
    actual: pa.Table | ArrowConvertible,
    expected: pa.Table | ArrowConvertible,
    sort_by: Sequence[str] | None = None,
    policy: Policy = Policy(),
    sort_results: bool = True,
) -> None:
    """Assert actual holds the same data as expected. See compare_tables for the sorting options.

    Raises TableMismatchError if the tables differ, and a ComparisonError if the comparison cannot run.
    """
    __tracebackhide__ = True  # pytest reports the failure at the caller's line
    comparison = compare_tables(actual, expected, sort_by, policy, sort_results)
    if not comparison.ok:
        raise TableMismatchError(comparison)


# --- inputs -------------------------------------------------------------------------------------


def _to_table(value: object, side: Side) -> pa.Table:
    if isinstance(value, pa.Table):
        return value
    if isinstance(value, ArrowConvertible):
        try:
            return value.to_arrow()
        except Exception as e:
            raise TableConversionError(side, e) from e
    raise TableConversionError(side, TypeError(f"expected a pyarrow.Table or an object with to_arrow(), got {type(value).__name__}"))


# --- sorting ------------------------------------------------------------------------------------


def _sortable(type_: pa.DataType) -> bool:
    """Arrow can sort scalars and structs of sortable fields, but not lists, maps or unions."""
    if pa.types.is_struct(type_):
        return all(_sortable(field.type) for field in type_)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
    return not pa.types.is_nested(type_)


def _resolve_sort_by(sort_by: Sequence[str], schema: pa.Schema, policy: Policy) -> list[str]:
    """The sort_by columns in the schema's spelling, without repeats. Raises SortColumnError."""
    fold = _exact if policy.case_sensitive_column_names else str.casefold
    by_key: dict[str, tuple[str, pa.DataType]] = {}
    for name, type_ in zip(schema.names, schema.types):
        by_key.setdefault(fold(name), (name, type_))

    resolved: list[str] = []
    for column in sort_by:
        match = by_key.get(fold(column))
        if match is None:
            raise SortColumnError(column, "missing", schema.names)
        name, type_ = match
        if not _sortable(type_):
            raise SortColumnError(column, "unsortable", schema.names, type_)
        if name not in resolved:
            resolved.append(name)
    return resolved


def _sort_keys(sort_by: Sequence[str], schema: pa.Schema, policy: Policy) -> list[str]:
    """The sort_by columns, then every other sortable column left to right."""
    primary = _resolve_sort_by(sort_by, schema, policy)
    rest = [name for name, type_ in zip(schema.names, schema.types) if name not in primary and _sortable(type_)]
    return primary + rest


def _sort(table: pa.Table, keys: Sequence[str]) -> pa.Table:
    if not keys:
        return table
    return table.take(pc.sort_indices(table, sort_keys=[(key, "ascending") for key in keys]))  # pyright: ignore[reportUnknownMemberType]


def _exact(name: str) -> str:
    return name


_FALSE = pa.scalar(False)


# --- values -------------------------------------------------------------------------------------


def _row_diffs(actual: pa.Table, expected: pa.Table) -> tuple[RowDiff, ...]:
    """The rows whose values differ. Both tables have the same schema and row count."""
    if actual.equals(expected):
        return ()

    masks = {
        name: _differs(actual.column(name).combine_chunks(), expected.column(name).combine_chunks())
        for name in actual.column_names
    }
    if not masks:
        return ()

    any_difference = reduce(pc.or_, masks.values())
    rows = cast("list[int]", pc.indices_nonzero(any_difference).to_pylist())  # pyright: ignore[reportUnknownMemberType]
    return tuple(
        RowDiff(row, tuple(name for name, mask in masks.items() if mask[row].as_py())) for row in rows
    )


def _differs(actual: pa.Array[Any], expected: pa.Array[Any]) -> pa.Array[Any]:
    """Boolean array, no nulls: True where the two values differ. null equals null and NaN equals NaN."""
    try:
        # The pyarrow stubs leave the type parameters of these compute functions unknown.
        not_equal = pc.not_equal(actual, expected)  # pyright: ignore[reportUnknownMemberType]
        actual_null = pc.is_null(actual)  # pyright: ignore[reportUnknownMemberType]
        expected_null = pc.is_null(expected)  # pyright: ignore[reportUnknownMemberType]
        differs = pc.or_(pc.fill_null(not_equal, _FALSE), pc.xor(actual_null, expected_null))
        if pa.types.is_floating(actual.type):
            actual_nan = pc.is_nan(actual)  # pyright: ignore[reportUnknownMemberType]
            expected_nan = pc.is_nan(expected)  # pyright: ignore[reportUnknownMemberType]
            both_nan = pc.and_(pc.fill_null(actual_nan, _FALSE), pc.fill_null(expected_nan, _FALSE))
            differs = pc.and_(differs, pc.invert(both_nan))
        return cast("pa.Array[Any]", differs)  # both inputs are arrays, so the result is an array
    except pa.ArrowNotImplementedError:
        # Nested and null-typed columns have no compute kernels. Compare the python values instead.
        return pa.array(
            [not _same(a, b) for a, b in zip(actual.to_pylist(), expected.to_pylist())],  # pyright: ignore[reportUnknownMemberType]
            pa.bool_(),
        )


def _same(a: object, b: object) -> bool:
    if a != a and b != b:  # NaN is the only value that is not equal to itself
        return True
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
    return a == b


# --- plain-text summary -------------------------------------------------------------------------

_MAX_SUMMARY_ROWS = 10


def _summary(comparison: TableComparison) -> str:
    """A plain message for str(error). A renderer builds a richer one from the TableComparison."""
    schema = comparison.schema
    if not schema.ok:
        lines = ["Schemas do not match:"]
        lines += [f"  {problem}" for problem in schema.problems]
        for column in schema.columns:
            if column.problem is not None:
                lines.append(f"  column '{column.name}': {column.problem} (expected {column.expected}, actual {column.actual})")
        return "\n".join(lines)

    if comparison.row_count is not None:
        counts = comparison.row_count
        return f"Row counts differ: expected {counts.expected} rows, actual {counts.actual} rows."

    assert comparison.actual is not None and comparison.expected is not None
    diffs = comparison.row_diffs or ()
    lines = [f"{len(diffs)} of {comparison.expected.num_rows} rows differ:"]
    for diff in diffs[:_MAX_SUMMARY_ROWS]:
        for name in diff.columns:
            actual_value = comparison.actual.column(name)[diff.row].as_py()
            expected_value = comparison.expected.column(name)[diff.row].as_py()
            lines.append(f"  row {diff.row}, column '{name}': expected {expected_value!r}, actual {actual_value!r}")
    if len(diffs) > _MAX_SUMMARY_ROWS:
        lines.append(f"  ... and {len(diffs) - _MAX_SUMMARY_ROWS} more rows")
    return "\n".join(lines)
