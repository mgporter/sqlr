"""Compare the schema of an actual query result against an expected one.

`compare_schemas` decides whether each column's actual type is compatible with the expected type
and, if so, which common type both sides should be cast to before comparing data.
`cast_to_common` applies those casts to real tables.

Compatibility rules. Each rule has a `Policy` flag. With the flag False (the default) the rule below
applies; with it True the two types must match exactly instead. Anything not listed must match exactly.

    null + X                            -> X                           strict_null_types
    int + int (same signedness)         -> the wider int               strict_int_widths
    int + decimal128(p, 0)              -> decimal128(max(p, digits    strict_integer_vs_decimal
                                           the int needs), 0)
    decimal128 + decimal128             -> larger precision            strict_decimal_precision
    any float + any float               -> float64                     strict_float_widths
    string / large_string / string_view -> large_string                strict_large_variants
    binary / large_binary / binary_view -> large_binary                strict_large_variants
    list / large_list                   -> large_list                  strict_large_variants
    time + time, timestamp + timestamp  -> finest unit                 strict_temporal_units
    timestamp with differing tz names   -> tz UTC                      strict_tz_names
    children of list / map / struct     -> compared by these rules     strict_nested_children

Always true, with no flag: identical types match and are left as they are (so two float32 columns
stay float32). Signed vs unsigned integers, int vs float, decimals with a different scale, and
timestamps with vs without a tz never match.

Nested types are compared one level deep: the children of a list, map or struct must be non-nested
(or match exactly). Every field in the result is nullable.

Casts are never checked value by value. A cast that does not fit (e.g. timestamp us -> ns for a year
outside 1677-2262) raises `SchemaCastError` from `cast_to_common`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Self, Sequence, cast

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, model_validator

from sqlr.errors import ComparisonError

__all__ = [
    "Policy",
    "SchemaPolicyError",
    "ColumnComparison",
    "SchemaComparison",
    "SchemaCastError",
    "compare_schemas",
    "cast_to_common",
]


class SchemaPolicyError(ComparisonError):
    """The Policy combines options that contradict each other."""


class Policy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- columns -------------------------------------------------------------------------------

    # True: column names must match exactly. False: any casing matches, and the common schema uses
    # the expected spelling.
    case_sensitive_column_names: bool = False

    # True: the actual columns must be in exactly the expected order. False: order is ignored.
    # Needs allow_extra_columns_in_result=False, since extra columns would shift the positions.
    strict_column_ordering: bool = False

    # Columns in the actual result that the expected result does not have. They are dropped before
    # the data is compared. A column the expected result has but the actual result lacks always fails.
    # Cannot be True when strict_column_ordering is True.
    allow_extra_columns_in_result: bool = False

    # --- structs -------------------------------------------------------------------------------
    # Struct field names always match case-sensitively.

    # True: struct fields must be in exactly the expected order. False: order is ignored.
    # Needs allow_extra_fields_in_structs=False.
    strict_struct_field_ordering: bool = False

    # Fields in an actual struct that the expected struct does not have. They are dropped.
    # A field the expected struct has but the actual struct lacks always fails.
    allow_extra_fields_in_structs: bool = False

    # --- types: True means the types must match exactly, False means cast to a common type ------

    strict_null_types: bool = False  # a null-typed column matches any type
    strict_int_widths: bool = False  # int8..int64 (same signedness)
    strict_integer_vs_decimal: bool = False  # an integer matches a decimal with scale 0
    strict_decimal_precision: bool = False  # decimal precision; the scale must always match
    strict_float_widths: bool = False  # float16/float32/float64; False casts every float to float64
    strict_large_variants: bool = False  # string/large_string/string_view, binary variants, list/large_list
    strict_temporal_units: bool = False  # s/ms/us/ns on time and timestamp
    strict_tz_names: bool = False  # tz-aware timestamps with different tz names
    strict_nested_children: bool = False  # children of list/map/struct are not promoted

    @model_validator(mode="after")
    def _check_exclusive_options(self) -> Self:
        if self.strict_column_ordering and self.allow_extra_columns_in_result:
            raise SchemaPolicyError(
                "strict_column_ordering=True cannot be combined with allow_extra_columns_in_result=True: "
                "extra columns shift the positions, so strict ordering needs the exact column list. "
                "Set allow_extra_columns_in_result=False."
            )
        if self.strict_struct_field_ordering and self.allow_extra_fields_in_structs:
            raise SchemaPolicyError(
                "strict_struct_field_ordering=True cannot be combined with allow_extra_fields_in_structs=True: "
                "extra fields shift the positions, so strict ordering needs the exact field list. "
                "Set allow_extra_fields_in_structs=False."
            )
        return self


@dataclass(frozen=True)
class ColumnComparison:
    name: str  # the expected spelling, or the actual spelling for an extra column
    expected: pa.DataType | None  # None: the expected result has no such column
    actual: pa.DataType | None  # None: the actual result has no such column
    common: pa.DataType | None  # type both sides are cast to; None for a mismatch or an ignored extra column
    problem: str | None = None
    actual_name: str | None = None  # the column's name in the actual result; differs from `name` only in case

    @property
    def ok(self) -> bool:
        return self.problem is None


@dataclass(frozen=True)
class SchemaComparison:
    columns: tuple[ColumnComparison, ...]  # expected columns in expected order, then extra actual columns
    problems: tuple[str, ...]  # whole-schema problems: duplicate column names, column order
    common_schema: pa.Schema | None  # None unless ok; expected columns only, in expected order

    @property
    def ok(self) -> bool:
        return not self.problems and all(column.ok for column in self.columns)


class SchemaCastError(ComparisonError):
    """A column could not be cast to its common type."""

    def __init__(
        self,
        column: str,
        side: Literal["actual", "expected"],
        source: pa.DataType,
        target: pa.DataType,
        cause: Exception,
    ) -> None:
        super().__init__(f"Cannot cast {side} column '{column}' from {source} to {target}: {cause}")
        self.column = column
        self.side = side
        self.source = source
        self.target = target


def compare_schemas(actual: pa.Schema, expected: pa.Schema, policy: Policy = Policy()) -> SchemaComparison:
    key = _exact if policy.case_sensitive_column_names else _fold_case
    actual_by_key = {key(name): (name, type_) for name, type_ in zip(actual.names, actual.types)}
    expected_keys = {key(name) for name in expected.names}
    columns: list[ColumnComparison] = []

    for name, expected_type in zip(expected.names, expected.types):
        match = actual_by_key.get(key(name))
        if match is None:
            columns.append(ColumnComparison(name, expected_type, None, None, "missing column"))
            continue
        actual_name, actual_type = match
        try:
            common = _common(actual_type, expected_type, policy)
        except _NoCommonType as e:
            columns.append(ColumnComparison(name, expected_type, actual_type, None, str(e), actual_name))
        else:
            columns.append(ColumnComparison(name, expected_type, actual_type, common, None, actual_name))

    for name, actual_type in zip(actual.names, actual.types):
        if key(name) not in expected_keys:
            problem = None if policy.allow_extra_columns_in_result else "unexpected column"
            columns.append(ColumnComparison(name, None, actual_type, None, problem, name))

    problems = _structure_problems(
        actual.names, expected.names, "column", strict_order=policy.strict_column_ordering, key=key
    )
    ok = not problems and all(column.ok for column in columns)

    common_schema: pa.Schema | None = None
    if ok:
        common_schema = pa.schema(
            [pa.field(c.name, c.common, nullable=True) for c in columns if c.expected is not None and c.common is not None]
        )

    return SchemaComparison(tuple(columns), tuple(problems), common_schema)


def cast_to_common(actual: pa.Table, expected: pa.Table, comparison: SchemaComparison) -> tuple[pa.Table, pa.Table]:
    """Cast both tables to the comparison's common schema. Extra actual columns are dropped.

    Columns come out in expected order with the expected spelling of their names.
    Raises SchemaCastError if a cast fails, and ValueError if the comparison was not ok.
    """
    schema = comparison.common_schema
    if schema is None:
        raise ValueError("The schemas do not match, so there is no common schema to cast to.")
    actual_names = [c.actual_name for c in comparison.columns if c.expected is not None and c.actual_name is not None]
    return (
        _cast_table(actual, actual_names, schema, "actual"),
        _cast_table(expected, schema.names, schema, "expected"),
    )


def _cast_table(
    table: pa.Table, source_names: Sequence[str], schema: pa.Schema, side: Literal["actual", "expected"]
) -> pa.Table:
    arrays: list[pa.ChunkedArray[pa.Scalar[pa.DataType]]] = []
    for source_name, target in zip(source_names, schema.types):
        column = table.column(source_name)
        if column.type != target:
            try:
                column = column.cast(target)  # safe cast: a value that does not fit raises
            except pa.ArrowException as e:
                raise SchemaCastError(source_name, side, column.type, target, e) from e
        arrays.append(column)
    return pa.Table.from_arrays(arrays, schema=schema)


def _exact(name: str) -> str:
    return name


def _fold_case(name: str) -> str:
    return name.casefold()


# --- type rules ---------------------------------------------------------------------------------


class _NoCommonType(Exception):
    """Internal: the two types are incompatible. The message says why."""


Unit = Literal["s", "ms", "us", "ns"]
_UNITS: tuple[Unit, ...] = ("s", "ms", "us", "ns")


def _finest(a: Unit, b: Unit) -> Unit:
    return a if _UNITS.index(a) >= _UNITS.index(b) else b


def _common(actual: pa.DataType, expected: pa.DataType, policy: Policy, depth: int = 0) -> pa.DataType:
    if actual == expected:  # most schemas match exactly
        return expected

    actual_is_null = pa.types.is_null(actual)
    expected_is_null = pa.types.is_null(expected)
    if actual_is_null or expected_is_null:
        if policy.strict_null_types:
            raise _NoCommonType(f"a null column only matches a null column ({actual} vs {expected})")
        return expected if actual_is_null else actual

    if pa.types.is_nested(actual) or pa.types.is_nested(expected):
        if depth == 0:
            if (pa.types.is_list(actual) or pa.types.is_large_list(actual)) and (
                pa.types.is_list(expected) or pa.types.is_large_list(expected)
            ):
                return _common_list(actual, expected, policy)
            if pa.types.is_map(actual) and pa.types.is_map(expected):
                return _common_map(actual, expected, policy)
            if pa.types.is_struct(actual) and pa.types.is_struct(expected):
                return _common_struct(actual, expected, policy)
        if depth > 0:
            raise _NoCommonType(f"nesting deeper than one level must match exactly ({actual} vs {expected})")
        raise _NoCommonType(f"no common type for {actual} and {expected}")

    return _common_leaf(actual, expected, policy)


def _child(where: str, actual: pa.DataType, expected: pa.DataType, policy: Policy) -> pa.DataType:
    try:
        if policy.strict_nested_children and actual != expected:
            raise _NoCommonType(f"children must match exactly ({actual} vs {expected})")
        return _common(actual, expected, policy, depth=1)
    except _NoCommonType as e:
        raise _NoCommonType(f"{where}: {e}") from None


def _common_list(
    actual: pa.ListType[Any] | pa.LargeListType[Any],
    expected: pa.ListType[Any] | pa.LargeListType[Any],
    policy: Policy,
) -> pa.DataType:
    if policy.strict_large_variants and pa.types.is_large_list(actual) != pa.types.is_large_list(expected):
        raise _NoCommonType(f"list and large_list must match exactly ({actual} vs {expected})")
    element = _child("list element", actual.value_type, expected.value_type, policy)
    if pa.types.is_large_list(actual) or pa.types.is_large_list(expected):
        return pa.large_list(element)
    return pa.list_(element)


def _common_map(actual: pa.MapType[Any, Any, Any], expected: pa.MapType[Any, Any, Any], policy: Policy) -> pa.DataType:
    key = _child("map key", actual.key_type, expected.key_type, policy)
    item = _child("map value", actual.item_type, expected.item_type, policy)
    return pa.map_(key, item)


def _common_struct(actual: pa.StructType, expected: pa.StructType, policy: Policy) -> pa.DataType:
    actual_fields = _struct_fields(actual)
    expected_fields = _struct_fields(expected)
    actual_names = [name for name, _ in actual_fields]
    expected_names = [name for name, _ in expected_fields]

    problems = _structure_problems(
        actual_names, expected_names, "struct field", strict_order=policy.strict_struct_field_ordering
    )
    missing = [name for name in expected_names if name not in actual_names]
    if missing:
        problems.append(f"struct fields missing from actual: {missing}")
    extra = [name for name in actual_names if name not in expected_names]
    if extra and not policy.allow_extra_fields_in_structs:
        problems.append(f"unexpected struct fields in actual: {extra}")
    if problems:
        raise _NoCommonType("; ".join(problems))

    actual_types = dict(actual_fields)
    return pa.struct(
        [
            pa.field(name, _child(f"struct field '{name}'", actual_types[name], expected_type, policy), nullable=True)
            for name, expected_type in expected_fields
        ]
    )


def _struct_fields(t: pa.StructType) -> list[tuple[str, pa.DataType]]:
    return [(f.name, cast(pa.DataType, f.type)) for f in t]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def _common_leaf(actual: pa.DataType, expected: pa.DataType, policy: Policy) -> pa.DataType:
    """Common type of two different, non-null, non-nested types."""
    if pa.types.is_integer(actual) and pa.types.is_integer(expected):
        if pa.types.is_signed_integer(actual) != pa.types.is_signed_integer(expected):
            raise _NoCommonType(f"signed and unsigned integers never match ({actual} vs {expected})")
        if policy.strict_int_widths:
            raise _NoCommonType(f"integer widths must match exactly ({actual} vs {expected})")
        return actual if actual.bit_width >= expected.bit_width else expected

    if pa.types.is_floating(actual) and pa.types.is_floating(expected):
        if policy.strict_float_widths:
            raise _NoCommonType(f"float widths must match exactly ({actual} vs {expected})")
        return pa.float64()

    if pa.types.is_decimal128(actual) and pa.types.is_decimal128(expected):
        if actual.scale != expected.scale:
            raise _NoCommonType(f"decimal scale differs ({actual} vs {expected})")
        if policy.strict_decimal_precision and actual.precision != expected.precision:
            raise _NoCommonType(f"decimal precision must match exactly ({actual} vs {expected})")
        return pa.decimal128(max(actual.precision, expected.precision), expected.scale)

    for integer, decimal in ((actual, expected), (expected, actual)):
        if pa.types.is_integer(integer) and pa.types.is_decimal128(decimal):
            if policy.strict_integer_vs_decimal:
                raise _NoCommonType(f"an integer never matches a decimal ({actual} vs {expected})")
            if decimal.scale != 0:
                raise _NoCommonType(f"an integer only matches a decimal with scale 0 ({actual} vs {expected})")
            return pa.decimal128(max(decimal.precision, _int_digits(integer)), 0)

    if _is_text(actual) and _is_text(expected):
        if policy.strict_large_variants:
            raise _NoCommonType(f"string variants must match exactly ({actual} vs {expected})")
        return pa.large_string()

    if _is_binary(actual) and _is_binary(expected):
        if policy.strict_large_variants:
            raise _NoCommonType(f"binary variants must match exactly ({actual} vs {expected})")
        return pa.large_binary()

    if pa.types.is_time(actual) and pa.types.is_time(expected):
        if policy.strict_temporal_units and actual.unit != expected.unit:
            raise _NoCommonType(f"time units must match exactly ({actual} vs {expected})")
        unit = _finest(actual.unit, expected.unit)
        if unit == "s" or unit == "ms":
            return pa.time32(unit)
        return pa.time64(unit)

    if pa.types.is_timestamp(actual) and pa.types.is_timestamp(expected):
        if (actual.tz is None) != (expected.tz is None):
            raise _NoCommonType(f"timestamps with and without a time zone never match ({actual} vs {expected})")
        if policy.strict_temporal_units and actual.unit != expected.unit:
            raise _NoCommonType(f"timestamp units must match exactly ({actual} vs {expected})")
        unit = _finest(actual.unit, expected.unit)
        if actual.tz is None or expected.tz is None:
            return pa.timestamp(unit)
        if actual.tz == expected.tz:
            return pa.timestamp(unit, tz=actual.tz)
        if policy.strict_tz_names:
            raise _NoCommonType(f"time zone names must match exactly ({actual} vs {expected})")
        return pa.timestamp(unit, tz="UTC")

    raise _NoCommonType(f"no common type for {actual} and {expected}")


def _int_digits(t: pa.DataType) -> int:
    """Decimal digits needed to hold any value of an integer type."""
    bits = t.bit_width
    if pa.types.is_unsigned_integer(t) and bits == 64:
        return 20
    return {8: 3, 16: 5, 32: 10, 64: 19}[bits]


def _is_text(t: pa.DataType) -> bool:
    return pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_string_view(t)


def _is_binary(t: pa.DataType) -> bool:
    return pa.types.is_binary(t) or pa.types.is_large_binary(t) or pa.types.is_binary_view(t)


def _structure_problems(
    actual: Sequence[str],
    expected: Sequence[str],
    what: str,
    *,
    strict_order: bool,
    key: Callable[[str], str] = _exact,
) -> list[str]:
    """Duplicate names, and (if strict_order) names shared by both sides appearing in a different order."""
    problems: list[str] = []
    for label, names in (("actual", actual), ("expected", expected)):
        keys = [key(name) for name in names]
        duplicates = sorted({name for name, k in zip(names, keys) if keys.count(k) > 1})
        if duplicates:
            problems.append(f"duplicate {what} names in {label}: {duplicates}")

    if strict_order:
        actual_keys = {key(name) for name in actual}
        expected_keys = {key(name) for name in expected}
        shared_in_actual = [name for name in actual if key(name) in expected_keys]
        shared_in_expected = [name for name in expected if key(name) in actual_keys]
        if [key(n) for n in shared_in_actual] != [key(n) for n in shared_in_expected]:
            problems.append(f"{what} order differs: expected {shared_in_expected}, actual {shared_in_actual}")
    return problems
