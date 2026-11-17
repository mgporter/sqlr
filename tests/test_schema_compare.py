import datetime as dt

import duckdb
import pyarrow as pa
import pytest
from pydantic import ValidationError

from sqlr.schema_compare import Policy, SchemaCastError, SchemaPolicyError, cast_to_common, compare_schemas


def one_column(actual: pa.DataType, expected: pa.DataType, **policy: bool):
    return compare_schemas(
        pa.schema([("c", actual)]),
        pa.schema([("c", expected)]),
        Policy(**policy),
    )


def common_of(actual: pa.DataType, expected: pa.DataType, **policy: bool) -> pa.DataType:
    comparison = one_column(actual, expected, **policy)
    assert comparison.ok, comparison.columns[0].problem
    assert comparison.common_schema is not None
    return comparison.common_schema.field("c").type


COMMON_TYPES = [
    # null
    pytest.param(pa.null(), pa.int32(), pa.int32(), id="null-int"),
    pytest.param(pa.string(), pa.null(), pa.string(), id="string-null"),
    # integers
    pytest.param(pa.int16(), pa.int32(), pa.int32(), id="int16-int32"),
    pytest.param(pa.int64(), pa.int8(), pa.int64(), id="int64-int8"),
    pytest.param(pa.int32(), pa.int32(), pa.int32(), id="int32-int32"),
    pytest.param(pa.uint8(), pa.uint16(), pa.uint16(), id="uint8-uint16"),
    # floats always become float64
    pytest.param(pa.float32(), pa.float64(), pa.float64(), id="float32-float64"),
    pytest.param(pa.float32(), pa.float32(), pa.float32(), id="float32-float32-identical-left-alone"),
    pytest.param(pa.float16(), pa.float32(), pa.float64(), id="float16-float32"),
    pytest.param(pa.float64(), pa.float64(), pa.float64(), id="float64-float64"),
    # decimals
    pytest.param(pa.decimal128(10, 2), pa.decimal128(18, 2), pa.decimal128(18, 2), id="decimal-precision"),
    pytest.param(pa.int32(), pa.decimal128(38, 0), pa.decimal128(38, 0), id="int32-decimal38"),
    pytest.param(pa.int64(), pa.decimal128(10, 0), pa.decimal128(19, 0), id="int64-small-decimal"),
    pytest.param(pa.uint64(), pa.decimal128(10, 0), pa.decimal128(20, 0), id="uint64-small-decimal"),
    pytest.param(pa.decimal128(38, 0), pa.int8(), pa.decimal128(38, 0), id="decimal-int8"),
    # text and binary
    pytest.param(pa.string(), pa.string(), pa.string(), id="string-string"),
    pytest.param(pa.string(), pa.large_string(), pa.large_string(), id="string-large_string"),
    pytest.param(pa.string_view(), pa.string(), pa.large_string(), id="string_view-string"),
    pytest.param(pa.binary(), pa.large_binary(), pa.large_binary(), id="binary-large_binary"),
    pytest.param(pa.binary(), pa.binary(), pa.binary(), id="binary-binary"),
    # exact-match types
    pytest.param(pa.bool_(), pa.bool_(), pa.bool_(), id="bool"),
    pytest.param(pa.date32(), pa.date32(), pa.date32(), id="date32"),
    # time
    pytest.param(pa.time32("ms"), pa.time64("ns"), pa.time64("ns"), id="time32ms-time64ns"),
    pytest.param(pa.time32("s"), pa.time32("ms"), pa.time32("ms"), id="time32s-time32ms"),
    pytest.param(pa.time64("us"), pa.time64("us"), pa.time64("us"), id="time64us-time64us"),
    # timestamps
    pytest.param(pa.timestamp("ms"), pa.timestamp("us"), pa.timestamp("us"), id="ts-ms-us"),
    pytest.param(pa.timestamp("ns"), pa.timestamp("s"), pa.timestamp("ns"), id="ts-ns-s"),
    pytest.param(pa.timestamp("us", tz="UTC"), pa.timestamp("ns", tz="UTC"), pa.timestamp("ns", tz="UTC"), id="ts-tz-same"),
    pytest.param(
        pa.timestamp("us", tz="UTC"),
        pa.timestamp("us", tz="America/Los_Angeles"),
        pa.timestamp("us", tz="UTC"),
        id="ts-tz-differs-becomes-utc",
    ),
    # lists
    pytest.param(pa.list_(pa.int16()), pa.list_(pa.int32()), pa.list_(pa.int32()), id="list-widen"),
    pytest.param(pa.list_(pa.int32()), pa.large_list(pa.int32()), pa.large_list(pa.int32()), id="list-large_list"),
    pytest.param(
        pa.list_(pa.field("item", pa.int32(), nullable=False)),
        pa.list_(pa.field("element", pa.int32())),
        pa.list_(pa.int32()),
        id="list-names-and-nullability-ignored",
    ),
    pytest.param(pa.list_(pa.null()), pa.list_(pa.string()), pa.list_(pa.string()), id="list-null-element"),
    # maps
    pytest.param(
        pa.map_(pa.string(), pa.int16()),
        pa.map_(pa.string(), pa.int32()),
        pa.map_(pa.string(), pa.int32()),
        id="map-widen-value",
    ),
    # structs
    pytest.param(
        pa.struct([("a", pa.int16()), ("b", pa.string())]),
        pa.struct([("a", pa.int32()), ("b", pa.large_string())]),
        pa.struct([("a", pa.int32()), ("b", pa.large_string())]),
        id="struct-widen-fields",
    ),
    # one level of nesting only compares by exact equality below that
    pytest.param(pa.list_(pa.list_(pa.int32())), pa.list_(pa.list_(pa.int32())), pa.list_(pa.list_(pa.int32())), id="deep-equal"),
]


@pytest.mark.parametrize("actual, expected, common", COMMON_TYPES)
def test_common_type_both_directions(actual: pa.DataType, expected: pa.DataType, common: pa.DataType) -> None:
    assert common_of(actual, expected) == common
    assert common_of(expected, actual) == common


MISMATCHES = [
    pytest.param(pa.int32(), pa.uint32(), id="signed-unsigned"),
    pytest.param(pa.int8(), pa.uint64(), id="signed-unsigned-wide"),
    pytest.param(pa.int64(), pa.float64(), id="int-float"),
    pytest.param(pa.float64(), pa.decimal128(10, 2), id="float-decimal"),
    pytest.param(pa.decimal128(10, 2), pa.decimal128(10, 4), id="decimal-scale"),
    pytest.param(pa.int32(), pa.decimal128(10, 2), id="int-decimal-with-scale"),
    pytest.param(pa.string(), pa.binary(), id="string-binary"),
    pytest.param(pa.string(), pa.int32(), id="string-int"),
    pytest.param(pa.date32(), pa.date64(), id="date32-date64"),
    pytest.param(pa.date32(), pa.timestamp("us"), id="date-timestamp"),
    pytest.param(pa.timestamp("us"), pa.timestamp("us", tz="UTC"), id="ntz-vs-aware"),
    pytest.param(pa.timestamp("ns"), pa.timestamp("ns", tz="America/Los_Angeles"), id="ntz-vs-ltz"),
    pytest.param(pa.list_(pa.int32()), pa.list_(pa.string()), id="list-element"),
    pytest.param(pa.list_(pa.int32()), pa.int32(), id="list-vs-scalar"),
    pytest.param(pa.list_(pa.list_(pa.int16())), pa.list_(pa.list_(pa.int32())), id="deep-needs-exact-match"),
    pytest.param(pa.list_(pa.struct([("a", pa.int16())])), pa.list_(pa.struct([("a", pa.int32())])), id="struct-in-list-needs-exact-match"),
    pytest.param(pa.map_(pa.string(), pa.int32()), pa.map_(pa.int32(), pa.int32()), id="map-key"),
    pytest.param(pa.struct([("a", pa.int32())]), pa.struct([("a", pa.string())]), id="struct-field-type"),
    pytest.param(pa.struct([("a", pa.int32())]), pa.map_(pa.string(), pa.int32()), id="struct-vs-map"),
    pytest.param(pa.struct([("a", pa.struct([("b", pa.int16())]))]), pa.struct([("a", pa.struct([("b", pa.int32())]))]), id="struct-in-struct"),
    pytest.param(pa.month_day_nano_interval(), pa.duration("ns"), id="interval-duration"),
]


@pytest.mark.parametrize("actual, expected", MISMATCHES)
def test_mismatch_both_directions(actual: pa.DataType, expected: pa.DataType) -> None:
    for a, e in ((actual, expected), (expected, actual)):
        comparison = one_column(a, e)
        assert not comparison.ok
        assert comparison.common_schema is None
        assert comparison.columns[0].problem


def test_struct_fields_missing_from_actual_always_fail() -> None:
    actual = pa.struct([("a", pa.int32())])
    expected = pa.struct([("a", pa.int32()), ("b", pa.string())])
    assert not one_column(actual, expected).ok
    assert not one_column(actual, expected, allow_extra_fields_in_structs=True).ok


def test_struct_extra_fields_follow_policy() -> None:
    actual = pa.struct([("a", pa.int16()), ("b", pa.string()), ("c", pa.bool_())])
    expected = pa.struct([("a", pa.int32()), ("b", pa.string())])

    strict = one_column(actual, expected)
    assert not strict.ok
    assert "unexpected struct fields" in (strict.columns[0].problem or "")

    assert common_of(actual, expected, allow_extra_fields_in_structs=True) == pa.struct(
        [("a", pa.int32()), ("b", pa.string())]
    )


def test_struct_field_order_follows_policy() -> None:
    actual = pa.struct([("b", pa.string()), ("a", pa.int32())])
    expected = pa.struct([("a", pa.int32()), ("b", pa.string())])
    assert common_of(actual, expected) == expected  # order is ignored by default

    strict = one_column(actual, expected, strict_struct_field_ordering=True)
    assert not strict.ok
    assert "struct field order differs" in (strict.columns[0].problem or "")
    assert one_column(expected, expected, strict_struct_field_ordering=True).ok


def test_struct_extra_field_may_sit_anywhere_if_shared_order_matches() -> None:
    actual = pa.struct([("x", pa.int32()), ("a", pa.int32()), ("y", pa.int32()), ("b", pa.string())])
    expected = pa.struct([("a", pa.int32()), ("b", pa.string())])
    assert common_of(actual, expected, allow_extra_fields_in_structs=True) == expected


# --- whole-schema behavior ---------------------------------------------------------------------


def test_identical_schemas_are_ok() -> None:
    schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    comparison = compare_schemas(schema, schema)
    assert comparison.ok
    assert comparison.common_schema == schema


def test_common_schema_is_nullable_and_in_expected_order() -> None:
    actual = pa.schema([("b", pa.int16()), ("a", pa.string())])
    expected = pa.schema([pa.field("a", pa.string(), nullable=False), pa.field("b", pa.int32(), nullable=False)])
    comparison = compare_schemas(actual, expected)
    assert comparison.ok
    assert comparison.common_schema is not None
    assert comparison.common_schema == pa.schema([("a", pa.string()), ("b", pa.int32())])
    assert all(field.nullable for field in comparison.common_schema)


def test_missing_column_always_fails() -> None:
    actual = pa.schema([("a", pa.int32())])
    expected = pa.schema([("a", pa.int32()), ("b", pa.string())])
    for allow in (False, True):
        comparison = compare_schemas(actual, expected, Policy(allow_extra_columns_in_result=allow))
        assert not comparison.ok
        assert [c.problem for c in comparison.columns] == [None, "missing column"]


def test_extra_column_follows_policy() -> None:
    actual = pa.schema([("a", pa.int32()), ("extra", pa.string())])
    expected = pa.schema([("a", pa.int32())])

    strict = compare_schemas(actual, expected)  # extra columns are not allowed by default
    assert not strict.ok
    assert strict.columns[1].problem == "unexpected column"

    lenient = compare_schemas(actual, expected, Policy(allow_extra_columns_in_result=True))
    assert lenient.ok
    assert lenient.common_schema == pa.schema([("a", pa.int32())])
    assert lenient.columns[1].expected is None and lenient.columns[1].actual == pa.string()


def test_column_order_follows_policy() -> None:
    actual = pa.schema([("b", pa.int32()), ("a", pa.int32())])
    expected = pa.schema([("a", pa.int32()), ("b", pa.int32())])
    assert compare_schemas(actual, expected).ok  # order is ignored by default

    strict = compare_schemas(actual, expected, Policy(strict_column_ordering=True))
    assert not strict.ok
    assert all(c.ok for c in strict.columns)
    assert "column order differs" in strict.problems[0]
    assert compare_schemas(expected, expected, Policy(strict_column_ordering=True)).ok


def test_extra_column_in_the_middle_is_fine_when_allowed() -> None:
    actual = pa.schema([("a", pa.int32()), ("extra", pa.int32()), ("b", pa.int32())])
    expected = pa.schema([("a", pa.int32()), ("b", pa.int32())])
    assert not compare_schemas(actual, expected).ok
    assert compare_schemas(actual, expected, Policy(allow_extra_columns_in_result=True)).ok


def test_duplicate_column_names_fail() -> None:
    actual = pa.schema([("a", pa.int32()), ("a", pa.int32())])
    expected = pa.schema([("a", pa.int32())])
    comparison = compare_schemas(actual, expected)
    assert not comparison.ok
    assert "duplicate column names in actual" in comparison.problems[0]


def test_column_names_follow_case_policy() -> None:
    actual = pa.schema([("NUM", pa.int32())])
    expected = pa.schema([("num", pa.int32())])

    insensitive = compare_schemas(actual, expected)  # case-insensitive by default
    assert insensitive.ok
    assert insensitive.common_schema == expected
    assert insensitive.columns[0].name == "num" and insensitive.columns[0].actual_name == "NUM"

    sensitive = compare_schemas(actual, expected, Policy(case_sensitive_column_names=True))
    assert not sensitive.ok
    assert [c.problem for c in sensitive.columns] == ["missing column", "unexpected column"]


def test_columns_differing_only_by_case_are_duplicates_unless_case_sensitive() -> None:
    actual = pa.schema([("a", pa.int32()), ("A", pa.int32())])
    expected = pa.schema([("a", pa.int32())])

    insensitive = compare_schemas(actual, expected)
    assert not insensitive.ok
    assert "duplicate column names in actual" in insensitive.problems[0]

    sensitive = Policy(case_sensitive_column_names=True, allow_extra_columns_in_result=True)
    assert compare_schemas(actual, expected, sensitive).ok


def test_strict_column_ordering_with_case_insensitive_names() -> None:
    expected = pa.schema([("a", pa.int32()), ("b", pa.int32())])
    policy = Policy(strict_column_ordering=True)
    assert compare_schemas(pa.schema([("A", pa.int32()), ("B", pa.int32())]), expected, policy).ok
    assert not compare_schemas(pa.schema([("B", pa.int32()), ("A", pa.int32())]), expected, policy).ok


# --- policy validation -------------------------------------------------------------------------


def test_strict_column_ordering_conflicts_with_extra_columns() -> None:
    with pytest.raises(SchemaPolicyError, match="allow_extra_columns_in_result"):
        Policy(strict_column_ordering=True, allow_extra_columns_in_result=True)
    with pytest.raises(SchemaPolicyError):
        Policy.model_validate({"strict_column_ordering": True, "allow_extra_columns_in_result": True})


def test_extra_columns_default_to_not_allowed() -> None:
    assert Policy().allow_extra_columns_in_result is False
    assert Policy(strict_column_ordering=True).allow_extra_columns_in_result is False
    assert Policy(strict_column_ordering=False, allow_extra_columns_in_result=True).allow_extra_columns_in_result is True


def test_strict_struct_field_ordering_conflicts_with_extra_fields() -> None:
    with pytest.raises(SchemaPolicyError, match="allow_extra_fields_in_structs"):
        Policy(strict_struct_field_ordering=True, allow_extra_fields_in_structs=True)
    assert Policy(strict_struct_field_ordering=True).allow_extra_fields_in_structs is False


# --- strict flags: promoted by default, exact match when True ------------------------------------

STRICT_FLAGS = [
    pytest.param("strict_null_types", pa.null(), pa.int32(), id="null"),
    pytest.param("strict_int_widths", pa.int16(), pa.int32(), id="int-widths"),
    pytest.param("strict_integer_vs_decimal", pa.int32(), pa.decimal128(38, 0), id="integer-vs-decimal"),
    pytest.param("strict_decimal_precision", pa.decimal128(10, 2), pa.decimal128(18, 2), id="decimal-precision"),
    pytest.param("strict_float_widths", pa.float32(), pa.float64(), id="float-widths"),
    pytest.param("strict_float_widths", pa.float16(), pa.float32(), id="float16-float32"),
    pytest.param("strict_large_variants", pa.string(), pa.large_string(), id="string-large_string"),
    pytest.param("strict_large_variants", pa.string_view(), pa.string(), id="string_view-string"),
    pytest.param("strict_large_variants", pa.binary(), pa.large_binary(), id="binary-large_binary"),
    pytest.param("strict_large_variants", pa.list_(pa.int32()), pa.large_list(pa.int32()), id="list-large_list"),
    pytest.param("strict_temporal_units", pa.timestamp("ms"), pa.timestamp("us"), id="timestamp-unit"),
    pytest.param("strict_temporal_units", pa.timestamp("ms", tz="UTC"), pa.timestamp("us", tz="UTC"), id="timestamp-tz-unit"),
    pytest.param("strict_temporal_units", pa.time32("ms"), pa.time64("ns"), id="time-unit"),
    pytest.param("strict_tz_names", pa.timestamp("us", tz="UTC"), pa.timestamp("us", tz="America/Los_Angeles"), id="tz-names"),
    pytest.param("strict_nested_children", pa.list_(pa.int16()), pa.list_(pa.int32()), id="list-child"),
    pytest.param("strict_nested_children", pa.map_(pa.string(), pa.int16()), pa.map_(pa.string(), pa.int32()), id="map-value"),
    pytest.param(
        "strict_nested_children",
        pa.struct([("a", pa.int16())]),
        pa.struct([("a", pa.int32())]),
        id="struct-field",
    ),
]


@pytest.mark.parametrize("flag, actual, expected", STRICT_FLAGS)
def test_strict_flag_requires_an_exact_match(flag: str, actual: pa.DataType, expected: pa.DataType) -> None:
    strict: dict[str, bool] = {flag: True}
    for a, e in ((actual, expected), (expected, actual)):
        assert one_column(a, e).ok  # promoted by default
        assert not one_column(a, e, **strict).ok
    assert one_column(actual, actual, **strict).ok
    assert one_column(expected, expected, **strict).ok


def test_strict_tz_names_only_affects_names() -> None:
    assert one_column(pa.timestamp("us", tz="UTC"), pa.timestamp("ns", tz="UTC"), strict_tz_names=True).ok
    assert not one_column(pa.timestamp("us", tz="UTC"), pa.timestamp("ns"), strict_tz_names=True).ok


def test_every_strict_flag_on_still_passes_identical_schemas() -> None:
    all_strict = {
        name: True
        for name in Policy.model_fields
        if name.startswith("strict_") and name not in ("strict_column_ordering", "strict_struct_field_ordering")
    }
    assert len(all_strict) == 9
    types = [
        pa.int16(), pa.float32(), pa.decimal128(10, 2), pa.string(), pa.large_binary(), pa.timestamp("us", tz="UTC"),
        pa.time64("ns"), pa.list_(pa.int32()), pa.map_(pa.string(), pa.int64()), pa.struct([("a", pa.int8())]), pa.null(),
    ]
    schema = pa.schema([(f"c{i}", t) for i, t in enumerate(types)])
    assert compare_schemas(schema, schema, Policy(**all_strict)).ok

    assert not one_column(pa.float32(), pa.float64(), **all_strict).ok
    assert not one_column(pa.int32(), pa.uint32()).ok  # signed vs unsigned has no flag and never matches


def test_policy_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Policy(allow_extra_columns=True)  # pyright: ignore[reportCallIssue]


# --- casting real data -----------------------------------------------------------------------


def test_cast_to_common_makes_tables_equal() -> None:
    actual = pa.table(
        {
            "i": pa.array([1, 2], pa.int16()),
            "s": pa.array(["a", "b"], pa.string()),
            "l": pa.array([[1], [2, 3]], pa.list_(pa.int16())),
            "t": pa.array([dt.datetime(2020, 1, 1)] * 2, pa.timestamp("ms")),
            "extra": pa.array([True, False]),
        }
    )
    expected = pa.table(
        {
            "i": pa.array([1, 2], pa.int64()),
            "s": pa.array(["a", "b"], pa.large_string()),
            "l": pa.array([[1], [2, 3]], pa.large_list(pa.int32())),
            "t": pa.array([dt.datetime(2020, 1, 1)] * 2, pa.timestamp("ns")),
        }
    )
    comparison = compare_schemas(actual.schema, expected.schema, Policy(allow_extra_columns_in_result=True))
    assert comparison.ok

    actual_cast, expected_cast = cast_to_common(actual, expected, comparison)
    assert actual_cast.schema == expected_cast.schema == comparison.common_schema
    assert actual_cast.equals(expected_cast)


def test_cast_to_common_drops_extra_struct_fields_and_widens_map_values() -> None:
    actual = pa.table(
        {
            "s": pa.array([{"a": 1, "b": "x", "c": True}], pa.struct([("a", pa.int16()), ("b", pa.string()), ("c", pa.bool_())])),
            "m": pa.array([[("k", 1)]], pa.map_(pa.string(), pa.int16())),
        }
    )
    expected = pa.table(
        {
            "s": pa.array([{"a": 1, "b": "x"}], pa.struct([("a", pa.int32()), ("b", pa.string())])),
            "m": pa.array([[("k", 1)]], pa.map_(pa.string(), pa.int64())),
        }
    )
    comparison = compare_schemas(actual.schema, expected.schema, Policy(allow_extra_fields_in_structs=True))
    assert comparison.ok

    actual_cast, expected_cast = cast_to_common(actual, expected, comparison)
    assert actual_cast.equals(expected_cast)


def test_cast_to_common_null_column_takes_the_expected_type() -> None:
    actual = pa.table({"x": pa.nulls(2)})
    expected = pa.table({"x": pa.array([None, None], pa.int32())})
    comparison = compare_schemas(actual.schema, expected.schema)
    actual_cast, expected_cast = cast_to_common(actual, expected, comparison)
    assert actual_cast.equals(expected_cast)


def test_cast_failure_raises_with_column_and_side() -> None:
    # Year 9999 does not fit in nanoseconds.
    actual = pa.table({"when": pa.array([dt.datetime(9999, 12, 31)], pa.timestamp("us"))})
    expected = pa.table({"when": pa.array([dt.datetime(2020, 1, 1)], pa.timestamp("ns"))})
    comparison = compare_schemas(actual.schema, expected.schema)
    assert comparison.ok

    with pytest.raises(SchemaCastError) as excinfo:
        cast_to_common(actual, expected, comparison)
    assert excinfo.value.column == "when"
    assert excinfo.value.side == "actual"
    assert excinfo.value.source == pa.timestamp("us")
    assert excinfo.value.target == pa.timestamp("ns")


def test_cast_to_common_refuses_a_failed_comparison() -> None:
    table = pa.table({"a": pa.array([1], pa.int32())})
    comparison = compare_schemas(table.schema, pa.schema([("a", pa.string())]))
    with pytest.raises(ValueError):
        cast_to_common(table, table, comparison)


def test_float32_widens_to_float64_but_values_are_not_rounded_back() -> None:
    actual = pa.table({"f": pa.array([0.5, 0.1], pa.float32())})
    expected = pa.table({"f": pa.array([0.5, 0.1], pa.float64())})
    comparison = compare_schemas(actual.schema, expected.schema)
    assert comparison.ok
    actual_cast, expected_cast = cast_to_common(actual, expected, comparison)
    assert actual_cast.schema == expected_cast.schema
    # 0.5 is exact in float32, 0.1 is not: exact comparison still differs.
    assert not actual_cast.equals(expected_cast)


# --- real DuckDB output ------------------------------------------------------------------------


def test_duckdb_nested_results_compare_and_cast() -> None:
    relation = duckdb.sql(
        """
        select
            [1, 2]::smallint[]                      as l,
            map {'a': 1}::map(varchar, smallint)    as m,
            {'x': 1::smallint, 'y': 'b', 'z': true} as s,
            1::smallint                             as i,
            timestamp '2020-01-01 00:00:00'         as t
        """
    )
    assert relation is not None
    actual = relation.to_arrow_table()

    expected = pa.table(
        {
            "l": pa.array([[1, 2]], pa.list_(pa.int64())),
            "m": pa.array([[("a", 1)]], pa.map_(pa.string(), pa.int64())),
            "s": pa.array([{"x": 1, "y": "b"}], pa.struct([("x", pa.int64()), ("y", pa.string())])),
            "i": pa.array([1], pa.int64()),
            "t": pa.array([dt.datetime(2020, 1, 1)], pa.timestamp("ns")),
        }
    )

    assert not compare_schemas(actual.schema, expected.schema).ok  # struct has an extra field 'z'

    comparison = compare_schemas(actual.schema, expected.schema, Policy(allow_extra_fields_in_structs=True))
    assert comparison.ok, [c.problem for c in comparison.columns]

    actual_cast, expected_cast = cast_to_common(actual, expected, comparison)
    assert actual_cast.equals(expected_cast)


def test_cast_to_common_renames_columns_to_the_expected_spelling_and_order() -> None:
    actual = pa.table({"B": pa.array(["x"], pa.string()), "NUM": pa.array([1], pa.int16())})
    expected = pa.table({"num": pa.array([1], pa.int32()), "b": pa.array(["x"], pa.string())})
    comparison = compare_schemas(actual.schema, expected.schema)
    assert comparison.ok

    actual_cast, expected_cast = cast_to_common(actual, expected, comparison)
    assert actual_cast.column_names == ["num", "b"]
    assert actual_cast.equals(expected_cast)


def test_cast_to_common_reorders_and_drops_struct_fields_in_one_cast() -> None:
    actual = pa.table(
        {"s": pa.array([{"c": True, "b": "x", "a": 1}], pa.struct([("c", pa.bool_()), ("b", pa.string()), ("a", pa.int16())]))}
    )
    expected = pa.table({"s": pa.array([{"a": 1, "b": "x"}], pa.struct([("a", pa.int32()), ("b", pa.string())]))})
    comparison = compare_schemas(actual.schema, expected.schema, Policy(allow_extra_fields_in_structs=True))
    assert comparison.ok

    actual_cast, expected_cast = cast_to_common(actual, expected, comparison)
    assert actual_cast.equals(expected_cast)


def test_struct_field_names_are_case_sensitive() -> None:
    # pyarrow would cast a differently-cased field to nulls, so it must be reported as missing.
    actual = pa.struct([("A", pa.int32())])
    expected = pa.struct([("a", pa.int32())])
    assert not one_column(actual, expected).ok
    assert not one_column(actual, expected, allow_extra_fields_in_structs=True).ok
