"""Inferred types held up against declared ones.

The assertions are about the *judgement* - which of the two types stands, and how loudly
the disagreement is reported - rather than about inference, which `test_schema_resolution`
already covers. The cases that matter are the asymmetric ones: a family pinned down by a
declaration is a pass, the same declaration losing information is a warning, and a column
typed only by its own name never contradicts anything.
"""

from pathlib import Path

from sqlr.catalog import find_yaml_files
from sqlr.declared import load_declared_schemas
from sqlr.schema_resolution import resolve_schema
from sqlr.sql_analysis import analyze_file
from sqlr.validation import validate_schema
from sqlr.validation.types import ColumnValidation, StatementValidation


def _validate(tmp_path: Path, sql: str, yml: str) -> StatementValidation:
    path = tmp_path / "orders.sql"
    path.write_text(sql)
    (tmp_path / "schema.yml").write_text(yml)
    declared = load_declared_schemas(find_yaml_files(tmp_path))
    return validate_schema(resolve_schema(analyze_file(path)), declared, path)


def _column(validation: StatementValidation, table: str, name: str) -> ColumnValidation:
    found = next(t for t in validation.all_tables if t.name == table)
    return next(column for column in found.columns if column.name == name)



def _yml(name: str, *columns: str, sql_file: str | None = None) -> str:
    """A declaration for one relation, in the shape a standalone project writes.

    Everything is a `sources:` table, including what this project builds: those name the
    file that builds them with `sql_file:`. `warehouse` declares no database and no
    schema, so its tables are referred to bare - `source_table`, not `db.schema.x`.
    """
    lines = [
        "version: 2",
        "sources:",
        "  - name: warehouse",
        "    tables:",
        f"      - name: {name}",
    ]
    if sql_file is not None:
        lines.append(f"        sql_file: {sql_file}")
    if columns:
        lines.append("        columns:")
        for column in columns:
            column_name, _, data_type = column.partition(" ")
            lines.append(f"          - name: {column_name}")
            if data_type:
                lines.append(f"            data_type: {data_type}")
    return "\n".join(lines) + "\n"


def _upstream(*columns: str) -> str:
    """A declaration for the table `orders.sql` reads."""
    return _yml("source_table", *columns)


# ---- the outcomes --------------------------------------------------------------------


def test_the_same_type_on_both_sides_is_an_exact_match(tmp_path: Path) -> None:
    validation = _validate(
        tmp_path,
        "select id\nfrom source_table\nwhere country = 'US'\n",
        _upstream("country varchar(2)"),
    )

    country = _column(validation, "source_table", "country")
    assert (country.outcome, country.detail) == ("pass", "exact match")
    assert country.inferred_type == "string"
    assert country.resolved_type == "string"


def test_a_declaration_that_pins_down_a_family_narrows_it(tmp_path: Path) -> None:
    # `revenue > 1000` proves a number, never which kind. The declaration knows.
    validation = _validate(
        tmp_path,
        "select id\nfrom source_table\nwhere revenue > 1000\n",
        _upstream("revenue decimal(10,2)"),
    )

    revenue = _column(validation, "source_table", "revenue")
    assert (revenue.outcome, revenue.detail) == ("pass", "type narrowed")
    assert revenue.inferred_type == "numeric"
    assert revenue.resolved_type == "decimal"


def test_a_declaration_looser_than_the_sql_is_a_widening_warning(tmp_path: Path) -> None:
    # The cast pins the zone down; a bare `timestamp` declaration throws that away. Safe
    # to generate against - it accepts what was inferred - but it is losing information.
    validation = _validate(
        tmp_path,
        "select cast(seen_at as timestamp_ntz) as seen_at\nfrom source_table\n",
        _yml("orders", "seen_at timestamp", sql_file="orders"),
    )

    seen_at = _column(validation, "orders", "seen_at")
    assert seen_at.inferred_type == "timestamp_ntz"
    assert (seen_at.outcome, seen_at.detail) == ("warning", "type widened")
    assert seen_at.resolved_type == "timestamp"


def test_a_declaration_narrower_than_an_inferred_family_wins(tmp_path: Path) -> None:
    validation = _validate(
        tmp_path,
        "select id\nfrom source_table\nwhere qty > 1\n",
        _upstream("qty bigint"),
    )

    qty = _column(validation, "source_table", "qty")
    assert (qty.outcome, qty.detail) == ("pass", "type narrowed")
    assert qty.resolved_type == "integer"


def test_incompatible_types_are_an_error(tmp_path: Path) -> None:
    validation = _validate(
        tmp_path,
        "select id\nfrom source_table\nwhere revenue like 'A%'\n",
        _upstream("revenue decimal(10,2)"),
    )

    revenue = _column(validation, "source_table", "revenue")
    assert revenue.outcome == "error"
    assert revenue.detail == "inferred type differs from declared type"
    # Nothing can be generated for a column whose two sides contradict each other.
    assert revenue.resolved_type is None
    assert validation.has_errors


def test_an_error_carries_the_evidence_and_the_declaration(tmp_path: Path) -> None:
    validation = _validate(
        tmp_path,
        "select id\nfrom source_table\nwhere revenue like 'A%'\n",
        _upstream("revenue decimal(10,2)"),
    )

    [(table, revenue)] = validation.errors()
    assert table.name == "source_table"
    assert revenue.inferred is not None and revenue.inferred.chosen is not None
    assert revenue.inferred.chosen.detail == "like"
    assert validation.source.slice(revenue.inferred.location) == "revenue like 'A%'"
    assert revenue.declaration is not None
    assert revenue.declaration.snippet == "decimal(10,2)"
    assert revenue.declaration.path == tmp_path / "schema.yml"


# ---- the cases that must never be an error -------------------------------------------


def test_a_name_pattern_never_contradicts_a_declaration(tmp_path: Path) -> None:
    # `*_at` says timestamp, the yml says varchar. A suffix convention is not evidence
    # worth calling a user's written type wrong over.
    validation = _validate(
        tmp_path,
        "select created_at\nfrom source_table\n",
        _upstream("created_at varchar(30)"),
    )

    created_at = _column(validation, "source_table", "created_at")
    assert (created_at.outcome, created_at.detail) == ("pass", "declared only")
    assert created_at.resolved_type == "string"


def test_a_name_pattern_backed_by_usage_still_contradicts(tmp_path: Path) -> None:
    # The suffix is no longer the only witness, so the disagreement is real.
    validation = _validate(
        tmp_path,
        "select id\nfrom source_table\nwhere created_at like 'A%'\n",
        _upstream("created_at timestamp"),
    )

    created_at = _column(validation, "source_table", "created_at")
    assert created_at.outcome == "error"


def test_an_untyped_column_leaves_the_declaration_standing(tmp_path: Path) -> None:
    validation = _validate(
        tmp_path,
        "select nickname\nfrom source_table\n",
        _upstream("nickname varchar(10)"),
    )

    nickname = _column(validation, "source_table", "nickname")
    assert nickname.inferred_type == "unknown"
    assert (nickname.outcome, nickname.detail) == ("pass", "declared only")
    assert nickname.resolved_type == "string"


def test_a_declared_column_the_sql_never_mentions_resolves_to_the_declaration(
    tmp_path: Path,
) -> None:
    validation = _validate(
        tmp_path,
        "select *\nfrom source_table\n",
        _upstream("revenue decimal(10,2)"),
    )

    revenue = _column(validation, "source_table", "revenue")
    assert (revenue.outcome, revenue.detail) == ("pass", "declared only")
    # Absent, not `unknown`: inference never saw the column to fail on it.
    assert revenue.inferred is None
    assert revenue.inferred_type is None
    # Nothing opposed the declaration, so it is what gets generated.
    assert revenue.resolved_type == "decimal"
    assert revenue.declared_type == "decimal(10,2)"


def test_an_undeclared_column_is_a_warning(tmp_path: Path) -> None:
    validation = _validate(
        tmp_path,
        "select id\nfrom source_table\nwhere revenue > 1000\n",
        _upstream("other_column varchar"),
    )

    revenue = _column(validation, "source_table", "revenue")
    assert (revenue.outcome, revenue.detail) == ("warning", "no declaration")
    assert revenue.resolved_type == "numeric"


def test_a_table_with_no_yml_at_all_warns_per_column(tmp_path: Path) -> None:
    validation = _validate(
        tmp_path,
        "select id\nfrom source_table\nwhere revenue > 1000\n",
        _yml("something_else"),
    )

    [table] = validation.tables
    assert table.declared_model is None
    assert {column.detail for column in table.columns} == {"no declaration"}


def test_an_unrecognized_declared_type_is_a_warning(tmp_path: Path) -> None:
    validation = _validate(
        tmp_path,
        "select id\nfrom source_table\nwhere region = 'US'\n",
        _upstream("region geography"),
    )

    region = _column(validation, "source_table", "region")
    assert (region.outcome, region.detail) == ("warning", "unrecognized declaration")
    assert region.resolved_type is None


def test_an_unrecognized_declared_type_wins_over_declared_only(tmp_path: Path) -> None:
    # The column is never referenced, but the declaration is unusable either way.
    validation = _validate(
        tmp_path,
        "select id\nfrom source_table\n",
        _upstream("region geography"),
    )

    region = _column(validation, "source_table", "region")
    assert (region.outcome, region.detail) == ("warning", "unrecognized declaration")


# ---- timestamps ----------------------------------------------------------------------


def test_timestamp_ntz_narrows_an_inferred_timestamp(tmp_path: Path) -> None:
    validation = _validate(
        tmp_path,
        "select id\nfrom source_table\nwhere updated_at = current_timestamp\n",
        _upstream("updated_at timestamp_ntz"),
    )

    updated_at = _column(validation, "source_table", "updated_at")
    assert updated_at.inferred_type == "timestamp"
    assert (updated_at.outcome, updated_at.detail) == ("pass", "type narrowed")
    assert updated_at.resolved_type == "timestamp_ntz"


def test_a_bare_timestamp_declaration_is_an_exact_match(tmp_path: Path) -> None:
    validation = _validate(
        tmp_path,
        "select id\nfrom source_table\nwhere updated_at = current_timestamp\n",
        _upstream("updated_at timestamp"),
    )

    updated_at = _column(validation, "source_table", "updated_at")
    assert (updated_at.outcome, updated_at.detail) == ("pass", "exact match")


def test_a_zoned_declaration_never_collides_with_an_unzoned_one(
    tmp_path: Path,
) -> None:
    # Both are timestamps, but not the same one, and generating either for the other is
    # wrong - which is why the family has concrete members rather than being one type.
    validation = _validate(
        tmp_path,
        "select cast(seen_at as timestamp_ntz) as seen_at\nfrom source_table\n",
        _yml("orders", "seen_at timestamptz", sql_file="orders"),
    )

    seen_at = _column(validation, "orders", "seen_at")
    assert seen_at.inferred_type == "timestamp_ntz"
    assert seen_at.outcome == "error"


# ---- the projection ------------------------------------------------------------------


def test_the_projection_is_checked_against_the_models_own_declaration(
    tmp_path: Path,
) -> None:
    validation = _validate(
        tmp_path,
        "select cast(total as integer) as total\nfrom source_table\n",
        _yml("orders", "total varchar(10)", sql_file="orders"),
    )

    assert validation.projection.declared_model == "warehouse.orders"
    [(_, total)] = validation.errors()
    assert total.name == "total"
    assert total.ordinal == 0
    assert total.inferred_type == "integer"


def test_a_declared_column_the_projection_omits_is_an_error(tmp_path: Path) -> None:
    # The declaration says what this file produces. A column it names that the SELECT list
    # does not produce is a claim the SQL contradicts, and neither side is guessable.
    validation = _validate(
        tmp_path,
        "select id\nfrom source_table\n",
        _yml("orders", "id bigint", "total decimal(10,2)", sql_file="orders"),
    )

    total = _column(validation, "orders", "total")
    assert (total.outcome, total.detail) == ("error", "declared but not projected")
    assert total.inferred is None
    assert total.ordinal is None
    # Still the type to generate, if the SQL is what gets fixed.
    assert total.resolved_type == "decimal"
    assert validation.has_errors


def test_an_unexpandable_star_softens_a_missing_projection_column(
    tmp_path: Path,
) -> None:
    # `select *` over a table whose columns cannot be enumerated: the SELECT list is a
    # lower bound on the output, so the declared column may well be produced.
    validation = _validate(
        tmp_path,
        "select *\nfrom source_table\n",
        _yml("orders", "total decimal(10,2)", sql_file="orders"),
    )

    total = _column(validation, "orders", "total")
    assert (total.outcome, total.detail) == (
        "warning",
        "declared with no explicit projection",
    )
    assert total.resolved_type == "decimal"
    assert not validation.has_errors


def test_a_star_over_a_cte_still_makes_a_missing_column_an_error(
    tmp_path: Path,
) -> None:
    # The `*` expands here - the CTE's outputs are known - so the list is the whole output
    # and the declaration names something that really is not produced.
    validation = _validate(
        tmp_path,
        "with rows as (select id from source_table)\nselect * from rows\n",
        _yml("orders", "total decimal(10,2)", sql_file="orders"),
    )

    total = _column(validation, "orders", "total")
    assert (total.outcome, total.detail) == ("error", "declared but not projected")


def test_a_column_the_projection_declares_nothing_about_is_only_a_warning(
    tmp_path: Path,
) -> None:
    # The asymmetry: an undeclared projected column is a hole, not a contradiction.
    validation = _validate(
        tmp_path,
        "select id, total\nfrom source_table\n",
        _yml("orders", "id bigint", sql_file="orders"),
    )

    total = _column(validation, "orders", "total")
    assert (total.outcome, total.detail) == ("warning", "no declaration")
    assert not validation.has_errors


def test_a_source_declaration_the_sql_never_mentions_is_not_an_error(
    tmp_path: Path,
) -> None:
    # The same absence on the other side of the comparison. Selecting four of a table's
    # twelve columns is normal.
    validation = _validate(
        tmp_path,
        "select id\nfrom source_table\n",
        _upstream("revenue decimal(10,2)"),
    )

    revenue = _column(validation, "source_table", "revenue")
    assert (revenue.outcome, revenue.detail) == ("pass", "declared only")
    assert revenue.resolved_type == "decimal"


def test_an_unrecognized_type_does_not_hide_a_missing_projection_column(
    tmp_path: Path,
) -> None:
    validation = _validate(
        tmp_path,
        "select id\nfrom source_table\n",
        _yml("orders", "total geography", sql_file="orders"),
    )

    total = _column(validation, "orders", "total")
    assert (total.outcome, total.detail) == ("error", "declared but not projected")
    # Unusable either way, so there is nothing to generate.
    assert total.resolved_type is None
