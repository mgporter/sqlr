"""Divergence between inferred and declared types.

The assertions here are mostly about *locations*, because that is what the eventual
editor integration consumes. A diagnostic that is right about the type but points at the
wrong range is worse than useless.
"""

from pathlib import Path

from sqlr.declared import load_declared_schemas
from sqlr.declared.types import DeclaredSchemas
from sqlr.catalog import find_yaml_files
from sqlr.diagnostics import (
    Diagnostic,
    DiagnosticReport,
    check_schema,
    codes,
    render_text,
    to_lsp,
    unresolved_types,
)
from sqlr.schema_resolution import resolve_schema
from sqlr.sql_analysis import analyze_file


def _project(tmp_path: Path, sql: str, yml: str) -> tuple[Path, DeclaredSchemas]:
    (tmp_path / "orders.sql").write_text(sql)
    (tmp_path / "schema.yml").write_text(yml)
    return tmp_path / "orders.sql", load_declared_schemas(find_yaml_files(tmp_path))


def _check(tmp_path: Path, sql: str, yml: str) -> list[Diagnostic]:
    path, declared = _project(tmp_path, sql, yml)
    schema = resolve_schema(analyze_file(path))
    return check_schema(schema, declared, path)



def _yml(name: str, *columns: str, source_file: str | None = None) -> str:
    """A declaration for one relation, in the shape a standalone project writes.

    Everything is a `sources:` table, including what this project builds: those name the
    file that builds them with `meta.source_file`. `warehouse` declares no database and no
    schema, so its tables are referred to bare - `source_table`, not `db.schema.x`.
    """
    lines = [
        "version: 2",
        "sources:",
        "  - name: warehouse",
        "    tables:",
        f"      - name: {name}",
    ]
    if source_file is not None:
        lines.append("        config:")
        lines.append("          meta:")
        lines.append(f"            source_file: {source_file}")
    if columns:
        lines.append("        columns:")
        for column in columns:
            column_name, _, data_type = column.partition(" ")
            lines.append(f"          - name: {column_name}")
            if data_type:
                lines.append(f"            data_type: {data_type}")
    return "\n".join(lines) + "\n"


UPSTREAM_YML = _yml("source_table", "revenue varchar")


# ---- type mismatch -------------------------------------------------------------------


def test_a_string_declaration_contradicted_by_numeric_usage_is_reported(
    tmp_path: Path,
) -> None:
    diagnostics = _check(
        tmp_path,
        "select id\nfrom source_table\nwhere revenue > 1000\n",
        UPSTREAM_YML,
    )

    [mismatch] = [d for d in diagnostics if d.code == codes.TYPE_MISMATCH]
    assert mismatch.severity == "warning"
    assert mismatch.column == "revenue"
    assert mismatch.location.snippet == "revenue > 1000"
    assert mismatch.location.span is not None
    assert mismatch.location.span.start_line == 2


def test_a_cast_contradicting_a_declaration_is_an_error(tmp_path: Path) -> None:
    # A cast names a type outright, so disagreeing with it is not a matter of opinion.
    diagnostics = _check(
        tmp_path,
        "select cast(x as integer) as revenue\nfrom source_table\n",
        _yml("orders", "revenue varchar", source_file="orders"),
    )

    [mismatch] = [d for d in diagnostics if d.code == codes.TYPE_MISMATCH]
    assert mismatch.severity == "error"


def test_a_name_pattern_contradicting_a_declaration_is_only_a_hint(
    tmp_path: Path,
) -> None:
    diagnostics = _check(
        tmp_path,
        "select created_at\nfrom source_table\n",
        _yml("source_table", "created_at integer"),
    )

    [mismatch] = [d for d in diagnostics if d.code == codes.TYPE_MISMATCH]
    assert mismatch.severity == "hint"


def test_a_declaration_narrower_than_the_inferred_family_is_not_a_mismatch(
    tmp_path: Path,
) -> None:
    # `revenue > 1000` infers `numeric`, deliberately a family. Declaring `integer` is
    # more specific, not contradictory, and the user is the authority on which.
    diagnostics = _check(
        tmp_path,
        "select id from source_table where revenue > 1000\n",
        _yml("source_table", "revenue integer"),
    )

    assert [d for d in diagnostics if d.code == codes.TYPE_MISMATCH] == []


def test_an_agreeing_declaration_produces_nothing(tmp_path: Path) -> None:
    diagnostics = _check(
        tmp_path,
        "select id from source_table where revenue in ('a', 'b')\n",
        UPSTREAM_YML,
    )

    assert [d for d in diagnostics if d.code == codes.TYPE_MISMATCH] == []


def test_an_untyped_column_is_not_a_mismatch(tmp_path: Path) -> None:
    diagnostics = _check(
        tmp_path, "select notes from source_table\n", UPSTREAM_YML
    )

    assert [d for d in diagnostics if d.code == codes.TYPE_MISMATCH] == []


# ---- related locations ---------------------------------------------------------------


def test_every_other_piece_of_evidence_becomes_a_related_location(
    tmp_path: Path,
) -> None:
    # The whole reason schema resolution keeps its losing evidence.
    diagnostics = _check(
        tmp_path,
        "select id\nfrom source_table\nwhere revenue > 1000\n  and revenue * 2 > 4\n",
        UPSTREAM_YML,
    )

    [mismatch] = [d for d in diagnostics if d.code == codes.TYPE_MISMATCH]
    messages = [r.message for r in mismatch.related]
    assert any("declared as varchar here" in m for m in messages)
    assert any("also used as" in m for m in messages)

    declaration = next(r for r in mismatch.related if "declared as" in r.message)
    assert declaration.snippet == "varchar"
    assert declaration.path is not None and declaration.path.name == "schema.yml"


def test_the_same_evidence_is_not_related_to_itself(tmp_path: Path) -> None:
    diagnostics = _check(
        tmp_path,
        "select id from source_table where revenue > 1000\n",
        UPSTREAM_YML,
    )

    [mismatch] = [d for d in diagnostics if d.code == codes.TYPE_MISMATCH]
    assert [r.message for r in mismatch.related] == ["declared as varchar here"]


def test_related_locations_span_two_files(tmp_path: Path) -> None:
    diagnostics = _check(
        tmp_path,
        "select id from source_table where revenue > 1000\n",
        UPSTREAM_YML,
    )

    [mismatch] = [d for d in diagnostics if d.code == codes.TYPE_MISMATCH]
    assert mismatch.location.path is not None
    assert mismatch.location.path.name == "orders.sql"
    assert mismatch.related[0].path is not None
    assert mismatch.related[0].path.name == "schema.yml"


# ---- coverage ------------------------------------------------------------------------


def test_an_unrecognised_declared_type_is_reported(tmp_path: Path) -> None:
    diagnostics = _check(
        tmp_path,
        "select revenue from source_table\n",
        _yml("source_table", "revenue blorp"),
    )

    [unknown] = [d for d in diagnostics if d.code == codes.UNKNOWN_DECLARED_TYPE]
    assert unknown.severity == "warning"
    assert unknown.location.snippet == "blorp"


def test_a_column_the_declaration_omits_is_reported(tmp_path: Path) -> None:
    diagnostics = _check(
        tmp_path,
        "select revenue, surprise from source_table\n",
        UPSTREAM_YML,
    )

    [undeclared] = [d for d in diagnostics if d.code == codes.UNDECLARED_COLUMN]
    assert undeclared.column == "surprise"
    assert undeclared.severity == "info"


def test_a_star_suppresses_undeclared_column_reports(tmp_path: Path) -> None:
    # Under a `*` the attributed column list is a subset of the real one, so absence
    # from it proves nothing.
    diagnostics = _check(tmp_path, "select * from source_table\n", UPSTREAM_YML)

    assert [d for d in diagnostics if d.code == codes.UNDECLARED_COLUMN] == []


def test_a_declared_column_the_model_does_not_produce_is_reported(
    tmp_path: Path,
) -> None:
    diagnostics = _check(
        tmp_path,
        "select id from source_table\n",
        _yml("orders", "id integer", "absent integer", source_file="orders"),
    )

    [missing] = [d for d in diagnostics if d.code == codes.MISSING_COLUMN]
    assert missing.column == "absent"
    assert missing.severity == "hint"


def test_upstream_declarations_do_not_produce_missing_column_noise(
    tmp_path: Path,
) -> None:
    # `source_table` declares two columns and this file selects one. That is normal.
    diagnostics = _check(
        tmp_path,
        "select revenue from source_table\n",
        _yml("source_table", "revenue varchar", "other varchar"),
    )

    assert [d for d in diagnostics if d.code == codes.MISSING_COLUMN] == []


def test_an_untyped_undeclared_column_is_reported(tmp_path: Path) -> None:
    path, declared = _project(tmp_path, "select notes from source_table\n", UPSTREAM_YML)
    schema = resolve_schema(analyze_file(path))

    [unresolved] = unresolved_types(schema, declared)

    assert unresolved.code == codes.UNRESOLVED_TYPE
    assert unresolved.column == "notes"


def test_a_declared_column_is_not_reported_as_untyped(tmp_path: Path) -> None:
    path, declared = _project(
        tmp_path, "select revenue from source_table\n", UPSTREAM_YML
    )
    schema = resolve_schema(analyze_file(path))

    assert unresolved_types(schema, declared) == []


# ---- rendering -----------------------------------------------------------------------


def test_text_rendering_underlines_the_offending_code(tmp_path: Path) -> None:
    diagnostics = _check(
        tmp_path,
        "select id\nfrom source_table\nwhere revenue > 1000\n",
        UPSTREAM_YML,
    )
    rendered = render_text(DiagnosticReport(diagnostics=diagnostics))

    assert "revenue > 1000" in rendered
    assert "^^^^^^^^^^^^^^" in rendered
    assert "declared as varchar here" in rendered


def test_an_empty_report_renders_as_nothing() -> None:
    assert render_text(DiagnosticReport()) == ""


def test_lsp_ranges_are_zero_based_and_slice_back_to_the_source(
    tmp_path: Path,
) -> None:
    path, declared = _project(
        tmp_path,
        "select id\nfrom source_table\nwhere revenue > 1000\n",
        UPSTREAM_YML,
    )
    schema = resolve_schema(analyze_file(path))
    report = DiagnosticReport(diagnostics=check_schema(schema, declared, path))

    [entry] = [e for e in to_lsp(report) if e["code"] == codes.TYPE_MISMATCH]

    assert entry["severity"] == 2  # LSP warning
    assert entry["range"]["start"] == {"line": 2, "character": 6}
    assert entry["range"]["end"] == {"line": 2, "character": 20}

    # The range has to select exactly the offending text in the real file.
    start_line: int = entry["range"]["start"]["line"]
    start: int = entry["range"]["start"]["character"]
    end: int = entry["range"]["end"]["character"]
    line = path.read_text().splitlines()[start_line]
    assert line[start:end] == "revenue > 1000"


def test_lsp_carries_related_information(tmp_path: Path) -> None:
    path, declared = _project(
        tmp_path, "select id from source_table where revenue > 1000\n", UPSTREAM_YML
    )
    schema = resolve_schema(analyze_file(path))
    report = DiagnosticReport(diagnostics=check_schema(schema, declared, path))

    [entry] = [e for e in to_lsp(report) if e["code"] == codes.TYPE_MISMATCH]

    assert entry["relatedInformation"][0]["message"] == "declared as varchar here"
    assert entry["relatedInformation"][0]["location"]["uri"].endswith("schema.yml")


def test_reports_sort_the_worst_problem_first() -> None:
    from sqlr.diagnostics.types import Diagnostic

    report = DiagnosticReport(
        diagnostics=[
            Diagnostic(code="a", severity="hint", message="h"),
            Diagnostic(code="b", severity="error", message="e"),
            Diagnostic(code="c", severity="warning", message="w"),
        ]
    )

    assert [d.severity for d in report.sorted()] == ["error", "warning", "hint"]
    assert report.has_errors
