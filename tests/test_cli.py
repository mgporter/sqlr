import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sqlr.cli import app
from sqlr.config import CONFIG_FILENAME

runner = CliRunner()


def test_infer_schema_errors_from_cwd_without_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["infer-schema"])
    assert result.exit_code == 1


def test_validate_schema_errors_from_cwd_without_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["validate-schema"])
    assert result.exit_code == 1


SCHEMA_SQL = """
select p.name, a.city
from person p
join address a on a.person_id = p.id
where a.country = 'US'
  and p.status in ('active', 'on_leave')
  and p.age > 20
  and a.county is null
"""

OTHER_SQL = "select id, sku from product where sku like 'AB%'\n"


def _project(tmp_path: Path, config: str = "version: 1\n") -> Path:
    (tmp_path / CONFIG_FILENAME).write_text(config)
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    (models / "customers.sql").write_text(SCHEMA_SQL)
    (models / "products.sql").write_text(OTHER_SQL)
    return tmp_path


def test_infer_schema_renders_a_table(tmp_path: Path) -> None:
    _project(tmp_path)

    result = runner.invoke(
        app, ["infer-schema", "--project-dir", str(tmp_path), "--select", "customers"]
    )

    assert result.exit_code == 0
    assert "address" in result.output
    assert "person" in result.output
    assert "projection" in result.output
    # The type, the nullability and the constraint all reach the table.
    assert "string" in result.output
    assert "in ('active', 'on_leave')" in result.output
    assert "= 'US'" in result.output
    assert "true" in result.output
    # Only the selected model was rendered.
    assert "product" not in result.output


def test_infer_schema_defaults_to_the_current_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _project(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["infer-schema", "--select", "products"])

    assert result.exit_code == 0
    assert "product" in result.output


def test_infer_schema_without_a_selector_covers_every_model(tmp_path: Path) -> None:
    _project(tmp_path)

    result = runner.invoke(app, ["infer-schema", "--project-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "person" in result.output
    assert "product" in result.output


def test_infer_schema_accepts_space_separated_selectors(tmp_path: Path) -> None:
    _project(tmp_path)

    result = runner.invoke(
        app,
        [
            "infer-schema",
            "--project-dir",
            str(tmp_path),
            "--select",
            "customers",
            "products",
        ],
    )

    assert result.exit_code == 0
    assert "person" in result.output
    assert "product" in result.output


def test_infer_schema_accepts_a_repeated_select_flag(tmp_path: Path) -> None:
    _project(tmp_path)

    result = runner.invoke(
        app,
        [
            "infer-schema",
            "--project-dir",
            str(tmp_path),
            "--select",
            "customers",
            "--select",
            "products",
        ],
    )

    assert result.exit_code == 0
    assert "person" in result.output
    assert "product" in result.output


def test_infer_schema_honours_model_paths(tmp_path: Path) -> None:
    _project(tmp_path, "version: 1\ngeneral:\n  model_paths:\n    - models\n")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "notes.sql").write_text("select 1 as x\n")

    result = runner.invoke(
        app, ["infer-schema", "--project-dir", str(tmp_path), "--select", "notes"]
    )

    assert result.exit_code == 1
    assert "no model named 'notes'" in result.output


def test_infer_schema_errors_on_duplicate_model_names(tmp_path: Path) -> None:
    _project(tmp_path)
    nested = tmp_path / "models" / "staging"
    nested.mkdir()
    (nested / "customers.sql").write_text(SCHEMA_SQL)

    result = runner.invoke(
        app, ["infer-schema", "--project-dir", str(tmp_path), "--select", "customers"]
    )

    assert result.exit_code == 1
    assert "found 2 models named 'customers'" in result.output


def test_infer_schema_json_format(tmp_path: Path) -> None:
    _project(tmp_path)

    result = runner.invoke(
        app,
        [
            "infer-schema",
            "--project-dir",
            str(tmp_path),
            "--select",
            "customers",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert len(payload) == 1
    assert {table["name"] for table in payload[0]["tables"]} == {"person", "address"}


def test_infer_schema_errors_on_an_unknown_model(tmp_path: Path) -> None:
    _project(tmp_path)

    result = runner.invoke(
        app, ["infer-schema", "--project-dir", str(tmp_path), "--select", "nope"]
    )

    assert result.exit_code == 1
    assert "no model named 'nope'" in result.output


def test_infer_schema_errors_without_a_config(tmp_path: Path) -> None:
    result = runner.invoke(app, ["infer-schema", "--project-dir", str(tmp_path)])
    assert result.exit_code == 1


def test_infer_schema_errors_on_unparseable_sql(tmp_path: Path) -> None:
    _project(tmp_path)
    (tmp_path / "models" / "broken.sql").write_text("select from from where;")

    result = runner.invoke(
        app, ["infer-schema", "--project-dir", str(tmp_path), "--select", "broken"]
    )

    assert result.exit_code == 1


# ---- validate-schema -----------------------------------------------------------------

AGREEING_YML = """\
version: 2
sources:
  - name: warehouse
    tables:
      - name: person
        columns:
          - name: age
            data_type: decimal(10,2)
          - name: status
            data_type: varchar(20)
"""

CONTRADICTING_YML = """\
version: 2
sources:
  - name: warehouse
    tables:
      - name: person
        columns:
          - name: age
            data_type: timestamp
"""


@pytest.fixture
def wide(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rich wraps to the terminal, and a wrapped row breaks a substring assertion."""
    monkeypatch.setenv("COLUMNS", "200")


def test_validate_schema_reports_each_column_against_its_declaration(
    tmp_path: Path, wide: None
) -> None:
    _project(tmp_path)
    (tmp_path / "models" / "schema.yml").write_text(AGREEING_YML)

    result = runner.invoke(
        app,
        ["validate-schema", "--project-dir", str(tmp_path), "--select", "customers"],
    )

    assert result.exit_code == 0
    # `age > 20` proves a number; the declaration says which kind.
    assert "decimal(10,2)" in result.output
    assert "type narrowed" in result.output
    assert "exact match" in result.output
    # A column the SQL uses that nothing declares is still worth saying out loud.
    assert "no declaration" in result.output


def test_validate_schema_exits_non_zero_and_explains_a_contradiction(
    tmp_path: Path, wide: None
) -> None:
    _project(tmp_path)
    (tmp_path / "models" / "schema.yml").write_text(CONTRADICTING_YML)

    result = runner.invoke(
        app,
        ["validate-schema", "--project-dir", str(tmp_path), "--select", "customers"],
    )

    assert result.exit_code == 1
    assert "1 error" in result.output
    assert "person.age is declared timestamp" in result.output
    # The comparison that forced the inferred type, quoted and underlined.
    assert "p.age > 20" in result.output
    assert "^" in result.output
    assert "declared here" in result.output


def test_validate_schema_passes_when_nothing_is_declared(tmp_path: Path) -> None:
    _project(tmp_path)

    result = runner.invoke(
        app,
        ["validate-schema", "--project-dir", str(tmp_path), "--select", "customers"],
    )

    assert result.exit_code == 0


def test_validate_schema_json_format(tmp_path: Path) -> None:
    _project(tmp_path)
    (tmp_path / "models" / "schema.yml").write_text(AGREEING_YML)

    result = runner.invoke(
        app,
        [
            "validate-schema",
            "--project-dir",
            str(tmp_path),
            "--select",
            "customers",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    [payload] = json.loads(result.output)
    person = next(t for t in payload["tables"] if t["name"] == "person")
    age = next(c for c in person["columns"] if c["name"] == "age")
    assert age["outcome"] == "pass"
    assert age["detail"] == "type narrowed"
    assert age["resolved_type"] == "decimal"


def test_validate_schema_shares_selection_with_infer_schema(tmp_path: Path) -> None:
    _project(tmp_path)

    result = runner.invoke(
        app, ["validate-schema", "--project-dir", str(tmp_path), "--select", "nope"]
    )

    assert result.exit_code == 1
    assert "no model named 'nope'" in result.output


def test_validate_schema_errors_on_unparseable_sql(tmp_path: Path) -> None:
    _project(tmp_path)
    (tmp_path / "models" / "broken.sql").write_text("select from from where;")

    result = runner.invoke(
        app, ["validate-schema", "--project-dir", str(tmp_path), "--select", "broken"]
    )

    assert result.exit_code == 1


# ---- declarations ----------------------------------------------------------------------


def test_validate_schema_errors_when_two_sources_describe_one_relation(
    tmp_path: Path, wide: None
) -> None:
    _project(tmp_path)
    (tmp_path / "models" / "schema.yml").write_text(AGREEING_YML)
    (tmp_path / "models" / "other.yml").write_text(
        "version: 2\nsources:\n  - name: other\n    tables:\n      - name: person\n"
    )

    result = runner.invoke(
        app,
        ["validate-schema", "--project-dir", str(tmp_path), "--select", "customers"],
    )

    assert result.exit_code == 1
    assert "both describe the relation 'person'" in result.output
    # It failed before comparing anything, so no grid was printed.
    assert "type narrowed" not in result.output


def test_validate_schema_errors_on_a_sources_key_that_is_not_a_list(
    tmp_path: Path, wide: None
) -> None:
    _project(tmp_path)
    (tmp_path / "models" / "schema.yml").write_text(
        "version: 2\nsources:\n  name: warehouse\n  tables:\n    - name: person\n"
    )

    result = runner.invoke(
        app,
        ["validate-schema", "--project-dir", str(tmp_path), "--select", "customers"],
    )

    assert result.exit_code == 1
    assert "`sources:` must be a list of sources" in result.output


def test_validate_schema_errors_when_sql_file_names_nothing(
    tmp_path: Path, wide: None
) -> None:
    _project(tmp_path)
    (tmp_path / "models" / "schema.yml").write_text(
        "version: 2\nsources:\n  - name: warehouse\n    tables:\n"
        "      - name: person\n        sql_file: nowhere\n"
    )

    result = runner.invoke(
        app,
        ["validate-schema", "--project-dir", str(tmp_path), "--select", "customers"],
    )

    assert result.exit_code == 1
    assert "sql_file 'nowhere' does not name a SQL file" in result.output


def test_validate_schema_checks_the_projection_of_a_declared_sql_file(
    tmp_path: Path, wide: None
) -> None:
    _project(tmp_path)
    (tmp_path / "models" / "schema.yml").write_text(
        "version: 2\nsources:\n  - name: warehouse\n    tables:\n"
        "      - name: customers\n        sql_file: customers\n        columns:\n"
        "          - name: city\n            data_type: varchar(40)\n"
    )

    result = runner.invoke(
        app,
        ["validate-schema", "--project-dir", str(tmp_path), "--select", "customers"],
    )

    assert result.exit_code == 0
    assert "varchar(40)" in result.output


def test_validate_schema_warns_that_models_are_ignored_without_dbt(
    tmp_path: Path, wide: None
) -> None:
    _project(tmp_path)
    (tmp_path / "models" / "schema.yml").write_text(
        "version: 2\nmodels:\n  - name: customers\n    columns:\n"
        "      - name: city\n        data_type: varchar(40)\n"
    )

    result = runner.invoke(
        app,
        ["validate-schema", "--project-dir", str(tmp_path), "--select", "customers"],
    )

    assert result.exit_code == 0
    assert "no dbt_project.yml was found" in result.output
    assert "customers (models/schema.yml:3)" in result.output
    # Ignored means ignored: the declaration did not reach the comparison.
    assert "varchar(40)" not in result.output


def test_validate_schema_names_the_relation_an_under_qualified_reference_meant(
    tmp_path: Path, wide: None
) -> None:
    _project(tmp_path)
    (tmp_path / "models" / "schema.yml").write_text(
        "version: 2\nsources:\n  - name: warehouse\n    database: mydb\n"
        "    tables:\n      - name: person\n"
    )

    result = runner.invoke(
        app,
        ["validate-schema", "--project-dir", str(tmp_path), "--select", "customers"],
    )

    assert result.exit_code == 0
    assert "Write mydb.person in the SQL" in result.output
