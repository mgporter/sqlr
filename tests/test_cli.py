import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sqlr.cli import app
from sqlr.config import CONFIG_FILENAME

runner = CliRunner()


def test_check_runs_from_cwd_with_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / CONFIG_FILENAME).write_text("version: 1\n")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["check"])
    assert result.exit_code == 0


def test_check_errors_from_cwd_without_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["check"])
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
