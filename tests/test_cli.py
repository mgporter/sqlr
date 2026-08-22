from pathlib import Path

import pytest
from typer.testing import CliRunner

from sqlr.cli import app
from sqlr.config import CONFIG_FILENAME

runner = CliRunner()


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

@pytest.fixture
def wide(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rich wraps to the terminal, and a wrapped row breaks a substring assertion."""
    monkeypatch.setenv("COLUMNS", "200")




def test_validate_schema_passes_when_nothing_is_declared(tmp_path: Path) -> None:
    _project(tmp_path)

    result = runner.invoke(
        app,
        ["validate-schema", "--project-dir", str(tmp_path), "--select", "customers"],
    )

    assert result.exit_code == 0



INCOMPLETE_DECLARATION_YML = """\
version: 2
sources:
  - name: warehouse
    tables:
      - name: raw_address
        columns:
          - name: person_id
            data_type: varchar(20)
"""

READS_MORE_THAN_IS_DECLARED = """\
with ranked as (select * from raw_address)
select person_id, street from ranked
"""


def test_validate_schema_reports_why_a_model_failed_to_qualify(
    tmp_path: Path, wide: None
) -> None:
    """The regression this whole change exists for.

    Steps 1-3 produce the findings and `validate-schema` carried them on the result without
    ever rendering them, so a model that failed qualification printed `not typed` and
    exited 1 with nothing else on screen. Everything a reader needs is here: what failed,
    where in the SQL, and which yml entry made it a failure.
    """
    _project(tmp_path)
    (tmp_path / "models" / "addresses.sql").write_text(READS_MORE_THAN_IS_DECLARED)
    (tmp_path / "models" / "schema.yml").write_text(INCOMPLETE_DECLARATION_YML)

    result = runner.invoke(
        app,
        ["validate-schema", "--project-dir", str(tmp_path), "--select", "addresses"],
    )

    assert result.exit_code == 1
    assert "found undeclared column 'street' in CTE 'ranked'" in result.output
    # The hop the reader cannot see: `ranked` is closed only because `raw_address` is, so
    # the fix names the table and not the CTE the column failed against.
    assert "Declare it on 'raw_address' at models/schema.yml:5" in result.output
    assert "not typed - fix the errors above" in result.output


def test_validate_schema_errors_on_an_unknown_model(tmp_path: Path) -> None:
    _project(tmp_path)

    result = runner.invoke(
        app, ["validate-schema", "--project-dir", str(tmp_path), "--select", "nope"]
    )

    assert result.exit_code == 1
    assert "no model named 'nope'" in result.output



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



