from pathlib import Path

import pytest
from pydantic import ValidationError

from sqlrunner.config import CONFIG_FILENAME, ConfigError, load, load_config, resolve_project_root

MINIMAL_CONFIG = """
version: 1

general:
  default_row_count: 10
  seed: 12345
  fixture_directory: fixtures
  sql_file_globs:
    - "*.sql"
    - "sql/**/*.sql"
"""


def _write_config(project_dir: Path, content: str = MINIMAL_CONFIG) -> None:
    (project_dir / CONFIG_FILENAME).write_text(content)


def test_load_finds_and_parses_config(tmp_path: Path) -> None:
    _write_config(tmp_path)

    cfg = load(tmp_path)

    assert cfg.version == 1
    assert cfg.general.default_row_count == 10
    assert cfg.general.seed == 12345
    assert cfg.general.fixture_directory == "fixtures"
    assert cfg.general.sql_file_globs == ["*.sql", "sql/**/*.sql"]


def test_load_ignores_trailing_slash(tmp_path: Path) -> None:
    _write_config(tmp_path)

    with_slash = load(Path(str(tmp_path) + "/"))
    without_slash = load(tmp_path)

    assert with_slash == without_slash


def test_resolve_project_root_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        resolve_project_root(tmp_path / "does_not_exist")


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path)


def test_star_over_join_behavior_defaults_to_guess(tmp_path: Path) -> None:
    _write_config(tmp_path)

    cfg = load(tmp_path)

    assert cfg.general.star_over_join_behavior == "guess"


def test_star_over_join_behavior_is_read_from_config(tmp_path: Path) -> None:
    _write_config(tmp_path, MINIMAL_CONFIG + "  star_over_join_behavior: error\n")

    cfg = load(tmp_path)

    assert cfg.general.star_over_join_behavior == "error"


def test_star_over_join_behavior_rejects_unknown_value(tmp_path: Path) -> None:
    _write_config(tmp_path, MINIMAL_CONFIG + "  star_over_join_behavior: shrug\n")

    with pytest.raises(ValidationError):
        load(tmp_path)
