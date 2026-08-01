from pathlib import Path

import pytest
from typer.testing import CliRunner

from sqlrunner.cli import app
from sqlrunner.config import CONFIG_FILENAME

runner = CliRunner()


def test_main_runs_from_cwd_with_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / CONFIG_FILENAME).write_text("version: 1\n")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app)
    assert result.exit_code == 0


def test_main_errors_from_cwd_without_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app)
    assert result.exit_code == 1
