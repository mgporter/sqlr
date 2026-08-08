from pathlib import Path

import pytest

from sqlr.config.types import GeneralConfig, SqlrConfig
from sqlr.selection import (
    SelectionError,
    build_model_index,
    model_search_paths,
    select_models,
)


def _config(**general: object) -> SqlrConfig:
    return SqlrConfig(general=GeneralConfig(**general))  # type: ignore[arg-type]


def _model(root: Path, relative: str, sql: str = "select 1\n") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sql)
    return path


def test_index_finds_models_in_subfolders(tmp_path: Path) -> None:
    _model(tmp_path, "models/staging/orders.sql")
    _model(tmp_path, "models/marts/customers.sql")

    index = build_model_index(tmp_path, _config())

    assert index.names == ["customers", "orders"]
    assert index.search_paths == [tmp_path.resolve()]


def test_index_is_narrowed_by_model_paths(tmp_path: Path) -> None:
    _model(tmp_path, "models/orders.sql")
    _model(tmp_path, "scratch/orders_backup.sql")

    index = build_model_index(tmp_path, _config(model_paths=["models"]))

    assert index.names == ["orders"]


def test_model_paths_accepts_a_bare_string(tmp_path: Path) -> None:
    _model(tmp_path, "models/orders.sql")
    _model(tmp_path, "scratch/other.sql")

    index = build_model_index(tmp_path, _config(model_paths="models"))

    assert index.names == ["orders"]


def test_model_paths_narrows_away_a_would_be_duplicate(tmp_path: Path) -> None:
    _model(tmp_path, "models/orders.sql")
    _model(tmp_path, "scratch/orders.sql")

    index = build_model_index(tmp_path, _config(model_paths=["models"]))

    assert index.names == ["orders"]


def test_overlapping_model_paths_do_not_duplicate_a_file(tmp_path: Path) -> None:
    _model(tmp_path, "models/staging/orders.sql")

    index = build_model_index(
        tmp_path, _config(model_paths=["models", "models/staging"])
    )

    assert index.names == ["orders"]


def test_missing_model_path_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SelectionError, match="model path not found"):
        model_search_paths(tmp_path, _config(model_paths=["models"]))


def test_model_path_outside_the_project_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "outside_project").mkdir()
    project = tmp_path / "project"
    project.mkdir()

    with pytest.raises(SelectionError, match="outside the project root"):
        model_search_paths(project, _config(model_paths=["../outside_project"]))


def test_duplicate_model_names_are_rejected(tmp_path: Path) -> None:
    _model(tmp_path, "models/staging/orders.sql")
    _model(tmp_path, "models/marts/orders.sql")

    with pytest.raises(SelectionError) as excinfo:
        build_model_index(tmp_path, _config())

    message = str(excinfo.value)
    assert "found 2 models named 'orders'" in message
    assert "models/staging/orders.sql" in message
    assert "models/marts/orders.sql" in message


def test_no_selector_selects_every_model(tmp_path: Path) -> None:
    _model(tmp_path, "a.sql")
    _model(tmp_path, "b.sql")
    index = build_model_index(tmp_path, _config())

    assert [model.name for model in select_models(index, None)] == ["a", "b"]
    assert [model.name for model in select_models(index, [])] == ["a", "b"]


def test_selectors_pick_models_in_the_order_given(tmp_path: Path) -> None:
    for name in ("a", "b", "c"):
        _model(tmp_path, f"{name}.sql")
    index = build_model_index(tmp_path, _config())

    selected = select_models(index, ["c", "a"])

    assert [model.name for model in selected] == ["c", "a"]


def test_selectors_are_split_and_deduplicated(tmp_path: Path) -> None:
    _model(tmp_path, "a.sql")
    _model(tmp_path, "b.sql")
    index = build_model_index(tmp_path, _config())

    selected = select_models(index, ["a b", "b,a", "a.sql"])

    assert [model.name for model in selected] == ["a", "b"]


def test_unknown_selectors_are_reported_together(tmp_path: Path) -> None:
    _model(tmp_path, "a.sql")
    index = build_model_index(tmp_path, _config())

    with pytest.raises(SelectionError) as excinfo:
        select_models(index, ["nope", "also_nope"])

    message = str(excinfo.value)
    assert "'nope'" in message
    assert "'also_nope'" in message
    assert "a" in message


def test_gitignored_models_are_not_indexed(tmp_path: Path) -> None:
    _model(tmp_path, "models/orders.sql")
    _model(tmp_path, "target/orders.sql")
    (tmp_path / ".gitignore").write_text("target/\n")

    index = build_model_index(tmp_path, _config())

    assert index.names == ["orders"]
