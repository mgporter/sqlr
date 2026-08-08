"""Turning `--select` into a set of files.

Every command that operates on models - inferring a schema, generating fixtures, checking
- takes the same selectors and has to resolve them the same way, so resolution lives here
rather than in any one command.

The addressing scheme is dbt's: a model is named by its bare filename, so `orders` finds
`models/marts/orders.sql` wherever it sits. That only works if the name is unique, and the
whole search space is indexed up front to prove that it is. Two `orders.sql` in different
folders is an error, not a coin flip - the alternative is a command silently operating on
the wrong file.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from sqlr.catalog import find_sql_files
from sqlr.config.types import SqlrConfig
from sqlr.selection.types import Model, ModelIndex, SelectionError

__all__ = [
    "Model",
    "ModelIndex",
    "SelectionError",
    "build_model_index",
    "model_search_paths",
    "select_models",
]

MODEL_SUFFIX = ".sql"
_SEPARATORS = re.compile(r"[,\s]+")


def model_search_paths(project_root: Path, cfg: SqlrConfig) -> list[Path]:
    """The directories models are looked for in.

    `model_paths` narrows the search; without it the whole project is the search space.
    A configured path that is missing is an error rather than an empty result, because a
    typo in `sqlr.yml` and a project with no models look identical otherwise.
    """
    configured = cfg.general.model_paths
    if not configured:
        return [project_root]

    paths: list[Path] = []
    for entry in configured:
        path = (project_root / entry).resolve()
        if not path.is_dir():
            raise SelectionError(f"model path not found: {entry} (in {project_root})")
        if project_root not in path.parents and path != project_root:
            raise SelectionError(
                f"model path is outside the project root: {entry} (in {project_root})"
            )
        if path not in paths:
            paths.append(path)
    return paths


def build_model_index(project_root: Path, cfg: SqlrConfig) -> ModelIndex:
    """Index every model under the search paths, by name.

    Discovery goes through the catalog from the project root - so gitignore rules and the
    configured globs apply exactly as they do everywhere else - and the result is then
    narrowed to the search paths. Raises `SelectionError` on a duplicate name.
    """
    project_root = Path(project_root).expanduser().resolve()
    search_paths = model_search_paths(project_root, cfg)

    inventory = find_sql_files(project_root, cfg.general.sql_file_globs)

    seen: set[Path] = set()
    models: list[Model] = []
    for sql_file in inventory.files:
        path = sql_file.path.resolve()
        if path in seen or not _within(path, search_paths):
            continue
        seen.add(path)
        models.append(Model(name=path.stem, file=sql_file))

    models.sort(key=lambda model: model.name)
    _reject_duplicates(models, project_root)

    return ModelIndex(
        project_root=project_root, search_paths=search_paths, models=models
    )


def select_models(index: ModelIndex, selectors: list[str] | None) -> list[Model]:
    """The models a run should operate on. No selectors means all of them.

    Selectors are model names; a `.sql` suffix is tolerated because it is what a shell's
    tab completion produces. Every unknown name is reported at once - fixing one typo per
    run is a poor way to spend a person's afternoon.
    """
    names = _normalize(selectors)
    if not names:
        return list(index.models)

    selected: list[Model] = []
    unknown: list[str] = []
    for name in names:
        model = index.get(name)
        if model is None:
            unknown.append(name)
        else:
            selected.append(model)

    if unknown:
        raise SelectionError(_unknown_message(unknown, index))
    return selected


def _normalize(selectors: list[str] | None) -> list[str]:
    """Flatten, split and de-duplicate, preserving the order given.

    `--select "a b"` and `--select a --select b` are the same request, and both are
    written in the wild.
    """
    names: list[str] = []
    for selector in selectors or []:
        for part in _SEPARATORS.split(selector.strip()):
            if not part:
                continue
            name = part[: -len(MODEL_SUFFIX)] if part.endswith(MODEL_SUFFIX) else part
            if name and name not in names:
                names.append(name)
    return names


def _within(path: Path, search_paths: list[Path]) -> bool:
    return any(path == root or root in path.parents for root in search_paths)


def _reject_duplicates(models: list[Model], project_root: Path) -> None:
    """Two models of the same name make every selector ambiguous, so fail the whole run.

    dbt does the same, and for the same reason: the name is the only handle a selector
    has, so the project has to be fixed before anything can act on it.
    """
    by_name: dict[str, list[Model]] = defaultdict(list)
    for model in models:
        by_name[model.name].append(model)

    collisions = {name: found for name, found in by_name.items() if len(found) > 1}
    if not collisions:
        return

    blocks: list[str] = []
    for name, found in sorted(collisions.items()):
        paths = "\n".join(f"  - {model.relative_path}" for model in found)
        blocks.append(f"found {len(found)} models named '{name}':\n{paths}")

    raise SelectionError(
        "\n\n".join(blocks)
        + f"\n\nModel names must be unique within {project_root}."
        + "\nRename one of them, or narrow `model_paths` in the config."
    )


def _unknown_message(unknown: list[str], index: ModelIndex) -> str:
    names = ", ".join(f"'{name}'" for name in unknown)
    where = ", ".join(_relative(path, index.project_root) for path in index.search_paths)
    plural = "s" if len(unknown) > 1 else ""
    message = f"no model{plural} named {names} under {where}"
    if index.models:
        message += f"\n{len(index.models)} models available: {', '.join(index.names)}"
    else:
        message += "\nno models found at all"
    return message


def _relative(path: Path, project_root: Path) -> str:
    if path == project_root:
        return str(project_root)
    return path.relative_to(project_root).as_posix()
