import hashlib
from pathlib import Path

import pathspec
from pathspec.patterns.gitignore.basic import GitIgnoreBasicPattern

from sqlrunner.catalog.types import FileInventory, SqlFile

GITIGNORE_FILENAME = ".gitignore"
ALWAYS_EXCLUDED_DIRS = {".git"}


def _load_gitignore_spec(project_root: Path) -> pathspec.PathSpec[GitIgnoreBasicPattern] | None:
    gitignore_path = project_root / GITIGNORE_FILENAME
    if not gitignore_path.is_file():
        return None
    lines = gitignore_path.read_text().splitlines()
    return pathspec.PathSpec.from_lines("gitignore", lines)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


DEFAULT_YAML_GLOBS = ["**/*.yml", "**/*.yaml"]


def find_sql_files(
    project_root: Path,
    include_globs: list[str],
    exclude_globs: list[str] | None = None,
) -> FileInventory:
    return find_files(project_root, include_globs, exclude_globs)


def find_yaml_files(
    project_root: Path,
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
) -> FileInventory:
    """Every yml in the project. Deciding which ones *mean* anything is not our job.

    Discovery is kept separate from interpretation so that a dbt project's existing
    `schema.yml` files and a hand-written one are found by the same code.
    """
    return find_files(
        project_root, include_globs or DEFAULT_YAML_GLOBS, exclude_globs
    )


def find_files(
    project_root: Path,
    include_globs: list[str],
    exclude_globs: list[str] | None = None,
) -> FileInventory:
    project_root = Path(project_root).expanduser().resolve()

    matched: set[Path] = set()
    for pattern in include_globs:
        matched.update(p for p in project_root.glob(pattern) if p.is_file())

    excluded: set[Path] = set()
    for pattern in exclude_globs or []:
        excluded.update(p for p in project_root.glob(pattern) if p.is_file())

    gitignore_spec = _load_gitignore_spec(project_root)

    files: list[SqlFile] = []
    for path in matched:
        if path in excluded:
            continue

        relative = path.relative_to(project_root)
        if any(part in ALWAYS_EXCLUDED_DIRS for part in relative.parts):
            continue

        relative_posix = relative.as_posix()
        if gitignore_spec is not None and gitignore_spec.match_file(relative_posix):
            continue

        stat = path.stat()
        files.append(
            SqlFile(
                path=path,
                relative_path=relative_posix,
                mtime=stat.st_mtime,
                content_hash=_hash_file(path),
            )
        )

    files.sort(key=lambda f: f.relative_path)
    return FileInventory(project_root=project_root, files=files)
