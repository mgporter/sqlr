from pathlib import Path

from sqlr.catalog import find_sql_files


def _write(path: Path, content: str = "select 1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_finds_files_matching_include_globs(tmp_path: Path) -> None:
    _write(tmp_path / "top.sql")
    _write(tmp_path / "sql" / "nested" / "deep.sql")
    _write(tmp_path / "notes.txt")

    inventory = find_sql_files(tmp_path, ["*.sql", "sql/**/*.sql"])

    assert [f.relative_path for f in inventory.files] == [
        "sql/nested/deep.sql",
        "top.sql",
    ]


def test_exclude_globs_win_over_include_globs(tmp_path: Path) -> None:
    _write(tmp_path / "keep.sql")
    _write(tmp_path / "generated.sql")

    inventory = find_sql_files(tmp_path, ["*.sql"], exclude_globs=["generated.sql"])

    assert [f.relative_path for f in inventory.files] == ["keep.sql"]


def test_honors_gitignore(tmp_path: Path) -> None:
    _write(tmp_path / "keep.sql")
    _write(tmp_path / "target" / "compiled.sql")
    _write(tmp_path / ".gitignore", "target/\n")

    inventory = find_sql_files(tmp_path, ["*.sql", "**/*.sql"])

    assert [f.relative_path for f in inventory.files] == ["keep.sql"]


def test_always_excludes_git_dir(tmp_path: Path) -> None:
    _write(tmp_path / "keep.sql")
    _write(tmp_path / ".git" / "hooks" / "weird.sql")

    inventory = find_sql_files(tmp_path, ["**/*.sql"])

    assert [f.relative_path for f in inventory.files] == ["keep.sql"]


def test_mtime_and_content_hash_populated(tmp_path: Path) -> None:
    _write(tmp_path / "a.sql", "select 1")

    inventory = find_sql_files(tmp_path, ["*.sql"])

    [f] = inventory.files
    assert f.mtime > 0
    assert len(f.content_hash) == 64


def test_content_hash_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "a.sql"
    _write(path, "select 1")
    first = find_sql_files(tmp_path, ["*.sql"]).files[0].content_hash

    _write(path, "select 2")
    second = find_sql_files(tmp_path, ["*.sql"]).files[0].content_hash

    assert first != second


def test_no_gitignore_file_is_fine(tmp_path: Path) -> None:
    _write(tmp_path / "a.sql")

    inventory = find_sql_files(tmp_path, ["*.sql"])

    assert [f.relative_path for f in inventory.files] == ["a.sql"]
