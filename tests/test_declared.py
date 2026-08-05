"""Loading dbt-shaped model schema yml.

Fixtures are written into `tmp_path` rather than committed, because these tests are about
*discovery* as much as parsing - what gets picked up and what gets ignored depends on the
whole directory, so each test owns a directory.
"""

from pathlib import Path

from sqlrunner.catalog import find_yaml_files
from sqlrunner.declared import load_declared_schemas


def _project(tmp_path: Path, **files: str) -> Path:
    for name, content in files.items():
        path = tmp_path / name.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return tmp_path


SCHEMA = """\
version: 2
models:
  - name: orders
    columns:
      - name: revenue
        data_type: varchar(50)
        description: money
      - name: qty
        data_type: integer
"""


# ---- discovery -----------------------------------------------------------------------


def test_a_yml_without_a_models_key_is_ignored(tmp_path: Path) -> None:
    root = _project(tmp_path, **{"schema.yml": SCHEMA, "ci.yml": "jobs:\n  build: x\n"})

    declared = load_declared_schemas(find_yaml_files(root))

    assert set(declared.models) == {"orders"}
    assert declared.warnings == []


def test_yaml_and_yml_are_both_found(tmp_path: Path) -> None:
    root = _project(
        tmp_path,
        **{
            "a.yml": SCHEMA,
            "b.yaml": "models:\n  - name: second\n    columns:\n"
            "      - name: c\n        data_type: date\n",
        },
    )

    declared = load_declared_schemas(find_yaml_files(root))

    assert set(declared.models) == {"orders", "second"}


def test_a_model_is_matched_to_a_sql_file_by_stem(tmp_path: Path) -> None:
    root = _project(tmp_path, **{"schema.yml": SCHEMA})

    declared = load_declared_schemas(find_yaml_files(root))

    assert declared.for_sql_file(Path("models/staging/orders.sql")) is not None
    assert declared.for_sql_file(Path("models/other.sql")) is None


def test_model_lookup_is_case_insensitive(tmp_path: Path) -> None:
    root = _project(tmp_path, **{"schema.yml": SCHEMA})

    declared = load_declared_schemas(find_yaml_files(root))

    assert declared.for_model("ORDERS") is not None


def test_a_duplicated_model_warns_and_keeps_the_first(tmp_path: Path) -> None:
    root = _project(tmp_path, **{"a.yml": SCHEMA, "b.yml": SCHEMA})

    declared = load_declared_schemas(find_yaml_files(root))

    assert len(declared.models) == 1
    assert len(declared.warnings) == 1
    assert "declared in both" in declared.warnings[0]


# ---- parsing -------------------------------------------------------------------------


def test_data_types_are_mapped_onto_the_lattice(tmp_path: Path) -> None:
    root = _project(tmp_path, **{"schema.yml": SCHEMA})

    model = load_declared_schemas(find_yaml_files(root)).for_model("orders")

    assert model is not None
    assert model.column("revenue") is not None
    assert model.column("revenue").written_type == "varchar(50)"  # type: ignore[union-attr]
    assert model.column("revenue").resolved_type_name == "string"  # type: ignore[union-attr]
    assert model.column("qty").resolved_type_name == "integer"  # type: ignore[union-attr]


def test_an_unrecognised_data_type_resolves_to_unknown(tmp_path: Path) -> None:
    root = _project(
        tmp_path,
        **{
            "schema.yml": "models:\n  - name: t\n    columns:\n"
            "      - name: c\n        data_type: blorp\n"
        },
    )

    model = load_declared_schemas(find_yaml_files(root)).for_model("t")

    assert model is not None
    assert model.column("c").resolved_type_name == "unknown"  # type: ignore[union-attr]


def test_a_column_without_a_data_type_is_skipped_silently(tmp_path: Path) -> None:
    # Documenting a column without typing it is normal in dbt; there is simply nothing
    # to check it against.
    root = _project(
        tmp_path,
        **{
            "schema.yml": "models:\n  - name: t\n    columns:\n"
            "      - name: documented\n        description: hello\n"
            "      - name: typed\n        data_type: date\n"
        },
    )

    declared = load_declared_schemas(find_yaml_files(root))
    model = declared.for_model("t")

    assert model is not None
    assert [c.name for c in model.columns] == ["typed"]
    assert declared.warnings == []


def test_a_column_without_a_name_warns(tmp_path: Path) -> None:
    root = _project(
        tmp_path,
        **{
            "schema.yml": "models:\n  - name: t\n    columns:\n"
            "      - data_type: date\n"
        },
    )

    declared = load_declared_schemas(find_yaml_files(root))

    assert len(declared.warnings) == 1
    assert "column entry has no name" in declared.warnings[0]


def test_malformed_yaml_warns_instead_of_raising(tmp_path: Path) -> None:
    root = _project(tmp_path, **{"broken.yml": "models: [\n  - name: x\n"})

    declared = load_declared_schemas(find_yaml_files(root))

    assert declared.models == {}
    assert len(declared.warnings) == 1
    assert "invalid yaml" in declared.warnings[0]


def test_a_broken_yml_does_not_hide_a_good_one(tmp_path: Path) -> None:
    root = _project(tmp_path, **{"broken.yml": "a: [", "good.yml": SCHEMA})

    declared = load_declared_schemas(find_yaml_files(root))

    assert set(declared.models) == {"orders"}


# ---- positions -----------------------------------------------------------------------


def test_the_declared_type_carries_its_position(tmp_path: Path) -> None:
    root = _project(tmp_path, **{"schema.yml": SCHEMA})

    model = load_declared_schemas(find_yaml_files(root)).for_model("orders")

    assert model is not None
    revenue = model.column("revenue")
    assert revenue is not None
    # Round-trip: the span has to slice back to the type the user wrote, or a diagnostic
    # would underline the wrong part of their yml.
    assert model.source.slice(revenue.type_span) == "varchar(50)"
    assert revenue.type_span is not None and revenue.type_span.start_line == 5


def test_the_column_name_carries_its_position(tmp_path: Path) -> None:
    root = _project(tmp_path, **{"schema.yml": SCHEMA})

    model = load_declared_schemas(find_yaml_files(root)).for_model("orders")

    assert model is not None
    revenue = model.column("revenue")
    assert revenue is not None
    assert model.source.slice(revenue.name_span) == "revenue"


def test_the_model_name_carries_its_position(tmp_path: Path) -> None:
    root = _project(tmp_path, **{"schema.yml": SCHEMA})

    model = load_declared_schemas(find_yaml_files(root)).for_model("orders")

    assert model is not None
    assert model.source.slice(model.name_span) == "orders"
