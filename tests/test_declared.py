"""Loading dbt-shaped yml.

Fixtures are written into `tmp_path` rather than committed, because these tests are about
*discovery* as much as parsing - what gets picked up and what gets ignored depends on the
whole directory, so each test owns a directory.

Two project shapes get exercised separately. Without a `dbt_project.yml` every relation is
a `sources:` table named by exactly the parts its declaration writes down; with one,
`models:` comes back and dbt's own defaults apply.
"""

from pathlib import Path

from sqlr.catalog import find_yaml_files
from sqlr.declared import (
    check_sql_file_links,
    ignored_models_warning,
    load_declared_schemas,
    near_miss_warnings,
)
from sqlr.declared.types import DeclaredSchemas


def _project(tmp_path: Path, **files: str) -> Path:
    for name, content in files.items():
        path = tmp_path / name.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return tmp_path


def _load(tmp_path: Path, **files: str) -> DeclaredSchemas:
    root = _project(tmp_path, **files)
    return load_declared_schemas(find_yaml_files(root))


def _dbt(tmp_path: Path, **files: str) -> DeclaredSchemas:
    root = _project(tmp_path, **files)
    return load_declared_schemas(find_yaml_files(root), mode="dbt")


SOURCES = """\
version: 2

sources:
  - name: mysource
    database: mydatabase
    schema: myschema
    tables:
      - name: raw_department
        columns:
          - name: department_id
            data_type: varchar(50)
            description: the key
          - name: budget
            data_type: integer
"""


# ---- discovery -----------------------------------------------------------------------


def test_a_yml_with_neither_key_is_ignored(tmp_path: Path) -> None:
    declared = _load(tmp_path, **{"s.yml": SOURCES, "ci.yml": "jobs:\n  build: x\n"})

    assert set(declared.sources) == {"mydatabase.myschema.raw_department"}
    assert declared.warnings == []
    assert declared.errors == []


def test_yaml_and_yml_are_both_found(tmp_path: Path) -> None:
    declared = _load(
        tmp_path,
        **{
            "a.yml": SOURCES,
            "b.yaml": "sources:\n  - name: other\n    tables:\n      - name: second\n",
        },
    )

    assert set(declared.sources) == {"mydatabase.myschema.raw_department", "second"}


def test_malformed_yaml_warns_instead_of_raising(tmp_path: Path) -> None:
    declared = _load(tmp_path, **{"broken.yml": "sources: [\n  - name: x\n"})

    assert declared.sources == {}
    assert len(declared.warnings) == 1
    assert "invalid yaml" in declared.warnings[0]


def test_a_broken_yml_does_not_hide_a_good_one(tmp_path: Path) -> None:
    declared = _load(tmp_path, **{"broken.yml": "a: [", "good.yml": SOURCES})

    assert set(declared.sources) == {"mydatabase.myschema.raw_department"}


# ---- how a source table is named -----------------------------------------------------


def test_a_table_is_named_by_the_parts_its_source_declares(tmp_path: Path) -> None:
    declared = _load(tmp_path, **{"s.yml": SOURCES})

    table = declared.for_relation("mydatabase.myschema.raw_department")
    assert table is not None
    assert table.display_name == "mysource.raw_department"
    assert table.column("budget") is not None


def test_a_declaration_without_a_database_is_written_without_one(tmp_path: Path) -> None:
    declared = _load(
        tmp_path,
        **{
            "s.yml": "sources:\n  - name: mysource\n    schema: myschema\n"
            "    tables:\n      - name: raw_department\n"
        },
    )

    assert declared.for_relation("myschema.raw_department") is not None
    # dbt would default the schema to the source's name; a standalone project has no
    # profile to make that mean anything, so nothing is filled in.
    assert declared.for_relation("mysource.raw_department") is None


def test_a_declaration_without_a_schema_is_written_without_one(tmp_path: Path) -> None:
    declared = _load(
        tmp_path,
        **{
            "s.yml": "sources:\n  - name: mysource\n    database: mydatabase\n"
            "    tables:\n      - name: raw_department\n"
        },
    )

    assert declared.for_relation("mydatabase.raw_department") is not None


def test_a_declaration_with_neither_is_written_bare(tmp_path: Path) -> None:
    declared = _load(
        tmp_path,
        **{
            "s.yml": "sources:\n  - name: mysource\n    tables:\n"
            "      - name: raw_department\n"
        },
    )

    assert declared.for_relation("raw_department") is not None


def test_an_under_qualified_reference_does_not_match(tmp_path: Path) -> None:
    # Strict on purpose: the declaration says which relation it describes, and a reference
    # that names a different one is not that relation just because the tail agrees.
    declared = _load(tmp_path, **{"s.yml": SOURCES})

    assert declared.for_relation("raw_department") is None
    assert declared.for_relation("myschema.raw_department") is None


def test_an_over_qualified_reference_does_not_match(tmp_path: Path) -> None:
    declared = _load(
        tmp_path,
        **{
            "s.yml": "sources:\n  - name: mysource\n    tables:\n"
            "      - name: raw_department\n"
        },
    )

    assert declared.for_relation("myschema.raw_department") is None


def test_matching_ignores_case_and_quoting(tmp_path: Path) -> None:
    declared = _load(tmp_path, **{"s.yml": SOURCES})

    assert declared.for_relation('"MyDatabase".MYSCHEMA.raw_department') is not None


def test_an_identifier_overrides_the_name_in_the_relation(tmp_path: Path) -> None:
    declared = _load(
        tmp_path,
        **{
            "s.yml": "sources:\n  - name: mysource\n    schema: myschema\n"
            "    tables:\n      - name: departments\n        identifier: dept_raw\n"
        },
    )

    table = declared.for_relation("myschema.dept_raw")
    assert table is not None
    # The name is still how a person refers to it; the identifier is only the relation.
    assert table.display_name == "mysource.departments"
    assert declared.for_relation("myschema.departments") is None


# ---- tables this project builds ------------------------------------------------------


SQL_FILE_YML = """\
version: 2

sources:
  - name: mysource
    database: mydatabase
    schema: myschema
    tables:
      - name: employee
        sql_file: employee
        columns:
          - name: employee_id
            data_type: varchar
"""


def test_sql_file_links_a_declaration_to_the_file_that_builds_it(tmp_path: Path) -> None:
    declared = _load(tmp_path, **{"s.yml": SQL_FILE_YML})

    table = declared.for_sql_file(Path("project/employee.sql"))
    assert table is not None
    assert table.column("employee_id") is not None


def test_a_built_table_is_still_read_by_its_full_relation_name(tmp_path: Path) -> None:
    # The point of declaring database and schema on a table this project builds: another
    # file has to write them out to reach it.
    declared = _load(tmp_path, **{"s.yml": SQL_FILE_YML})

    assert declared.for_relation("mydatabase.myschema.employee") is not None
    assert declared.for_relation("employee") is None


def test_sql_file_accepts_the_file_name_as_well_as_the_stem(tmp_path: Path) -> None:
    declared = _load(
        tmp_path,
        **{
            "s.yml": "sources:\n  - name: mysource\n    tables:\n"
            "      - name: employee\n        sql_file: employee.sql\n"
        },
    )

    assert declared.for_sql_file(Path("employee.sql")) is not None


def test_a_sql_file_naming_nothing_in_the_project_is_an_error(tmp_path: Path) -> None:
    declared = _load(tmp_path, **{"s.yml": SQL_FILE_YML})

    [error] = check_sql_file_links(declared, ["department", "sales"])
    assert "sql_file 'employee' does not name a SQL file" in error
    assert "s.yml:9" in error

    assert check_sql_file_links(declared, ["employee"]) == []


# ---- columns -------------------------------------------------------------------------


def test_data_types_are_mapped_onto_the_lattice(tmp_path: Path) -> None:
    table = _load(tmp_path, **{"s.yml": SOURCES}).for_relation(
        "mydatabase.myschema.raw_department"
    )

    assert table is not None
    department_id = table.column("department_id")
    assert department_id is not None
    assert department_id.written_type == "varchar(50)"
    assert department_id.resolved_type_name == "string"
    assert department_id.description == "the key"
    assert table.column("budget").resolved_type_name == "integer"  # type: ignore[union-attr]


def test_an_unrecognised_data_type_resolves_to_unknown(tmp_path: Path) -> None:
    declared = _load(
        tmp_path,
        **{
            "s.yml": "sources:\n  - name: s\n    tables:\n      - name: t\n"
            "        columns:\n          - name: c\n            data_type: blorp\n"
        },
    )

    table = declared.for_relation("t")
    assert table is not None
    assert table.column("c").resolved_type_name == "unknown"  # type: ignore[union-attr]


def test_a_column_without_a_data_type_is_skipped_silently(tmp_path: Path) -> None:
    # Documenting a column without typing it is normal in dbt; there is simply nothing
    # to check it against.
    declared = _load(
        tmp_path,
        **{
            "s.yml": "sources:\n  - name: s\n    tables:\n      - name: t\n"
            "        columns:\n"
            "          - name: documented\n            description: hi\n"
            "          - name: typed\n            data_type: date\n"
        },
    )

    table = declared.for_relation("t")
    assert table is not None
    assert [c.name for c in table.columns] == ["typed"]
    assert declared.warnings == []


def test_a_column_without_a_name_warns(tmp_path: Path) -> None:
    declared = _load(
        tmp_path,
        **{
            "s.yml": "sources:\n  - name: s\n    tables:\n      - name: t\n"
            "        columns:\n          - data_type: date\n"
        },
    )

    assert len(declared.warnings) == 1
    assert "column entry has no name" in declared.warnings[0]


# ---- descriptions that contradict each other -----------------------------------------


def test_two_sources_that_resolve_to_the_same_relation_are_an_error(
    tmp_path: Path,
) -> None:
    other = (
        "sources:\n  - name: othersource\n    database: mydatabase\n"
        "    schema: myschema\n    tables:\n      - name: raw_department\n"
    )
    declared = _load(tmp_path, **{"a.yml": SOURCES, "b.yml": other})

    [error] = declared.errors
    assert "both describe the relation 'mydatabase.myschema.raw_department'" in error
    assert "mysource.raw_department at a.yml:8" in error
    assert "othersource.raw_department at b.yml:6" in error


def test_the_same_table_name_in_two_schemas_is_fine(tmp_path: Path) -> None:
    other = (
        "sources:\n  - name: mysource\n    database: mydatabase\n"
        "    schema: otherschema\n    tables:\n      - name: raw_department\n"
    )
    declared = _load(tmp_path, **{"a.yml": SOURCES, "b.yml": other})

    assert declared.errors == []
    assert len(declared.sources) == 2


def test_a_column_described_twice_is_an_error(tmp_path: Path) -> None:
    declared = _load(
        tmp_path,
        **{
            "s.yml": "sources:\n  - name: s\n    tables:\n      - name: t\n"
            "        columns:\n          - name: c\n            data_type: date\n"
            "          - name: c\n            data_type: integer\n"
        },
    )

    [error] = declared.errors
    assert "column 'c' of s.t is described more than once" in error
    # The first one still stands, so the rest of the run has something to work with.
    assert declared.for_relation("t").column("c").written_type == "date"  # type: ignore[union-attr]


def test_two_tables_claiming_one_sql_file_are_an_error(tmp_path: Path) -> None:
    other = (
        "sources:\n  - name: othersource\n    tables:\n"
        "      - name: employee\n        sql_file: employee\n"
    )
    declared = _load(tmp_path, **{"a.yml": SQL_FILE_YML, "b.yml": other})

    [error] = declared.errors
    assert "sql_file 'employee' is claimed by both" in error


def test_a_sources_key_that_is_not_a_list_is_an_error(tmp_path: Path) -> None:
    # The mistake everyone makes once: dbt's `sources:` is a list of sources, not one.
    declared = _load(
        tmp_path,
        **{
            "s.yml": "version: 2\nsources:\n  name: mysource\n  schema: myschema\n"
            "  tables:\n    - name: raw_department\n"
        },
    )

    [error] = declared.errors
    assert "`sources:` must be a list of sources" in error
    assert declared.sources == {}


def test_a_tables_key_that_is_not_a_list_is_an_error(tmp_path: Path) -> None:
    declared = _load(
        tmp_path,
        **{"s.yml": "sources:\n  - name: mysource\n    tables:\n      name: raw\n"},
    )

    [error] = declared.errors
    assert "`tables:` of source 'mysource' must be a list" in error


def test_a_source_without_a_name_warns(tmp_path: Path) -> None:
    declared = _load(
        tmp_path, **{"s.yml": "sources:\n  - tables:\n      - name: raw_department\n"}
    )

    assert declared.sources == {}
    assert len(declared.warnings) == 1
    assert "source entry has no name" in declared.warnings[0]


def test_a_source_with_no_tables_says_nothing(tmp_path: Path) -> None:
    declared = _load(tmp_path, **{"s.yml": "sources:\n  - name: mysource\n"})

    assert declared.sources == {}
    assert declared.warnings == []
    assert declared.errors == []


# ---- the reference that was meant to match -------------------------------------------


def test_a_reference_missing_a_database_is_pointed_at_the_right_name(
    tmp_path: Path,
) -> None:
    declared = _load(
        tmp_path,
        **{
            "s.yml": "sources:\n  - name: mysource\n    database: mydb\n"
            "    tables:\n      - name: mytable\n"
        },
    )

    [warning] = near_miss_warnings(declared, ["mytable"])
    assert "mytable is not declared" in warning
    assert "Write mydb.mytable in the SQL" in warning
    assert "drop the database from the declaration" in warning


def test_a_declaration_something_else_uses_is_not_a_near_miss(tmp_path: Path) -> None:
    # The declaration is doing its job for another file, so the unmatched reference here
    # is a different relation, not the same one written short.
    declared = _load(
        tmp_path,
        **{
            "s.yml": "sources:\n  - name: mysource\n    database: mydb\n"
            "    tables:\n      - name: mytable\n"
        },
    )

    assert near_miss_warnings(declared, ["mytable", "mydb.mytable"]) == []


def test_an_unmatched_reference_with_no_lookalike_says_nothing(tmp_path: Path) -> None:
    declared = _load(tmp_path, **{"s.yml": SOURCES})

    assert near_miss_warnings(declared, ["something_entirely_different"]) == []


# ---- models: outside a dbt project ---------------------------------------------------


MODELS_YML = """\
version: 2
models:
  - name: employee
    columns:
      - name: employee_id
        data_type: varchar
  - name: unrelated
    alias: department
"""


def test_models_are_not_read_without_a_dbt_project(tmp_path: Path) -> None:
    declared = _load(tmp_path, **{"schema.yml": MODELS_YML})

    assert declared.models == {}
    assert [entry.name for entry in declared.ignored_models] == ["employee", "unrelated"]


def test_ignored_models_are_only_reported_when_they_name_a_real_file(
    tmp_path: Path,
) -> None:
    declared = _load(tmp_path, **{"schema.yml": MODELS_YML})

    assert ignored_models_warning(declared, ["something_else"], tmp_path) is None

    warning = ignored_models_warning(declared, ["employee"], tmp_path)
    assert warning is not None
    assert "no dbt_project.yml was found" in warning
    assert "employee (schema.yml:3)" in warning


def test_an_ignored_model_matches_on_its_alias_too(tmp_path: Path) -> None:
    declared = _load(tmp_path, **{"schema.yml": MODELS_YML})

    warning = ignored_models_warning(declared, ["department"], tmp_path)
    assert warning is not None
    assert "department (schema.yml:7)" in warning


def test_the_ignored_model_list_is_capped(tmp_path: Path) -> None:
    names = [f"m{index}" for index in range(13)]
    yml = "models:\n" + "".join(f"  - name: {name}\n" for name in names)
    declared = _load(tmp_path, **{"schema.yml": yml})

    warning = ignored_models_warning(declared, names, tmp_path)
    assert warning is not None
    assert "m9 (schema.yml:11)" in warning
    assert "m10" not in warning
    assert "and 3 others" in warning


# ---- a dbt project -------------------------------------------------------------------


def test_models_are_read_in_a_dbt_project(tmp_path: Path) -> None:
    declared = _dbt(tmp_path, **{"schema.yml": MODELS_YML})

    assert set(declared.models) == {"employee", "unrelated"}
    assert declared.ignored_models == []
    assert declared.for_sql_file(Path("models/employee.sql")) is not None


def test_a_model_described_twice_in_a_dbt_project_is_an_error(tmp_path: Path) -> None:
    declared = _dbt(tmp_path, **{"a.yml": MODELS_YML, "b.yml": MODELS_YML})

    assert len(declared.errors) == 2
    assert "model 'employee' is described more than once" in declared.errors[0]
    assert "a.yml:3 and b.yml:3" in declared.errors[0]


def test_a_dbt_source_fills_in_the_parts_a_profile_would(tmp_path: Path) -> None:
    # dbt defaults the schema to the source's name and takes the database from the target
    # profile, which sqlr cannot read - so an unwritten part matches anything.
    declared = _dbt(
        tmp_path,
        **{
            "s.yml": "sources:\n  - name: myschema\n    tables:\n"
            "      - name: raw_department\n"
        },
    )

    assert declared.for_relation("myschema.raw_department") is not None
    assert declared.for_relation("anydb.myschema.raw_department") is not None
    assert declared.for_relation("raw_department") is not None


def test_a_dbt_model_wins_an_unqualified_name(tmp_path: Path) -> None:
    declared = _dbt(
        tmp_path,
        **{
            "m.yml": "models:\n  - name: employee\n    columns:\n"
            "      - name: from_model\n        data_type: date\n",
            "s.yml": "sources:\n  - name: raw\n    tables:\n      - name: employee\n",
        },
    )

    table = declared.for_relation("employee")
    assert table is not None and table.column("from_model") is not None


def test_a_models_key_that_is_not_a_list_is_an_error(tmp_path: Path) -> None:
    declared = _dbt(tmp_path, **{"schema.yml": "models:\n  name: orders\n"})

    [error] = declared.errors
    assert "`models:` must be a list of model entries" in error


# ---- positions -----------------------------------------------------------------------


def test_the_declared_type_carries_its_position(tmp_path: Path) -> None:
    table = _load(tmp_path, **{"s.yml": SOURCES}).for_relation(
        "mydatabase.myschema.raw_department"
    )

    assert table is not None
    department_id = table.column("department_id")
    assert department_id is not None
    # Round-trip: the span has to slice back to the type the user wrote, or a diagnostic
    # would underline the wrong part of their yml.
    assert table.source.slice(department_id.type_span) == "varchar(50)"
    assert department_id.type_span is not None
    assert department_id.type_span.start_line == 10


def test_the_column_name_carries_its_position(tmp_path: Path) -> None:
    table = _load(tmp_path, **{"s.yml": SOURCES}).for_relation(
        "mydatabase.myschema.raw_department"
    )

    assert table is not None
    department_id = table.column("department_id")
    assert department_id is not None
    assert table.source.slice(department_id.name_span) == "department_id"


def test_the_table_name_carries_its_position(tmp_path: Path) -> None:
    table = _load(tmp_path, **{"s.yml": SOURCES}).for_relation(
        "mydatabase.myschema.raw_department"
    )

    assert table is not None
    assert table.source.slice(table.name_span) == "raw_department"
    assert table.where == "s.yml:8"


# ---- declaration_is_partial ----------------------------------------------------------
PARTIAL_SOURCES = """\
version: 2

sources:
  - name: mysource
    database: mydatabase
    schema: myschema
    config:
      meta:
        declaration_is_partial: true
    tables:
      - name: inherits_it
        columns:
          - name: a
            data_type: integer
      - name: overrides_it
        config:
          meta:
            declaration_is_partial: false
        columns:
          - name: b
            data_type: integer
      - name: writes_the_older_spelling
        meta:
          declaration_is_partial: false
        columns:
          - name: c
            data_type: integer
"""


def test_a_source_meta_flag_is_inherited_by_its_tables(tmp_path: Path) -> None:
    """dbt already defines `meta:` on a source as inherited by its tables, so a whole
    partly-documented source says so in one line rather than once per table."""
    declared = _load(tmp_path, **{"s.yml": PARTIAL_SOURCES})
    partial = {
        table.name: table.declaration_is_partial for table in declared.sources.values()
    }
    assert partial == {
        "inherits_it": True,
        "overrides_it": False,
        "writes_the_older_spelling": False,
    }
    assert declared.errors == []


def test_the_flag_defaults_to_complete(tmp_path: Path) -> None:
    """Completeness is the payoff for declaring, so it is what a declaration means unless
    it says otherwise."""
    declared = _load(tmp_path, **{"s.yml": SOURCES})
    (table,) = declared.sources.values()
    assert table.declaration_is_partial is False


def test_a_meta_flag_that_is_not_a_boolean_is_warned_about(tmp_path: Path) -> None:
    """YAML 1.1 reads `yes` as true and sqlr deliberately does not. A flag that quietly
    did nothing is worse than one that was never written."""
    declared = _load(
        tmp_path,
        **{"s.yml": """\
version: 2

sources:
  - name: mysource
    tables:
      - name: t
        config:
          meta:
            declaration_is_partial: yes
        columns:
          - name: a
            data_type: integer
"""},
    )
    (table,) = declared.sources.values()
    assert table.declaration_is_partial is False
    assert any("is not `true` or `false`" in warning for warning in declared.warnings)


def test_a_partial_declaration_with_no_columns_is_warned_about(tmp_path: Path) -> None:
    declared = _load(
        tmp_path,
        **{"s.yml": """\
version: 2

sources:
  - name: mysource
    tables:
      - name: t
        config:
          meta:
            declaration_is_partial: true
"""},
    )
    assert any(
        "unnecessary declaration_is_partial flag set for mysource.t" in warning
        for warning in declared.warnings
    )
