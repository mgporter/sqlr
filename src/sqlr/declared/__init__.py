"""Load the types the user declared, from dbt-shaped yml.

The format is dbt's, deliberately: a project that already has property files gets checked
with no extra authoring, and a project without dbt can write the same thing. Only two
fields are required of each column - `name` and `data_type`.

Which key is read depends on the project. Without a `dbt_project.yml` there are no models,
only sources, because a standalone project has no `ref()`/`source()` distinction to
inherit - every relation the SQL reads is a table someone has to describe, and the ones
this project builds say so with `sql_file:`:

```yml
sources:
  - name: mysource
    database: mydatabase
    schema: myschema
    tables:
      - name: raw_department      # read as mydatabase.myschema.raw_department
        columns:
          - name: department_id
            data_type: string
      - name: employee
        sql_file: employee        # ...and this one is built by employee.sql
```

The parts a table writes down are the relation, exactly: `database` and `schema` appear in
the name only when they are given. Nothing is defaulted in, since the value dbt would take
from a target profile is not knowable here, and a guess would silently fail to match the
name the user wrote. An unmatched relation is not fatal - its columns simply have nothing
to check against - but `near_miss_warnings` catches the case that is almost always a typo:
a reference whose table name matches a declaration nothing else uses.

A declaration's `columns:` is the *complete* list of what the relation has, which is what
makes a name the SQL reads and the yml omits an error rather than a column invented from
the read. A relation that is only partly described says so under `config: meta:`:

```yml
sources:
  - name: mysource
    config:
      meta:
        declaration_is_partial: true      # every table below is partly described...
    tables:
      - name: raw_address
        config:
          meta:
            declaration_is_partial: true  # ...or just this one
        columns:
          - name: person_id
            data_type: varchar(20)        # typed; everything else is inferred
```

`meta:` is where every sqlr-specific setting goes, and inherits source-to-table the way dbt
already defines it - see `META_KEY`.

Discovery is decoupled from interpretation. `DeclarationProvider` is the seam: this module
implements it over yml files, and a future dbt `manifest.json` reader can implement it
without any consumer changing.

Parsing goes through `yaml.compose` rather than `yaml.safe_load`, because the composed
node tree carries source marks. Without them a diagnostic could say a declaration was
contradicted but not where the declaration was written.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, cast

import yaml
from pydantic import BaseModel

from sqlr.catalog.types import FileInventory
from sqlr.declared.types import (
    DeclarationMode,
    DeclaredColumn,
    DeclaredModel,
    DeclaredSchemas,
    DeclaredSourceTable,
    IgnoredModel,
    relation_parts,
)
from sqlr.source import Positions, SourceDoc, SourceSpan
from sqlr.typemap import resolve_type_name

__all__ = [
    "DeclarationProvider",
    "Declarations",
    "YamlDeclarationProvider",
    "check_sql_file_links",
    "ignored_models_warning",
    "load_declared_schemas",
    "near_miss_warnings",
]

MODELS_KEY = "models"
SOURCES_KEY = "sources"
"""The discriminators. A yml with neither is not ours and is skipped in silence."""

MAX_LISTED = 10
"""How many entries a warning names before it summarises the rest."""

CONFIG_KEY = "config"
META_KEY = "meta"
"""Where sqlr's own settings are written: `config: meta:`, or a bare `meta:`.

dbt validates the keys it knows and rejects the ones it does not, but `meta:` is free-form
by design and carries any key through untouched. A setting written there leaves the file a
valid dbt file, so a real dbt project can be checked without editing its yml. Every
sqlr-specific setting that attaches to a declaration goes here, for that reason.

Both spellings are read because dbt moved `meta:` under `config:` in 1.10 and still accepts
the older top-level form. `config: meta:` wins where a file writes both, matching which one
dbt itself would apply.
"""

DECLARATION_IS_PARTIAL_KEY = "declaration_is_partial"

BOOLEANS = {"true": True, "false": False}
"""The only two words a sqlr `meta:` flag accepts, case-insensitively.

`yaml.compose` leaves a scalar as text rather than a Python value, so the resolution is
ours to make. YAML 1.1's wider set - `yes`, `on`, `1` - is deliberately not honoured:
anything else is warned about and ignored, which tells a user their setting did nothing
instead of silently reading `on` as False.
"""


class Declarations(BaseModel):
    """Everything one provider found, before names are checked against each other."""

    models: list[DeclaredModel] = []
    source_tables: list[DeclaredSourceTable] = []
    ignored_models: list[IgnoredModel] = []
    errors: list[str] = []
    warnings: list[str] = []


class DeclarationProvider(Protocol):
    """Anything that can produce declared schemas."""

    def declarations(self) -> Declarations: ...


class YamlDeclarationProvider:
    """Reads dbt-shaped yml out of a file inventory."""

    def __init__(
        self, inventory: FileInventory, mode: DeclarationMode = "standalone"
    ) -> None:
        self.inventory = inventory
        self.mode = mode

    def declarations(self) -> Declarations:
        found = Declarations()
        for file in self.inventory.files:
            self._file(file.path, file.relative_path, found)
        return found

    # ---- one file --------------------------------------------------------------------

    def _file(self, path: Path, label: str, found: Declarations) -> None:
        try:
            text = path.read_text()
        except OSError as e:
            found.warnings.append(f"{label}: could not be read ({e})")
            return

        try:
            root = _compose(text)
        except yaml.YAMLError as e:
            # A broken yml is a problem with that file, not a reason to stop analysing
            # the SQL it was meant to describe.
            found.warnings.append(f"{label}: invalid yaml ({_terse(e)})")
            return

        if not isinstance(root, yaml.MappingNode):
            return

        models_node = _entry(root, MODELS_KEY)
        sources_node = _entry(root, SOURCES_KEY)
        if models_node is None and sources_node is None:
            return

        source = SourceDoc(path=path, text=text)
        positions = Positions(text)

        if models_node is not None:
            self._models(models_node, path, label, source, positions, found)
        if sources_node is not None:
            self._sources(sources_node, path, label, source, positions, found)

    # ---- models ----------------------------------------------------------------------

    def _models(
        self,
        node: yaml.Node,
        path: Path,
        label: str,
        source: SourceDoc,
        positions: Positions,
        found: Declarations,
    ) -> None:
        if not isinstance(node, yaml.SequenceNode):
            found.errors.append(
                f"{label}:{_line(node)}: `{MODELS_KEY}:` must be a list of model entries"
            )
            return

        for entry in node.value:
            if not isinstance(entry, yaml.MappingNode):
                continue
            name = _scalar(_entry(entry, "name"))
            if not name:
                found.warnings.append(
                    f"{label}:{_line(entry)}: model entry has no name; skipped"
                )
                continue

            if self.mode != "dbt":
                # Kept, not dropped: the warning has to be able to name the entries that
                # would have applied, or it is noise the user cannot act on.
                found.ignored_models.append(
                    IgnoredModel(
                        name=name,
                        alias=_alias(entry),
                        label=label,
                        line=_line(entry),
                    )
                )
                continue

            name_node = _entry(entry, "name")
            found.models.append(
                DeclaredModel(
                    name=name,
                    path=path,
                    label=label,
                    source=source,
                    columns=self._columns(entry, label, name, positions, found),
                    declaration_is_partial=bool(
                        _meta_flag(
                            entry, DECLARATION_IS_PARTIAL_KEY, label, name, found
                        )
                    ),
                    span=_span(positions, entry),
                    name_span=_span(positions, name_node),
                )
            )

    # ---- sources ---------------------------------------------------------------------

    def _sources(
        self,
        node: yaml.Node,
        path: Path,
        label: str,
        source: SourceDoc,
        positions: Positions,
        found: Declarations,
    ) -> None:
        if not isinstance(node, yaml.SequenceNode):
            found.errors.append(
                f"{label}:{_line(node)}: `{SOURCES_KEY}:` must be a list of sources, "
                f"each with a `name:` and a `tables:` list"
            )
            return

        for entry in node.value:
            if not isinstance(entry, yaml.MappingNode):
                continue
            self._source(entry, path, label, source, positions, found)

    def _source(
        self,
        node: yaml.MappingNode,
        path: Path,
        label: str,
        source: SourceDoc,
        positions: Positions,
        found: Declarations,
    ) -> None:
        name = _scalar(_entry(node, "name"))
        if not name:
            found.warnings.append(
                f"{label}:{_line(node)}: source entry has no name; skipped"
            )
            return

        tables_node = _entry(node, "tables")
        if tables_node is None:
            # A source that lists no tables describes nothing. Legal, and not worth a word.
            return
        if not isinstance(tables_node, yaml.SequenceNode):
            found.errors.append(
                f"{label}:{_line(tables_node)}: `tables:` of source {name!r} must be a list"
            )
            return

        database = _scalar(_entry(node, "database"))
        schema = _scalar(_entry(node, "schema"))
        if self.mode == "dbt" and schema is None:
            # dbt's own default. Standalone leaves it absent: an omitted part is a part
            # the SQL does not write, not one to be filled in from somewhere else.
            schema = name

        # dbt already defines a source's `meta:` as inherited by its tables, with a
        # table's own entry winning. Following that rule rather than inventing one is what
        # lets a whole partially-documented source say so in a single line.
        inherited_partial = _meta_flag(
            node, DECLARATION_IS_PARTIAL_KEY, label, f"source {name!r}", found
        )

        for entry in tables_node.value:
            if not isinstance(entry, yaml.MappingNode):
                continue
            table = self._table(
                entry,
                name,
                database,
                schema,
                inherited_partial,
                path,
                label,
                source,
                positions,
                found,
            )
            if table is not None:
                found.source_tables.append(table)

    def _table(
        self,
        node: yaml.MappingNode,
        source_name: str,
        database: str | None,
        schema: str | None,
        inherited_partial: bool | None,
        path: Path,
        label: str,
        source: SourceDoc,
        positions: Positions,
        found: Declarations,
    ) -> DeclaredSourceTable | None:
        name_node = _entry(node, "name")
        name = _scalar(name_node)
        if not name:
            found.warnings.append(
                f"{label}:{_line(node)}: table entry in source {source_name!r} has no "
                f"name; skipped"
            )
            return None

        sql_file_node = _entry(node, "sql_file")
        sql_file = _scalar(sql_file_node)
        if sql_file is not None:
            # `sql_file: employee.sql` is what a person writes half the time, and the two
            # spellings naming the same file should not be two different answers.
            sql_file = Path(sql_file).stem

        own_partial = _meta_flag(
            node, DECLARATION_IS_PARTIAL_KEY, label, f"{source_name}.{name}", found
        )
        declaration_is_partial = bool(
            own_partial if own_partial is not None else inherited_partial
        )

        columns = self._columns(node, label, f"{source_name}.{name}", positions, found)
        if declaration_is_partial and not columns:
            # A partial declaration with nothing in it describes the relation exactly as
            # well as no declaration at all.
            found.warnings.append(
                f"{label}:{_line(node)}: unnecessary declaration_is_partial flag set for "
                f"{source_name}.{name} - no columns are declared so this flag will have "
                f"no effect"
            )

        return DeclaredSourceTable(
            name=name,
            path=path,
            label=label,
            source=source,
            source_name=source_name,
            database=database,
            schema_name=schema,
            identifier=_scalar(_entry(node, "identifier")),
            sql_file=sql_file,
            sql_file_span=_span(positions, sql_file_node),
            columns=columns,
            declaration_is_partial=declaration_is_partial,
            span=_span(positions, node),
            name_span=_span(positions, name_node),
        )

    # ---- columns ---------------------------------------------------------------------

    def _columns(
        self,
        node: yaml.MappingNode,
        label: str,
        owner: str,
        positions: Positions,
        found: Declarations,
    ) -> list[DeclaredColumn]:
        columns_node = _entry(node, "columns")
        if not isinstance(columns_node, yaml.SequenceNode):
            return []

        columns: list[DeclaredColumn] = []
        seen: dict[str, int] = {}
        for entry in columns_node.value:
            if not isinstance(entry, yaml.MappingNode):
                continue

            name_node = _entry(entry, "name")
            name = _scalar(name_node)
            if not name:
                found.warnings.append(
                    f"{label}:{_line(entry)}: column entry has no name; skipped"
                )
                continue

            first = seen.get(name.lower())
            if first is not None:
                found.errors.append(
                    f"{label}:{_line(entry)}: column {name!r} of {owner} is described "
                    f"more than once, first at line {first}; a column may only be "
                    f"described once"
                )
                continue
            seen[name.lower()] = _line(entry)

            type_node = _entry(entry, "data_type")
            written_type = _scalar(type_node)
            if not written_type:
                # A documented column with no declared type is normal in dbt and simply
                # has nothing to check against. Not a warning.
                continue

            columns.append(
                DeclaredColumn(
                    name=name,
                    written_type=written_type,
                    resolved_type_name=resolve_type_name(written_type),
                    description=_scalar(_entry(entry, "description")),
                    span=_span(positions, entry),
                    name_span=_span(positions, name_node),
                    type_span=_span(positions, type_node),
                )
            )
        return columns


# ---- assembly --------------------------------------------------------------------------


def load_declared_schemas(
    inventory: FileInventory,
    mode: DeclarationMode = "standalone",
    providers: Iterable[DeclarationProvider] | None = None,
) -> DeclaredSchemas:
    """Collect declarations from every provider into one index.

    A relation described twice is an error rather than a warning that picks a winner: dbt
    rejects the same thing, and when two entries give one column two types there is no way
    to tell which the user meant. Sources collide on the *relation they resolve to*, not on
    their `source.table` names - two sources may well both have a `raw_department` as long
    as they land in different schemas.
    """
    provider_list: list[DeclarationProvider] = (
        [YamlDeclarationProvider(inventory, mode)] if providers is None else list(providers)
    )

    models: dict[str, DeclaredModel] = {}
    sources: dict[str, DeclaredSourceTable] = {}
    claimed: dict[str, DeclaredSourceTable] = {}
    ignored: list[IgnoredModel] = []
    errors: list[str] = []
    warnings: list[str] = []

    for provider in provider_list:
        found = provider.declarations()
        errors.extend(found.errors)
        warnings.extend(found.warnings)
        ignored.extend(found.ignored_models)

        for model in found.models:
            existing = models.get(model.name.lower())
            if existing is not None:
                errors.append(
                    f"model {model.name!r} is described more than once, at "
                    f"{existing.where} and {model.where}; a model may only be described "
                    f"once"
                )
                continue
            models[model.name.lower()] = model

        for table in found.source_tables:
            existing = sources.get(table.key)
            if existing is not None:
                errors.append(
                    f"{existing.display_name} at {existing.where} and "
                    f"{table.display_name} at {table.where} both describe the relation "
                    f"{table.relation_name!r}; a relation may only be described once"
                )
                continue
            sources[table.key] = table

            if table.sql_file is None:
                continue
            owner = claimed.get(table.sql_file.lower())
            if owner is not None:
                errors.append(
                    f"sql_file {table.sql_file!r} is claimed by both "
                    f"{owner.display_name} at {owner.where} and {table.display_name} at "
                    f"{table.where}; a SQL file may only be described once"
                )
                continue
            claimed[table.sql_file.lower()] = table

    return DeclaredSchemas(
        mode=mode,
        models=models,
        sources=sources,
        ignored_models=ignored,
        errors=errors,
        warnings=warnings,
    )


# ---- checks that need the rest of the project ----------------------------------------


def check_sql_file_links(
    declared: DeclaredSchemas, model_names: Iterable[str]
) -> list[str]:
    """Errors for `sql_file:` values that name no SQL file in the project.

    Separate from loading because it is the one thing a yml cannot answer on its own: the
    set of models is built from the config's search paths, not from the yml.
    """
    known = {name.lower() for name in model_names}
    return [
        f"{_at(table)}: sql_file {table.sql_file!r} does not name a SQL file in this "
        f"project"
        for table in declared.sources.values()
        if table.sql_file is not None and table.sql_file.lower() not in known
    ]


def ignored_models_warning(
    declared: DeclaredSchemas, model_names: Iterable[str], project_root: Path
) -> str | None:
    """One warning for `models:` entries that would have described a real SQL file.

    A `models:` key outside a dbt project is not read, and staying silent about it means a
    user watching their declarations do nothing with no idea why. Entries that match no
    file are left alone: those are as likely to belong to some other tool's yml as to be a
    mistake, and warning about them would make the message unusable in a mixed repo.
    """
    if not declared.ignored_models:
        return None

    known = {name.lower() for name in model_names}
    matched = [
        (entry, hit)
        for entry in declared.ignored_models
        if (hit := next((n for n in entry.names if n.lower() in known), None)) is not None
    ]
    if not matched:
        return None

    listed = ", ".join(
        f"{hit} ({entry.where})" for entry, hit in matched[:MAX_LISTED]
    )
    rest = len(matched) - MAX_LISTED
    if rest > 0:
        listed += f", and {rest} other{'s' if rest != 1 else ''}"

    return (
        f"no dbt_project.yml was found at {project_root}, so this is not a dbt project "
        f"and `models:` entries are ignored - sqlr reads declarations from `sources:` "
        f"only. {len(matched)} ignored entr{'y' if len(matched) == 1 else 'ies'} "
        f"name{'s' if len(matched) == 1 else ''} a SQL file in this project: {listed}. "
        f"Move them under `sources:`, adding `sql_file:`, to have them applied."
    )


def near_miss_warnings(
    declared: DeclaredSchemas, referenced: Iterable[str]
) -> list[str]:
    """Warnings for relations that look like a declaration written short.

    An undeclared relation is normally just undeclared. But when its table name matches a
    declaration that *nothing* in the run references, the two are almost certainly meant to
    be the same thing and the reference is missing a database or a schema - so the warning
    says which name to write rather than leaving the user to work out that the declaration
    they can see is not the one being applied.
    """
    if declared.mode != "standalone":
        return []

    names = list(dict.fromkeys(referenced))
    used: set[str] = set()
    unmatched: list[str] = []
    for name in names:
        table = declared.for_source(name)
        if table is None:
            unmatched.append(name)
        else:
            used.add(table.key)

    warnings: list[str] = []
    for name in unmatched:
        parts = relation_parts(name)
        if not parts:
            continue
        for table in declared.sources.values():
            if table.key in used or table.table_name.lower() != parts[-1]:
                continue
            warnings.append(
                f"{name} is not declared, but {table.display_name} at {table.where} "
                f"describes a table named {table.table_name!r} and nothing references it. "
                f"Write {table.relation_name} in the SQL, or drop the "
                f"{_omitted(table)} from the declaration."
            )
    return warnings


def _omitted(table: DeclaredSourceTable) -> str:
    """The parts of the relation name a short reference left out."""
    given = [
        word
        for word, part in (("database", table.database), ("schema", table.schema_name))
        if part
    ]
    return " and ".join(given) if given else "extra parts"


def _at(table: DeclaredSourceTable) -> str:
    """Where a table's `sql_file:` was written, falling back to the entry itself."""
    if table.sql_file_span is None:
        return table.where
    base = table.label or str(table.path)
    return f"{base}:{table.sql_file_span.start_line + 1}"


# ---- yaml node helpers ---------------------------------------------------------------


def _alias(node: yaml.MappingNode) -> str | None:
    """A model's `alias:`, top level or under `config:` - both spellings are dbt's."""
    alias = _scalar(_entry(node, "alias"))
    if alias is not None:
        return alias
    config = _entry(node, "config")
    if isinstance(config, yaml.MappingNode):
        return _scalar(_entry(config, "alias"))
    return None


def _compose(text: str) -> yaml.Node | None:
    """`yaml.compose`, with the return type its type stub omits.

    Typeshed declares `compose` without one, so its result is untyped and infects every
    expression downstream of it. Pinning it here keeps the ignore to a single line.
    """
    composed = yaml.compose(text)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return cast("yaml.Node | None", composed)


def _entry(node: yaml.MappingNode, key: str) -> yaml.Node | None:
    for key_node, value_node in node.value:
        if isinstance(key_node, yaml.ScalarNode) and key_node.value == key:
            return value_node
    return None


def _scalar(node: yaml.Node | None) -> str | None:
    if not isinstance(node, yaml.ScalarNode):
        return None
    value = str(node.value).strip()
    return value or None


def _meta_entry(node: yaml.MappingNode | None, key: str) -> yaml.Node | None:
    """One key of an entry's `meta:` mapping, under `config:` or at the top level.

    dbt moved `meta:` under `config:` in 1.10 and still accepts the older spelling, so both
    are read and `config: meta:` wins - which is the one dbt itself would apply.
    """
    if node is None:
        return None
    for owner in (_entry(node, CONFIG_KEY), node):
        if not isinstance(owner, yaml.MappingNode):
            continue
        meta = _entry(owner, META_KEY)
        if isinstance(meta, yaml.MappingNode) and (found := _entry(meta, key)) is not None:
            return found
    return None


def _meta_flag(
    node: yaml.MappingNode | None,
    key: str,
    label: str,
    owner: str,
    found: Declarations,
) -> bool | None:
    """One boolean out of an entry's `meta:` mapping, or None when it is not written there.

    None is not False: it means the key was absent, which is what lets a table's `meta:`
    fall back to its source's rather than overriding it with a default.

    A value that is neither `true` nor `false` is a warning and reads as absent. A flag
    that quietly did nothing is worse than one that was never written - the user believes
    the setting is in effect and every finding that follows looks like a different bug.
    """
    value_node = _meta_entry(node, key)
    if value_node is None:
        return None

    written = _scalar(value_node)
    resolved = BOOLEANS.get(written.lower()) if written is not None else None
    if resolved is None:
        found.warnings.append(
            f"{label}:{_line(value_node)}: `meta.{key}` of {owner} is "
            f"{written!r}, which is not `true` or `false`; ignored"
        )
    return resolved


def _span(positions: Positions, node: yaml.Node | None) -> SourceSpan | None:
    """Convert a composed node's marks into a span.

    PyYAML's end mark for a block scalar runs to the start of the next token, trailing
    newline included, so it is trimmed back to the value itself.
    """
    if node is None:
        return None
    start = node.start_mark.index
    end = node.end_mark.index
    text = positions.text
    while end > start and text[end - 1].isspace():
        end -= 1
    if end <= start:
        return None
    return positions.span(start, end)


def _line(node: yaml.Node) -> int:
    return node.start_mark.line + 1


def _terse(error: yaml.YAMLError) -> str:
    return " ".join(str(error).split())
