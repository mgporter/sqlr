"""Types the user wrote down, as opposed to types the analyser inferred.

Two shapes of project, and the difference runs through everything here:

- **standalone** - no `dbt_project.yml`. Every relation the SQL reads is a `sources:`
  table, including the ones this project produces, which mark themselves with `sql_file:`.
  A declaration names a relation *exactly*: the parts it writes down, joined, are what the
  SQL has to write.
- **dbt** - a `dbt_project.yml` is present. `models:` entries come back, matched by file
  stem, and a source's missing parts are the ones dbt would fill from a profile, so they
  match anything.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from sqlr.source import SourceDoc, SourceSpan
from sqlr.typemap import ResolvedTypeName

DeclarationMode = Literal["standalone", "dbt"]
"""Which set of rules a project's yml is read under."""


class DeclaredColumn(BaseModel):
    name: str
    written_type: str
    """The yml's `data_type:`, verbatim - `varchar(50)`, not `string`."""
    resolved_type_name: ResolvedTypeName
    """`written_type` mapped onto the lattice. `unknown` when the name is unrecognised."""
    description: str | None = None
    span: SourceSpan | None = None
    """The whole column entry in the yml."""
    name_span: SourceSpan | None = None
    type_span: SourceSpan | None = None
    """The `data_type` scalar - what to point at when a declaration is contradicted."""


class DeclaredRelation(BaseModel):
    """One thing a yml describes, plus the columns it gives types to.

    A `models:` entry and a `sources:` table entry differ in how they are *found* - by the
    stem of a `.sql` file, or by the relation name written into one - not in what they
    say. Everything downstream of the lookup works on this.
    """

    name: str
    path: Path
    label: str = ""
    """`path` relative to the project root, for messages. Falls back to `path`."""
    source: SourceDoc = Field(default_factory=SourceDoc)
    columns: list[DeclaredColumn] = []
    span: SourceSpan | None = None
    name_span: SourceSpan | None = None

    @property
    def display_name(self) -> str:
        """What to call this in a message."""
        return self.name

    @property
    def relation_name(self) -> str:
        """The name the SQL has to write to reach this."""
        return self.name

    @property
    def where(self) -> str:
        """`models/schema.yml:12` - where the entry was written."""
        located = self.name_span or self.span
        base = self.label or str(self.path)
        return base if located is None else f"{base}:{located.start_line + 1}"

    def column(self, name: str) -> DeclaredColumn | None:
        lowered = name.lower()
        return next((c for c in self.columns if c.name.lower() == lowered), None)


class DeclaredModel(DeclaredRelation):
    """A `models:` entry. Matches the stem of the `.sql` file, as in dbt.

    Only read in a dbt project; see `DeclaredSchemas.ignored_models`.
    """


class DeclaredSourceTable(DeclaredRelation):
    """One `tables:` entry of a `sources:` entry.

    The parts the yml writes down, in order, are the relation: `database` and `schema` are
    each included only when given, so a table with neither is written bare in the SQL and
    one with both has to be written out in full. Nothing is defaulted in - a standalone
    project has no profile for sqlr to guess a database from, and guessing would mean
    silently failing to match the name the user actually wrote.
    """

    source_name: str
    database: str | None = None
    schema_name: str | None = None
    identifier: str | None = None
    """dbt's escape hatch: the real table name, when `name` is what you want to call it."""
    sql_file: str | None = None
    """The stem of the `.sql` file that builds this table - what dbt would call a model."""
    sql_file_span: SourceSpan | None = None

    @property
    def table_name(self) -> str:
        return self.identifier or self.name

    @property
    def parts(self) -> list[str]:
        parts = (self.database, self.schema_name, self.table_name)
        return [part for part in parts if part]

    @property
    def relation_name(self) -> str:
        """The relation as the SQL has to write it."""
        return ".".join(self.parts)

    @property
    def key(self) -> str:
        """`relation_name`, normalised - the identity two declarations can collide on."""
        return ".".join(part.lower() for part in self.parts)

    @property
    def display_name(self) -> str:
        """dbt's own way of naming one: the `source()` arguments, dotted."""
        return f"{self.source_name}.{self.name}"

    def matches_loosely(self, parts: list[str]) -> bool:
        """True when `parts` names this table, allowing an under-qualified reference.

        dbt-project rule only. A part the yml omits is one dbt would fill from the target
        profile, which sqlr cannot read, so it matches anything.
        """
        chain = [self.database, self.schema_name, self.table_name]
        if not parts or len(parts) > len(chain):
            return False
        tail = chain[len(chain) - len(parts) :]
        return all(
            mine is None or mine.lower() == written
            for written, mine in zip(parts, tail)
        )


class IgnoredModel(BaseModel):
    """A `models:` entry found in a project that is not a dbt project.

    Kept rather than dropped so the warning can name the entries that would have applied,
    which is the difference between a message a user can act on and noise.
    """

    name: str
    alias: str | None = None
    label: str = ""
    line: int | None = None

    @property
    def where(self) -> str:
        return self.label if self.line is None else f"{self.label}:{self.line}"

    @property
    def names(self) -> list[str]:
        """Every name this entry could match a `.sql` file by."""
        return [self.name, *([self.alias] if self.alias else [])]


class DeclaredSchemas(BaseModel):
    """Every declaration found in the project, indexed for lookup."""

    mode: DeclarationMode = "standalone"
    models: dict[str, DeclaredModel] = {}
    """Keyed by model name, lowercased. Empty outside a dbt project."""
    sources: dict[str, DeclaredSourceTable] = {}
    """Keyed by normalised relation name - `mydatabase.myschema.raw_department`."""
    ignored_models: list[IgnoredModel] = []
    errors: list[str] = []
    """Declarations that contradict each other. Fatal: there is no winner to pick."""
    warnings: list[str] = []

    def for_model(self, name: str) -> DeclaredModel | None:
        return self.models.get(name.lower())

    def for_sql_file(self, path: Path) -> DeclaredRelation | None:
        """The declaration describing what a `.sql` file produces.

        Standalone: the source table that claims the file with `sql_file:`. dbt: the
        `models:` entry of the same stem.
        """
        stem = Path(path).stem.lower()
        if self.mode == "dbt":
            return self.for_model(stem)
        return next(
            (
                table
                for table in self.sources.values()
                if table.sql_file is not None and table.sql_file.lower() == stem
            ),
            None,
        )

    def for_source(self, name: str) -> DeclaredSourceTable | None:
        """The source table a relation name refers to, if exactly one does."""
        parts = relation_parts(name)
        if not parts:
            return None
        if self.mode == "standalone":
            return self.sources.get(".".join(parts))

        matches = [
            table for table in self.sources.values() if table.matches_loosely(parts)
        ]
        # Two sources that both answer to an under-qualified name resolve to neither: the
        # query names one of them and nothing here can say which.
        return matches[0] if len(matches) == 1 else None

    def for_relation(self, name: str) -> DeclaredRelation | None:
        """The declaration a relation written in SQL should be checked against.

        Standalone projects have only sources, so the lookup is exact and total. In a dbt
        project an unqualified name is a model first - `from employee` is how one model
        reads another before templating - and a qualified one a source first.
        """
        if self.mode == "standalone":
            return self.for_source(name)

        parts = relation_parts(name)
        if not parts:
            return None
        if len(parts) == 1:
            return self.for_model(parts[0]) or self.for_source(name)
        return self.for_source(name) or self.for_model(parts[-1])


def relation_parts(name: str) -> list[str]:
    """`"MyDatabase".myschema.orders` -> `['mydatabase', 'myschema', 'orders']`."""
    return [
        stripped.lower()
        for part in name.split(".")
        if (stripped := part.strip().strip('"`[]').strip())
    ]
