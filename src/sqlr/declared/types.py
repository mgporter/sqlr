"""Types the user wrote down, as opposed to types the analyser inferred."""

from pathlib import Path

from pydantic import BaseModel, Field

from sqlr.source import SourceDoc, SourceSpan
from sqlr.typemap import ResolvedTypeName


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


class DeclaredModel(BaseModel):
    name: str
    """Matches the stem of the `.sql` file, as in dbt."""
    path: Path
    source: SourceDoc = Field(default_factory=SourceDoc)
    columns: list[DeclaredColumn] = []
    span: SourceSpan | None = None
    name_span: SourceSpan | None = None

    def column(self, name: str) -> DeclaredColumn | None:
        lowered = name.lower()
        return next((c for c in self.columns if c.name.lower() == lowered), None)


class DeclaredSchemas(BaseModel):
    """Every declaration found in the project, indexed by model name."""

    models: dict[str, DeclaredModel] = {}
    warnings: list[str] = []

    def for_model(self, name: str) -> DeclaredModel | None:
        return self.models.get(name.lower())

    def for_sql_file(self, path: Path) -> DeclaredModel | None:
        """The declaration matching a `.sql` file, by stem - dbt's convention."""
        return self.for_model(Path(path).stem)
