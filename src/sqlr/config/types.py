from typing import Literal

from pydantic import BaseModel, Field, field_validator

from sqlr.config.defaults import (
    DEFAULT_FIXTURE_DIRECTORY,
    DEFAULT_MODEL_PATHS,
    DEFAULT_ROW_COUNT,
    DEFAULT_SEED,
    DEFAULT_SQL_DIALECT,
    DEFAULT_SQL_FILE_GLOBS,
    DEFAULT_STAR_OVER_JOIN_BEHAVIOR,
    DEFAULT_VERSION,
    DEFAULT_WARN_ON_COLUMN_WITHOUT_SOURCE,
)

StarOverJoinBehavior = Literal["error", "guess"]
"""What to do when `select *` reads a join and a column cannot be attributed.

Both settings first look for evidence: a source that declares the column, or a qualified
reference elsewhere in the scope that proves ownership. They differ only in what happens
when there is none.

- `error`: fail the file, reporting the offending SELECT.
- `guess`: attribute the column to the leftmost source.
"""


class GeneralConfig(BaseModel):
    default_row_count: int = DEFAULT_ROW_COUNT
    seed: int = DEFAULT_SEED
    fixture_directory: str = DEFAULT_FIXTURE_DIRECTORY
    sql_file_globs: list[str] = Field(default_factory=lambda: list(DEFAULT_SQL_FILE_GLOBS))
    sql_dialect: str | None = DEFAULT_SQL_DIALECT
    star_over_join_behavior: StarOverJoinBehavior = DEFAULT_STAR_OVER_JOIN_BEHAVIOR  # type: ignore[assignment]
    warn_on_column_without_source: bool = DEFAULT_WARN_ON_COLUMN_WITHOUT_SOURCE
    """Whether to warn when an unqualified column is attributed by guesswork.

    A scope with several sources can still attribute a bare column with certainty, and
    those stay silent. The warning is for the case where certainty is unreachable: some
    source in the scope has an *unknown* column set, so the name might have come from it
    instead of from the source it was credited to.

    A source's column set is known when it is a CTE or derived table - its projections are
    the whole set - or when it is a table declared in a `sources.yml` with at least one
    column, which is read as the complete list rather than a sample. A table that is
    undeclared, or declared with no columns at all, is the unknown case that triggers this.
    """
    model_paths: list[str] | None = DEFAULT_MODEL_PATHS
    """Directories, relative to the project root, that hold models.

    None - the default - means the whole project is searched. Naming paths narrows model
    *selection* only: a model can still read a SQL file that lives elsewhere.
    """

    @field_validator("model_paths", "sql_file_globs", mode="before")
    @classmethod
    def _allow_a_bare_string(cls, value: object) -> object:
        """`model_paths: models` is what a person writes; treat it as a one-item list."""
        return [value] if isinstance(value, str) else value


class SqlrConfig(BaseModel):
    version: int = DEFAULT_VERSION
    general: GeneralConfig = Field(default_factory=GeneralConfig)
