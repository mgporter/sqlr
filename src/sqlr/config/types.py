from typing import Literal

from pydantic import BaseModel, Field

from sqlr.config.defaults import (
    DEFAULT_FIXTURE_DIRECTORY,
    DEFAULT_ROW_COUNT,
    DEFAULT_SEED,
    DEFAULT_SQL_DIALECT,
    DEFAULT_SQL_FILE_GLOBS,
    DEFAULT_STAR_OVER_JOIN_BEHAVIOR,
    DEFAULT_VERSION,
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


class SqlrConfig(BaseModel):
    version: int = DEFAULT_VERSION
    general: GeneralConfig = Field(default_factory=GeneralConfig)
