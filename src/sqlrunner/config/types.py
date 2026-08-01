from pydantic import BaseModel, Field

from sqlrunner.config.defaults import (
    DEFAULT_FIXTURE_DIRECTORY,
    DEFAULT_ROW_COUNT,
    DEFAULT_SEED,
    DEFAULT_SQL_DIALECT,
    DEFAULT_SQL_FILE_GLOBS,
    DEFAULT_VERSION,
)


class GeneralConfig(BaseModel):
    default_row_count: int = DEFAULT_ROW_COUNT
    seed: int = DEFAULT_SEED
    fixture_directory: str = DEFAULT_FIXTURE_DIRECTORY
    sql_file_globs: list[str] = Field(default_factory=lambda: list(DEFAULT_SQL_FILE_GLOBS))
    sql_dialect: str | None = DEFAULT_SQL_DIALECT


class SQLRunnerConfig(BaseModel):
    version: int = DEFAULT_VERSION
    general: GeneralConfig = Field(default_factory=GeneralConfig)
