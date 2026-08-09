from pathlib import Path

import yaml

from sqlr.config.types import SqlrConfig
from sqlr.declared.types import DeclarationMode

CONFIG_FILENAME = "sqlr.yml"

DBT_PROJECT_FILENAME = "dbt_project.yml"
"""What tells sqlr it is looking at a dbt project rather than a folder of SQL."""


class ConfigError(Exception):
    pass


def declaration_mode(project_root: Path) -> DeclarationMode:
    """Which set of yml rules the project's declarations are read under.

    The presence of a `dbt_project.yml` is the whole test. In a dbt project `models:` is
    read and dbt's own defaults apply; without one there are only `sources:`, and a
    relation is named by exactly the parts its declaration writes down.
    """
    return "dbt" if (project_root / DBT_PROJECT_FILENAME).is_file() else "standalone"


def resolve_project_root(project_dir: Path) -> Path:
    root = Path(project_dir).expanduser().resolve()
    if not root.is_dir():
        raise ConfigError(f"project dir not found: {root}")
    return root


def load_config(project_root: Path) -> SqlrConfig:
    config_path = project_root / CONFIG_FILENAME
    if not config_path.is_file():
        raise ConfigError(f"no {CONFIG_FILENAME} found at {project_root}")

    data: dict[str, object] = yaml.safe_load(config_path.read_text()) or {}
    return SqlrConfig.model_validate(data)


def load(project_dir: Path) -> SqlrConfig:
    root = resolve_project_root(project_dir)
    return load_config(root)
