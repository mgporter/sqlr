from pathlib import Path

import yaml

from sqlrunner.config.types import SQLRunnerConfig

CONFIG_FILENAME = "sqlrunner.yml"


class ConfigError(Exception):
    pass


def resolve_project_root(project_dir: Path) -> Path:
    root = Path(project_dir).expanduser().resolve()
    if not root.is_dir():
        raise ConfigError(f"project dir not found: {root}")
    return root


def load_config(project_root: Path) -> SQLRunnerConfig:
    config_path = project_root / CONFIG_FILENAME
    if not config_path.is_file():
        raise ConfigError(f"no {CONFIG_FILENAME} found at {project_root}")

    data: dict[str, object] = yaml.safe_load(config_path.read_text()) or {}
    return SQLRunnerConfig.model_validate(data)


def load(project_dir: Path) -> SQLRunnerConfig:
    root = resolve_project_root(project_dir)
    return load_config(root)
