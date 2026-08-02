import logging
from pathlib import Path

import typer

from sqlrunner import config as config_module
from sqlrunner.catalog import find_sql_files
from sqlrunner.schema_resolution import resolve_schema
from sqlrunner.sql_analysis import analyze_file

app = typer.Typer()
logger = logging.getLogger("sqlrunner")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def main(
    project_dir: Path | None = typer.Option(
        None, "--project-dir", help="Path to the root of the project."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    _configure_logging(verbose)

    if project_dir is None:
        project_dir = Path.cwd()

    logger.debug("resolving project root from %s", project_dir)

    try:
        root = config_module.resolve_project_root(project_dir)
        cfg = config_module.load_config(root)
    except config_module.ConfigError as e:
        logger.debug("config load failed", exc_info=e)
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e

    logger.debug("resolved project root to %s", root)
    # typer.echo(f"loaded config from {root / config_module.CONFIG_FILENAME}")
    # typer.echo(cfg.model_dump_json(indent=2))

    file_inventory = find_sql_files(root, cfg.general.sql_file_globs)
    logger.info("found %d SQL files in project root %s", len(file_inventory.files), root)

    # typer.echo(f"found {len(file_inventory.files)} SQL files:")
    # typer.echo(file_inventory.model_dump_json(indent=2))

    for sql_file in file_inventory.files:
        logger.debug("analyzing %s", sql_file.relative_path)
        result = analyze_file(sql_file.path, dialect=cfg.general.sql_dialect)
        if result.errors:
            logger.error(
                "errors analyzing %s: %s", sql_file.relative_path, result.errors
            )
        else:
            logger.info(
                "analysis of %s: %d CTEs, %d external sources",
                sql_file.relative_path,
                len(result.ctes),
                len(result.external_sources),
            )
        # typer.echo(result.model_dump_json(indent=2))

        table_schemas = resolve_schema(result)

        # for table_schema in table_schemas:
        #     typer.echo(table_schema.model_dump_json(indent=2))





if __name__ == "__main__":
    app()
