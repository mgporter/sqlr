import json
import logging
from pathlib import Path

import typer
from rich.console import Console

from sqlr import config as config_module
from sqlr.catalog import find_yaml_files
from sqlr.config.types import SqlrConfig
from sqlr.declared import load_declared_schemas
from sqlr.schema_resolution import resolve_schema
from sqlr.schema_resolution.render import render_schema
from sqlr.schema_resolution.types import StatementSchema
from sqlr.selection import (
    Model,
    SelectionError,
    build_model_index,
    select_models,
)
from sqlr.sql_analysis import analyze_file
from sqlr.validation import render_errors, render_validation, validate_schema
from sqlr.validation.types import StatementValidation

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger("sqlr")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@app.command(
    "infer-schema",
    context_settings={"allow_extra_args": True},
)
def infer_schema(
    ctx: typer.Context,
    select: list[str] = typer.Option(
        [],
        "--select",
        "-s",
        help="Model names to operate on, e.g. `--select orders customers`. "
        "Omit to operate on every model.",
    ),
    project_dir: Path | None = typer.Option(
        None, "--project-dir", help="Path to the root of the project."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
    output_format: str = typer.Option(
        "table", "--format", help="Schema output format: table or json."
    ),
) -> None:
    """Infer and print the schema of the selected models."""
    _configure_logging(verbose)

    root, cfg = _load_project(project_dir)
    # Click has no variadic option, so `--select a b` leaves `b` in the extra args. They
    # are selectors too; anything starting with `-` still fails as an unknown option.
    analyzed = _analyze(root, cfg, [*select, *ctx.args])
    schemas = [schema for _, schema in analyzed]

    if output_format == "json":
        typer.echo(
            json.dumps([schema.model_dump(mode="json") for schema in schemas], indent=2)
        )
        return

    console = Console()
    for schema in schemas:
        console.print(render_schema(schema), new_line_start=True)


@app.command(
    "validate-schema",
    context_settings={"allow_extra_args": True},
)
def validate(
    ctx: typer.Context,
    select: list[str] = typer.Option(
        [],
        "--select",
        "-s",
        help="Model names to operate on, e.g. `--select orders customers`. "
        "Omit to operate on every model.",
    ),
    project_dir: Path | None = typer.Option(
        None, "--project-dir", help="Path to the root of the project."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
    output_format: str = typer.Option(
        "table", "--format", help="Validation output format: table or json."
    ),
) -> None:
    """Infer the schema of the selected models and check it against what they declare.

    Exits non-zero when a column's inferred and declared types cannot both be true.
    """
    _configure_logging(verbose)

    root, cfg = _load_project(project_dir)
    analyzed = _analyze(root, cfg, [*select, *ctx.args])

    # Declarations are a project-wide index: a selected model's source tables are usually
    # declared in some other model's yml, so every yml is loaded regardless of selection.
    declared = load_declared_schemas(find_yaml_files(root))
    logger.info("found %d declared models", len(declared.models))
    for warning in declared.warnings:
        typer.echo(f"warning: {warning}", err=True)

    validations: list[StatementValidation] = [
        validate_schema(schema, declared, model.path) for model, schema in analyzed
    ]

    if output_format == "json":
        typer.echo(
            json.dumps(
                [validation.model_dump(mode="json") for validation in validations],
                indent=2,
            )
        )
    else:
        console = Console()
        for validation in validations:
            console.print(render_validation(validation), new_line_start=True)

        # Errors go last, after every model's table, so the summary is the last thing on
        # screen rather than buried above the grids it refers to.
        errors = render_errors(validations)
        if errors is not None:
            console.print(errors, new_line_start=True)

    if any(validation.has_errors for validation in validations):
        raise typer.Exit(code=1)


def _load_project(project_dir: Path | None) -> tuple[Path, SqlrConfig]:
    """The project root and its config, or exit 1 with the reason."""
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
    return root, cfg


def _analyze(
    root: Path, cfg: SqlrConfig, selectors: list[str]
) -> list[tuple[Model, StatementSchema]]:
    """Resolve selectors, analyse each model, and resolve its schema.

    The whole of `infer-schema`; `validate-schema` is this plus a comparison. An analysis
    error is fatal for the run rather than skipped, because a schema resolved from a
    statement that did not parse would be quietly wrong rather than visibly absent.
    """
    models = _select(root, cfg, selectors)
    analyzed: list[tuple[Model, StatementSchema]] = []

    for model in models:
        logger.debug("analyzing %s", model.relative_path)
        result = analyze_file(
            model.path,
            dialect=cfg.general.sql_dialect,
            star_over_join_behavior=cfg.general.star_over_join_behavior,
        )

        for warning in result.warnings:
            logger.warning("%s: %s", model.relative_path, warning)

        if result.errors:
            for error in result.errors:
                typer.echo(f"error: {model.relative_path}: {error}", err=True)
            raise typer.Exit(code=1)

        logger.info(
            "analysis of %s: %d relations, %d external sources, %d projected columns",
            model.relative_path,
            len(result.relations),
            len(result.sources),
            len(result.projection),
        )

        analyzed.append((model, resolve_schema(result)))

    return analyzed


def _select(root: Path, cfg: SqlrConfig, selectors: list[str]) -> list[Model]:
    """Resolve selectors against the project, or exit 1 with the reason.

    A duplicate model name and an unknown selector are both user-fixable project
    problems, not crashes, so they are reported as messages and not tracebacks.
    """
    try:
        index = build_model_index(root, cfg)
        models = select_models(index, selectors)
    except SelectionError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e

    if not models:
        typer.echo(f"no models found under {root}", err=True)
        raise typer.Exit(code=1)

    logger.info("selected %d of %d models", len(models), len(index.models))
    return models


if __name__ == "__main__":
    app()
