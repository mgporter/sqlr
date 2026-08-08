import json
import logging
from pathlib import Path

import typer
from rich.console import Console

from sqlr import config as config_module
from sqlr.catalog import find_sql_files, find_yaml_files
from sqlr.config.types import SqlrConfig
from sqlr.declared import load_declared_schemas
from sqlr.diagnostics import (
    DiagnosticReport,
    check_schema,
    codes,
    from_analysis,
    render_text,
    to_lsp,
    unresolved_types,
)
from sqlr.diagnostics.types import Diagnostic, Location
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

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger("sqlr")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def check(
    project_dir: Path | None = typer.Option(
        None, "--project-dir", help="Path to the root of the project."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
    output_format: str = typer.Option(
        "text", "--format", help="Diagnostic output format: text or json (LSP shape)."
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero when any diagnostic is an error."
    ),
) -> None:
    """Analyse every SQL file in the project and report diagnostics."""
    _configure_logging(verbose)

    root, cfg = _load_project(project_dir)

    file_inventory = find_sql_files(root, cfg.general.sql_file_globs)
    logger.info("found %d SQL files in project root %s", len(file_inventory.files), root)

    # Declarations are a project-wide index: a mismatch in one file is usually against a
    # type declared for a different one, so they are loaded once, before the loop.
    declared = load_declared_schemas(find_yaml_files(root))
    logger.info("found %d declared models", len(declared.models))

    report = DiagnosticReport()
    report.extend(
        [
            Diagnostic(
                code=codes.DECLARATION_WARNING,
                severity="warning",
                message=warning,
                location=Location(),
            )
            for warning in declared.warnings
        ]
    )

    for sql_file in file_inventory.files:
        logger.debug("analyzing %s", sql_file.relative_path)
        result = analyze_file(
            sql_file.path,
            dialect=cfg.general.sql_dialect,
            star_over_join_behavior=cfg.general.star_over_join_behavior,
        )

        report.extend(
            from_analysis(
                result.source, result.errors, result.warnings, result.ambiguities
            )
        )
        if result.errors:
            continue

        logger.info(
            "analysis of %s: %d relations, %d external sources, %d projected columns",
            sql_file.relative_path,
            len(result.relations),
            len(result.sources),
            len(result.projection),
        )

        schema = resolve_schema(result)
        report.extend(check_schema(schema, declared, sql_file.path))
        report.extend(unresolved_types(schema, declared))

    if output_format == "json":
        typer.echo(json.dumps(to_lsp(report), indent=2))
    else:
        rendered = render_text(report)
        if rendered:
            typer.echo(rendered)

    if strict and report.has_errors:
        raise typer.Exit(code=1)


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
    models = _select(root, cfg, [*select, *ctx.args])

    console = Console()
    schemas: list[StatementSchema] = []

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

        schemas.append(resolve_schema(result))

    if output_format == "json":
        typer.echo(
            json.dumps([schema.model_dump(mode="json") for schema in schemas], indent=2)
        )
        return

    for schema in schemas:
        console.print(render_schema(schema), new_line_start=True)


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
