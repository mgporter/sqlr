import json
import logging
from pathlib import Path

import typer

from sqlrunner import config as config_module
from sqlrunner.catalog import find_sql_files, find_yaml_files
from sqlrunner.declared import load_declared_schemas
from sqlrunner.diagnostics import (
    DiagnosticReport,
    check_schema,
    codes,
    from_analysis,
    render_text,
    to_lsp,
    unresolved_types,
)
from sqlrunner.diagnostics.types import Diagnostic, Location
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
    output_format: str = typer.Option(
        "text", "--format", help="Diagnostic output format: text or json (LSP shape)."
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero when any diagnostic is an error."
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
        print(f"{schema.model_dump_json(indent=2)}")
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


if __name__ == "__main__":
    app()
