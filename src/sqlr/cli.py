# import json
import logging
from pathlib import Path

import typer
# from rich.console import Console

from sqlr import config as config_module
from sqlr.catalog import find_yaml_files
from sqlr.config.types import SqlrConfig
from sqlr.declared import (
    check_sql_file_links,
    ignored_models_warning,
    load_declared_schemas,
)
from sqlr.declared.types import DeclaredSchemas
# from sqlr.schema_resolution import resolve_schema
# from sqlr.schema_resolution.render import render_schema
# from sqlr.schema_resolution.types import StatementSchema
from sqlr.selection import (
    Model,
    SelectionError,
    build_model_index,
    select_models,
)
from sqlr.selection.types import ModelIndex
# from sqlr.sql_analysis import analyze_file
from sqlr.sql_analysis2 import any_model_has_errors, qualify_schema, validate_schema
from sqlr.sql_analysis2.render import print_annotations, print_qualification

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger("sqlr")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

# To be removed
# @app.command(
#     "infer-schema",
#     context_settings={"allow_extra_args": True},
# )
# def infer_schema(
#     ctx: typer.Context,
#     select: list[str] = typer.Option(
#         [],
#         "--select",
#         "-s",
#         help="Model names to operate on, e.g. `--select orders customers`. "
#         "Omit to operate on every model.",
#     ),
#     project_dir: Path | None = typer.Option(
#         None, "--project-dir", help="Path to the root of the project."
#     ),
#     verbose: bool = typer.Option(
#         False, "--verbose", "-v", help="Enable debug logging."
#     ),
#     output_format: str = typer.Option(
#         "table", "--format", help="Schema output format: table or json."
#     ),
# ) -> None:
#     """Infer and print the schema of the selected models."""
#     _configure_logging(verbose)

#     root, cfg = _load_project(project_dir)
#     # Click has no variadic option, so `--select a b` leaves `b` in the extra args. They
#     # are selectors too; anything starting with `-` still fails as an unknown option.
#     _, models = _select(root, cfg, [*select, *ctx.args])
#     schemas = [schema for _, schema in _analyze(cfg, models)]

#     if output_format == "json":
#         typer.echo(
#             json.dumps([schema.model_dump(mode="json") for schema in schemas], indent=2)
#         )
#         return

#     console = Console()
#     for schema in schemas:
#         console.print(render_schema(schema), new_line_start=True)


@app.command(
    "qualify-schema",
    context_settings={"allow_extra_args": True},
)
def qualify(
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
) -> None:
    """Resolve every column in the selected models to the source it reads from.

    The first half of `validate-schema`, on its own: stars are expanded and every column
    is attributed, but nothing is typed and nothing is compared against a declaration.

    Exits non-zero when a column cannot be attributed at all.
    """
    _configure_logging(verbose)

    root, cfg = _load_project(project_dir)
    index, models = _select(root, cfg, [*select, *ctx.args])
    declared = _load_declarations(root, index)

    qualified = qualify_schema(cfg, declared, models)
    print_qualification(qualified)

    if any_model_has_errors(qualified):
        raise typer.Exit(code=1)


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
) -> None:
    """Type every expression in the selected models and check it against what they declare.

    Runs `qualify-schema`'s resolution first, then annotates types on top of it.

    Exits non-zero when a column's inferred and declared types cannot both be true.
    """
    _configure_logging(verbose)

    root, cfg = _load_project(project_dir)
    index, models = _select(root, cfg, [*select, *ctx.args])

    # Declarations are loaded before anything is analysed: a yml that describes one
    # relation twice has no right answer to pick, and running the comparison anyway would
    # report whichever of the two happened to be read first as though it were the rule.
    declared = _load_declarations(root, index)

    results = validate_schema(cfg, declared, models)

    for result in results:
        print(result._asdict())

    # Both halves are printed: where each column comes from, then what type it is. The
    # second is unreadable without the first - a type nobody can trace back to a source is
    # a number on a page.
    # print_qualification([result.qualified for result in results])
    print_annotations(results)

    if any_model_has_errors(results):
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


def _load_declarations(root: Path, index: ModelIndex) -> DeclaredSchemas:
    """Every declaration in the project, or exit 1 with what is wrong with them.

    Project-wide regardless of selection: a selected model's upstream tables are usually
    described in some other file's yml. Two of the checks need the model index rather than
    the yml alone - whether a `sql_file:` names a real file, and whether an ignored
    `models:` entry would have described one - so they run here.
    """
    mode = config_module.declaration_mode(root)
    logger.debug("reading declarations in %s mode", mode)

    declared = load_declared_schemas(find_yaml_files(root), mode)
    logger.info(
        "found %d declared sources and %d declared models",
        len(declared.sources),
        len(declared.models),
    )

    warnings = list(declared.warnings)
    ignored = ignored_models_warning(declared, index.names, root)
    if ignored is not None:
        warnings.append(ignored)
    for warning in warnings:
        typer.echo(f"warning: {warning}", err=True)

    errors = [*declared.errors, *check_sql_file_links(declared, index.names)]
    for error in errors:
        typer.echo(f"error: {error}", err=True)
    if errors:
        raise typer.Exit(code=1)

    return declared


# def _analyze(
#     cfg: SqlrConfig, models: list[Model]
# ) -> list[tuple[Model, StatementSchema]]:
#     """Analyse each model and resolve its schema.

#     The whole of `infer-schema`; `validate-schema` is this plus a comparison. An analysis
#     error is fatal for the run rather than skipped, because a schema resolved from a
#     statement that did not parse would be quietly wrong rather than visibly absent.
#     """
#     analyzed: list[tuple[Model, StatementSchema]] = []

#     for model in models:
#         logger.debug("analyzing %s", model.relative_path)
#         result = analyze_file(
#             model.path,
#             dialect=cfg.general.sql_dialect,
#             star_over_join_behavior=cfg.general.star_over_join_behavior,
#         )

#         for warning in result.warnings:
#             logger.warning("%s: %s", model.relative_path, warning)

#         if result.errors:
#             for error in result.errors:
#                 typer.echo(f"error: {model.relative_path}: {error}", err=True)
#             raise typer.Exit(code=1)

#         logger.info(
#             "analysis of %s: %d relations, %d external sources, %d projected columns",
#             model.relative_path,
#             len(result.relations),
#             len(result.sources),
#             len(result.projection),
#         )

#         analyzed.append((model, resolve_schema(result)))

#     return analyzed


def _select(
    root: Path, cfg: SqlrConfig, selectors: list[str]
) -> tuple[ModelIndex, list[Model]]:
    """Resolve selectors against the project, or exit 1 with the reason.

    The whole index comes back beside the selection because declarations are checked
    against every model in the project, not only the ones being run.

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
    return index, models


if __name__ == "__main__":
    app()
