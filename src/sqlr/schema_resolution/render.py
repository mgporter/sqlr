"""Human-readable rendering of a `StatementSchema`.

The model keeps every piece of evidence it saw, which is what a diagnostic needs and what
a person reading a terminal does not. This module keeps the four things a reader asks for
first - which column, what type, can it be null, what has to be true of its values - and
drops the rest.

Cells are built as `Text` rather than markup strings: constraint values come from the
user's SQL, and a literal containing `[` would otherwise be read as a style tag.
"""

from __future__ import annotations

from rich import box
from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from sqlr.schema_resolution.types import (
    ColumnSchema,
    ProjectionSchema,
    StatementSchema,
    TableSchema,
    ValueConstraint,
)
from sqlr.sql_analysis.types import ColumnNode

__all__ = ["render_schema"]

_UNKNOWN = "dim italic"
_CONFIDENCE_STYLE = {"explicit": "", "inferred": "yellow", "guessed": "red"}


def render_schema(schema: StatementSchema) -> RenderableType:
    """Every source table, then the statement's projection."""
    blocks: list[RenderableType] = [_source_header(schema)]

    if schema.tables:
        blocks.extend(_table_block(table) for table in schema.tables)
    else:
        blocks.append(Text("no source tables resolved", style=_UNKNOWN))

    blocks.append(_projection_block(schema.projection))
    return Group(*blocks)


def _source_header(schema: StatementSchema) -> Text:
    return Text(schema.source.name, style="bold white")


def _table_block(table: TableSchema) -> Table:
    title = Text(table.name, style="bold cyan")
    if table.star_expanded:
        title.append("  (star-expanded, may be incomplete)", style=_UNKNOWN)

    grid = Table(
        title=title,
        title_justify="left",
        box=box.SIMPLE_HEAVY,
        header_style="bold",
        pad_edge=False,
        expand=False,
    )
    grid.add_column("column", style="bold", overflow="fold")
    grid.add_column("type")
    grid.add_column("nullable", justify="center")
    grid.add_column("constraints", overflow="fold")

    if not table.columns:
        grid.add_row(Text("no columns attributed", style=_UNKNOWN), "", "", "")
        return grid

    for column in table.columns:
        grid.add_row(
            _column_name(column),
            _type_cell(column),
            _nullable_cell(column),
            _constraints_cell(column.constraints),
        )
    return grid


def _projection_block(projection: list[ProjectionSchema]) -> Table:
    grid = Table(
        title=Text("projection", style="bold cyan"),
        title_justify="left",
        box=box.SIMPLE_HEAVY,
        header_style="bold",
        pad_edge=False,
        expand=False,
    )
    grid.add_column("#", justify="right", style=_UNKNOWN)
    grid.add_column("column", style="bold", overflow="fold")
    grid.add_column("type")
    grid.add_column("from", overflow="fold")

    if not projection:
        grid.add_row("", Text("nothing projected", style=_UNKNOWN), "", "")
        return grid

    for column in projection:
        name = (
            Text(column.name)
            if column.name
            else Text("<unnamed>", style=_UNKNOWN)
        )
        origins = Text(
            ", ".join(_origin_name(origin) for origin in column.origins) or "-",
            style=_UNKNOWN,
        )
        grid.add_row(str(column.ordinal), name, _type_cell(column), origins)
    return grid


def _origin_name(origin: ColumnNode) -> str:
    """`table:person.id` reads as `person.id`; other kinds keep their prefix.

    A base table is the common case and its kind carries no information, but a `cte:` or
    `derived:` prefix is the whole point of the line.
    """
    if origin.relation.kind == "table":
        return f"{origin.relation.name}.{origin.column}"
    return str(origin)


def _column_name(column: ColumnSchema) -> Text:
    """The name, tagged when the column was not written out explicitly."""
    text = Text(column.name)
    # if column.confidence != "explicit":
    #     text.append(f" ({column.confidence})", style=_CONFIDENCE_STYLE[column.confidence])
    return text


def _type_cell(column: ColumnSchema | ProjectionSchema) -> Text:
    """The resolved type, plus the coarse bucket of the evidence that decided it."""
    resolved = column.resolved_type
    if resolved.type_name == "unknown":
        return Text("unknown", style=_UNKNOWN)

    text = Text(resolved.type_name, style="yellow")
    if resolved.source != "unknown":
        text.append(f" ({resolved.source})", style=_UNKNOWN)
    return text


def _nullable_cell(column: ColumnSchema) -> Text:
    """`-` means the statement said nothing, not that the column is non-null."""
    nullable = column.nullable
    if nullable is None:
        return Text("-", style=_UNKNOWN)
    if nullable:
        return Text("true", style="green")
    return Text("false", style="red")


def _constraints_cell(constraints: list[ValueConstraint]) -> Text:
    if not constraints:
        return Text("")
    text = Text()
    for index, constraint in enumerate(constraints):
        if index:
            text.append(", ", style=_UNKNOWN)
        text.append_text(_constraint_text(constraint))
    return text


def _constraint_text(constraint: ValueConstraint) -> Text:
    """A predicate written back out roughly as it was in the SQL.

    Values are stored unquoted, so string literals are re-quoted here; without it a
    `= US` reads as a column reference rather than a value.
    """
    operator = constraint.operator
    values = [_literal(value, constraint) for value in constraint.values]

    if operator == "is_null":
        return Text("is null", style="magenta")
    if operator == "is_not_null":
        return Text("is not null", style="magenta")

    text = Text()
    if operator in ("in", "not_in"):
        text.append("in " if operator == "in" else "not in ", style="magenta")
        text.append("(", style=_UNKNOWN)
        for index, value in enumerate(values):
            if index:
                text.append(", ", style=_UNKNOWN)
            text.append(value, style="cyan")
        text.append(")", style=_UNKNOWN)
        return text

    if operator == "between" and len(values) == 2:
        text.append("between ", style="magenta")
        text.append(values[0], style="cyan")
        text.append(" and ", style="magenta")
        text.append(values[1], style="cyan")
        return text

    text.append(f"{operator} ", style="magenta")
    text.append(" ".join(values), style="cyan")
    return text


def _literal(value: str, constraint: ValueConstraint) -> str:
    if constraint.literal_kind == "string":
        return f"'{value}'"
    return value
