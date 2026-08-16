"""Human-readable rendering of steps 1-3.

One block per model: what went wrong with it, then one table per scope showing where each
column ended up. The table answers the question the qualification pass exists to answer -
*which source is this name actually reading from* - and says how confident that answer is,
because "you wrote it" and "we worked it out" are not the same claim.

Cells are built as `Text` rather than markup strings for the same reason as in
`schema_resolution.render`: names come from the user's SQL, and a literal `[` would
otherwise be read as a style tag.
"""

from __future__ import annotations

from rich import box
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from sqlr.sql_analysis2.qualify import (
    ColumnQualifierOrigin,
    ColumnReference,
    ColumnSource,
    ProjectedColumn,
    QualifiedModel,
    ScopeColumns,
    columns_per_scope,
)
from sqlr.sql_analysis2.reporting import ColumnFinding
from sqlr.sql_analysis2.types import TableName

__all__ = ["print_qualification", "render_qualification"]

_UNKNOWN = "dim italic"
_SEVERITY_STYLE = {"error": "bold red", "warning": "yellow"}
_ORIGIN_STYLE: dict[ColumnQualifierOrigin, str] = {
    # Written is the uninteresting case and gets no colour; the other two are claims the
    # pass made on the user's behalf, and are the reason to read the column at all.
    "written": "",
    "inferred": "yellow",
    "star": "cyan",
}


def render_qualification(result: QualifiedModel) -> RenderableType:
    """One model: its findings, then its scopes."""
    blocks: list[RenderableType] = [
        Text(str(result.model.relative_path), style="bold white")
    ]

    blocks.extend(Text(f"error: {error}", style="bold red") for error in result.errors)
    blocks.extend(_finding_line(finding) for finding in result.findings)

    if result.statement is None:
        blocks.append(Text("not qualified", style=_UNKNOWN))
        return Group(*blocks)

    scopes = columns_per_scope(result.statement)
    if not scopes:
        blocks.append(Text("no columns read", style=_UNKNOWN))
        return Group(*blocks)

    for scope in scopes:
        blocks.extend(_scope_blocks(scope))
    return Group(*blocks)


def print_qualification(results: list[QualifiedModel]) -> None:
    """The CLI's view. An editor consumes the results themselves instead."""
    console = Console()
    for result in results:
        console.print(render_qualification(result), new_line_start=True)


def _finding_line(finding: ColumnFinding) -> Text:
    """`error 9:3: ...`. The file is the block header, so only the position is repeated."""
    where = f" {finding.span}" if finding.span is not None else ""
    return Text.assemble(
        (finding.severity, _SEVERITY_STYLE[finding.severity]),
        (f"{where}: ", _UNKNOWN),
        (finding.message, ""),
    )


def _scope_blocks(scope: ScopeColumns) -> list[RenderableType]:
    """A scope header, then whichever of its two column tables has rows.

    An empty section is dropped rather than printed empty: "this CTE filters on nothing"
    is not a fact worth a header, and the blank grid reads as a bug.
    """
    title = Text(scope.name, style="bold cyan")
    if scope.kind != "final":
        # `<final>  (final)` says one thing twice; every other name needs its kind.
        title.append(f"  ({scope.kind})", style=_UNKNOWN)

    blocks: list[RenderableType] = [Text(), title]
    if scope.projected:
        blocks.append(_projected_table(scope.projected))
    if scope.non_projected:
        blocks.append(_non_projected_table(scope.non_projected))
    return blocks


def _grid(section: str) -> Table:
    grid = Table(
        title=Text(section, style="bold"),
        title_justify="left",
        box=box.SIMPLE_HEAVY,
        header_style="bold",
        pad_edge=False,
        expand=False,
    )
    grid.add_column("column", style="bold", overflow="fold")
    grid.add_column("source", overflow="fold")
    grid.add_column("qualifier")
    grid.add_column("declared")
    return grid


def _projected_table(columns: list[ProjectedColumn]) -> RenderableType:
    """The scope's output schema, in projection order."""
    grid = _grid("projected")
    for column in columns:
        name = Text(column.name)
        if column.engine_named:
            # The name is the dialect's rule applied to an unnamed projection, not
            # something in the file. A reader quoting it downstream should know that.
            name.append("  (engine-named)", style=_UNKNOWN)
        grid.add_row(
            name,
            _sources_cell([read.source for read in column.reads]),
            Text(column.origin, style=_ORIGIN_STYLE[column.origin]),
            Text("declared") if column.declared else Text("-", style=_UNKNOWN),
        )
    return Padding(grid, (0, 0, 0, 2))


def _non_projected_table(columns: list[ColumnReference]) -> RenderableType:
    grid = _grid("non-projected")
    for column in columns:
        grid.add_row(
            Text(column.name),
            _sources_cell([column.source]),
            Text(column.origin, style=_ORIGIN_STYLE[column.origin]),
            Text("declared") if column.declared else Text("-", style=_UNKNOWN),
        )
    return Padding(grid, (0, 0, 0, 2))


def _sources_cell(sources: list[ColumnSource]) -> Text:
    """Every relation the column reads, deduplicated, or `-` when it reads none.

    `1 + 1` reads nothing and gets the dash; `a.x + b.y` reads two and names both, because
    the column depends on both relations and a single answer would have to pick one.
    """
    text = Text()
    seen: set[TableName] = set()
    for source in sources:
        if not source.alias or source.alias in seen:
            continue
        seen.add(source.alias)
        if text:
            text.append(", ", style=_UNKNOWN)
        text.append(source.alias)
        if source.name is not None:
            text.append(f" ({source.name})", style=_UNKNOWN)
        elif source.kind != "table":
            text.append(f" ({source.kind})", style=_UNKNOWN)
    return text if text else Text("-", style=_UNKNOWN)
