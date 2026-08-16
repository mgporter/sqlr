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
    QualifiedModel,
    ScopeColumns,
    column_references_per_scope,
)
from sqlr.sql_analysis2.reporting import ColumnFinding

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

    scopes = column_references_per_scope(result.statement)
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
        blocks.append(_column_table("projected", scope.projected))
    if scope.non_projected:
        blocks.append(_column_table("non-projected", scope.non_projected))
    return blocks


def _column_table(section: str, columns: list[ColumnReference]) -> RenderableType:
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

    for column in columns:
        grid.add_row(
            Text(column.name),
            _source_cell(column),
            Text(column.origin, style=_ORIGIN_STYLE[column.origin]),
            Text("declared") if column.declared else Text("-", style=_UNKNOWN),
        )
    return Padding(grid, (0, 0, 0, 2))


def _source_cell(column: ColumnReference) -> Text:
    """The alias the column carries, plus whatever the alias is hiding.

    An alias that is also the table name says everything already; one that is not hides
    the relation the reader is looking for, and a CTE alias hides that it is not a table
    at all.
    """
    if not column.source_alias:
        return Text("-", style=_UNKNOWN)

    text = Text(column.source_alias)
    if column.source_name is not None:
        text.append(f" ({column.source_name})", style=_UNKNOWN)
    elif column.source_kind != "table":
        text.append(f" ({column.source_kind})", style=_UNKNOWN)
    return text
