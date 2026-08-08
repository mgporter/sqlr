"""Human-readable rendering of a `StatementValidation`.

Two surfaces, and the split matters: the per-table grid answers "what will be generated
for this column", one line each, for every column; the error report at the end answers
"why can these two not both be true", and is allowed to be long because there are usually
none of them.

The report quotes the SQL rather than describing it. "amount is used as a string" sends a
user hunting through the file; the comparison that proves it, with a caret under it, does
not. Cells are built as `Text` rather than markup strings for the same reason as in
`schema_resolution.render`: values come from the user's SQL, and a literal `[` would
otherwise be read as a style tag.
"""

from __future__ import annotations

from collections.abc import Iterable

from rich import box
from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from sqlr.diagnostics.types import Location
from sqlr.schema_resolution.types import ResolvedType, TypeEvidence
from sqlr.source import SourceDoc, SourceSpan
from sqlr.validation.types import (
    ColumnValidation,
    StatementValidation,
    TableValidation,
    ValidationOutcome,
)

__all__ = ["render_errors", "render_validation"]

_UNKNOWN = "dim italic"
_OUTCOME_STYLE: dict[ValidationOutcome, str] = {
    "pass": "green",
    "warning": "yellow",
    "error": "bold red",
}


def render_validation(validation: StatementValidation) -> RenderableType:
    """Every source table, then the statement's projection."""
    blocks: list[RenderableType] = [Text(validation.source.name, style="bold white")]

    if validation.tables:
        blocks.extend(_table_block(table) for table in validation.tables)
    else:
        blocks.append(Text("no source tables resolved", style=_UNKNOWN))

    blocks.append(_table_block(validation.projection))
    return Group(*blocks)


def _table_block(table: TableValidation) -> Table:
    title = Text(table.name, style="bold cyan")
    if table.star_expanded:
        title.append("  (star-expanded, may be incomplete)", style=_UNKNOWN)
    if table.declared_model is None:
        title.append("  (no declaration)", style=_UNKNOWN)

    grid = Table(
        title=title,
        title_justify="left",
        box=box.SIMPLE_HEAVY,
        header_style="bold",
        pad_edge=False,
        expand=False,
    )
    projection = table.kind == "projection"
    if projection:
        grid.add_column("#", justify="right", style=_UNKNOWN)
    grid.add_column("column", style="bold", overflow="fold")
    grid.add_column("inferred")
    grid.add_column("declared", overflow="fold")
    grid.add_column("result")
    grid.add_column("details", overflow="fold")
    grid.add_column("resolved")

    if not table.columns:
        empty = Text(
            "nothing projected" if projection else "no columns attributed",
            style=_UNKNOWN,
        )
        grid.add_row(*([""] if projection else []), empty, "", "", "", "", "")
        return grid

    for column in table.columns:
        cells = [
            Text(column.name),
            _inferred_cell(column),
            _dim_or(column.declared_type),
            Text(column.outcome, style=_OUTCOME_STYLE[column.outcome]),
            Text(column.detail, style=_UNKNOWN),
            _resolved_cell(column),
        ]
        if projection:
            cells.insert(
                0, Text("" if column.ordinal is None else str(column.ordinal))
            )
        grid.add_row(*cells)
    return grid


def _inferred_cell(column: ColumnValidation) -> Text:
    """`-` means the SQL never mentioned the column; `unknown` means it said nothing."""
    inferred = column.inferred_type
    if inferred is None:
        return Text("-", style=_UNKNOWN)
    if inferred == "unknown":
        return Text("unknown", style=_UNKNOWN)

    text = Text(inferred, style="yellow")
    source = column.inferred.source if column.inferred is not None else "unknown"
    if source != "unknown":
        text.append(f" ({source})", style=_UNKNOWN)
    return text


def _resolved_cell(column: ColumnValidation) -> Text:
    if column.resolved_type is None:
        return Text("-", style=_UNKNOWN)
    return Text(column.resolved_type, style="bold")


def _dim_or(value: str | None) -> Text:
    return Text(value) if value else Text("-", style=_UNKNOWN)


# ---- the error report ------------------------------------------------------------------


def render_errors(
    validations: Iterable[StatementValidation],
) -> RenderableType | None:
    """Every contradiction across the run, or None when there were none."""
    blocks: list[RenderableType] = []
    count = 0

    for validation in validations:
        for table, column in validation.errors():
            count += 1
            blocks.append(_error_block(validation.source, table, column))

    if not blocks:
        return None

    header = Text(
        f"{count} error{'s' if count != 1 else ''}", style="bold red"
    )
    return Group(header, Text(), *blocks)


def _error_block(
    source: SourceDoc, table: TableValidation, column: ColumnValidation
) -> RenderableType:
    declared = column.declared
    lines: list[RenderableType] = [
        Text.assemble(
            (f"{table.name}.{column.name}", "bold"),
            (" is declared ", ""),
            (declared.written_type if declared is not None else "?", "cyan"),
            (
                f" ({declared.resolved_type_name})" if declared is not None else "",
                _UNKNOWN,
            ),
            (" but the SQL uses it as ", ""),
            (column.inferred_type or "unknown", "yellow"),
        )
    ]

    inferred = column.inferred
    chosen = inferred.chosen if inferred is not None else None
    if chosen is not None:
        lines.extend(
            _evidence_block(
                source,
                chosen,
                f"inferred {chosen.type_name} here ({chosen.detail})",
            )
        )
    if inferred is not None:
        lines.extend(_supporting(source, inferred, chosen))

    if column.declaration is not None:
        lines.extend(_declaration_block(column.declaration))

    lines.append(Text())
    return Group(*lines)


def _supporting(
    source: SourceDoc, inferred: ResolvedType, chosen: TypeEvidence | None
) -> list[RenderableType]:
    """The rest of the case, deduped by what it says and where.

    The same observation reaches a column by more than one path - a derived column is typed
    both as its relation's output and again as the projected column - and repeating it
    buries the disagreement it was meant to explain.
    """
    seen: set[tuple[int, int, str, str]] = set()
    if chosen is not None:
        seen.add(_key(chosen))

    lines: list[RenderableType] = []
    for item in inferred.evidence:
        if item.location is None:
            continue
        key = _key(item)
        if key in seen:
            continue
        seen.add(key)
        lines.extend(
            _evidence_block(
                source, item, f"also used as {item.type_name} here ({item.detail})"
            )
        )
    return lines


def _key(evidence: TypeEvidence) -> tuple[int, int, str, str]:
    span = evidence.location
    return (
        span.start if span is not None else -1,
        span.end if span is not None else -1,
        evidence.type_name,
        evidence.detail or "",
    )


def _evidence_block(
    source: SourceDoc, evidence: TypeEvidence, message: str
) -> list[RenderableType]:
    span = evidence.location
    lines: list[RenderableType] = [
        Text.assemble(("  ", ""), (message, ""), (f"  {_where(source, span)}", _UNKNOWN))
    ]
    lines.extend(_excerpt(source, span))
    return lines


def _declaration_block(location: Location) -> list[RenderableType]:
    lines: list[RenderableType] = [
        Text.assemble(("  declared here", ""), (f"  {location}", _UNKNOWN))
    ]
    if location.snippet is not None:
        lines.append(Text(f"      {location.snippet}", style="cyan"))
    return lines


def _where(source: SourceDoc, span: SourceSpan | None) -> str:
    if span is None:
        return source.name
    return f"{source.name}:{span}"


def _excerpt(source: SourceDoc, span: SourceSpan | None) -> list[RenderableType]:
    """The line the evidence sits on, with a caret run under the span itself.

    Only single-line spans get carets; a multi-line span is shown as its first line with an
    ellipsis, since a caret run across a wrapped range reads as noise.
    """
    if span is None:
        return []
    lines = source.lines_of(span)
    if not lines:
        return []

    if span.start_line != span.end_line:
        return [Text(f"      {lines[0].rstrip()} ...", style="cyan")]

    line = lines[0]
    width = max(1, span.end_col - span.start_col)
    return [
        Text(f"      {line.rstrip()}", style="cyan"),
        Text(f"      {' ' * span.start_col}{'^' * width}", style="red"),
    ]
