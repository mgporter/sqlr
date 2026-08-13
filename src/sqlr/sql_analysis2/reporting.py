"""Turning what the pipeline found into something with a place in the source.

A finding carries spans rather than a formatted line, so the same value serves the CLI and
an editor that wants to underline the offending text.
"""

from typing import Literal

from pydantic import BaseModel
from sqlglot import exp

from sqlr.sql_analysis2.sourcedoc import Positions, SourceSpan
from sqlr.sql_analysis2.types import (
    ColumnName,
    ColumnTypeName,
    ParsedColumn,
    TableName,
)
from sqlr.typemap import resolve_type_name

type FindingCode = Literal[
    "unresolvable-column",
    "dot-access-unsupported",
    "structured-column-declared-scalar",
]


class ColumnFinding(BaseModel):
    """One reportable problem, and where in the SQL it was written.

    Two spans, because an editor wants both: `span` covers the offending name, which is
    what to underline, and `access_span` covers the whole read it appeared in.
    """

    code: FindingCode
    column_name: ColumnName
    message: str
    span: SourceSpan | None = None
    access_span: SourceSpan | None = None

    def where(self) -> str:
        """`:10:3`, or nothing when the node carries no position."""
        return f":{self.span}" if self.span is not None else ""


def span_of_access(column: exp.Column, positions: Positions) -> SourceSpan | None:
    """The span of the read a column sits inside - `x.field`, `x['field']`.

    sqlglot puts no position on a closing bracket, so the hull stops one character short
    of it; that character is added back here.
    """
    parent = column.parent
    if not isinstance(parent, (exp.Dot, exp.Bracket)):
        return None
    span = positions.span_of(parent)
    if span is None:
        return None
    if isinstance(parent, exp.Bracket) and positions.text[span.end : span.end + 1] == "]":
        return positions.span(span.start, span.end + 1)
    return span


def findings_for_unresolvable_columns(
    columns: list[exp.Column], positions: Positions
) -> list[ColumnFinding]:
    """A column no source can own - a typo, or a missing join."""
    return [
        ColumnFinding(
            code="unresolvable-column",
            column_name=column.name,
            message=f"column '{column.name}' could not be resolved to any source",
            span=positions.span_of(column),
        )
        for column in columns
    ]


def findings_for_columns_read_with_unsupported_dot_notation(
    columns: list[exp.Column], positions: Positions, dialect_name: str
) -> list[ColumnFinding]:
    """One finding per read site.

    The message covers both readings of the same tree, because nothing distinguishes them
    once sqlglot's `_convert_columns_to_dots` has run: the name is either a mistyped
    source or a structured column read with syntax the dialect does not have.
    """
    return [
        ColumnFinding(
            code="dot-access-unsupported",
            column_name=column.name,
            message=(
                f"'{column.name}' matches no source, and {dialect_name} does not read a "
                "dotted name as a field of a structured column"
            ),
            span=positions.span_of(column),
            access_span=span_of_access(column, positions),
        )
        for column in columns
    ]


def findings_for_columns_declared_as_scalar_but_read_as_structured(
    declared: dict[TableName, dict[ColumnName, ColumnTypeName]],
    columns_per_table: dict[TableName, dict[ColumnName, ParsedColumn]],
    positions: Positions,
) -> list[ColumnFinding]:
    """A column read as `x.field` or `x['field']` cannot hold a scalar.

    Only a declaration that lands on the lattice is contradicted: every `ResolvedTypeName`
    is scalar, so `varchar` is a mistake, while `json`, `struct(...)`, `map` and `variant`
    all resolve to `unknown` and say nothing. An unrecognised name resolves to `unknown`
    too, and is somebody else's finding.
    """
    findings: list[ColumnFinding] = []
    for table_name, columns in columns_per_table.items():
        for column in columns.values():
            if not column.requires_structured_type:
                continue
            written = declared.get(table_name, {}).get(column.name)
            if written is None or resolve_type_name(written) == "unknown":
                continue
            findings.extend(
                ColumnFinding(
                    code="structured-column-declared-scalar",
                    column_name=column.name,
                    message=(
                        f"column '{column.name}' of '{table_name}' is declared "
                        f"'{written}', but is read as a structured value; declare it as "
                        "a struct, json, map or variant type"
                    ),
                    span=positions.span_of(node),
                    access_span=span_of_access(node, positions),
                )
                for _, node in column.structured_access
            )
    return findings


def print_findings(findings: list[ColumnFinding], relative_path: str) -> None:
    """The CLI's view. An editor consumes the findings themselves instead."""
    for finding in findings:
        print(f"error: {relative_path}{finding.where()}: {finding.message}")
