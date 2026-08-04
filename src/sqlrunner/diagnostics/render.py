"""Two renderings of the same report: one for a terminal, one for an editor.

`to_lsp` exists now rather than later on purpose. It is the shape the eventual VS Code
extension consumes, and producing it is what proves the spans are actually right - a
range that does not slice back to the offending text shows up immediately here.
"""

from __future__ import annotations

from typing import Any

from sqlrunner.diagnostics.types import (
    LSP_SEVERITY,
    Diagnostic,
    DiagnosticReport,
    Location,
)

def render_text(report: DiagnosticReport) -> str:
    blocks = [_render_one(diagnostic) for diagnostic in report.sorted()]
    if not blocks:
        return ""
    counts = report.counts()
    summary = ", ".join(
        f"{count} {name}{'s' if count != 1 else ''}"
        for name, count in sorted(counts.items())
    )
    return "\n\n".join(blocks) + f"\n\n{summary}"


def _render_one(diagnostic: Diagnostic) -> str:
    lines = [
        f"{diagnostic.severity}: {diagnostic.message} [{diagnostic.code}]",
        f"  {diagnostic.location}",
    ]
    underline = _underline(diagnostic.location)
    if underline:
        lines.append(underline)

    for related in diagnostic.related:
        lines.append(f"  {related.message}:")
        lines.append(f"    {related}")
        related_underline = _underline(related, indent="    ")
        if related_underline:
            lines.append(related_underline)

    return "\n".join(lines)


def _underline(location: Location, indent: str = "  ") -> str:
    """The offending text with a caret run under it.

    Only single-line spans get carets; a multi-line span is shown as its first line with
    an ellipsis, since a caret run across a wrapped range reads as noise.
    """
    if location.span is None or location.snippet is None:
        return ""
    span = location.span
    if span.start_line != span.end_line:
        first = location.snippet.splitlines()[0]
        return f"{indent}  {first} ..."
    width = max(1, span.end_col - span.start_col)
    return f"{indent}  {location.snippet}\n{indent}  {'^' * width}"


def to_lsp(report: DiagnosticReport) -> list[dict[str, Any]]:
    """LSP `Diagnostic` objects, 0-based, one per finding.

    `uri` is left as a plain path string; whichever transport carries these knows how to
    turn a path into a URI, and this module should not guess a scheme.
    """
    return [_to_lsp_one(diagnostic) for diagnostic in report.sorted()]


def _to_lsp_one(diagnostic: Diagnostic) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "code": diagnostic.code,
        "severity": LSP_SEVERITY[diagnostic.severity],
        "message": diagnostic.message,
        "source": "sqlrunner",
        "range": _range(diagnostic.location),
    }
    if diagnostic.location.path is not None:
        entry["uri"] = str(diagnostic.location.path)
    if diagnostic.related:
        entry["relatedInformation"] = [
            {
                "message": related.message,
                "location": {
                    "uri": str(related.path) if related.path is not None else "",
                    "range": _range(related),
                },
            }
            for related in diagnostic.related
        ]
    return entry


def _range(location: Location) -> dict[str, dict[str, int]]:
    span = location.span
    if span is None:
        return {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 0}}
    return {
        "start": {"line": span.start_line, "character": span.start_col},
        "end": {"line": span.end_line, "character": span.end_col},
    }
