"""Structured, locatable findings.

Every other module writes into this one. A diagnostic is deliberately more than a string:
it carries the range to underline, and a list of *related* ranges, because the interesting
findings are the ones with several sites. "This is declared varchar but used as a number"
is only actionable when it can show the declaration and every use at once.

The shape is chosen to convert to an LSP diagnostic without loss - see `render.to_lsp`.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from sqlrunner.source import SourceDoc, SourceSpan

Severity = Literal["error", "warning", "info", "hint"]

SEVERITY_RANK: dict[Severity, int] = {"error": 0, "warning": 1, "info": 2, "hint": 3}

LSP_SEVERITY: dict[Severity, int] = {"error": 1, "warning": 2, "info": 3, "hint": 4}


class Location(BaseModel):
    path: Path | None = None
    span: SourceSpan | None = None
    snippet: str | None = None
    """The source text the span covers, resolved when the location is built."""

    @classmethod
    def of(cls, source: SourceDoc, span: SourceSpan | None) -> "Location":
        return cls(path=source.path, span=span, snippet=source.slice(span))

    @property
    def line(self) -> int | None:
        """1-based, for humans."""
        return None if self.span is None else self.span.start_line + 1

    def __str__(self) -> str:
        name = str(self.path) if self.path is not None else "<sql>"
        if self.span is None:
            return name
        return f"{name}:{self.span.start_line + 1}:{self.span.start_col + 1}"


class Related(Location):
    """Another place that bears on the finding, with a note on why it matters."""

    message: str = ""

    @classmethod
    def at(
        cls, source: SourceDoc, span: SourceSpan | None, message: str
    ) -> "Related":
        return cls(
            path=source.path,
            span=span,
            snippet=source.slice(span),
            message=message,
        )


class Diagnostic(BaseModel):
    code: str
    severity: Severity
    message: str
    location: Location = Location()
    related: list[Related] = []
    table: str | None = None
    column: str | None = None


class DiagnosticReport(BaseModel):
    diagnostics: list[Diagnostic] = []

    @property
    def has_errors(self) -> bool:
        return any(d.severity == "error" for d in self.diagnostics)

    def counts(self) -> dict[Severity, int]:
        counts: dict[Severity, int] = {}
        for diagnostic in self.diagnostics:
            counts[diagnostic.severity] = counts.get(diagnostic.severity, 0) + 1
        return counts

    def extend(self, diagnostics: list[Diagnostic]) -> None:
        self.diagnostics.extend(diagnostics)

    def sorted(self) -> list[Diagnostic]:
        """Most severe first, then by position, so the worst problem reads first."""
        return sorted(
            self.diagnostics,
            key=lambda d: (
                SEVERITY_RANK[d.severity],
                str(d.location.path or ""),
                d.location.span.start if d.location.span is not None else -1,
            ),
        )
