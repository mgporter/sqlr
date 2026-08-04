"""Provenance, warnings and the explain surface.

Cross-cutting: schema resolution writes divergences here today, and constraint
extraction, relationship inference and execution will write theirs here later. One
channel, one renderer, one shape that converts to an editor diagnostic.
"""

from sqlrunner.diagnostics import codes
from sqlrunner.diagnostics.check import (
    check_schema,
    from_analysis,
    unresolved_types,
)
from sqlrunner.diagnostics.render import render_text, to_lsp
from sqlrunner.diagnostics.types import (
    Diagnostic,
    DiagnosticReport,
    Location,
    Related,
    Severity,
)

__all__ = [
    "Diagnostic",
    "DiagnosticReport",
    "Location",
    "Related",
    "Severity",
    "check_schema",
    "codes",
    "from_analysis",
    "render_text",
    "to_lsp",
    "unresolved_types",
]
