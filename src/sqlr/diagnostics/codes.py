"""Diagnostic codes. Stable strings - they end up in editor UI and in `# noqa`-style
suppressions later, so they are named once, here."""

TYPE_MISMATCH = "type-mismatch"
"""Declared type and inferred type have no concrete type in common."""

UNKNOWN_DECLARED_TYPE = "unknown-declared-type"
"""A `data_type` this tool does not recognise, so it cannot be checked or generated."""

UNDECLARED_COLUMN = "undeclared-column"
"""The SQL uses a column the declaration does not mention."""

MISSING_COLUMN = "missing-column"
"""The declaration mentions a column the SQL never touches."""

UNRESOLVED_TYPE = "unresolved-type"
"""Nothing typed the column and nothing declared it, so nothing can generate it."""

AMBIGUOUS_COLUMN = "ambiguous-column"
"""A column reference could not be attributed with confidence."""

ANALYSIS_ERROR = "analysis-error"
ANALYSIS_WARNING = "analysis-warning"
DECLARATION_WARNING = "declaration-warning"
