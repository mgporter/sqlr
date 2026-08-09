"""Compare what was inferred against what was declared.

Two comparisons, one rule set:

- the statement's **projection** against the declaration for the `.sql` file itself - the
  source table that claims it with `sql_file:`, or a dbt project's `models:` entry;
- each **relation it reads** against the declaration of that name, which is what catches
  the interesting case - `orders.sql` declares `revenue` a varchar, and
  `revenue_report.sql` writes `where revenue > 1000`.

A mismatch is only reported when the declared and inferred types have no concrete type in
common. Declaring `integer` where `number` was inferred is not a disagreement: inference
deliberately widens to a family when the evidence does not distinguish, and the user's
declaration is the more specific of the two.
"""

from __future__ import annotations

from pathlib import Path

from sqlr.declared.types import DeclaredColumn, DeclaredRelation, DeclaredSchemas
from sqlr.diagnostics import codes
from sqlr.diagnostics.types import Diagnostic, Location, Related, Severity
from sqlr.schema_resolution.types import (
    ColumnSchema,
    ResolvedType,
    StatementSchema,
    TypeEvidence,
)
from sqlr.source import SourceDoc, SourceSpan
from sqlr.sql_analysis.types import Ambiguity
from sqlr.typemap import compatible

# Above this, the evidence names a type outright - a cast, or a function whose return
# type the dialect fixes. Contradicting one of those is a real error.
STRONG_EVIDENCE_WEIGHT = 80

# Below this, the only thing that spoke was the column's name. A name pattern losing to a
# declaration is unremarkable, so it is a hint rather than a warning.
WEAK_EVIDENCE_WEIGHT = 10

_EvidenceKey = tuple[int, int, str, str]
"""What makes two pieces of evidence the same finding: where, and what it says."""


def check_schema(
    schema: StatementSchema,
    declared: DeclaredSchemas,
    path: Path | None = None,
) -> list[Diagnostic]:
    """Every divergence between `schema` and the declarations that bear on it."""
    diagnostics: list[Diagnostic] = []

    own = declared.for_sql_file(path) if path is not None else None
    if own is not None:
        diagnostics.extend(_check_projection(schema, own))
        # Only the file's own model can be *missing* a column. A declaration for an
        # upstream table listing columns this file happens not to select is normal.
        diagnostics.extend(_check_missing(schema, own))

    for table in schema.tables:
        model = declared.for_relation(table.name)
        if model is None:
            continue
        diagnostics.extend(_check_table(schema, table.name, table.columns, model))
        if not table.star_expanded:
            # A `*` means the attributed column list is a subset of the real one, so
            # absence from it proves nothing.
            diagnostics.extend(
                _check_undeclared(schema, table.name, table.columns, model)
            )

    return diagnostics


# ---- the comparison ------------------------------------------------------------------


def _check_table(
    schema: StatementSchema,
    table: str,
    columns: list[ColumnSchema],
    model: DeclaredRelation,
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for column in columns:
        declaration = model.column(column.name)
        if declaration is None:
            continue
        diagnostics.extend(
            _compare(
                schema=schema,
                model=model,
                declaration=declaration,
                table=table,
                name=column.name,
                resolved=column.resolved_type,
                location=column.location,
            )
        )
    return diagnostics


def _check_projection(
    schema: StatementSchema, model: DeclaredRelation
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for column in schema.projection:
        if column.name is None:
            continue
        declaration = model.column(column.name)
        if declaration is None:
            continue
        diagnostics.extend(
            _compare(
                schema=schema,
                model=model,
                declaration=declaration,
                table=model.display_name,
                name=column.name,
                resolved=column.resolved_type,
                location=column.location,
            )
        )
    return diagnostics


def _compare(
    schema: StatementSchema,
    model: DeclaredRelation,
    declaration: DeclaredColumn,
    table: str,
    name: str,
    resolved: ResolvedType,
    location: SourceSpan | None,
) -> list[Diagnostic]:
    if declaration.resolved_type_name == "unknown":
        return [
            Diagnostic(
                code=codes.UNKNOWN_DECLARED_TYPE,
                severity="warning",
                message=(
                    f"{table}.{name} is declared {declaration.written_type!r}, which is not "
                    f"a type sqlr recognises; it cannot be checked or generated"
                ),
                location=Location.of(model.source, declaration.type_span),
                table=table,
                column=name,
            )
        ]

    inferred = resolved.type_name
    if inferred == "unknown" or compatible(declaration.resolved_type_name, inferred):
        return []

    return [
        Diagnostic(
            code=codes.TYPE_MISMATCH,
            severity=_severity(resolved.chosen),
            message=(
                f"{table}.{name} is declared {declaration.written_type} "
                f"({declaration.resolved_type_name}) but the SQL uses it as {inferred}"
            ),
            location=Location.of(schema.source, location),
            related=_related(schema, model, declaration, resolved),
            table=table,
            column=name,
        )
    ]


def _severity(chosen: TypeEvidence | None) -> Severity:
    """How loud to be, based on how sure the SQL was.

    A cast contradicting a declaration is a real error. A `*_at` suffix contradicting one
    is barely worth mentioning.
    """
    if chosen is None:
        return "info"
    if chosen.weight >= STRONG_EVIDENCE_WEIGHT:
        return "error"
    if chosen.weight >= WEAK_EVIDENCE_WEIGHT:
        return "warning"
    return "hint"


def _related(
    schema: StatementSchema,
    model: DeclaredRelation,
    declaration: DeclaredColumn,
    resolved: ResolvedType,
) -> list[Related]:
    """Every other place that bears on the mismatch.

    This is the reason schema resolution keeps its losing evidence: a user looking at one
    squiggle needs to see the whole case against their declaration, not one instance of
    it.
    """
    related: list[Related] = []

    # Deduped by what the evidence *says* and where, not by object identity: the same
    # observation reaches here by more than one path - a derived column is typed both as
    # its relation's output and again as the statement's projected column - and repeating
    # it in the report just makes the real disagreement harder to see.
    def key(item: TypeEvidence) -> _EvidenceKey:
        span = item.location
        return (
            span.start if span is not None else -1,
            span.end if span is not None else -1,
            item.type_name,
            item.detail or "",
        )

    # Annotated because the empty-set branch has nothing to infer an element type from.
    chosen = resolved.chosen
    seen: set[_EvidenceKey] = {key(chosen)} if chosen is not None else set()

    for item in resolved.evidence:
        if item.location is None:
            continue
        item_key = key(item)
        if item_key in seen:
            continue
        seen.add(item_key)
        related.append(
            Related.at(
                schema.source,
                item.location,
                f"also used as {item.type_name} here ({item.detail})",
            )
        )

    related.append(
        Related.at(
            model.source,
            declaration.type_span or declaration.span,
            f"declared as {declaration.written_type} here",
        )
    )
    return related


# ---- coverage ------------------------------------------------------------------------


def _check_undeclared(
    schema: StatementSchema,
    table: str,
    columns: list[ColumnSchema],
    model: DeclaredRelation,
) -> list[Diagnostic]:
    return [
        Diagnostic(
            code=codes.UNDECLARED_COLUMN,
            severity="info",
            message=(
                f"{table}.{column.name} is used here but not declared in "
                f"{model.path.name}"
            ),
            location=Location.of(
                schema.source, column.references[0] if column.references else None
            ),
            table=table,
            column=column.name,
        )
        for column in columns
        if model.column(column.name) is None
    ]


def _check_missing(schema: StatementSchema, model: DeclaredRelation) -> list[Diagnostic]:
    """Columns the model declares that its own SELECT list does not produce."""
    if not schema.projection_is_complete:
        # An unexpandable `*` is in the projection, so the real output is wider than the
        # list here and nothing can be called missing.
        return []

    produced = {
        column.name.lower() for column in schema.projection if column.name is not None
    }
    return [
        Diagnostic(
            code=codes.MISSING_COLUMN,
            severity="hint",
            message=(
                f"{model.display_name}.{declaration.name} is declared but "
                f"{model.display_name} does not "
                f"produce it"
            ),
            location=Location.of(model.source, declaration.name_span),
            table=model.display_name,
            column=declaration.name,
        )
        for declaration in model.columns
        if declaration.name.lower() not in produced
    ]


# ---- adapters ------------------------------------------------------------------------


def from_analysis(
    source: SourceDoc,
    errors: list[str],
    warnings: list[str],
    ambiguities: list[Ambiguity],
) -> list[Diagnostic]:
    """Fold the analyser's loose strings into the same channel as everything else."""
    diagnostics: list[Diagnostic] = [
        Diagnostic(
            code=codes.ANALYSIS_ERROR,
            severity="error",
            message=message,
            location=Location(path=source.path),
        )
        for message in errors
    ]

    diagnostics.extend(
        Diagnostic(
            code=codes.ANALYSIS_WARNING,
            severity="warning",
            message=message,
            location=Location(path=source.path),
        )
        for message in warnings
    )

    for ambiguity in ambiguities:
        chosen = ambiguity.chosen
        diagnostics.append(
            Diagnostic(
                code=codes.AMBIGUOUS_COLUMN,
                severity="info" if ambiguity.resolution == "attributed" else "warning",
                message=(
                    f"column {ambiguity.column!r} in {ambiguity.relation} was "
                    + (
                        f"attributed to {chosen} ({ambiguity.confidence})"
                        if chosen is not None
                        else "dropped"
                    )
                    + f" - {ambiguity.reason}"
                ),
                location=Location.of(source, ambiguity.span),
                column=ambiguity.column,
            )
        )

    return diagnostics


def unresolved_types(
    schema: StatementSchema, declared: DeclaredSchemas
) -> list[Diagnostic]:
    """Columns nothing typed and nothing declared - generation has nothing to go on."""
    diagnostics: list[Diagnostic] = []
    for table in schema.tables:
        model = declared.for_relation(table.name)
        for column in table.columns:
            if column.resolved_type.type_name != "unknown":
                continue
            if model is not None and model.column(column.name) is not None:
                # Declared, so it is typed after all - just not by the SQL.
                continue
            diagnostics.append(
                Diagnostic(
                    code=codes.UNRESOLVED_TYPE,
                    severity="info",
                    message=(
                        f"{table.name}.{column.name} could not be typed from the SQL; "
                        f"declare it to control what gets generated"
                    ),
                    location=Location.of(
                        schema.source,
                        column.references[0] if column.references else None,
                    ),
                    table=table.name,
                    column=column.name,
                )
            )
    return diagnostics
