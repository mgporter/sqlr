"""Compare what the SQL implies against what the yml declares.

The rule set is small and sits in `_judge`; everything else here is about deciding *which*
declaration a column should be held against, and about the columns that exist on only one
of the two sides.

Three asymmetries are deliberate:

- A **source** declaration with no matching column in the SQL is not a finding. A `select`
  that touches four of a table's twelve columns is normal, and after a `*` expansion the
  SQL's list is a subset by construction. Those rows exist so the declaration is visible,
  and they resolve to the declared type: inference never saw the column, so the
  declaration is unopposed rather than unsupported.
- A **projection** declaration with no matching column is a finding, and how loud depends
  on whether the SELECT list is the whole output. With no unexpandable `*` in it, the list
  is exhaustive and the file genuinely does not produce the declared column: an error.
  Under `select * from a`, where `a`'s columns cannot be enumerated, the column may well be
  produced and simply be invisible from here, so it is only a warning - declaring those
  columns explicitly is better practice, not a rule to fail a run over.
- A column with no declaration **is** a finding, at warning. Generation is driven by the
  declared type where there is one, so a column nothing declares is a hole - whether the
  model has no yml at all or its yml simply omits the column.

The declaration also wins every compatible disagreement: a `numeric` inferred from
`amount > 0` against a declared `decimal(10,2)` resolves to the declaration, because the
user wrote down something inference could not see.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from sqlr.declared.types import (
    DeclaredColumn,
    DeclaredRelation,
    DeclaredSchemas,
    DeclaredSourceTable,
)
from sqlr.diagnostics.types import Location
from sqlr.schema_resolution.types import (
    ProjectionSchema,
    ResolvedType,
    StatementSchema,
    TableSchema,
)
from sqlr.typemap import ResolvedTypeName, compatible, covers
from sqlr.validation.render import render_errors, render_validation
from sqlr.validation.types import (
    ColumnValidation,
    StatementValidation,
    TableValidation,
    ValidationDetail,
    ValidationOutcome,
)

__all__ = [
    "ColumnValidation",
    "StatementValidation",
    "TableValidation",
    "ValidationDetail",
    "ValidationOutcome",
    "render_errors",
    "render_validation",
    "validate_schema",
]

_Judgement = tuple[ValidationOutcome, ValidationDetail, ResolvedTypeName | None]

_Side = Literal["source", "projection"]
"""Which of the two comparisons a column belongs to. Only matters when it is missing."""


def validate_schema(
    schema: StatementSchema,
    declared: DeclaredSchemas,
    path: Path | None = None,
) -> StatementValidation:
    """Check `schema` against every declaration that bears on it.

    `path` is the `.sql` file this statement came from; the declaration describing what it
    produces is the one that claims it with `meta.source_file`, or - in a dbt project - the
    `models:` entry of the same stem. Every relation it *reads* is matched by the name the
    SQL writes.
    """
    return StatementValidation(
        source=schema.source,
        tables=[
            _validate_table(table, declared.for_relation(table.name))
            for table in schema.tables
        ],
        projection=_validate_projection(
            schema.projection,
            declared.for_source_file(path) if path is not None else None,
            complete=schema.projection_is_complete,
        ),
    )


def _kind(model: DeclaredRelation | None) -> Literal["model", "source"] | None:
    if model is None:
        return None
    return "source" if isinstance(model, DeclaredSourceTable) else "model"


# ---- the two sides ---------------------------------------------------------------------


def _validate_table(
    table: TableSchema, model: DeclaredRelation | None
) -> TableValidation:
    columns = [
        _validate_column(
            name=column.name,
            inferred=column.resolved_type,
            model=model,
        )
        for column in table.columns
    ]
    columns.extend(
        _declared_only(model, seen={column.name.lower() for column in table.columns})
    )

    return TableValidation(
        name=table.name,
        kind="source",
        star_expanded=table.star_expanded,
        declared_model=model.display_name if model is not None else None,
        declared_kind=_kind(model),
        declared_path=model.path if model is not None else None,
        columns=columns,
    )


def _validate_projection(
    projection: list[ProjectionSchema],
    model: DeclaredRelation | None,
    complete: bool = True,
) -> TableValidation:
    columns = [
        _validate_column(
            # An unnamed projected column cannot be matched to a declaration, so it is
            # named for the reader and falls through to `no declaration`.
            name=column.name or f"<column {column.ordinal}>",
            inferred=column.resolved_type,
            model=model if column.name is not None else None,
            ordinal=column.ordinal,
        )
        for column in projection
    ]
    columns.extend(
        _declared_only(
            model,
            seen={
                column.name.lower() for column in projection if column.name is not None
            },
            kind="projection",
            complete=complete,
        )
    )

    return TableValidation(
        # Named by the relation rather than by the declaration: this block is what the
        # file produces, and the relation is what another file has to write to read it.
        name=model.relation_name if model is not None else "projection",
        kind="projection",
        declared_model=model.display_name if model is not None else None,
        declared_kind=_kind(model),
        declared_path=model.path if model is not None else None,
        columns=columns,
    )


def _validate_column(
    name: str,
    inferred: ResolvedType,
    model: DeclaredRelation | None,
    ordinal: int | None = None,
) -> ColumnValidation:
    declaration = model.column(name) if model is not None else None
    outcome, detail, resolved = _judge(inferred, declaration)
    return ColumnValidation(
        name=name,
        outcome=outcome,
        detail=detail,
        resolved_type=resolved,
        inferred=inferred,
        declared=declaration,
        declaration=_declaration_location(model, declaration),
        ordinal=ordinal,
    )


def _declared_only(
    model: DeclaredRelation | None,
    seen: set[str],
    kind: _Side = "source",
    complete: bool = True,
) -> list[ColumnValidation]:
    """Rows for declarations the SQL never mentioned.

    `inferred` is None rather than an empty `ResolvedType`: inference did not fail on
    these columns, it never saw them. They still go through `_judge`, because a
    declaration can be wrong on its own - an unrecognised type is unusable whether or not
    any SQL refers to it, and on the projection side the absence is itself the finding.
    """
    if model is None:
        return []

    rows: list[ColumnValidation] = []
    for declaration in model.columns:
        if declaration.name.lower() in seen:
            continue
        outcome, detail, resolved = _judge(None, declaration, kind, complete)
        rows.append(
            ColumnValidation(
                name=declaration.name,
                outcome=outcome,
                detail=detail,
                resolved_type=resolved,
                inferred=None,
                declared=declaration,
                declaration=_declaration_location(model, declaration),
            )
        )
    return rows


def _declaration_location(
    model: DeclaredRelation | None, declaration: DeclaredColumn | None
) -> Location | None:
    if model is None or declaration is None:
        return None
    return Location.of(model.source, declaration.type_span or declaration.span)


# ---- the rule set ----------------------------------------------------------------------


def _judge(
    inferred: ResolvedType | None,
    declaration: DeclaredColumn | None,
    kind: _Side = "source",
    complete: bool = True,
) -> _Judgement:
    """Which of the two types stands, and how loudly to say so."""
    if declaration is None:
        resolved = inferred.type_name if inferred is not None else None
        return "warning", "no declaration", _known(resolved)

    declared_name = declaration.resolved_type_name
    missing = inferred is None and kind == "projection"
    # An unexpandable `*` means the SELECT list is a lower bound on the output, so a column
    # missing from it may still be produced - it was just never written down. Without one
    # the list is the whole answer, and the column really is not there.
    missing_outcome: ValidationOutcome = "error" if complete else "warning"
    missing_detail: ValidationDetail = (
        "declared but not projected" if complete else "declared with no explicit projection"
    )

    if declared_name == "unknown":
        # A type sqlr cannot map cannot be compared against, or generated from - which is
        # true of a column the SQL never mentioned too, so this is judged before that. Not
        # before a column the projection is provably missing, though: that one outranks it,
        # and reporting the weaker of two true things is the wrong default.
        if missing:
            return missing_outcome, missing_detail, None
        return "warning", "unrecognized declaration", None

    if inferred is None:
        # Declared but never referenced. Nothing opposed the declaration, so it stands -
        # generation has the type it needs even though inference never saw the column.
        if missing:
            return missing_outcome, missing_detail, declared_name
        return "pass", "declared only", declared_name

    inferred_name = inferred.type_name
    if inferred_name == "unknown":
        return "pass", "declared only", declared_name

    if inferred_name == declared_name:
        return "pass", "exact match", declared_name
    if covers(inferred_name, declared_name):
        # `numeric` inferred, `decimal` declared: the user knows which kind of number.
        return "pass", "type narrowed", declared_name
    if compatible(inferred_name, declared_name):
        # `integer` inferred, `numeric` declared. Generating to the declaration is safe -
        # it accepts what was inferred - but the declaration is losing information.
        return "warning", "type widened", declared_name

    if _name_pattern_only(inferred):
        # The only thing that spoke was the column's name, and a suffix convention losing
        # to a written declaration is unremarkable rather than a contradiction.
        return "pass", "declared only", declared_name

    return "error", "inferred type differs from declared type", None


def _known(resolved: ResolvedTypeName | None) -> ResolvedTypeName | None:
    """`unknown` is not something to resolve to, so it reads as nothing at all."""
    return None if resolved is None or resolved == "unknown" else resolved


def _name_pattern_only(inferred: ResolvedType) -> bool:
    """True when every piece of evidence was the column's name.

    Not `chosen.source == "name_pattern"`: a name pattern that outweighed nothing is a
    guess, but a name pattern agreeing with a `like` is a guess the SQL backs up, and that
    one is worth contradicting a declaration with.
    """
    return bool(inferred.evidence) and all(
        item.source == "name_pattern" for item in inferred.evidence
    )
