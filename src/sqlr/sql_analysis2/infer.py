"""Step 5 - type the columns nobody declared, from the facts about them.

Only columns of real tables, and only the ones whose schema slot is `UNKNOWN`. A CTE's
column is computed from its own projection and a declared column is the user's to state, so
neither is ever inferred - which is also what keeps widening monotone and the steps 5+6 skip
sound.

Evidence comes from the same facts the checker reads:

- a **claim** names a family directly (`upper(x)` -> STRING).
- a **link** carries whatever the other end turned out to be, which is more precise than a
  family: comparing against a `DATE` column infers `DATE`, not "some temporal type".

**Conflicting evidence infers nothing.** Two facts pointing at different families is a
defect - sqlr's position is that engine autocasting is never something to rely on - so every
site is reported and the column keeps `UNKNOWN`. UNKNOWN is absorbing in sqlglot, so
everything downstream of it goes quiet rather than inheriting a coin-flip.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlglot import exp

from sqlr.sql_analysis2.annotate import FAMILY_DEFAULT_TYPE, family_of_type
from sqlr.sql_analysis2.catalog import DialectName, FamilyName
from sqlr.sql_analysis2.facts import Facts, ValueSite
from sqlr.sql_analysis2.resolve import is_declared
from sqlr.sql_analysis2.sourcedoc import SourceSpan
from sqlr.sql_analysis2.types import ColumnName, ColumnTypeName, RelationKey


@dataclass(frozen=True)
class TypeEvidence:
    """One fact's opinion about one undeclared column, and where it was written."""

    site: ValueSite
    family: FamilyName
    concrete_type: exp.DataType | None
    """The exact type the other end of a link had, when the evidence came from one. A claim
    names a family and nothing more, so this is None for claims."""
    detail: str
    """How the fact reads in a message: `UPPER argument 1`, `compared with 'customers'.'id'`."""

    @property
    def span(self) -> SourceSpan | None:
        return self.site.span


@dataclass(frozen=True)
class InferredColumnType:
    table: RelationKey
    column: ColumnName
    type_name: ColumnTypeName
    """What goes into the widened schema - concrete, because sqlglot's schema speaks types
    and not families."""
    family: FamilyName
    evidence: list[TypeEvidence]


@dataclass(frozen=True)
class ColumnTypeConflict:
    """One undeclared column whose facts disagree. Infers nothing, reports every site."""

    table: RelationKey
    column: ColumnName
    evidence: list[TypeEvidence]

    @property
    def families(self) -> list[FamilyName]:
        """The families claimed, in first-seen order - the reader needs them in the order
        the sites will be listed in."""
        seen: list[FamilyName] = []
        for item in self.evidence:
            if item.family not in seen:
                seen.append(item.family)
        return seen


@dataclass
class Inference:
    inferred: list[InferredColumnType] = field(default_factory=list[InferredColumnType])
    conflicts: list[ColumnTypeConflict] = field(
        default_factory=list[ColumnTypeConflict]
    )

    def types_per_table(self) -> dict[RelationKey, dict[ColumnName, InferredColumnType]]:
        out: dict[RelationKey, dict[ColumnName, InferredColumnType]] = {}
        for entry in self.inferred:
            out.setdefault(entry.table, {})[entry.column] = entry
        return out


def undeclared_column_slots(
    declared_types_per_relation: dict[RelationKey, dict[ColumnName, ColumnTypeName]],
) -> set[tuple[RelationKey, ColumnName]]:
    """Every schema slot step 2 gap-filled rather than read from a yml."""
    return {
        (table, column)
        for table, columns in declared_types_per_relation.items()
        for column, type_name in columns.items()
        if not is_declared(type_name)
    }


def evidence_per_undeclared_column(
    facts: Facts,
    declared_types_per_relation: dict[RelationKey, dict[ColumnName, ColumnTypeName]],
) -> dict[tuple[RelationKey, ColumnName], list[TypeEvidence]]:
    """Collect what every fact says about the columns nobody described.

    Facts about anything else are collected too - fixture generation reads them - they just
    have no schema slot to fill, so they are skipped here.
    """
    undeclared = undeclared_column_slots(declared_types_per_relation)
    evidence: dict[tuple[RelationKey, ColumnName], list[TypeEvidence]] = {}

    for claim in facts.type_claims:
        key = claim.site.source_table_column
        if key is None or key not in undeclared or len(claim.families) != 1:
            # Several families is a disjunction: `x * 2` accepts numeric or interval, and
            # picking one would be the coin-flip this pass exists to avoid. It can still
            # contradict a known type, which is `check.py`'s half of the same claim.
            continue
        evidence.setdefault(key, []).append(
            TypeEvidence(
                site=claim.site,
                family=next(iter(claim.families)),
                concrete_type=None,
                detail=claim.because.describe(),
            )
        )

    for link in facts.type_links:
        for near, far in ((link.left, link.right), (link.right, link.left)):
            key = near.source_table_column
            if key is None or key not in undeclared:
                continue
            far_type = far.node.type
            family = family_of_type(far_type)
            if family is None:
                # The other end is unknown too. A link between two unknowns says nothing,
                # and one round of inference is deliberate: see the plan revision.
                continue
            evidence.setdefault(key, []).append(
                TypeEvidence(
                    site=near,
                    family=family,
                    concrete_type=far_type,
                    detail=f"{link.describe()} with {far.describe()}",
                )
            )

    return evidence


def infer_types_for_undeclared_columns(
    facts: Facts,
    declared_types_per_relation: dict[RelationKey, dict[ColumnName, ColumnTypeName]],
    dialect_name: DialectName,
) -> Inference:
    """One verdict per undeclared column: a type, or a conflict."""
    inference = Inference()

    for (table, column), evidence in sorted(
        evidence_per_undeclared_column(facts, declared_types_per_relation).items()
    ):
        families = {item.family for item in evidence}
        if len(families) > 1:
            inference.conflicts.append(
                ColumnTypeConflict(table=table, column=column, evidence=evidence)
            )
            continue
        family = families.pop()
        inference.inferred.append(
            InferredColumnType(
                table=table,
                column=column,
                type_name=type_name_for_evidence(family, evidence, dialect_name),
                family=family,
                evidence=evidence,
            )
        )
    return inference


def type_name_for_evidence(
    family: FamilyName, evidence: list[TypeEvidence], dialect_name: DialectName
) -> ColumnTypeName:
    """The concrete type an agreed family becomes.

    A link's known end wins over the family default, because it is strictly more
    informative: `where d = order_date` infers `DATE` rather than widening a date column to
    `TIMESTAMP`, and `where n > 1.5` infers `DOUBLE` rather than `BIGINT`.
    """
    for item in evidence:
        if item.concrete_type is not None:
            return item.concrete_type.sql(dialect=dialect_name)
    return FAMILY_DEFAULT_TYPE.get(family, "UNKNOWN")


def widen_schema_with_inferred_types(
    declared_types_per_relation: dict[RelationKey, dict[ColumnName, ColumnTypeName]],
    inference: Inference,
) -> dict[RelationKey, dict[ColumnName, ColumnTypeName]]:
    """The gap-filled schema with inferred types written into its UNKNOWN slots.

    ONLY fills UNKNOWN slots. Never overwrites a declared type. This is what makes the
    step-5/6 skip sound, and what makes widening converge in one round. Breaking it breaks
    both - see the revision section of `type_check_plan.md`.
    """
    widened = {table: dict(columns) for table, columns in declared_types_per_relation.items()}
    for entry in inference.inferred:
        current = widened.get(entry.table, {}).get(entry.column)
        if current is None or is_declared(current):
            continue
        widened[entry.table][entry.column] = entry.type_name
    return widened
