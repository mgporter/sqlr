"""Step 5 - resolve every value in the statement to a type, or say why it could not be.

## The one rule

> A fact is a claim about a value. A claim contradicted by that value's actual type is an
> error; a claim about a value with no type at all is the inference.

What that rule does not settle - and what this module exists to settle - is **how strong an
answer is**, and therefore which of the two readings applies when a value's type came from
another fact rather than from a declaration.

> **Strength is a property of the evidence, and it crosses a link intact.** A value linked
> to a stated type is stated. A value linked to an inferred type is inferred. Disagreement
> among stated values is an **error**; disagreement among inferred values is a **resolution
> failure**.

| origin | strength | example |
|---|---|---|
| `declared` | stated | `data_type: varchar(20)` in a yml |
| `computed` | stated | sqlglot annotated it bottom-up - a return type, a CTE projection |
| `literal` | stated | `'2024-01-01'`, `0`, `true` |
| `linked` from a stated member | stated | `x = some_declared_date_col` |
| `claimed` | inferred | `upper(x)` says x is STRING |

**A literal is a stated type.** `where order_date >= '2024-01-01'` types `order_date` as
STRING, so `date_trunc('day', order_date)` elsewhere is an error rather than a shrug. sqlr
does not model engine autocasting: if the column is really a date, the SQL should say
`date '2024-01-01'` or the yml should declare it.

**A claim can never contradict an inferred type**, because an inferred type is derived from
the claim set. If the claims disagreed the column is unresolved and there is nothing to
contradict. That is what lets `check.py` report contradictions without a special case: it
reports only against `stated` values, and everything else was already reconciled here.

## The mechanism

Links make an undirected graph over value sites, so the answer is a connected component and
not a column: union-find, one pass, transitive by construction, no iteration cap to justify.

Sites are additionally unioned by **schema slot**, because `orders.amount` at line 4 and
`orders.amount` at line 10 are two `exp.Column` nodes and one column. Leaving that out is the
easiest mistake to make here and the hardest to notice: a claim at one site and a link at the
other land in different components and neither ever sees the other.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Literal

from sqlglot import exp

from sqlr.sql_analysis2.catalog import DialectName
from sqlr.sql_analysis2.families import (
    ANY,
    FamilyName,
    describe_families,
    family_of_type,
    family_satisfies,
    nearest_common_family,
    reported_type_for,
    root_family_of,
    schema_type_for,
)
from sqlr.sql_analysis2.facts import Facts, ValueSite
from sqlr.sql_analysis2.resolve import is_declared
from sqlr.sql_analysis2.sourcedoc import SourceSpan
from sqlr.sql_analysis2.types import ColumnName, ColumnTypeName, RelationKey

logger = logging.getLogger(__name__)

type TypeStrength = Literal["stated", "inferred", "unresolved"]
"""How much an answer is worth when answers disagree.

- `stated`     - somebody wrote this type down, or it was computed from types that were.
  Disagreement is an **error**: two things are written and one of them is wrong.
- `inferred`   - sqlr derived it from usage. Disagreement is a **warning**: the guesses
  fought, the SQL may be fine, and the honest answer is "I could not tell".
- `unresolved` - nothing could say, or the evidence conflicted.
"""

type TypeOrigin = Literal["declared", "computed", "literal", "linked", "claimed"]
"""Where one piece of evidence came from. Reported to the user; `TypeStrength` is what the
resolution actually branches on."""

STATED_ORIGINS: frozenset[TypeOrigin] = frozenset({"declared", "computed", "literal"})


@dataclass(frozen=True)
class TypeEvidence:
    """One fact's opinion about one value, and where it was written."""

    site: ValueSite
    families: frozenset[FamilyName]
    """What this fact says the value may be. A set because one position of an overloaded
    call genuinely accepts several - `x * 2` says numeric or interval and nothing narrower.
    Always a single family for an anchor, which has an actual type."""
    concrete_type: exp.DataType | None
    """The exact type behind the family, when there is one. None for a claim, which names a
    family and nothing more, and None for a literal - `> 0` proves the column is numeric and
    proves nothing about its width."""
    origin: TypeOrigin
    detail: str
    """How the fact reads in a message: `UPPER argument 1`, `compared with 'customers'.'id'`."""

    @property
    def strength(self) -> TypeStrength:
        return "stated" if self.origin in STATED_ORIGINS else "inferred"

    @property
    def span(self) -> SourceSpan | None:
        return self.site.span

    def describe_family(self) -> str:
        return describe_families(self.families)


@dataclass(frozen=True)
class InferredColumnType:
    """One source column this pass gave a type to."""

    table: RelationKey
    column: ColumnName
    type_name: ColumnTypeName
    """What a report calls this column. An anchor's exact type when one pinned it down -
    `where d = order_date` gives `DATE(3)` if that is what was declared - and otherwise the
    family, parameter-free, because usage evidence proves a kind and never a width."""
    schema_type_name: ColumnTypeName
    """What goes into the widened schema instead. Differs from `type_name` only for a family
    whose reported name sqlglot cannot parse or would narrow on - see `SCHEMA_TYPE_FOR_FAMILY`."""
    family: FamilyName
    strength: TypeStrength
    evidence: list[TypeEvidence]


@dataclass(frozen=True)
class ColumnTypeConflict:
    """One component whose evidence disagrees. Infers nothing, reports every site.

    `strength` decides how loudly. Two stated types in one component means two things were
    written down and one is wrong, which is an error. Two inferred families means sqlr
    guessed twice and the guesses fought, which is a warning about sqlr's confidence rather
    than an accusation about the SQL.
    """

    table: RelationKey | None
    column: ColumnName | None
    """None when the component touches no source column at all - `upper(cte_col) = 5` is a
    real conflict about values that have no schema slot between them."""
    strength: TypeStrength
    evidence: list[TypeEvidence]

    def described_families(self) -> list[str]:
        """The families in dispute, in first-seen order - the reader needs them in the
        order the sites will be listed in."""
        seen: list[str] = []
        for item in self.evidence:
            described = item.describe_family()
            if described not in seen:
                seen.append(described)
        return seen

    def describe_subject(self) -> str:
        """`column 'amount' of 'mydb.sch.orders'`, or the value itself when it is not one."""
        if self.table is not None and self.column is not None:
            return f"column '{self.column}' of '{self.table}'"
        first = self.evidence[0].site.describe() if self.evidence else "this value"
        return f"the value {first}"


@dataclass
class Inference:
    inferred: list[InferredColumnType] = field(default_factory=list[InferredColumnType])
    conflicts: list[ColumnTypeConflict] = field(default_factory=list[ColumnTypeConflict])
    strength_of_node: dict[int, TypeStrength] = field(default_factory=dict[int, TypeStrength])
    """Every value site that appeared in a fact, keyed by `id(node)`.

    `check.py` reads it to decide whether a contradicted claim is reportable: only a
    `stated` value can contradict anything. A site in a conflicted component is
    `unresolved`, which keeps the claim quiet - the conflict finding already points at every
    site involved and a second finding on the same span would be noise.
    """

    def types_per_table(self) -> dict[RelationKey, dict[ColumnName, InferredColumnType]]:
        out: dict[RelationKey, dict[ColumnName, InferredColumnType]] = {}
        for entry in self.inferred:
            out.setdefault(entry.table, {})[entry.column] = entry
        return out


# ------------------------------------------------------------------------ union-find
class _Components:
    """Disjoint sets over value sites, keyed by `id(node)`."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._sites: dict[int, ValueSite] = {}

    def add(self, site: ValueSite) -> int:
        key = id(site.node)
        self._parent.setdefault(key, key)
        # First site wins: several facts describe one node and any of their ValueSites
        # answers the same questions, so re-recording buys nothing.
        self._sites.setdefault(key, site)
        return key

    def find(self, key: int) -> int:
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[key] != root:  # path compression
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[left_root] = right_root

    def groups(self) -> list[list[ValueSite]]:
        out: dict[int, list[ValueSite]] = {}
        for key in self._parent:
            out.setdefault(self.find(key), []).append(self._sites[key])
        return list(out.values())

    def sites(self) -> Iterable[tuple[int, ValueSite]]:
        return list(self._sites.items())


# -------------------------------------------------------------------------- evidence
def _origin_of_anchor(
    site: ValueSite,
    declared_types_per_relation: dict[RelationKey, dict[ColumnName, ColumnTypeName]],
) -> TypeOrigin:
    """Which kind of stated thing a typed node is.

    The distinction is for the message, not the arithmetic - all three are `stated` - but a
    reader told "declared varchar(20)" knows where the fix goes and a reader told "is TEXT
    here" does not.
    """
    if isinstance(site.node, (exp.Literal, exp.Boolean)):
        return "literal"
    slot = site.source_table_column
    if slot is not None and is_declared(
        declared_types_per_relation.get(slot[0], {}).get(slot[1])
    ):
        return "declared"
    return "computed"


def _anchor_evidence_of(
    site: ValueSite,
    declared_types_per_relation: dict[RelationKey, dict[ColumnName, ColumnTypeName]],
) -> TypeEvidence | None:
    """What a node that already has a type contributes, or None when it has none."""
    family = family_of_type(site.node.type)
    if family is None:
        return None

    origin = _origin_of_anchor(site, declared_types_per_relation)
    written = site.node.type.sql() if site.node.type is not None else family
    if origin == "literal":
        # A literal proves the family and nothing else. `where amount > 0` says numeric;
        # inferring INT from it would falsely contradict `amount * 1.5` three CTEs later.
        return TypeEvidence(
            site=site,
            families=frozenset({root_family_of(family)}),
            concrete_type=None,
            origin="literal",
            detail=f"compared with the literal {site.node.sql()}",
        )
    return TypeEvidence(
        site=site,
        families=frozenset({family}),
        concrete_type=site.node.type,
        origin=origin,
        detail=(
            f"compared with {describe_declared_slot(site)} ({written})"
            if origin == "declared"
            else f"{site.describe()} is {written} here"
        ),
    )


def describe_declared_slot(site: ValueSite) -> str:
    """`raw_department.department_id` - the declared column this evidence came from.

    The bare table name rather than the full `db.schema.table` key: the reader is looking at
    a narrow evidence column, and the qualifier never disambiguates anything they can see.
    """
    slot = site.source_table_column
    if slot is None:
        return site.describe()
    relation_key, column_name = slot
    return f"{relation_key.split('.')[-1]}.{column_name}"


def _intersect_family_sets(
    left: set[FamilyName], right: set[FamilyName]
) -> set[FamilyName]:
    """The families satisfying both sides, keeping the narrower of any pair on one path.

    `{NUMERIC}` against `{INTEGER}` is `{INTEGER}`: a claim of numeric is satisfied by an
    integer, so the two agree and the narrower is what they agree on. `{STRING}` against
    `{NUMERIC}` is empty, which is the conflict.
    """
    out: set[FamilyName] = set()
    for a in left:
        for b in right:
            if family_satisfies(a, b):
                out.add(a)
            elif family_satisfies(b, a):
                out.add(b)
    return out


# ------------------------------------------------------------------------ resolution
@dataclass(frozen=True)
class ComponentVerdict:
    """One component of linked values, resolved."""

    members: list[ValueSite]
    family: FamilyName | None
    concrete_type_name: ColumnTypeName | None
    """The exact type an anchor pinned down, or None when the family is all that is known -
    which is the usual case, since usage evidence proves a kind and never a width."""
    strength: TypeStrength
    evidence: list[TypeEvidence]
    conflict: ColumnTypeConflict | None = None

    @property
    def type_name(self) -> ColumnTypeName | None:
        """What a report calls this. None when the component resolved to nothing nameable."""
        if self.concrete_type_name is not None:
            return self.concrete_type_name
        return reported_type_for(self.family) if self.family is not None else None

    @property
    def schema_type_name(self) -> ColumnTypeName | None:
        """What sqlglot's schema gets instead. None for the same components `type_name` is."""
        if self.concrete_type_name is not None:
            return self.concrete_type_name
        return schema_type_for(self.family) if self.family is not None else None


class _Resolver:
    def __init__(
        self,
        facts: Facts,
        declared_types_per_relation: dict[RelationKey, dict[ColumnName, ColumnTypeName]],
        dialect_name: DialectName,
    ) -> None:
        self.facts = facts
        self.schema = declared_types_per_relation
        self.dialect_name = dialect_name
        self.components = _Components()
        self.claims_at: dict[int, list[tuple[frozenset[FamilyName], str, ValueSite]]] = {}
        self.family_only_links: set[int] = set()
        """Component members touched by a `@family(argN)` link, by site id. A component
        containing one loses its concrete type - see `_resolve_component`."""

    # ---- I1: build the components -----------------------------------------------
    def build(self) -> None:
        for link in self.facts.type_links:
            left = self.components.add(link.left)
            right = self.components.add(link.right)
            self.components.union(left, right)
            if link.family_only:
                self.family_only_links.update((left, right))

        for claim in self.facts.type_claims:
            key = self.components.add(claim.site)
            self.claims_at.setdefault(key, []).append(
                (claim.families, claim.because.describe(), claim.site)
            )

        self._union_sites_sharing_a_schema_slot()

    def _union_sites_sharing_a_schema_slot(self) -> None:
        """Two reads of one column are one value, however far apart they were written.

        Without this a claim at line 4 and a link at line 10 land in different components
        and neither ever sees the other, which is exactly the evidence-pooling the whole
        design is for.
        """
        by_slot: dict[tuple[RelationKey, ColumnName], int] = {}
        for key, site in self.components.sites():
            slot = site.source_table_column
            if slot is None:
                continue
            first = by_slot.setdefault(slot, key)
            self.components.union(first, key)

    # ---- I2: resolve each component ---------------------------------------------
    def resolve(self) -> Inference:
        inference = Inference()
        for members in self.components.groups():
            verdict = self._resolve_component(members)
            self._record(verdict, inference)
        return inference

    def _resolve_component(self, members: list[ValueSite]) -> ComponentVerdict:
        anchors = [
            evidence
            for member in members
            if (evidence := _anchor_evidence_of(member, self.schema)) is not None
        ]
        if anchors:
            return self._resolve_from_anchors(members, anchors)
        return self._resolve_from_claims(members)

    def _resolve_from_anchors(
        self, members: list[ValueSite], anchors: list[TypeEvidence]
    ) -> ComponentVerdict:
        """Somebody wrote these types down. They agree, or one of them is wrong."""
        family = next(iter(anchors[0].families))
        for anchor in anchors[1:]:
            family = nearest_common_family(family, next(iter(anchor.families)))

        if family == ANY:
            return ComponentVerdict(
                members=members,
                family=None,
                concrete_type_name=None,
                strength="unresolved",
                evidence=anchors,
                conflict=self._conflict(members, anchors, "stated"),
            )

        return ComponentVerdict(
            members=members,
            family=family,
            concrete_type_name=self._concrete_type_name_for(family, anchors, members),
            strength="stated",
            evidence=anchors,
        )

    def _concrete_type_name_for(
        self, family: FamilyName, anchors: list[TypeEvidence], members: list[ValueSite]
    ) -> ColumnTypeName | None:
        """The exact type an agreed family pinned down, or None when it pinned down none.

        A single anchor type that *is* the agreed family wins, because it is strictly more
        informative: `where d = order_date` gives `DATE` rather than widening a date column
        to `TIMESTAMP`. Anything else is the family and nothing more - two anchors that
        merely agree on a family agree on no width between them, and picking one of their
        widths would report a precision the SQL never established.

        ⚠️ A `@family` link anywhere in the component forfeits the concrete type for all of
        it. `sum(x)` is numeric because `x` is, but its width is the engine's business, and
        carrying a width across that link would invent one. Coarse on purpose - a component
        is resolved as a whole and there is no half-precision.
        """
        if any(id(member.node) in self.family_only_links for member in members):
            return None
        concrete = {
            anchor.concrete_type.sql()
            for anchor in anchors
            if anchor.concrete_type is not None
            and any(family_satisfies(named, family) for named in anchor.families)
        }
        return concrete.pop() if len(concrete) == 1 else None

    def _resolve_from_claims(self, members: list[ValueSite]) -> ComponentVerdict:
        """Nobody stated anything, so the claims are all there is."""
        claims = [
            claim
            for member in members
            for claim in self.claims_at.get(id(member.node), [])
        ]
        if not claims:
            return ComponentVerdict(members, None, None, "unresolved", [])

        families = set(claims[0][0])
        for other, _, _ in claims[1:]:
            families = _intersect_family_sets(families, set(other))

        evidence = [
            TypeEvidence(
                site=site,
                families=claimed,
                concrete_type=None,
                origin="claimed",
                detail=detail,
            )
            for claimed, detail, site in claims
        ]

        if not families:
            return ComponentVerdict(
                members=members,
                family=None,
                concrete_type_name=None,
                strength="unresolved",
                evidence=evidence,
                conflict=self._conflict(members, evidence, "inferred"),
            )
        if len(families) > 1:
            # A disjunction nobody narrowed: `x * 2` accepts numeric or interval, and
            # picking one would be the coin-flip this whole design exists to avoid.
            return ComponentVerdict(members, None, None, "unresolved", evidence)

        family = families.pop()
        return ComponentVerdict(
            members=members,
            family=family,
            concrete_type_name=None,
            strength="inferred",
            evidence=evidence,
        )

    def _conflict(
        self,
        members: list[ValueSite],
        evidence: list[TypeEvidence],
        strength: TypeStrength,
    ) -> ColumnTypeConflict:
        slot = next(
            (
                member.source_table_column
                for member in members
                if member.source_table_column is not None
            ),
            None,
        )
        return ColumnTypeConflict(
            table=slot[0] if slot else None,
            column=slot[1] if slot else None,
            strength=strength,
            evidence=evidence,
        )

    # ---- recording ----------------------------------------------------------------
    def _record(self, verdict: ComponentVerdict, inference: Inference) -> None:
        for member in verdict.members:
            inference.strength_of_node[id(member.node)] = verdict.strength

        if verdict.conflict is not None:
            inference.conflicts.append(verdict.conflict)
            return
        type_name, schema_type_name = verdict.type_name, verdict.schema_type_name
        if verdict.family is None or type_name is None or schema_type_name is None:
            return

        # Only source columns have a schema slot to widen. A CTE column's type is computed
        # from its own projection and overwriting it would fight forward propagation.
        for slot in {
            member.source_table_column
            for member in verdict.members
            if member.source_table_column is not None
        }:
            table, column = slot
            if is_declared(self.schema.get(table, {}).get(column)):
                continue
            inference.inferred.append(
                InferredColumnType(
                    table=table,
                    column=column,
                    type_name=type_name,
                    schema_type_name=schema_type_name,
                    family=verdict.family,
                    strength=verdict.strength,
                    evidence=verdict.evidence,
                )
            )


def infer_types_for_undeclared_columns(
    facts: Facts,
    declared_types_per_relation: dict[RelationKey, dict[ColumnName, ColumnTypeName]],
    dialect_name: DialectName,
) -> Inference:
    """One verdict per component of linked values: a type, or a conflict, or silence."""
    resolver = _Resolver(facts, declared_types_per_relation, dialect_name)
    resolver.build()
    inference = resolver.resolve()
    inference.inferred.sort(key=lambda entry: (entry.table, entry.column))
    return inference


def widen_schema_with_inferred_types(
    declared_types_per_relation: dict[RelationKey, dict[ColumnName, ColumnTypeName]],
    inference: Inference,
) -> dict[RelationKey, dict[ColumnName, ColumnTypeName]]:
    """The gap-filled schema with inferred types written into its UNKNOWN slots.

    ONLY fills UNKNOWN slots. Never overwrites a stated type. This is what makes widening
    converge in one round and what makes the phase-D skip sound; it is also the coupling
    most likely to be broken later by a reasonable-sounding feature ("the SQL says string,
    the yml says int, trust the SQL"). If that feature is ever built, the skip predicate
    changes with it rather than getting a config bolted on top.
    """
    widened = {
        table: dict(columns) for table, columns in declared_types_per_relation.items()
    }
    for entry in inference.inferred:
        current = widened.get(entry.table, {}).get(entry.column)
        if current is None or is_declared(current):
            continue
        widened[entry.table][entry.column] = entry.schema_type_name
    return widened
