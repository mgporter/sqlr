"""Pass B: resolve column references backwards through the relation graph.

There is no catalog, so a `*` cannot be expanded forwards. Instead every reference is
resolved on demand: `projected` asks `dedupped` for `name`, `dedupped` has no explicit
`name` but has one star source, so the request is forwarded to `person`, which is an
external table and therefore terminal. Star chains recurse naturally.

Facts stop at derived columns: `resolve` returns the derived node itself rather than its
inputs, so evidence about `sum(x)` can never reach `x`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from sqlglot import exp

from sqlrunner.sql_analysis.relations import (
    InputRef,
    RelationGraph,
    RelationInfo,
    line_of,
)
from sqlrunner.sql_analysis.types import (
    CONFIDENCE_RANK,
    Ambiguity,
    AmbiguityReason,
    ColumnNode,
    ColumnOrigin,
    Confidence,
    RelationRef,
)

StarOverJoinBehavior = Literal["error", "guess"]
ResolutionStatus = Literal["resolved", "ambiguous", "unresolved"]

SNIPPET_MAX_LINES = 12


class StarOverJoinAbort(Exception):
    """Raised in `error` mode when a `*` over a join would have to be guessed.

    Attribution backed by evidence (see `Resolver._guess_star_source` rules 1 and 2) does
    not raise; only a fall-through to the leftmost source does.
    """

    def __init__(
        self,
        relation: RelationRef,
        column: str,
        candidates: list[RelationRef],
        snippet: str,
        line: int | None,
    ) -> None:
        self.relation = relation
        self.column = column
        self.candidates = candidates
        self.snippet = snippet
        self.line = line
        location = f" at line {line}" if line is not None else ""
        names = ", ".join(candidate.name for candidate in candidates)
        super().__init__(
            f'star over join is ambiguous: nothing proves which of [{names}] column '
            f'"{column}" in {relation} comes from{location} '
            '(star_over_join_behavior = "error")\n' + snippet
        )


def weakest(left: Confidence, right: Confidence) -> Confidence:
    return left if CONFIDENCE_RANK[left] <= CONFIDENCE_RANK[right] else right


# A Resolution is usually a single resolved column, but could have multiple ColumnNodes in set operations
@dataclass
class Resolution:
    nodes: list[ColumnNode] = field(default_factory=list[ColumnNode])
    confidence: Confidence = "explicit"
    status: ResolutionStatus = "resolved"

    @property
    def resolved(self) -> bool:
        return self.status == "resolved" and bool(self.nodes)

    @property
    def origins(self) -> list[ColumnOrigin]:
        return [
            ColumnOrigin(node=node, confidence=self.confidence) for node in self.nodes
        ]


UNRESOLVED = Resolution(nodes=[], confidence="explicit", status="unresolved")


def _dedupe(refs: list[RelationRef]) -> list[RelationRef]:
    seen: set[RelationRef] = set()
    out: list[RelationRef] = []
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


class Resolver:
    def __init__(
        self,
        graph: RelationGraph,
        star_over_join_behavior: StarOverJoinBehavior = "guess",
        dialect: str | None = None,
    ) -> None:
        self.graph = graph
        self.behavior = star_over_join_behavior
        self.dialect = dialect
        self.ambiguities: list[Ambiguity] = []
        self._memo: dict[tuple[RelationRef, str], Resolution] = {}
        self._active: set[tuple[RelationRef, str]] = set()
        self._star_memo: dict[
            tuple[RelationRef, str], tuple[RelationRef | None, Confidence]
        ] = {}
        self._recorded: set[tuple[RelationRef, str, str]] = set()

    # ---- entry points ---------------------------------------------------------

    def resolve_column_expr(
        self, info: RelationInfo, column: exp.Column
    ) -> Resolution:
        """Resolve a column reference written inside `info`'s scope."""
        bound = self.bind_alias(info, column.table or None, column.name, column)
        if bound is None:
            return UNRESOLVED
        ref, confidence = bound
        result = self.resolve(ref, column.name)
        return Resolution(
            result.nodes, weakest(confidence, result.confidence), result.status
        )

    def resolve(self, ref: RelationRef, column: str) -> Resolution:
        key = (ref, column.lower())
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        if key in self._active:
            # Recursive CTE, or a cycle we cannot unwind.
            return UNRESOLVED

        self._active.add(key)
        try:
            result = self._resolve(ref, column)
        finally:
            self._active.discard(key)

        self._memo[key] = result
        return result

    # ---- resolution -----------------------------------------------------------

    def _resolve(self, ref: RelationRef, column: str) -> Resolution:
        # If `ref` is a table, we cannot resolve any further.
        info = self.graph.get(ref)
        if info is None or info.is_table:
            return Resolution([ColumnNode(relation=ref, column=column)], "explicit")

        # If `ref` is a derived relation, we can only resolve against its outputs. If the
        # column is not present, we cannot resolve further.
        output = info.outputs_by_name.get(column.lower())
        if output is not None:
            if output.kind == "derived":
                return Resolution(
                    [ColumnNode(relation=ref, column=output.name or column)], "explicit"
                )
            return self._resolve_inputs(info, output.inputs)

        if info.is_setop:
            # A branch selected `*`, so positional matching was impossible. Ask every
            # branch by name; a set operation column legitimately has N origins.
            return self._resolve_branches(info, column)

        if info.star_sources:
            target, confidence = self._pick_star_source(info, column)
            if target is None:
                return Resolution([], confidence, "ambiguous")
            result = self.resolve(target, column)
            return Resolution(
                result.nodes, weakest(confidence, result.confidence), result.status
            )

        return UNRESOLVED

    def _resolve_inputs(
        self, info: RelationInfo, inputs: list[InputRef]
    ) -> Resolution:
        nodes: list[ColumnNode] = []
        confidence: Confidence = "explicit"
        worst: ResolutionStatus = "resolved"

        for item in inputs:
            if item.ref is not None:
                result = self.resolve(item.ref, item.column)
            else:
                bound = self.bind_alias(info, item.alias, item.column)
                if bound is None:
                    worst = "unresolved"
                    continue
                ref, bind_confidence = bound
                result = self.resolve(ref, item.column)
                result = Resolution(
                    result.nodes,
                    weakest(bind_confidence, result.confidence),
                    result.status,
                )

            nodes.extend(result.nodes)
            confidence = weakest(confidence, result.confidence)
            if result.status != "resolved":
                worst = result.status

        if not nodes:
            return Resolution([], confidence, worst if worst != "resolved" else "unresolved")
        return Resolution(nodes, confidence, "resolved")

    def _resolve_branches(self, info: RelationInfo, column: str) -> Resolution:
        nodes: list[ColumnNode] = []
        confidence: Confidence = "inferred"
        for branch in info.branches:
            result = self.resolve(branch, column)
            if result.resolved:
                nodes.extend(result.nodes)
                confidence = weakest(confidence, result.confidence)
        if not nodes:
            return UNRESOLVED
        return Resolution(nodes, confidence, "resolved")

    # ---- alias binding --------------------------------------------------------

    def bind_alias(
        self,
        info: RelationInfo,
        alias: str | None,
        column: str,
        expression: exp.Expr | None = None,
    ) -> tuple[RelationRef, Confidence] | None:
        """Map a reference to the relation it reads from."""
        if alias:
            binding = info.binding_for(alias)
            if binding is not None:
                return binding.ref, "explicit"
            outer = self._bind_outer(info, alias)
            if outer is not None:
                return outer, "explicit"
            self._record(
                info, column, [], "unknown_alias", "dropped", expression=expression
            )
            return None

        declaring = _dedupe(
            [b.ref for b in info.bindings if self._declares(b.ref, column)]
        )
        if len(declaring) == 1:
            return declaring[0], "explicit"
        if len(declaring) > 1:
            self._record(
                info,
                column,
                declaring,
                "unqualified_multi_source",
                "dropped",
                expression=expression,
            )
            return None

        if len(info.bindings) == 1:
            return info.bindings[0].ref, "explicit"

        if info.star_sources:
            target, confidence = self._pick_star_source(info, column)
            return (target, confidence) if target is not None else None

        if not info.bindings:
            outer = self._bind_outer_unqualified(info, column)
            if outer is not None:
                return outer, "explicit"
            self._record(
                info, column, [], "unresolved_column", "dropped", expression=expression
            )
            return None

        self._record(
            info,
            column,
            _dedupe([b.ref for b in info.bindings]),
            "unqualified_multi_source",
            "dropped",
            expression=expression,
        )
        return None

    def _bind_outer(self, info: RelationInfo, alias: str) -> RelationRef | None:
        """Correlated reference: the alias belongs to an enclosing scope."""
        parent_ref = info.parent
        while parent_ref is not None:
            parent = self.graph.get(parent_ref)
            if parent is None:
                return None
            binding = parent.binding_for(alias)
            if binding is not None:
                return binding.ref
            parent_ref = parent.parent
        return None

    def _bind_outer_unqualified(
        self, info: RelationInfo, column: str
    ) -> RelationRef | None:
        parent_ref = info.parent
        while parent_ref is not None:
            parent = self.graph.get(parent_ref)
            if parent is None:
                return None
            if len(parent.bindings) == 1:
                return parent.bindings[0].ref
            declaring = _dedupe(
                [b.ref for b in parent.bindings if self._declares(b.ref, column)]
            )
            if len(declaring) == 1:
                return declaring[0]
            parent_ref = parent.parent
        return None

    def _declares(self, ref: RelationRef, column: str) -> bool:
        """Does `ref` explicitly declare `column`?"""
        info = self.graph.get(ref)
        return info is not None and not info.is_table and info.declares(column)

    # ---- star over join -------------------------------------------------------

    def _pick_star_source(
        self, info: RelationInfo, column: str
    ) -> tuple[RelationRef | None, Confidence]:
        key = (info.ref, column.lower())
        cached = self._star_memo.get(key)
        if cached is not None:
            return cached

        candidates = info.star_sources
        if not candidates:
            result: tuple[RelationRef | None, Confidence] = (None, "guessed")
            self._star_memo[key] = result
            return result

        if len(candidates) == 1:
            result = (candidates[0], "inferred")
            self._star_memo[key] = result
            return result

        chosen, confidence = self._guess_star_source(info, column, candidates)

        # Rules 1 and 2 are evidence, not guesswork, so `error` mode lets them stand and
        # only fails when attribution would fall through to the leftmost source.
        if confidence == "guessed" and self.behavior == "error":
            raise StarOverJoinAbort(
                relation=info.ref,
                column=column,
                candidates=candidates,
                snippet=self._snippet(info),
                line=self._star_line(info),
            )

        self._record(
            info,
            column,
            candidates,
            "star_over_join",
            "attributed",
            chosen=chosen,
            confidence=confidence,
        )
        result = (chosen, confidence)
        self._star_memo[key] = result
        return result

    def _guess_star_source(
        self, info: RelationInfo, column: str, candidates: list[RelationRef]
    ) -> tuple[RelationRef, Confidence]:
        """Rules are ordered, first match wins, leftmost candidate breaks ties."""
        # 1. A candidate relation explicitly declares the column.
        for ref in candidates:
            if self._declares(ref, column):
                return ref, "inferred"

        # 2. The column appears qualified against a candidate elsewhere in the scope,
        #    e.g. `on a.id = b.person_id` proves `person_id` belongs to `b`.
        qualified = info.qualified_refs.get(column.lower(), [])
        for ref in candidates:
            if ref in qualified:
                return ref, "inferred"

        # 3. No evidence: the leftmost source takes it.
        return candidates[0], "guessed"

    def _snippet(self, info: RelationInfo) -> str:
        if info.expression is None:
            return ""
        rendered = info.expression.sql(dialect=self.dialect, pretty=True)
        lines = rendered.splitlines()
        if len(lines) > SNIPPET_MAX_LINES:
            lines = lines[:SNIPPET_MAX_LINES] + ["  ..."]
        return "\n".join(f"    {line}" for line in lines)

    def _star_line(self, info: RelationInfo) -> int | None:
        for output in info.outputs:
            if output.kind == "star" and output.expression is not None:
                line = line_of(output.expression)
                if line is not None:
                    return line
        return line_of(info.expression) if info.expression is not None else None

    # ---- diagnostics ----------------------------------------------------------

    def _record(
        self,
        info: RelationInfo,
        column: str,
        candidates: list[RelationRef],
        reason: AmbiguityReason,
        resolution: Literal["attributed", "dropped"],
        chosen: RelationRef | None = None,
        confidence: Confidence | None = None,
        expression: exp.Expr | None = None,
    ) -> None:
        key = (info.ref, column.lower(), reason)
        if key in self._recorded:
            return
        self._recorded.add(key)

        line = line_of(expression) if expression is not None else None
        if line is None and reason == "star_over_join":
            line = self._star_line(info)

        self.ambiguities.append(
            Ambiguity(
                relation=info.ref,
                column=column,
                candidates=candidates,
                reason=reason,
                resolution=resolution,
                chosen=chosen,
                confidence=confidence,
                line=line,
            )
        )
