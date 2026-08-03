"""Step 2 of the pipeline: build the relation graph.

    statement (parsed, table-qualified) -> RelationGraph

One `RelationInfo` per SELECT scope (plus one per external table), holding the ordered
FROM/JOIN bindings and the SELECT list decomposed into passthrough / derived / star
columns. Nothing is resolved here; this is the raw material the resolver walks backwards.

See `__init__.py` for the pipeline as a whole.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope

from sqlrunner.sql_analysis.types import RelationRef

_SET_OPERATION_BASE = getattr(exp, "SetOperation", None)
SET_OPERATION_CLASSES: tuple[type[exp.Expr], ...] = (
    (_SET_OPERATION_BASE,) if _SET_OPERATION_BASE else (exp.Union, exp.Intersect, exp.Except)
)

UNSUPPORTED_SOURCES: tuple[type[exp.Expr], ...] = (
    exp.Unnest,
    exp.Lateral,
    exp.Values,
)


def qualified_name(table: exp.Table) -> str:
    parts = [part for part in (table.catalog, table.db, table.name) if part]
    return ".".join(parts)


def line_of(expression: exp.Expr) -> int | None:
    """Best-effort source line. sqlglot only records position on leaf tokens."""
    line = expression.meta.get("line")
    if isinstance(line, int):
        return line
    for node in expression.walk():
        line = node.meta.get("line")
        if isinstance(line, int):
            return line
    return None


@dataclass(frozen=True)
class InputRef:
    """One input of an output column, before resolution.

    Either `alias` (resolve through the owning relation's bindings) or `ref` (a direct
    relation, used for set-operation branches which have no alias).
    """

    column: str
    alias: str | None = None
    ref: RelationRef | None = None


@dataclass
class Binding:
    """One FROM or JOIN item, in textual order."""

    alias: str
    ref: RelationRef
    position: int
    join_side: str = ""
    join_kind: str = ""
    join: exp.Join | None = None

    @property
    def is_outer_padded(self) -> bool:
        """True when this side of the join can be null-padded."""
        return self.join_side in ("LEFT", "FULL")


@dataclass
class OutputCol:
    name: str | None
    ordinal: int
    kind: str

    # The inputs that contribute to this output column. A column passed through
    # will have one input; but a column derived from others 
    # will have multiple inputs (e.g. `SELECT a + b AS c` -> inputs=[a, b]).
    inputs: list[InputRef] = field(default_factory=list[InputRef])

    # Empty for passthrough and derived columns; single element for qualified stars (a.*),
    # multiple elements for unqualified stars (select * from a join b).
    star_sources: list[RelationRef] = field(default_factory=list[RelationRef])
    function: str | None = None
    expression: exp.Expr | None = None


@dataclass
class RelationInfo:

    """ A relationRef with its associated metadata, 
    including the scope, expression, bindings, outputs, and other relevant information.
    """

    ref: RelationRef

    # The scope the relation came from
    scope: Scope | None = None
    expression: exp.Expr | None = None

    # set operations
    is_setop: bool = False
    branches: list[RelationRef] = field(default_factory=list[RelationRef])

    # For CTEs, what relations (tables/cte references) are within this relation
    bindings: list[Binding] = field(default_factory=list[Binding])

    # Columns output by this CTE's select statement
    outputs: list[OutputCol] = field(default_factory=list[OutputCol])
    outputs_by_name: dict[str, OutputCol] = field(default_factory=dict[str, OutputCol])

    # The relations that could contribute to a star output.
    # Eg. select * from a join b -> yields [table:a, table:b] (both tables' columns could contribute to the star's columns)
    # E.g. select a.*, b.id from a join b -> yields [table:a] (only a's columns could contribute to the star's columns)
    star_sources: list[RelationRef] = field(default_factory=list[RelationRef])

    # Within this cte, dict of column names to the relations that could provide them.
    # Usually there will be only one relation per column, but there could be more when...
    #  - select * from a join b where a.id = b.id  -> 'id' could come from either a or b
    #  - The same column is referenced in the cte more than once, than list could have duplicates: e.g. qualified_refs["id"] = [table:a, table:a]
    qualified_refs: dict[str, list[RelationRef]] = field(
        default_factory=dict[str, list[RelationRef]]
    )

    # list of columns that are defined in this relation's scope, but not necessarily outputted by it.
    own_columns: list[exp.Column] = field(default_factory=list[exp.Column])

    # Generated ids of all columns in own_columns
    scope_column_ids: set[int] = field(default_factory=set[int])

    parent: RelationRef | None = None

    @property
    def kind(self) -> str:
        return self.ref.kind

    @property
    def is_table(self) -> bool:
        return self.ref.kind == "table"

    def binding_for(self, alias: str) -> Binding | None:
        """ Return the binding for the given alias, or None if not found. """
        lowered = alias.lower()
        for binding in self.bindings:
            if binding.alias.lower() == lowered:
                return binding
        return None

    def declares(self, column: str) -> bool:
        """ True if this relation outputs a column with the given name. """
        return column.lower() in self.outputs_by_name


@dataclass
class RelationGraph:
    root: RelationRef
    relations: dict[RelationRef, RelationInfo] = field(
        default_factory=dict[RelationRef, RelationInfo]
    )
    order: list[RelationRef] = field(default_factory=list[RelationRef])
    warnings: list[str] = field(default_factory=list[str])

    def get(self, ref: RelationRef) -> RelationInfo | None:
        return self.relations.get(ref)

    def expand_outputs(
        self, ref: RelationRef, seen: frozenset[RelationRef] = frozenset()
    ) -> list[tuple[RelationRef, OutputCol]] | None:
        """Flatten a relation's SELECT list, splicing in `*` sources recursively.

        Each result pairs an output column with the relation that owns it, so the caller
        can resolve the column against the right scope.

        Returns None when a `*` reads a relation whose column list is unknown - an
        external table, or a set operation with a starred branch.
        """
        info = self.get(ref)
        if info is None or info.is_table or ref in seen:
            return None
        if not info.outputs:
            return None

        seen = seen | {ref}
        expanded: list[tuple[RelationRef, OutputCol]] = []

        for output in info.outputs:
            if output.kind != "star":
                expanded.append((ref, output))
                continue
            if not output.star_sources:
                return None
            for source in output.star_sources:
                nested = self.expand_outputs(source, seen)
                if nested is None:
                    return None
                expanded.extend(nested)

        return expanded


def ordered_dedupe(refs: list[RelationRef]) -> list[RelationRef]:
    """Drop duplicate refs, keeping first-seen order."""
    seen: set[RelationRef] = set()
    out: list[RelationRef] = []
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


def _function_name(expression: exp.Expr) -> str:
    inner = expression
    while isinstance(inner, exp.Paren):
        inner = inner.this
    if isinstance(inner, exp.Window):
        return type(inner.this).__name__
    return type(inner).__name__


class _GraphBuilder:
    def __init__(self, statement: exp.Expr) -> None:
        self.statement = statement
        self.scopes: list[Scope] = traverse_scope(statement)
        self.ref_by_scope_id: dict[int, RelationRef] = {}
        self.ref_by_expr_id: dict[int, RelationRef] = {}
        self.used_names: set[str] = set()
        self.graph = RelationGraph(root=RelationRef(kind="root", name=""))
        self._anon = 0

    # ---- naming ---------------------------------------------------------------

    def _unique(self, kind: str, name: str) -> RelationRef:
        candidate = name
        suffix = 2
        while f"{kind}:{candidate}" in self.used_names:
            candidate = f"{name}#{suffix}"
            suffix += 1
        self.used_names.add(f"{kind}:{candidate}")
        return RelationRef(kind=kind, name=candidate)  # type: ignore[arg-type]

    def _next_anon(self) -> int:
        self._anon += 1
        return self._anon

    def _ref_for_scope(self, scope: Scope) -> RelationRef:
        expression = scope.expression
        parent = expression.parent

        if scope.is_root:
            return self._unique("root", "")
        if isinstance(parent, exp.CTE):
            return self._unique("cte", parent.alias)
        if scope.is_derived_table:
            alias = getattr(parent, "alias", "") or f"_derived_{self._next_anon()}"
            return self._unique("derived", alias)
        if isinstance(parent, SET_OPERATION_CLASSES):
            return self._unique("derived", f"_setop_branch_{self._next_anon()}")
        alias = getattr(parent, "alias", "") or f"_subquery_{self._next_anon()}"
        return self._unique("subquery", alias)

    def _table_ref(self, table: exp.Table) -> RelationRef:
        ref = RelationRef(kind="table", name=qualified_name(table))
        if ref not in self.graph.relations:
            self.graph.relations[ref] = RelationInfo(ref=ref)
            self.graph.order.append(ref)
        return ref

    # ---- build ----------------------------------------------------------------

    def build(self) -> RelationGraph:
        for scope in self.scopes:
            ref = self._ref_for_scope(scope)
            self.ref_by_scope_id[id(scope)] = ref
            self.ref_by_expr_id[id(scope.expression)] = ref
            if scope.is_root:
                self.graph.root = ref

        for scope in self.scopes:
            self._build_relation(scope)

        return self.graph

    def _build_relation(self, scope: Scope) -> None:
        ref = self.ref_by_scope_id[id(scope)]
        info = RelationInfo(ref=ref, scope=scope, expression=scope.expression)
        # `Scope.columns` leaks columns belonging to nested subqueries, which would
        # otherwise be attributed to this scope's sources.
        info.own_columns = [
            column for column in scope.columns if self._owning_ref(column) == ref
        ]
        info.scope_column_ids = {id(column) for column in info.own_columns}
        info.parent = self._parent_ref(scope.expression)

        self.graph.relations[ref] = info
        self.graph.order.append(ref)

        if isinstance(scope.expression, SET_OPERATION_CLASSES):
            info.is_setop = True
            self._build_setop_outputs(scope, info)
            return

        self._build_bindings(scope, info)
        self._build_qualified_refs(info)
        self._build_outputs(info)

    def _parent_ref(self, expression: exp.Expr) -> RelationRef | None:
        node = expression.parent
        while node is not None:
            ref = self.ref_by_expr_id.get(id(node))
            if ref is not None:
                return ref
            node = node.parent
        return None

    def _owning_ref(self, expression: exp.Expr) -> RelationRef | None:
        """The relation whose scope this expression sits directly inside."""
        node: exp.Expr | None = expression
        while node is not None:
            ref = self.ref_by_expr_id.get(id(node))
            if ref is not None:
                return ref
            node = node.parent
        return None

    def _build_bindings(self, scope: Scope, info: RelationInfo) -> None:
        expression = scope.expression
        items: list[tuple[exp.Expr, exp.Join | None]] = []

        # sqlglot renamed this arg to `from_` in v30; keep reading both.
        from_clause: exp.From | None = expression.args.get("from_") or expression.args.get(
            "from"
        )
        if from_clause is not None:
            items.append((from_clause.this, None))

        joins: list[exp.Join] = expression.args.get("joins") or []
        for join in joins:
            items.append((join.this, join))

        sources = {alias.lower(): source for alias, source in scope.sources.items()}

        for position, (item, join) in enumerate(items):
            if join is not None:
                if join.args.get("using"):
                    self.graph.warnings.append(
                        f"USING join is not supported; columns of "
                        f"{item.alias_or_name!r} may be unresolved"
                    )
                if (join.args.get("method") or "").upper() == "NATURAL":
                    self.graph.warnings.append(
                        f"NATURAL join is not supported; columns of "
                        f"{item.alias_or_name!r} may be unresolved"
                    )

            if isinstance(item, UNSUPPORTED_SOURCES):
                self.graph.warnings.append(
                    f"unsupported FROM item {type(item).__name__} in {info.ref}"
                )
                continue

            alias = item.alias_or_name
            source = sources.get(alias.lower())
            ref: RelationRef | None = None

            if isinstance(source, Scope):
                ref = self.ref_by_scope_id.get(id(source))
            elif isinstance(source, exp.Table):
                ref = self._table_ref(source)
            elif isinstance(item, exp.Table):
                ref = self._table_ref(item)

            if ref is None:
                self.graph.warnings.append(
                    f"could not resolve FROM item {alias!r} in {info.ref}"
                )
                continue

            info.bindings.append(
                Binding(
                    alias=alias,
                    ref=ref,
                    position=position,
                    join_side=(join.side if join is not None else "") or "",
                    join_kind=(join.kind if join is not None else "") or "",
                    join=join,
                )
            )

    def _build_qualified_refs(self, info: RelationInfo) -> None:
        for column in info.own_columns:
            if not column.table:
                continue
            binding = info.binding_for(column.table)
            if binding is None:
                continue
            info.qualified_refs.setdefault(column.name.lower(), []).append(binding.ref)

    def _build_outputs(self, info: RelationInfo) -> None:
        expression = info.expression
        if not isinstance(expression, exp.Select):
            return

        binding_refs = ordered_dedupe([b.ref for b in info.bindings])

        for ordinal, select in enumerate(expression.selects):
            output = self._build_output(info, select, ordinal, binding_refs)
            info.outputs.append(output)
            if output.name is not None:
                info.outputs_by_name.setdefault(output.name.lower(), output)

        info.star_sources = self._order_by_binding(
            info, ordered_dedupe([r for o in info.outputs for r in o.star_sources])
        )

    def _build_output(
        self,
        info: RelationInfo,
        select: exp.Expr,
        ordinal: int,
        binding_refs: list[RelationRef],
    ) -> OutputCol:
        if isinstance(select, exp.Star):
            return OutputCol(
                name=None,
                ordinal=ordinal,
                kind="star",
                star_sources=list(binding_refs),
                expression=select,
            )

        if isinstance(select, exp.Column) and isinstance(select.this, exp.Star):
            binding = info.binding_for(select.table) if select.table else None
            return OutputCol(
                name=None,
                ordinal=ordinal,
                kind="star",
                star_sources=[binding.ref] if binding else [],
                expression=select,
            )

        name = select.alias_or_name or f"_col_{ordinal}"
        inner = select.this if isinstance(select, exp.Alias) else select
        while isinstance(inner, exp.Paren):
            inner = inner.this

        if isinstance(inner, exp.Column) and not isinstance(inner.this, exp.Star):
            return OutputCol(
                name=name,
                ordinal=ordinal,
                kind="passthrough",
                inputs=[InputRef(column=inner.name, alias=inner.table or None)],
                expression=select,
            )

        inputs = [
            InputRef(column=column.name, alias=column.table or None)
            for column in inner.find_all(exp.Column)
            if id(column) in info.scope_column_ids
        ]
        return OutputCol(
            name=name,
            ordinal=ordinal,
            kind="derived",
            inputs=inputs,
            function=_function_name(inner),
            expression=select,
        )

    def _order_by_binding(
        self, info: RelationInfo, refs: list[RelationRef]
    ) -> list[RelationRef]:
        """Star sources must be in FROM order; `scope.sources` order is not reliable."""
        position = {binding.ref: binding.position for binding in info.bindings}
        return sorted(refs, key=lambda ref: position.get(ref, len(position)))

    def _build_setop_outputs(self, scope: Scope, info: RelationInfo) -> None:
        branches = [
            self.ref_by_scope_id[id(branch)]
            for branch in scope.union_scopes
            if id(branch) in self.ref_by_scope_id
        ]
        info.branches = branches
        branch_infos = [self.graph.relations[ref] for ref in branches]

        if not branch_infos:
            return

        if any(any(o.kind == "star" for o in bi.outputs) for bi in branch_infos):
            # Positional matching is impossible when a branch column list is unknown.
            # Fall back to resolving names against every branch.
            self.graph.warnings.append(
                f"set operation {info.ref} has a branch selecting *; "
                "output columns matched by name instead of position"
            )
            return

        width = min(len(bi.outputs) for bi in branch_infos)
        if len({len(bi.outputs) for bi in branch_infos}) > 1:
            self.graph.warnings.append(
                f"set operation {info.ref} has branches with differing column counts"
            )

        for ordinal in range(width):
            inputs = [
                InputRef(column=bi.outputs[ordinal].name or f"_col_{ordinal}", ref=ref)
                for ref, bi in zip(branches, branch_infos)
            ]
            output = OutputCol(
                name=branch_infos[0].outputs[ordinal].name,
                ordinal=ordinal,
                kind="passthrough",
                inputs=inputs,
            )
            info.outputs.append(output)
            if output.name is not None:
                info.outputs_by_name.setdefault(output.name.lower(), output)


def build_graph(statement: exp.Expr) -> RelationGraph:
    return _GraphBuilder(statement).build()
