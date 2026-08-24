"""Steps 4-7: annotate, extract facts, infer, re-annotate, check.

Takes what `qualify_schema` produced and types it. Nothing here re-parses or re-qualifies -
steps 1-3 happen once, in `qualify.py`, and the two commands must not be able to disagree
about where a column comes from.

The step numbers are the plan's, and the ordering between them is load-bearing:

    4  annotate pass 1     establishes what is ALREADY known
    -  extract facts       claims and links, plus the non-type facts
    5  infer               facts about UNKNOWN source columns become types, or conflict
    6  annotate pass 2     same tree, widened schema, in place
    7  check               F001, F002, and every contradicted claim

Pass 1's real job is not producing types. It is establishing what is already known, so the
fact walk only speaks about columns nobody has described - see `facts.py`.
"""

from __future__ import annotations

import logging
from typing import Literal, NamedTuple

from sqlglot import exp
from sqlglot.optimizer.annotate_types import annotate_types as annotate_types_with_sqlglot
from sqlglot.optimizer.scope import Scope
from sqlglot.schema import ensure_schema
from sqlglot.typing import ExprMetadataType

from sqlr.sql_analysis2.annotate import arguments_of_call
from sqlr.sql_analysis2.check import (
    findings_for_calls_with_wrong_arity,
    findings_for_columns_with_conflicting_facts,
    findings_for_contradicted_claims,
    findings_for_unknown_functions,
)
from sqlr.sql_analysis2.facts import Facts, extract_facts_from_annotated_tree
from sqlr.sql_analysis2.families import FamilyName
from sqlr.sql_analysis2.infer import (
    Inference,
    InferredColumnType,
    TypeEvidence,
    infer_types_for_undeclared_columns,
    widen_schema_with_inferred_types,
)
from sqlr.sql_analysis2.qualify import (
    QualifiedModel,
    QualifiedStatement,
    name_and_kind_of_scope,
    output_name_of,
    qualifier_of_a_star_projection,
)
from sqlr.sql_analysis2.relations import relation_key_of
from sqlr.sql_analysis2.reporting import TypeFinding
from sqlr.sql_analysis2.resolve import (
    is_declared,
    needs_inference,
    nested_schema_for_sqlglot,
)
from sqlr.sql_analysis2.types import (
    ColumnName,
    ColumnTypeName,
    RelationKey,
    ScopeKind,
)

logger = logging.getLogger(__name__)

type TypeProvenance = Literal["declared", "inferred", "computed", "unknown"]
"""Where a type came from, which is what a reader needs to know how much to trust it.

- `declared` - the user wrote it in a yml. The source of truth; nothing was inferred.
- `computed` - sqlglot derived it from the expression, bottom-up.
- `inferred` - step 5 read it off how the SQL uses the column.
- `unknown`  - nothing could say. Absorbing: anything computed from it is unknown too.
"""

UNKNOWN_TYPE_NAME = "unknown"


class ColumnTypeAnnotation(NamedTuple):
    """One output column of one scope, typed."""

    name: ColumnName
    type_name: str
    provenance: TypeProvenance


class ScopeTypes(NamedTuple):
    name: str
    kind: ScopeKind
    columns: list[ColumnTypeAnnotation]


class SourceColumnType(NamedTuple):
    """One column of a real table: what it is, and how that was settled."""

    table: RelationKey
    column: ColumnName
    type_name: str
    provenance: TypeProvenance
    evidence: list[TypeEvidence]
    """Empty for a declared column - a declaration needs no evidence, it *is* the answer."""
    family: FamilyName | None = None
    """The lattice node behind an inferred type. None for anything else.

    What a consumer that needs a real column type should read: `type_name` may be the family
    name itself (`NUMERIC`, `TEMPORAL`), and matching on that string is a parser where a
    lookup will do."""


class AnnotatedModel(NamedTuple):
    """One model's trip through steps 4-7, whether or not it got there.

    `qualified` is the whole of steps 1-3, carried rather than copied: a reader of this
    result still needs the findings and the source document that produced it.
    """

    qualified: QualifiedModel
    facts: Facts
    inference: Inference
    findings: list[TypeFinding]
    scopes: list[ScopeTypes]
    source_columns: list[SourceColumnType]

    @property
    def has_errors(self) -> bool:
        return self.qualified.has_errors or any(
            finding.severity == "error" for finding in self.findings
        )


def annotate_types(
    qualified_models: list[QualifiedModel], metadata: ExprMetadataType
) -> list[AnnotatedModel]:
    """Steps 4-7 over every model `qualify_schema` returned, in the same order."""
    return [annotate_one_model(result, metadata) for result in qualified_models]


def annotate_one_model(
    result: QualifiedModel, metadata: ExprMetadataType
) -> AnnotatedModel:
    """Steps 4-7 for one model. Never raises: a model that failed step 3 comes back empty."""
    statement = result.statement
    if statement is None:
        return AnnotatedModel(
            qualified=result,
            facts=Facts(),
            inference=Inference(),
            findings=[],
            scopes=[],
            source_columns=[],
        )

    dialect_name = statement.dialect_name
    schema = statement.declared_types_per_relation

    # Step 4 - annotate, pass 1. Types every expression from what is currently known, and
    # equally importantly establishes what is ALREADY known, so the fact walk does not
    # over-claim about columns the user already declared.
    tree = annotate_types_with_sqlglot(
        statement.qualified,
        schema=statement.mapped_schema,
        expression_metadata=metadata,
        dialect=dialect_name,
    )
    log_type_coverage(tree, result.model.relative_path)

    # Facts. After pass 1, never before it: with every `.type` still None, every argument
    # position would generate a claim about a column the user had already described.
    facts = extract_facts_from_annotated_tree(statement, result.positions)
    logger.info(
        "%s: %d type claims, %d type links, %d predicates, %d joins",
        result.model.relative_path,
        len(facts.type_claims),
        len(facts.type_links),
        len(facts.predicates),
        len(facts.joins),
    )

    # Step 5 - resolve every value to a type, or say why it could not be.
    #
    # ⚠️ This runs even when the schema is fully declared, and the earlier plan's skip is
    # gone with it. That skip rested on step 5 doing one job - filling UNKNOWN slots - which
    # nothing can do when there are none. Step 5 now does a second job that a complete
    # declaration does not make vacuous: it detects **stated** values that disagree, and
    # `select a.s from a join b on a.s = b.n` with `s` declared varchar and `n` declared
    # bigint is exactly the case a fully-declared project most needs reported. The pass is
    # union-find over facts already extracted, so what the skip used to save was never here.
    if not needs_inference(schema):
        logger.info("schema fully declared; nothing to infer, still checking for conflicts")
    inference = infer_types_for_undeclared_columns(facts, schema, dialect_name)
    for entry in inference.inferred:
        logger.info(
            "inferred %s.%s %s (%s, %s) from %d fact(s)",
            entry.table,
            entry.column,
            entry.type_name,
            entry.family,
            entry.strength,
            len(entry.evidence),
        )

    # Step 6 - annotate, pass 2. Same tree, no re-parse, no re-qualify: qualification
    # depends on which columns *exist*, and widening never changes the column set.
    #
    # Gated on the schema having actually changed. An undeclared project where inference
    # finds nothing - a file with no predicates and no calls - would otherwise pay a full
    # second annotation pass for a guaranteed no-op. This is the only skip left, and it is
    # sound for the reason the old one was not: it compares the two inputs directly rather
    # than predicting that they will match.
    widened = widen_schema_with_inferred_types(schema, inference)
    if widened != schema:
        tree = annotate_types_with_sqlglot(
            tree,
            # The nested form, for the same reason step 3 needed it: a flat dotted key
            # only ever matches a table written as one identifier.
            schema=ensure_schema(
                nested_schema_for_sqlglot(widened, dialect_name), dialect=dialect_name
            ),
            expression_metadata=metadata,
            dialect=dialect_name,
        )
        log_type_coverage(tree, result.model.relative_path)

    # Step 7 - check. Reads node.type only, never the SQL text: everything upstream has
    # been flattened into one uniform annotated tree.
    findings = [
        *findings_for_unknown_functions(tree, result.positions, dialect_name),
        *findings_for_calls_with_wrong_arity(tree, result.positions, dialect_name),
        *findings_for_contradicted_claims(facts, schema, inference),
        *findings_for_columns_with_conflicting_facts(inference.conflicts),
    ]
    logger.info(
        "%s: %d type findings", result.model.relative_path, len(findings)
    )

    return AnnotatedModel(
        qualified=result,
        facts=facts,
        inference=inference,
        findings=findings,
        scopes=types_per_scope(statement, inference),
        source_columns=types_per_source_column(statement, inference),
    )


# ------------------------------------------------------------------ reading the types
def type_name_of(node: exp.Expr | None) -> str:
    """A node's type as a reader would write it, or `unknown`.

    Prints whatever parameters the type carries and never fills any in - the schema was built
    to hold none that nobody wrote, see `type_held_without_parameters_nobody_wrote`.
    """
    if node is None or node.type is None or node.type.is_type(exp.DType.UNKNOWN):
        return UNKNOWN_TYPE_NAME
    return node.type.sql()


def types_per_scope(statement: QualifiedStatement, inference: Inference) -> list[ScopeTypes]:
    """Every scope's output schema, typed, in the order sqlglot resolves them.

    Dependency order - a CTE before whatever selects from it - so the final projection comes
    last, which is where a reader looks for it.
    """
    inferred = inference.types_per_table()
    out: list[ScopeTypes] = []
    for scope in statement.scopes:
        select = scope.expression
        if not isinstance(select, exp.Select):
            continue
        name, kind = name_and_kind_of_scope(scope)
        columns = [
            _column_type_annotation(projection, scope, statement, inferred)
            for projection in select.selects
            if qualifier_of_a_star_projection(projection) is False
        ]
        if columns:
            out.append(ScopeTypes(name=name, kind=kind, columns=columns))
    return out


def _column_type_annotation(
    projection: exp.Expr,
    scope: Scope,
    statement: QualifiedStatement,
    inferred: dict[RelationKey, dict[ColumnName, InferredColumnType]],
) -> ColumnTypeAnnotation:
    """One projected column, named and typed.

    A projection that is nothing but a read of an inferred source column is reported as that
    column. sqlglot only ever saw the stand-in widening wrote into the schema, so asking it
    would print `DECIMAL` under a column the source table calls `NUMERIC` - the same value,
    named two ways, one line apart. Anything computed keeps sqlglot's answer: `ifnull(bonus,
    0)` really is a decimal expression, whatever the column feeding it is known as.
    """
    provenance, source_column = provenance_and_inferred_column_of_projection(
        projection, scope, statement, inferred
    )
    return ColumnTypeAnnotation(
        name=output_name_of(projection, statement.engine_named_projections),
        type_name=(
            source_column.type_name if source_column is not None else type_name_of(projection)
        ),
        provenance=provenance,
    )


def provenance_and_inferred_column_of_projection(
    projection: exp.Expr,
    scope: Scope,
    statement: QualifiedStatement,
    inferred: dict[RelationKey, dict[ColumnName, InferredColumnType]],
) -> tuple[TypeProvenance, InferredColumnType | None]:
    """How much to trust one projected column's type, and the inferred column behind it.

    A projection that *is* a source column carries that column's provenance; anything built
    from one is computed, whatever its inputs were. `unknown` wins over everything, because
    an untyped column is not a computed answer - it is the absence of one.

    The second half of the answer is non-None only for a passthrough of a column step 5
    inferred - the one case where the scope's type and the source column's type are the same
    fact and the caller should print the source column's name for it.
    """
    if type_name_of(projection) == UNKNOWN_TYPE_NAME:
        return "unknown", None
    inner = projection.this if isinstance(projection, exp.Alias) else projection
    if not isinstance(inner, exp.Column):
        return "computed", None

    source = scope.sources.get(inner.table)
    if not isinstance(source, exp.Table):
        # A CTE or derived table: its column was computed by the scope that produced it.
        return "computed", None

    # The full dotted key, never the bare table name: the schema is keyed on `RelationKey`
    # so that two schemas may each hold a `raw_department`, and a bare-name lookup misses
    # every time - which reported every declared column as `unknown`.
    table = relation_key_of(source)
    column = inner.name.lower()
    if is_declared(statement.declared_types_per_relation.get(table, {}).get(column)):
        return "declared", None
    entry = inferred.get(table, {}).get(column)
    if entry is not None:
        return "inferred", entry
    return "unknown", None


def types_per_source_column(
    statement: QualifiedStatement, inference: Inference
) -> list[SourceColumnType]:
    """Every column of every real table the statement reads, and how its type was settled.

    The gap-filled schema is the column list, so a column that is declared but never read
    still appears - a declaration nothing uses is worth seeing. A column whose facts
    conflicted is `unknown` here and reported by its findings; naming a type for it would
    be the guess step 5 refused to make.
    """
    inferred = inference.types_per_table()
    return [
        _source_column_type(table, column, written, inferred)
        for table, columns in sorted(statement.declared_types_per_relation.items())
        for column, written in sorted(columns.items())
    ]


def _source_column_type(
    table: RelationKey,
    column: ColumnName,
    written: ColumnTypeName,
    inferred: dict[RelationKey, dict[ColumnName, InferredColumnType]],
) -> SourceColumnType:
    if is_declared(written):
        return SourceColumnType(
            table=table,
            column=column,
            type_name=written,
            provenance="declared",
            evidence=[],
        )

    entry = inferred.get(table, {}).get(column)
    if entry is not None:
        return SourceColumnType(
            table=table,
            column=column,
            type_name=entry.type_name,
            provenance="inferred",
            evidence=entry.evidence,
            family=entry.family,
        )

    return SourceColumnType(
        table=table,
        column=column,
        type_name=UNKNOWN_TYPE_NAME,
        provenance="unknown",
        evidence=[],
    )


# ------------------------------------------------------------------------ diagnostics
def log_type_coverage(tree: exp.Expr, path: object) -> None:
    """Coverage debt: a Func whose arguments are all typed but whose result is UNKNOWN is a
    catalog gap, never a user error. That second number is the one that matters."""
    total = 0
    typed = 0
    gaps: list[str] = []
    for node in tree.walk():
        total += 1
        node_type = node.type
        if node_type is not None and not node_type.is_type(exp.DType.UNKNOWN):
            typed += 1
        elif isinstance(node, exp.Func) and node.is_type(exp.DType.UNKNOWN):
            args = arguments_of_call(node)
            if args and all(
                a.type is not None and not a.type.is_type(exp.DType.UNKNOWN) for a in args
            ):
                gaps.append(node.name if isinstance(node, exp.Anonymous) else node.sql_name())
    logger.info(
        "%s: typed %d/%d nodes; %d UNKNOWN with fully-typed arguments (catalog gaps)",
        path,
        typed,
        total,
        len(gaps),
    )
    if gaps:
        logger.debug("catalog gaps: %s", ", ".join(sorted(set(gaps))))
