"""The typing pipeline. Steps are numbered as in `type_check_plan.md`.

Two entrypoints, and the second contains the first:

- `qualify_schema` (steps 1-3, in `qualify.py`) resolves every column to its source and
  expands stars. It answers *where does this name come from*, and answers it without
  needing a single type.
- `validate_schema` (steps 1-7) runs that, then annotates and checks types on top of it.

Steps 4-7 (annotate, backward evidence, re-annotate, check) are marked where they belong
and do nothing yet.
"""

import logging

from pydantic import BaseModel
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from sqlr.config.types import SqlrConfig
from sqlr.declared.types import DeclaredSchemas
from sqlr.selection.types import Model
from sqlr.sql_analysis2.annotate import expression_metadata, func_args
from sqlr.sql_analysis2.qualify import (
    DEFAULT_DIALECT,
    QualifiedModel,
    QualifiedStatement,
    any_model_has_errors,
    qualify_schema,
)
from sqlr.sql_analysis2.render import print_qualification
from sqlr.sql_analysis2.sourcedoc import SourceSpan

__all__ = [
    "DEFAULT_DIALECT",
    "QualifiedModel",
    "QualifiedStatement",
    "any_model_has_errors",
    "qualify_schema",
    "validate_schema",
]

logger = logging.getLogger(__name__)


class TypeFact(BaseModel):
    span: SourceSpan
    type: str


# ------------------------------------------------------------------- diagnostics
def log_coverage(tree: exp.Expr) -> None:
    """Coverage debt: a Func whose arguments are all typed but whose result is UNKNOWN is
    a catalog gap, never a user error. That second number is the one that matters."""
    total = 0
    typed = 0
    gaps: list[str] = []
    for node in tree.walk():
        total += 1
        node_type = node.type
        if node_type is not None and not node_type.is_type(exp.DType.UNKNOWN):
            typed += 1
        elif isinstance(node, exp.Func) and node.is_type(exp.DType.UNKNOWN):
            args = func_args(node)
            if args and all(
                a.type is not None and not a.type.is_type(exp.DType.UNKNOWN) for a in args
            ):
                gaps.append(node.name if isinstance(node, exp.Anonymous) else node.sql_name())
    logger.info(
        "typed %d/%d nodes; %d UNKNOWN with fully-typed arguments (catalog gaps)",
        typed,
        total,
        len(gaps),
    )
    if gaps:
        logger.debug("catalog gaps: %s", ", ".join(sorted(set(gaps))))


def log_projections(tree: exp.Expr) -> None:
    """Per-scope name -> type table. More useful than dumping an annotated tree, and a
    fraction of the size."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    for scope in traverse_scope(tree):
        select = scope.expression
        if not isinstance(select, exp.Select):
            continue
        parent = select.parent
        name = parent.alias if isinstance(parent, exp.CTE) else "<final>"
        logger.debug("-- %s", name)
        for projection in select.selects:
            logger.debug("   %-20s %s", projection.alias_or_name, projection.type)


# ---------------------------------------------------------------------- pipeline
def validate_schema(
    cfg: SqlrConfig, declared: DeclaredSchemas, models: list[Model]
) -> list[QualifiedModel]:
    """Steps 1-7 over every selected model. Returns one result per model.

    Step 0, the annotation metadata, is built once per run rather than per file: it is a
    copy of the dialect's own map with our catalog layered on top, so the dialect itself is
    never mutated and two runs with different dialects cannot interfere (plan Q6).
    """
    dialect_name = cfg.general.sql_dialect or DEFAULT_DIALECT
    metadata = expression_metadata(dialect_name)

    # Steps 1-3. Identical to what `qualify-schema` runs and prints on its own - the two
    # commands must not be able to disagree about where a column comes from.
    qualified = qualify_schema(cfg, declared, models)
    print_qualification(qualified)

    # TODO steps 4-7: `annotate_types(qualified, metadata)`, which types every expression,
    # extracts the non-type facts (plan step 3b) and prints the annotations per model. It
    # takes the results above, so nothing here needs to re-parse or re-qualify.
    _ = metadata

    return qualified
