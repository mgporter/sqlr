"""The typing pipeline. Steps are numbered as in `type_check_plan.md`.

Two entrypoints, and the second contains the first:

- `qualify_schema` (steps 1-3, in `qualify.py`) resolves every column to its source and
  expands stars. It answers *where does this name come from*, and answers it without
  needing a single type.
- `validate_schema` (steps 1-7) runs that, then annotates and checks types on top of it.

Steps 4-7 live in `annotate_types.py`, with the facts they run on in `facts.py`. The one
rule they share: a fact is a claim about a value, a claim contradicted by that value's
actual type is an error, and a claim about a value with no type at all is the inference.
"""

import logging

from sqlr.config.types import SqlrConfig
from sqlr.declared.types import DeclaredSchemas
from sqlr.selection.types import Model
from sqlr.sql_analysis2.annotate import expression_metadata
from sqlr.sql_analysis2.annotate_types import AnnotatedModel, annotate_types
from sqlr.sql_analysis2.qualify import (
    DEFAULT_DIALECT,
    QualifiedModel,
    QualifiedStatement,
    any_model_has_errors,
    qualify_schema,
)

__all__ = [
    "DEFAULT_DIALECT",
    "AnnotatedModel",
    "QualifiedModel",
    "QualifiedStatement",
    "annotate_types",
    "any_model_has_errors",
    "qualify_schema",
    "validate_schema",
]

logger = logging.getLogger(__name__)


def validate_schema(
    cfg: SqlrConfig, declared: DeclaredSchemas, models: list[Model]
) -> list[AnnotatedModel]:
    """Steps 1-7 over every selected model. Returns one result per model.

    Step 0, the annotation metadata, is built once per run rather than per file: it is a
    copy of the dialect's own map with our catalog layered on top, so the dialect itself is
    never mutated and two runs with different dialects cannot interfere (plan Q6).
    """
    dialect_name = cfg.general.sql_dialect or DEFAULT_DIALECT
    metadata = expression_metadata(dialect_name)

    # Steps 1-3. Identical to what `qualify-schema` runs on its own - the two commands must
    # not be able to disagree about where a column comes from.
    qualified = qualify_schema(cfg, declared, models)

    # Steps 4-7. Takes the results above, so nothing here re-parses or re-qualifies.
    return annotate_types(qualified, metadata)
