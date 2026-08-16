"""What an engine calls a projection nobody named.

`select upper(first_name) from t` produces a column, and that column has a name even
though the user wrote none. Every engine invents a different one, and the invented name is
what downstream SQL has to quote, so it is not a display detail - it is part of the
relation's schema.

sqlglot will not answer this. `qualify_outputs` labels the projection `_col_1`, a
placeholder no engine produces, and after qualification it is indistinguishable from a
hand-written `AS _col_1`. So the name has to be derived here, from the projection as it was
*written*, before step 3 rewrites it.

Names are case-sensitive. An engine that folds unquoted identifiers folds this name too,
and a caller quoting `"UPPER(FIRST_NAME)"` against a column named `upper(first_name)` gets
nothing back.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlglot import exp

from sqlr.sql_analysis2.types import ColumnName

type OutputColumnNamer = Callable[[exp.Expr, str | None], ColumnName]
"""Names one unaliased projection. Takes the expression and the exact text it was written
as - the text is preferred where an engine echoes the source, since sqlglot's generator
normalises spacing and keyword case and the engine does not."""


def name_from_the_text_as_written(expression: exp.Expr, written: str | None) -> ColumnName:
    """The projection echoed back exactly as typed. DuckDB, MySQL, SQLite, and the default.

    The source slice rather than a re-generated string: DuckDB names the column
    `upper(first_name)`, lower-case, while sqlglot's generator would render
    `UPPER(first_name)`.
    """
    return written if written is not None else expression.sql()


def name_upper_cased(expression: exp.Expr, written: str | None) -> ColumnName:
    """Snowflake: the same text, folded the way it folds every unquoted identifier."""
    return name_from_the_text_as_written(expression, written).upper()


POSTGRES_UNNAMED_OUTPUT = "?column?"

NAMES_MEANING_UNNAMED = frozenset({POSTGRES_UNNAMED_OUTPUT})
"""Names an engine gives a projection it declined to name.

Not real names, and the difference matters in exactly one place: two of them in one
projection list is not two columns colliding, it is two columns nobody named. Postgres
happily returns `?column?` twice and only complains if something tries to reference it.
"""


def is_a_name_meaning_unnamed(name: ColumnName) -> bool:
    """Whether a name is an engine's way of saying the projection has none."""
    return name in NAMES_MEANING_UNNAMED


def name_from_the_function_called(expression: exp.Expr, written: str | None) -> ColumnName:
    """Postgres: the function that produced the value, or `?column?` when none did.

    `upper(first_name)` is `upper`; `a + b` and `1` are `?column?`. Postgres names an
    output after the thing that computed it, never after the whole expression.
    """
    _ = written
    if isinstance(expression, exp.Anonymous):
        return expression.name.lower()
    if isinstance(expression, exp.Func):
        return expression.sql_name().lower()
    return POSTGRES_UNNAMED_OUTPUT


OUTPUT_COLUMN_NAMERS: dict[str, OutputColumnNamer] = {
    "snowflake": name_upper_cased,
    "postgres": name_from_the_function_called,
    "redshift": name_from_the_function_called,
}
"""Dialects whose naming differs from echoing the written text. Add an entry when a
dialect is found to disagree; the default is right for more engines than not."""


def output_column_namer(dialect_name: str) -> OutputColumnNamer:
    """The namer for a dialect. Unlisted dialects echo the written text."""
    return OUTPUT_COLUMN_NAMERS.get(dialect_name.lower(), name_from_the_text_as_written)
