"""Building `DeclaredSchemas` from a plain dict, for tests that only care about types.

Every pipeline entry point takes `DeclaredSchemas` rather than a bare mapping, because a
declaration carries more than types: which relation it names, whether its column list is
complete, and where it was written. A test that only needs the types should not have to
spell all of that out, so it writes `{relation: {column: type}}` and this fills the rest in.
"""

from pathlib import Path

from sqlr.declared.types import DeclaredColumn, DeclaredSchemas, DeclaredSourceTable
from sqlr.sql_analysis2.types import ColumnName, ColumnTypeName, RelationKey
from sqlr.typemap import resolve_type_name


def q(name: str) -> RelationKey:
    """`test` -> `mydatabase.myschema.test`, the relation most fixtures write.

    A `RelationKey` is the parts the SQL has to write, so a declaration and a reference
    only meet when they are spelled the same way. Fixtures that write a table bare use the
    bare name as its key instead.
    """
    return f"mydatabase.myschema.{name}"


def declarations(
    declared: dict[RelationKey, dict[ColumnName, ColumnTypeName]] | None = None,
    partial: frozenset[RelationKey] = frozenset(),
) -> DeclaredSchemas:
    """A `DeclaredSchemas` from `{relation: {column: type}}`.

    The key is split back into the parts a `sources:` entry writes, so the declaration
    resolves to the same relation the SQL names. `partial` holds the keys whose entry
    carries `meta.declaration_is_partial: true` - the declared columns keep their types and
    everything else stays open to inference.
    """
    sources: dict[RelationKey, DeclaredSourceTable] = {}
    for key, columns in (declared or {}).items():
        *prefix, name = key.split(".")
        database, schema = ([None, None] + prefix)[-2:]
        sources[key] = DeclaredSourceTable(
            name=name,
            path=Path("schema.yml"),
            source_name="mysource",
            database=database,
            schema_name=schema,
            declaration_is_partial=key in partial,
            columns=[
                DeclaredColumn(
                    name=column,
                    written_type=written,
                    resolved_type_name=resolve_type_name(written),
                )
                for column, written in columns.items()
            ],
        )
    return DeclaredSchemas(mode="standalone", sources=sources)
