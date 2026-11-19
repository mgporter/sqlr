
from typing import Iterable, Sequence
import re

import typer
from rich.console import Console
# from rich.table import Table as RichTable
# from rich.text import Text
import sqlglot
from sqlglot import Expr
import sqlglot.expressions as exp
from sqlglot.optimizer.scope import Scope, traverse_scope, build_scope
from pydantic import BaseModel, ConfigDict, Field

import duckdb
import pandas as pd
import pyarrow as pa
from enum import Enum

from sqlr.errors import ComparisonError
from sqlr.schema_compare import Policy
from sqlr.table_compare import TableMismatchError, assert_equals

def run_query(query: str) -> pa.Table:
    result = duckdb.sql(query)

    if result is None:
        raise Exception("Query execution encountered an error.")

    return result.to_arrow_table()


app = typer.Typer(help="sqlr command line interface.", no_args_is_help=True)
console = Console()

SQLR_TAG_REGEXP = re.compile(r"^\s*@sqlr\s*\(\s*([^()\s]+)\s*\)\s*")

class SqlType(Enum):
    # Defaults match what DuckDB hands back through to_arrow_table().
    INT = pa.int32()      # DuckDB INTEGER/INT; also what int literals and n + 1 produce
    BIGINT = pa.int64()   # DuckDB BIGINT; also count(*)
    FLOAT = pa.float64()  # DuckDB DOUBLE (Snowflake FLOAT is 64-bit; DuckDB FLOAT is 32-bit)
    STRING = pa.string()
    DATE = pa.date32()
    BOOLEAN = pa.bool_()

    DECIMAL1 = pa.decimal128(18,1)
    DECIMAL2 = pa.decimal128(18,2)
    DECIMAL3 = pa.decimal128(18,3)
    DECIMAL4 = pa.decimal128(18,4)
    DECIMAL5 = pa.decimal128(18,5)

    TIMESTAMP_NTZ0 = pa.timestamp("s")
    TIMESTAMP_NTZ3 = pa.timestamp("ms")
    TIMESTAMP_NTZ6 = pa.timestamp("us")
    TIMESTAMP_NTZ9 = pa.timestamp("ns")

    TIME = pa.time64("us")  # DuckDB has a single TIME type (microseconds)

    # TIMESTAMP_LTZ0 = pa.timestamp("s")
    # TIMESTAMP_LTZ3 = pa.timestamp("ms")
    # TIMESTAMP_LTZ6 = pa.timestamp("us")
    # TIMESTAMP_LTZ9 = pa.timestamp("ns")



class FixtureColumn(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    type: pa.DataType

    def __init__(self, name: str, type: SqlType | pa.DataType) -> None:
        super().__init__(
            name=name,
            type=type.value if isinstance(type, SqlType) else type,
        )

class FixtureData(BaseModel):

    model_config = ConfigDict(frozen=True)

    data: Sequence[tuple[object, ...]]
    columns: list[FixtureColumn]

    _paTable: pa.Table | None = None

    def to_arrow(self) -> pa.Table:

        if self._paTable is None:
            self._paTable = pa.Table.from_pandas(
                pd.DataFrame(
                    columns=[col.name for col in self.columns],
                    data=self.data
                ),
                schema=pa.schema([(col.name, col.type) for col in self.columns])
            )
        return self._paTable

class SourceFixture(BaseModel):

    model_config = ConfigDict(frozen=True)

    alias: str
    data: FixtureData


def source_fixture_to_values_expr(fixture: SourceFixture) -> exp.Values:
    return exp.values(fixture.data.data, alias=fixture.alias, columns=[col.name for col in fixture.data.columns])


def get_scope_name(expression: Expr) -> str | None:
    # expression.comments only contains the first comment before a select statement
    for comment in expression.comments or []:
        match = SQLR_TAG_REGEXP.fullmatch(comment)
        if match:
            return match.group(1)
    return None

def replace_sources(scope: Scope, fixtures: Iterable[SourceFixture], tag: str = '') -> Scope:
    """Return copy of scope with each selected source swapped for the corresponding fixture."""
    expression = scope.expression.copy()
    scope_copy = build_scope(expression)  # nodes must belong to the copy

    if scope_copy is None:
        # print("Failed to build scope copy.")
        raise Exception("Failed to build scope copy.")

    for alias, (node, _source) in list(scope_copy.selected_sources.items()):
        fixture_dict = {fixture.alias: source_fixture_to_values_expr(fixture) for fixture in fixtures}
        if alias not in fixture_dict:
            raise Exception(f"Fixture for alias '{alias}' not found for tag '{tag}'.")
        scope_copy.replace(node, fixture_dict[alias])

    return scope_copy


# def schema_diff_table(actual: pa.Schema, expected: pa.Schema) -> RichTable | None:
#     """Table of expected columns vs actual types, mismatches in red. None if schemas match."""
#     table = RichTable(box=None, padding=(0, 2))
#     table.add_column()
#     table.add_column("expected")
#     table.add_column("actual")

#     actual_types = dict(zip(actual.names, actual.types))
#     expected_names = set(expected.names)
#     differs = False

#     # Text (not str) cells so rich doesn't parse markup, e.g. the "[s]" in "timestamp[s]".
#     for name, expected_type in zip(expected.names, expected.types):
#         actual_type = actual_types.get(name)
#         if actual_type is None:
#             actual_cell = Text("<missing column>", style="red")
#             differs = True
#         elif actual_type != expected_type:
#             actual_cell = Text(str(actual_type), style="red")
#             differs = True
#         else:
#             actual_cell = Text(str(actual_type))
#         table.add_row(Text(name), Text(str(expected_type)), actual_cell)

#     for name in actual.names:
#         if name not in expected_names:
#             table.add_row("", "", Text(f"<extra column '{name}'>", style="red"))
#             differs = True

#     if not differs:
#         if actual.names == expected.names:
#             return None
#         table.caption = f"Names and types match but column order differs. Actual order: {actual.names}"

#     return table


class TestableSql(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    namespace: str = ""
    tag: str
    original_scope: Scope = Field(alias="scope")
    fixtured_scope: Scope | None = None

    def get_source_aliases(self) -> list[str]:
        return list(self.original_scope.selected_sources.keys())

    def run_with_fixtures(self, fixtures: Iterable[SourceFixture]) -> pa.Table:
        fixtured_scope = replace_sources(self.original_scope, fixtures, self.tag)
        return run_query(fixtured_scope.expression.sql())


def by_tag(tag: str) -> TestableSql | None:
    path = "examples/exp1.sql"
    with open(path, "r") as f:
        sql = f.read()

    sql_statements = sqlglot.parse(sql, dialect="snowflake")

    for statement in sql_statements:
        if statement is None:
            continue

        for scope in traverse_scope(statement):

            scope_name = get_scope_name(scope.expression)

            if scope_name != tag:
                continue

            return TestableSql(
                scope=scope,
                namespace='',
                tag=tag
            )

    return None

@app.command()
def te() -> None:
    # testable_sql = by_tag("inner_select")
    testable_sql = by_tag("plus_two")

    if testable_sql is None:
        print("Testable SQL not found.")
        return

    # print(testable_sql.get_source_aliases())

    numbers_fixture = SourceFixture(
        alias="numbers",
        data=FixtureData(
            data=[(1,'one'), (5,'five'), (5,'five??'), (17,'seventeen')],
            columns=[
                FixtureColumn(name="num", type=SqlType.INT), 
                FixtureColumn(name="name", type=SqlType.STRING)
            ]
        ),
    )

    # inner_select_expected = FixtureData(
    #     data=[(1,), (5,), (17,)],
    #     columns=[FixtureColumn(name="num", type=SqlType.INT)]
    # )

    plus_two_expected = FixtureData(
        data=[(17,19,'seventeen'), (1,3,'ont'), (5,7,'five'), (5,7,'five??'),],
        columns=[
            FixtureColumn(name="num", type=SqlType.INT),
            FixtureColumn(name="num_plus_two", type=SqlType.INT),
            FixtureColumn(name="name", type=SqlType.STRING)
        ]
    )

    result = testable_sql.run_with_fixtures([numbers_fixture])

    print(result)

    policy = Policy()

    try:
        assert_equals(result, plus_two_expected, sort_by=["num"], policy=policy)
    except TableMismatchError as e:
        typer.echo(f"FAILED\n{e}")
        raise typer.Exit(code=1)
    except ComparisonError as e:
        typer.echo(f"ERROR\n{e}", err=True)
        raise typer.Exit(code=2)

    print("Tables are equal")



if __name__ == "__main__":
    app()
