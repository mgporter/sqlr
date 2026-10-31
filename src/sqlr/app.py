from typing import Iterable
import re

import typer
import sqlglot
from sqlglot import Expr
import sqlglot.expressions as exp
from sqlglot.optimizer.scope import Scope, traverse_scope, build_scope
from pydantic import BaseModel

import duckdb
import pandas as pd

def run_query(query: str) -> pd.DataFrame:
    result = duckdb.sql(query)

    if result is None:
        raise Exception("Query execution encountered an error.")

    return result.to_df()


app = typer.Typer(help="sqlr command line interface.", no_args_is_help=True)

SQLR_TAG = re.compile(r"^\s*@sqlr\s*\(\s*([^()\s]+)\s*\)\s*")

class SourceFixture(BaseModel):
    alias: str
    rows: Iterable[tuple[object, ...]]
    columns: list[str]

def source_fixture_to_expr(fixture: SourceFixture) -> exp.Values:
    return exp.values(fixture.rows, alias=fixture.alias, columns=fixture.columns)

fixtures: dict[str, Expr] = {
    "numbers": source_fixture_to_expr(
        SourceFixture(alias="numbers", rows=[(1,'one'), (5,'five'), (5,'five??'), (17,'seventeen')], columns=["num", "name"])
    ),
}


def get_scope_name(expression: Expr) -> str | None:
    # expression.comments only contains the first comment before a select statement
    for comment in expression.comments or []:
        match = SQLR_TAG.fullmatch(comment)
        if match:
            return match.group(1)
    return None

def replace_sources(scope: Scope, fixtures: dict[str, Expr]) -> Expr | None:
    """Return copy of scope.expression with each selected source swapped for the corresponding fixture."""
    expression = scope.expression.copy()
    scope_copy = build_scope(expression)  # nodes must belong to the copy

    if scope_copy is None:
        print("Failed to build scope copy.")
        return None

    for alias, (node, _source) in list(scope_copy.selected_sources.items()):
        if alias not in fixtures:
            print(f"Fixture for alias '{alias}' not found.")
            return None
        scope_copy.replace(node, fixtures[alias])

    return expression



@app.command()
def mycommand() -> None:
    """Test."""

    path = "examples/exp1.sql"
    with open(path, "r") as f:
        sql = f.read()

    sql_statements = sqlglot.parse(sql, dialect="snowflake")

    for statement in sql_statements:
        if statement is None:
            continue

        for scope in traverse_scope(statement):

            scope_name = get_scope_name(scope.expression)

            if scope_name is None:
                continue

            print(f' === {scope_name} ===')
            print(scope.expression.sql())

            expression_with_numbers = replace_sources(scope, fixtures)

            if expression_with_numbers is None:
                continue

            replaced_sql = expression_with_numbers.sql()

            print(replaced_sql)

            run_results = run_query(replaced_sql)
            print(run_results)

            print("\n\n")


# @app.command()
# def q(query: str) -> None:
#     # """Run a query and print the result."""
#     # df = run_query(query)
#     # print(df)

#     my_expr = exp.values([(1,2), (5,10), (17,34)], alias="t", columns=["num", "num_plus_one"])

#     print(my_expr)

    # typer.echo("\n\ndone")
    # for testing_scope in testing_scopes:
    #     typer.echo(f"Testing scope: {testing_scope}")


# @app.command()
# def greet(
#     name: str = typer.Argument(..., help="Name to greet."),
#     shout: bool = typer.Option(False, "--shout", "-s", help="Uppercase the greeting."),
# ) -> None:
#     """Print a greeting."""
#     message = f"Hello, {name}!"
#     typer.echo(message.upper() if shout else message)


# @app.command()
# def add(
#     a: int = typer.Argument(..., help="First number."),
#     b: int = typer.Argument(..., help="Second number."),
# ) -> None:
#     """Add two integers."""
#     typer.echo(a + b)


if __name__ == "__main__":
    app()
