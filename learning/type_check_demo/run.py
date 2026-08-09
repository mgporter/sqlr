"""Reproduces every output in learning/type_check_walkthrough.html."""
import json

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope
from sqlglot.schema import MappingSchema

import pipeline as P

SQL = open("example.sql").read()
DECLARED = {"orders": {"order_id": "BIGINT", "order_ts": "TIMESTAMP", "amount_cents": "INT"}}
TREE = sqlglot.parse_one(SQL, read="duckdb")


def table_columns(tree=TREE):
    """Columns read straight off a real table, per table. Works before qualify."""
    out: dict[str, set[str]] = {}
    for scope in traverse_scope(tree):
        tables = {n: s for n, s in scope.sources.items() if isinstance(s, exp.Table)}
        if not tables:
            continue
        for col in scope.columns:
            # unqualified column in a single-table scope belongs to that table
            name = col.table or (next(iter(tables)) if len(tables) == 1 else None)
            if name in tables:
                out.setdefault(tables[name].name, set()).add(col.name)
    return out


def gapfilled(declared, tree=TREE):
    """Every column the SQL reads off a table: declared type if known, UNKNOWN otherwise."""
    return {
        table: {c: declared.get(table, {}).get(c, "UNKNOWN") for c in sorted(cols)}
        for table, cols in table_columns(tree).items()
    }


def run(schema, label):
    ms = MappingSchema(schema, dialect="duckdb")
    q = qualify(sqlglot.parse_one(SQL, read="duckdb"), schema=ms, dialect="duckdb")
    a = annotate_types(q, schema=ms, dialect="duckdb")
    print(f"########## {label}")
    for scope in traverse_scope(a):
        parent = scope.expression.parent
        name = parent.alias if isinstance(parent, exp.CTE) else "<final projection>"
        print(f"-- {name}")
        for s in scope.expression.selects:
            inner = s.this if isinstance(s, exp.Alias) else s
            print(f"   {s.alias_or_name:16} {s.type.sql('duckdb'):16} <- {type(inner).__name__}")
    return a


a0 = run(gapfilled(DECLARED), "PASS 0  stock sqlglot, declared schema gap-filled with UNKNOWN")

P.install_catalog()
a1 = run(gapfilled(DECLARED), "PASS 1  catalog installed, source types still partial")

ev = P.backward_evidence(a1)
print("\nbackward evidence:", {f"{t}.{c}": f for (t, c), f in ev.items()})
wide = P.widen_schema(gapfilled(DECLARED), ev)
print("widened schema:", json.dumps(wide, indent=2))

a2 = run(wide, "PASS 2  schema widened by backward evidence")

# widening is monotone: a second round of evidence changes nothing
assert P.widen_schema(wide, P.backward_evidence(a2)) == wide, "evidence did not converge"

print("\n########## FINDINGS")
for f in P.check(a2):
    print(f"  {f.code}  chars {f.start}-{f.end}  {f.message}")
    if f.start is not None:
        print(f"           | {SQL[f.start:f.end + 1]}")
