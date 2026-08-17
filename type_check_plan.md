# Type checking pipeline — implementation plan

Status: **design agreed, not implemented.** Open questions are marked `❓Qn` and sit in the
step they affect. Answer them before writing the code in that step.

Everything factual here was verified by running it against **sqlglot 30.14.0**. Runnable proof
lives in `learning/type_check_demo/` (`pipeline.py`, `run.py`, `example.sql`); the narrative
version is `learning/type_check_walkthrough.html`. When this document says "measured" or
"verified", it means a script was run and the output pasted, not that it seemed likely.

---

## What we are building and why

Today sqlr infers types at the two ends of a model — source columns coming in, projection
going out. **The middle is dark.** A CTE column like `upper(status_code)` never gets a type,
so nothing downstream of it can be checked, and a bug in an intermediate column is invisible.

The goal is a checker that types *every expression in the file*, and reports a real error when
a function is called wrongly, using the file's own SQL plus whatever the user declared.

The load-bearing insight is that **no parser needs writing**. sqlglot already does parsing,
scope resolution, star expansion, cross-CTE type propagation and dialect-specific coercion.
What it lacks is a function signature catalog, and that is a data problem. The new code is
~200 lines plus a generated data file.

### The one thing to internalise before reading further

`annotate_types` **infers; it never rejects.** There is no error path anywhere in it. Verified:
with an empty catalog, `round(status, 2)` where `status` is `VARCHAR` was annotated
`DOUBLE` and reported nothing.

Type *checking* is a separate pass we write, over the tree sqlglot annotated. sqlglot supplies
the types; we supply the judgement. Steps 1–6 are inference. Only step 7 has an opinion.

---

## Decisions already settled

| Decision | Reason |
|---|---|
| Declared types are the source of truth and always override inferred | User's call. Declarations may still be *wrong*, which is what step 7 catches. |
| **Steps 5 and 6 are skipped when the schema has no `UNKNOWN` slots** — no config option | Proven no-op, see step 5. A config would offer a choice between a result and the identical result, 22% slower. |
| `traverse_scope` result is cached per run and threaded through | Measured: the scope graph is built **6 times** per file today. |
| Cache is a per-run object, never a module-level `id()` memo | `id()` is reused after GC; a stale hit would be a silent wrong answer. |
| Full rewrite of the typing path, not incremental edits | Unreleased app. Cleaner to delete and rewrite. |
| One labelled entrypoint, steps 1–7 in comments | Flow clarity is the maintenance property that matters most here. |

---

## Architecture questions — answer these first

These are not specific to one step. They change the shape of everything.

### ❓Q1 — Does this replace `sql_analysis`, or only the typing layer?

**The data flow inverts.** Today declarations enter at the *end*:

```python
# src/sqlr/cli.py:_analyze  — current
result = analyze_file(model.path, dialect=...)      # no schema at all
schema = resolve_schema(result)                      # no declarations
validation = validate_schema(schema, declared)       # declarations FIRST appear here
```

The new pipeline needs the declared schema at **step 2**, before qualify. So `declared/` must
load *before* analysis, not after. That reordering is unavoidable and touches `cli.py`.

Two things I checked before recommending a scope, because both would have changed the answer:

**`Resolver` does not die.** I assumed `qualify` with a schema would make backward star
resolution obsolete. It does not — with an undeclared table, the star survives silently:

```
table IS in schema      -> SELECT "orders"."a" AS "a", "orders"."b" AS "b" FROM "orders" AS "orders"
table NOT in schema     -> SELECT * FROM "orders" AS "orders"          # no error, no expansion
star over join, unknown -> SELECT * FROM "a" AS "a" JOIN "b" AS "b" ON "a"."id" = "b"."id"
```

Design principle 2 is *zero mandatory metadata*, so the undeclared case is a first-class path.
`Resolver` (461 lines) and `star_over_join_behavior` both survive.

**`facts.py` is only ~30% typing.** Predicates, joins, cardinality and nullability feed
constraint extraction and fixture generation. None of that overlaps with this work.

> **Recommendation:** rewrite the typing path only. Keep `relations.py`, `resolver.py`,
> `assemble.py`, and the predicate/join/cardinality half of `facts.py`.

### ❓Q2 — What happens to `ResolvedType` and its evidence lists?

`annotate_types` returns a bare `DataType` with **zero provenance**. Design principle 5 says
everything explains itself; straight adoption loses that.

> **Recommendation:** keep `ResolvedType` + evidence. A sqlglot annotation becomes one more
> evidence kind, weighted just below `declared` and above everything else. Then
> `bogus: UNKNOWN` explains itself as *"no `round` overload accepts TEXT, at line 15"*
> instead of going silently untyped.

### ❓Q3 — Does `typemap`'s lattice stay the public type?

sqlglot speaks `VARCHAR` / `BIGINT` / `DOUBLE` / `DECIMAL(10,2)`. `typemap` speaks the families
`string` / `numeric` / `number`. Fixture generation needs DECIMAL-vs-INT, which families lose.

> **Recommendation:** sqlglot `DataType` becomes the internal currency. `typemap` converts at
> the two boundaries (declared yml in, reports out) and keeps `compatible` / `widen` for
> declared-vs-inferred comparison. Families stop being a *type* and become a *query over* one.

### ❓Q4 — Multi-statement files

`analyze_sql` today parses N statements, merges, and warns *"file contains multiple statements;
projection reflects the last statement"*. `qualify` and `annotate_types` are per-statement.

> **Recommendation:** run steps 1–7 per statement, merge findings, keep the warning.
> Alternative: make one-statement-per-file a hard rule and error. Say which.

---

## Cost budget

Measured on 16 chained CTEs, N=1094 nodes post-qualify, 64 `Func` nodes, 18 scopes.
Unit: **1.0 = one operation per expression node.**

| Step | Passes | % time | Scope builds |
|---|---|---|---|
| 1 parse | ~1.0 | 17% | 0 |
| 2 gap-fill | 0.2 | 2% | 1 |
| 3 qualify | ~5 | **39%** | **3** |
| 4 annotate | ~2.0 | 17% | 1 |
| 5 backward | 0.5 | 4% | 1 |
| 6 annotate | ~2.0 | 18% | 1 |
| 7 check | 0.3 | 3% | 0 |

**All linear in N.** Verified by doubling: 1367→2711 nodes gives 2.02–2.35× on every step.
Nothing superlinear anywhere.

Absolute: the 64-CTE file runs the whole pipeline in **104 ms**. A 100-model project is roughly
10 s single-threaded.

Two things this table says loudly:

- **Our code is cheap.** Steps 5 and 7 are 7% combined. The catalog touches only `Func` nodes,
  ~6% of the tree. Do not optimise the checker; there is nothing there.
- **Getting the tree ready costs more than using it.** Parse + qualify are 56% and are entirely
  sqlglot's.

---

## Step 0 — Install the catalog

**Purpose.** Teach sqlglot the return types it does not know. Once per process, not per file.

**Why it exists.** `annotate_types` dispatches on a plain dict keyed by expression class. That
dict is the entire extension point ([`annotate_types.py:528`](.venv/lib/python3.12/site-packages/sqlglot/optimizer/annotate_types.py)):

```python
spec = self.expression_metadata.get(expr.__class__)
if spec and (annotator := spec.get("annotator")):
    annotator(self, expr)
elif spec and (returns := spec.get("returns")):
    self._set_type(expr, returns)
else:
    self._set_type(expr, exp.DType.UNKNOWN)     # <- everything we don't cover lands here
```

DuckDB ships 307 entries. The gaps we hit in one small example: `TimestampTrunc` (a real
coverage hole — DuckDB parses `date_trunc` to `TimestampTrunc`, but only `DateTrunc` is
registered), `Split`, and every `Anonymous` function.

```python
@dataclass(frozen=True)
class Sig:
    params: tuple[str, ...]   # families: NUMERIC STRING TEMPORAL BOOLEAN ARRAY ANY
    returns: str              # a type string, or "@element" / "@argN"


def install_catalog(dialect_name: str) -> None:
    """Register one annotator per catalog entry."""
    d = Dialect.get_or_raise(dialect_name)
    for key in CATALOG[dialect_name]:
        cls = exp.FUNCTION_BY_NAME.get(key)     # sqlglot's own name -> class map
        if cls is not None:
            d.EXPRESSION_METADATA[cls] = {"annotator": _annotate}
    # everything with no sqlglot class arrives as Anonymous; route it by written name
    d.EXPRESSION_METADATA[exp.Anonymous] = {"annotator": _annotate}
```

`exp.FUNCTION_BY_NAME` is sqlglot's own `"ROUND" → exp.Round` map, so the catalog never mentions
a class name.

### Four design rules for the catalog

**Key on the SQL name, never the class name.** Class names change between sqlglot versions and
do not exist for `Anonymous` at all. Today's `FUNCTION_TYPE_MAP` is keyed on class name — that
is the coupling being removed.

```python
def catalog_key(node: exp.Expr) -> str | None:
    if isinstance(node, exp.Anonymous):
        return node.name.upper()
    if isinstance(node, exp.Func):
        return node.sql_name()
    return None
```

**Families, not concrete types.** `round` accepts any numeric; enumerating nine types per
parameter is unwritable. Families map onto sqlglot's own constants:

```python
FAMILIES = {
    "NUMERIC":  set(exp.DataType.NUMERIC_TYPES),
    "STRING":   set(exp.DataType.TEXT_TYPES),
    "TEMPORAL": set(exp.DataType.TEMPORAL_TYPES),
    "BOOLEAN":  {exp.DType.BOOLEAN},
    "ARRAY":    {exp.DType.ARRAY},
}
```

**Polymorphic returns as markers.** `"@arg0"` = "whatever the first argument was", so
`round(DECIMAL)` stays DECIMAL and `round(DOUBLE)` stays DOUBLE with one entry. `"@element"`
unwraps `ARRAY<T>` to `T`.

**Three-valued family test.** This is the single most important function in the design.

```python
def in_family(dtype: exp.DataType | None, family: str) -> bool | None:
    """True / False / None. None means unknown, so no judgement is made."""
    if family == "ANY":
        return True
    if dtype is None or dtype.is_type(exp.DType.UNKNOWN):
        return None                       # <- NOT False
    return dtype.this in FAMILIES[family]
```

`False` means "definitely wrong, report it". `None` means "cannot say, stay quiet". Conflating
them is how a checker gets a reputation for lying. Every false positive I hit while building the
demo traced back to this distinction.

### ⚠️ The gotcha that cost a false positive

**sqlglot normalises argument order into its node shape, which is not the written order.**

```python
# written:  date_trunc('day', order_ts)
# parsed:   TimestampTrunc(this=order_ts, unit='day')     <- unit second, though written first
```

A signature written as `(STRING, TEMPORAL)` — matching the SQL you can *see* — produces a
confident, wrong error on valid SQL. This happened on the first run of the demo. **Catalog
signatures must be written against sqlglot's node order**, and `func_args` must extract in that
order:

```python
def func_args(node: exp.Func) -> list[exp.Expr]:
    """Positional arguments of a call, in sqlglot's node arg order."""
    if isinstance(node, exp.Anonymous):
        return list(node.args.get("expressions") or [])
    out: list[exp.Expr] = []
    for name in node.arg_types:                 # declaration order, stable
        val = node.args.get(name)
        if val is None or isinstance(val, (str, bool)):
            continue                            # flags like `big_int`, not arguments
        out.extend(val if isinstance(val, list) else [val])
    return out
```

Write a test that asserts each catalog entry's arity matches its class's `arg_types`. This is a
whole class of silent bug.

### ❓Q5 — Catalog source, storage, dialects

Three sub-decisions:

- **Source.** `select * from duckdb_functions()` gives ~1,200 overloads with `parameter_types`
  and `return_type` — exact, free, machine-readable. Postgres has the same via `pg_proc`.
  Snowflake and BigQuery have **no introspection at all**; those are hand-written or scraped.
  That asymmetry is the real cost of this entire design.
- **Storage.** Generated file checked in, or built at runtime? Checked-in needs a regeneration
  story; runtime needs DuckDB importable at analysis time.
- **Granularity.** `duckdb_functions()` gives concrete types. Store those and derive families,
  or collapse at generation time?

> **Recommendation:** generated from DuckDB, checked in as data, concrete types preserved,
> families derived at load. DuckDB only at launch; add dialects when a real project needs one.

### ❓Q6 — Global dialect mutation

`install_catalog` writes into a process-global `Dialect` class. Fine for a demo; in sqlr two
runs with different dialects could interfere.

> **Recommendation:** pass `expression_metadata=` directly to `annotate_types()` in steps 4 and
> 6 instead. Then it is a per-call argument and there is no global state at all.

### Keep the catalog a gap-filler

Not in the demo catalog: `datediff`, `current_timestamp`, `/`, `>`. sqlglot already types all of
those correctly. Adding them buys nothing and inherits 300 entries of maintenance.

**Logging.** INFO once at startup: `loaded 1187 signatures for dialect duckdb, 412 mapped to
sqlglot classes, 775 anonymous-only`. DEBUG: the names that failed to map.

---

## Step 1 — Parse

**Purpose.** SQL text → AST. Nothing type-related happens.

```python
try:
    statements = sqlglot.parse(sql, read=dialect)
except ParseError as e:
    return TypeCheckResult(source=source, errors=[str(e)])
```

Positions come free and are absolute across the whole file, so **one `Positions` index per
document stays valid across every statement** — as today. Keep that.

A fresh parse carries no type information anywhere. `node.type` is `None` until step 4.

**Logging.** INFO: `parsed <path>: N statements, M expression nodes`. DEBUG: per-statement node
counts by class, top 10 — not a full tree dump.

---

## Step 2 — Gap-fill the declared schema

**Purpose.** Make the schema's *column set* complete so step 3 does not raise. Declared types
where the user wrote them, `UNKNOWN` everywhere else.

**Why it exists.** This is not optional and it is not obvious. `qualify` validates that every
column resolves; a schema covering 3 of 5 columns fails before `annotate_types` is ever reached:

```
OptimizeError: Column 'status_code' could not be resolved. Line: 6, Col: 25
```

**I tried every built-in escape first. None work:**

```
{}                                                          OptimizeError: Column 'status_code' ...
{'allow_partial_qualification': True}                       OptimizeError: Column 'status_code' ...
{'infer_schema': True}                                      OptimizeError: Column 'status_code' ...
{'allow_partial_qualification': True, 'infer_schema': True} OptimizeError: Column 'status_code' ...
```

The other tempting escape, `validate_qualify_columns=False`, *does* let qualification proceed —
but the undeclared columns come back **unqualified**: `UPPER("status_code")`, no table prefix.
It silently degrades exactly the columns you most need to reason about. Do not use it.

So: gap-fill. `UNKNOWN` is a first-class `DType` in sqlglot, not an invented sentinel, and every
downstream pass already handles it.

```python
def table_columns(tree: exp.Expr, scopes: list[Scope]) -> dict[str, set[str]]:
    """Columns read straight off a real table, per table. Runs before qualify."""
    out: dict[str, set[str]] = {}
    for scope in scopes:
        tables = {n: s for n, s in scope.sources.items() if isinstance(s, exp.Table)}
        if not tables:
            continue                       # scope reads CTEs only; nothing to declare
        for col in scope.columns:
            # an unqualified column in a single-table scope belongs to that table
            name = col.table or (next(iter(tables)) if len(tables) == 1 else None)
            if name in tables:
                out.setdefault(tables[name].name, set()).add(col.name)
    return out


def gapfilled(declared: dict, tree: exp.Expr, scopes: list[Scope]) -> dict:
    return {
        t: {c: declared.get(t, {}).get(c, "UNKNOWN") for c in sorted(cols)}
        for t, cols in table_columns(tree, scopes).items()
    }
```

The `isinstance(source, exp.Table)` filter is what stops CTE-internal names like `amount` and
`is_large` from being invented as columns of `orders`. Without it the schema fills with garbage
that happens to work but is wrong the moment anything reads it.

### The typo trap

Gap-fill will happily invent `orders.custmer_id` if the user typos it. **The check for
unresolvable columns must run here, before gap-filling** — compare the tree's column set against
the declared one and report the difference. That is a different diagnostic from anything else in
this document and it has no other natural home.

**Skip predicate for later — compute it here:**

```python
needs_inference = any(t == "UNKNOWN" for cols in schema.values() for t in cols.values())
```

O(columns), not O(N). Free.

**Logging.** INFO: `orders: 3 of 5 columns declared, 2 gap-filled UNKNOWN`. DEBUG: the column
names in each bucket.

---

## Step 3 — Qualify

**Purpose.** Every column names its relation; `select *` becomes a real projection list.
Prerequisite for all type inference.

```python
qualified = qualify(statement, schema=gapfilled_schema, dialect=dialect)
```

**What it produces** — `*` gone, six real projections, every column carrying its relation:

```sql
), "flagged" AS (
  SELECT
    "enriched"."order_id" AS "order_id",
    ...
    ROUND("enriched"."status", 2) AS "bogus",
    ZEROIFNULL("enriched"."amount") AS "safe_amount",
    LIST_EXTRACT("enriched"."first_promo") AS "broken"
  FROM "enriched" AS "enriched"
)
```

The most expensive step in the pipeline: **39% of runtime, 3 internal scope-graph rebuilds**
(`normalize_identifiers`, `qualify_tables`, `isolate_table_selects`, `qualify_columns`,
`quote_identifiers`). Those three rebuilds are internal to sqlglot and cannot be cached away.

### ⚠️ qualify lowercases identifiers

```
input:   select Order_Id from MyDb.MySchema.Orders
output:  SELECT "orders"."order_id" AS "order_id" FROM "mydb"."myschema"."orders" AS "orders"
```

DuckDB's `NORMALIZATION_STRATEGY` is `CASE_INSENSITIVE`, so qualify folds case. `table.name`,
`table.db` and `table.catalog` all come back lowercased.

This matters because `declared/` matches a relation by **the name the user wrote** — the design
doc is explicit that "the parts it writes down, joined, are what the SQL has to write". Case
folding either fixes that (case-insensitive matching becomes free) or breaks it (a declaration
written `MySchema` no longer matches). **Decide deliberately and test both spellings.**

### ❓Q7 — Which tree does `build_graph` receive?

Today `sql_analysis` calls `qualify_tables` only — a much lighter pass. The new pipeline produces
a fully-qualified tree with stars already expanded (when the schema knows them). That is *better*
input for `build_graph`, but it is different input:

- `relations.py` may assume unexpanded stars in places; star handling is a large part of it.
- Column names arrive lowercased and quoted.
- `RelationRef` names may change shape.

> **Recommendation:** feed `build_graph` the fully-qualified tree and fix the fallout. Running
> two differently-qualified trees side by side would double the parse cost and create two
> sources of truth about the same file. But this needs a test pass over `test_sql_analysis.py`
> before committing — flag it as the highest-risk integration point in the whole plan.

**Logging.** INFO: `qualified <path>: N scopes, M columns, K stars expanded`. DEBUG: the
qualified SQL, pretty-printed, per statement — this one *is* worth a full dump, it is the single
most useful artefact when debugging a wrong type.

---

## Step 3b — Non-type analysis (unchanged work, new input)

This is where the existing lineage and fact extraction runs. **None of it is being rewritten** —
it feeds constraint extraction, relationship inference and fixture generation, which have nothing
to do with function signatures.

```python
# unchanged modules, now fed the fully-qualified tree from step 3
graph    = build_graph(qualified)                       # relations.py
resolver = Resolver(graph,
                    star_over_join_behavior=cfg.general.star_over_join_behavior,
                    dialect=dialect,
                    positions=positions)                # resolver.py — survives, see Q1
facts    = extract_facts(graph, resolver, positions)    # facts.py
analysis = assemble(graph, resolver, facts, positions, source)   # assemble.py
```

`facts` carries `predicates`, `joins`, `nullability`, `cardinality`. Those go on to constraint
extraction and the generation planner exactly as today:

```python
# downstream consumers, unchanged
analysis.predicates    # -> constraint extraction  (where amount > 20, straddle the literal)
analysis.joins         # -> relationship inference (address.person_id references person.id)
analysis.cardinality   # -> generation planner     (group by / distinct / partition by)
analysis.nullability   # -> generator null rates
```

### What gets deleted from `facts.py`

Only the type-evidence half. These five constants and the two functions reading them:

| Location | Symbol |
|---|---|
| `facts.py:41` | `DATE_FUNCTION_CLASSES` |
| `facts.py:70` | `STRING_ARGUMENT_FUNCTIONS` |
| `facts.py:81` | `STRING_FIRST_ARGUMENT_FUNCTIONS` |
| `facts.py:83` | `NUMERIC_ARGUMENT_FUNCTIONS` |
| `facts.py:95` | `NUMERIC_FIRST_ARGUMENT_FUNCTIONS` |

Read at `facts.py:176` and `facts.py:192-198`, inside `classify_usage` /
`_function_argument_usage`. Those produce the `UsageKind` values `date_function`,
`string_function`, `numeric_function`, `function_argument` — all four become dead once the
catalog does this properly, and step 5 replaces them.

Everything else in `classify_usage` — comparisons, IN lists, LIKE, boolean context, casts —
stays. It types columns from *literals*, which no function catalog covers.

---

## Step 4 — Annotate, pass 1

**Purpose.** Type every expression in the file from what is currently known. Also establishes
**what is already known**, which step 5 depends on.

```python
annotated = annotate_types(qualified, schema=ms, dialect=dialect)
```

**Measured behaviour:** `_set_type` fires exactly **1094 times against N=1094 nodes** — one type
write per node, no revisits. The `_visited` memo keyed by `id()` works. Our catalog code touches
only the 64 `Func` nodes, ~6% of the tree.

### What you get free, and it is a lot

```
-- enriched
   order_id         BIGINT           <- Column
   order_day        TIMESTAMP        <- TimestampTrunc
   amount           DOUBLE           <- Div          (INT / float literal, TYPED_DIVISION applied)
   status           TEXT             <- Upper
   age_days         BIGINT           <- DateDiff     (sqlglot's own table, not ours)
   first_promo      TEXT             <- Anonymous    (split -> ARRAY<VARCHAR> -> @element)
-- flagged
   is_large         BOOLEAN          <- GT
   bogus            UNKNOWN          <- Round        (no overload accepts TEXT)
```

**Types flow across CTE boundaries automatically.** `amount` is `DOUBLE` in `flagged` because
that scope resolved it back into `enriched`'s already-annotated projection. That is the part
that would take months to rebuild by hand and it works out of the box.

**Composition is free.** `first_promo` needed *both* catalog entries: `split` produced
`ARRAY<VARCHAR>`, and only then could `list_extract`'s `@element` unwrap it. Annotation is
bottom-up, so nested calls just work.

**Sound inference and error detection are the same mechanism.** `bogus` flipped from a confident
wrong `DOUBLE` to `UNKNOWN` purely because `pick_overload` found no signature that `TEXT`
satisfies. Step 7 then reports it.

```python
def pick_overload(node: exp.Func) -> tuple[Sig | None, list[exp.Expr]]:
    """First overload no argument definitively contradicts."""
    args = func_args(node)
    for sig in CATALOG.get(catalog_key(node) or "", []):
        if len(sig.params) != len(args):
            continue
        if any(in_family(a.type, f) is False for a, f in zip(args, sig.params)):
            continue                      # definitely the wrong overload
        return sig, args
    return None, args
```

### ⚠️ UNKNOWN is absorbing

`_annotate_by_args` returns immediately with UNKNOWN the moment **any** argument is UNKNOWN. No
partial recovery. One uncatalogued function low in a file blanks out everything above it — in
the demo, one missing `TimestampTrunc` entry made `order_day` UNKNOWN in all three scopes.

**Consequence for diagnostics:** five UNKNOWNs in the demo had four distinct causes.

| Column | Cause | Whose problem |
|---|---|---|
| `order_day` | coverage gap in sqlglot's metadata | ours — fill it |
| `first_promo` | no signature in our catalog | ours — catalog it |
| `safe_amount` | function does not exist in dialect | user — report F001 |
| `broken` | wrong arity | user — report F002 |
| downstream copies | propagation from any of the above | derived — do not report twice |

Telling these apart is the whole difference between a useful tool and a noisy one, and it gets
much harder to retrofit later.

### Add the coverage counter now

```python
typed   = sum(1 for n in walk(tree) if n.type and not n.is_type(exp.DType.UNKNOWN))
unknown = [n for n in walk(tree) if isinstance(n, exp.Func) and n.is_type(exp.DType.UNKNOWN)
           and all(a.type and not a.type.is_type(exp.DType.UNKNOWN) for a in func_args(n))]
logger.info("typed %d/%d nodes; %d UNKNOWN with fully-typed arguments (catalog gaps)",
            typed, total, len(unknown))
```

The second number is the one that matters: **a function whose arguments are all typed but whose
result is UNKNOWN is a catalog gap, never a user error.** That is the coverage debt metric.

**Logging.** INFO: the counter above. DEBUG: per-scope projection name → type table, exactly as
printed in the demo. Do not dump annotated trees; the name → type table is strictly more useful
and a fraction of the size.

---

## Step 5 — Backward evidence *(skipped when nothing is UNKNOWN)*

**Purpose.** Type the columns nobody declared, by reading the catalog in reverse: an argument
position constrains what feeds it. If a column is passed to `upper()`, it is a string.

This is what makes partially-declared schemas work, and it replaces `STRING_ARGUMENT_FUNCTIONS`
and friends from `facts.py`.

```python
def backward_evidence(tree: exp.Expr, scopes: list[Scope]) -> dict[tuple[str, str], str]:
    real = source_columns(tree, scopes)
    by_node = {id(c): k for k, c in real.items()}
    evidence: dict[tuple[str, str], str] = {}
    for node in tree.walk():
        if not isinstance(node, exp.Func):
            continue
        for sig in CATALOG.get(catalog_key(node) or "", []):
            args = func_args(node)
            if len(sig.params) != len(args):
                continue
            for arg, family in zip(args, sig.params):
                key = by_node.get(id(arg))
                # only claim something about a column we know nothing about
                if key and family != "ANY" and in_family(arg.type, family) is None:
                    evidence.setdefault(key, family)
            break
    return evidence
```

`source_columns` filters on `isinstance(source, exp.Table)`. That line separates *"this column is
stored data, so my claim is about the table"* from *"this column comes from a CTE, so its type is
already computed and I must not overwrite it"*. Without it, backward evidence fights forward
inference.

```
>>> backward_evidence(pass1_tree)
{('orders', 'status_code'): 'STRING', ('orders', 'promo_csv'): 'STRING'}
```

### Why step 4 must run first

I checked whether the backward pass could run straight off the qualified tree and skip pass 1.
It cannot — it over-claims:

```
qualify only:   {order_ts: TEMPORAL, status_code: STRING, promo_csv: STRING}
after pass 1:   {              status_code: STRING, promo_csv: STRING}
```

`order_ts` is declared `TIMESTAMP`. Before annotation `.type` is `None`, so `in_family` cannot
distinguish "undeclared" from "declared and fine" — every column looks unknown and every argument
position generates a claim. Harmless today only because `widen_schema` drops claims about
non-UNKNOWN slots. **The moment evidence is weighted, recorded with provenance, or shown to a
user, that spurious `order_ts → TEMPORAL` becomes a lie in a report.**

So step 4's real job is not producing types. It is establishing what is *already known*, so the
backward pass only speaks about columns nobody has described.

### The monotonicity invariant — protect this

```python
def widen_schema(schema: dict, evidence: dict[tuple[str, str], str]) -> dict:
    out = {t: dict(cols) for t, cols in schema.items()}
    for (table, column), family in evidence.items():
        # ONLY fill UNKNOWN slots. Never overwrite a type.
        # This is what makes the step-5/6 skip sound, and what makes widening converge
        # in one round. Breaking it breaks both. See type_check_plan.md step 5.
        if out.get(table, {}).get(column, "UNKNOWN").upper() == "UNKNOWN":
            out.setdefault(table, {})[column] = FAMILY_DEFAULT[family]
    return out
```

That comment goes in the code. It is the coupling point most likely to be broken by someone
adding a reasonable-sounding feature later (see the warning at the end of step 6).

### The skip — proven, not assumed

I ran the full pipeline with steps 5+6 on and off, against a correct complete schema *and*
against a deliberately **wrong** complete schema:

```
===== FULLY DECLARED
  steps 5+6 OFF: 3 findings
  steps 5+6 ON : evidence={} schema_changed=False, 3 findings
  IDENTICAL? types=True  findings=True

===== WRONG DECLARATION   (status_code declared INT, SQL calls upper() on it)
  steps 5+6 OFF: UPPER argument 1 expects STRING, got INT
  steps 5+6 ON : evidence={} schema_changed=False, same
  IDENTICAL? types=True  findings=True
```

Two structural reasons, not just empirical equality:

1. **Step 5 cannot fire.** Evidence is recorded only when `in_family(...) is None`. No UNKNOWN
   slots → no argument of a source column is ever UNKNOWN → evidence is empty by construction.
2. **Step 6 cannot differ.** Empty evidence → `widen_schema` returns its input → step 6
   annotates the same tree with an equal schema.

The wrong-declaration case is worth dwelling on, because it is the one that looks like it needs
the backward pass and does not. Declaration-vs-usage contradiction is a **checking** concern
(step 7, `UPPER argument 1 expects STRING, got INT`), not an inference one. Running inference
more often was never going to answer it.

Note also from that run: with `status_code` wrongly typed, `status` went UNKNOWN and
`round(status, 2)` produced **no** finding. Only the root cause reported. That is three-valued
`in_family` preventing cascade blame, working as intended.

### ❓Q8 — Does evidence go through sqlr's weighting?

The demo uses `setdefault` — first writer wins. Two functions can constrain one column
differently (`upper(x)` and `x + 1`), and picking by first-seen is exactly the arbitrary
coin-flip the design doc argues against.

> **Recommendation:** route backward evidence through the existing `TypeEvidence` weighting in
> `schema_resolution` rather than `setdefault`. Related to ❓Q2.

**Logging.** INFO: `backward evidence: 2 columns inferred from usage (orders.status_code STRING,
orders.promo_csv STRING)`. When skipped, INFO: `schema fully declared; skipping inference passes
5-6`. DEBUG: every candidate claim including the ones dropped for already-known columns — that
list is how you debug a wrong inference.

---

## Step 6 — Annotate, pass 2 *(skipped when nothing is UNKNOWN)*

**Purpose.** Re-run inference against the widened schema. Declared and inferred types become
indistinguishable to everything downstream — the same `MappingSchema`, whatever the provenance.

That interchangeability is the property worth protecting. It means the checker behaves
identically on a fully-documented project and an undocumented one, differing only in confidence.

```python
annotated = annotate_types(annotated, schema=widened_ms, dialect=dialect)
```

**No re-parse, no re-qualify.** The demo's `run.py` re-parses each pass; that was demo
convenience. Verified equivalent:

```
re-annotate same tree:  {order_day: TIMESTAMP, first_promo: TEXT, ...}
re-parse + re-qualify:  {order_day: TIMESTAMP, first_promo: TEXT, ...}
```

Each `annotate_types()` call builds a fresh `TypeAnnotator` with an empty `_visited`, and
`overwrite_types=True` is the default. Qualification depends on which columns *exist*, not their
types, and widening never changes the column set.

### ⚠️ What would break the skip

The skip rests entirely on `widen_schema` writing only into UNKNOWN slots. If someone later adds
*"the SQL says string, the yml says int, trust the SQL"* — evidence **overriding** a declared
type — then step 5 can fire on a fully-declared schema and the skip becomes unsound.

That is a plausible feature. If it is built, the skip predicate has to change with it, not get a
config bolted on top.

**Logging.** INFO: the coverage counter again, so the before/after is visible in one log.
DEBUG: only the columns whose type *changed* between pass 1 and pass 2 — a full re-dump of
identical tables is noise.

---

## Step 7 — Check

**Purpose.** The only step with an opinion. Everything before it infers.

```python
def check(tree: exp.Expr) -> list[Finding]:
    out: list[Finding] = []
    for node in tree.walk():
        if not isinstance(node, exp.Func):
            continue
        key  = catalog_key(node)
        sigs = CATALOG.get(key or "")

        # F001 - no signature at all. Only report for Anonymous: a function with a
        # sqlglot class exists somewhere, so it is our catalog that is thin, not the SQL.
        if not sigs:
            if isinstance(node, exp.Anonymous):
                s, e = name_span(node)
                out.append(Finding(UNKNOWN_FUNCTION,
                    f"unknown function {node.name!r} in dialect {dialect}", s, e))
            continue

        # F002 - arity. Checked BEFORE types and `continue`s, so a 1-arg call never
        # also reports a bogus type mismatch against a 2-arg signature.
        args = func_args(node)
        if all(len(sig.params) != len(args) for sig in sigs):
            arities = sorted({len(s.params) for s in sigs})
            s, e = name_span(node)
            out.append(Finding(FUNCTION_ARITY,
                f"{key} takes {arities} arguments, {len(args)} given", s, e))
            continue

        # F003 - right arity, no overload accepts these types.
        sig, _ = pick_overload(node)
        if sig is None:
            for cand in sigs:
                if len(cand.params) != len(args):
                    continue
                for i, (arg, family) in enumerate(zip(args, cand.params)):
                    if in_family(arg.type, family) is False:      # False, never None
                        s, e = span(arg)
                        out.append(Finding(FUNCTION_ARGUMENT_TYPE,
                            f"{key} argument {i+1} expects {family}, "
                            f"got {arg.type.sql(dialect)}", s, e))
                break
    return out
```

**Output on the demo — three bugs planted, three found, nothing else:**

```
SQLR-F003  chars 492-497  ROUND argument 1 expects NUMERIC, got TEXT      | status
SQLR-F001  chars 530-539  unknown function 'zeroifnull' in dialect duckdb | zeroifnull
SQLR-F002  chars 580-591  LIST_EXTRACT takes [2] arguments, 1 given       | list_extract
```

### Why there are zero false positives — each of these is load-bearing

- `order_day` was UNKNOWN for two whole passes and produced nothing. `in_family` returned
  `None`, and `None` is not `False`.
- `promo_csv` was undeclared and produced nothing, same reason.
- `datediff`, `current_timestamp` and the arithmetic are not in the catalog at all, and absence
  from the catalog is **never** an error for a class-backed function — only for `Anonymous`.
- `broken` got exactly one finding, not two, because the arity branch `continue`s.

**The checker never looks at SQL text, only at `node.type`.** Everything upstream — declared yml,
backward inference, sqlglot's own tables, cross-CTE propagation — has already been flattened into
one uniform annotated tree. That is what makes this a foundation other modules can build on:
relationship inference, constraint extraction and fixture generation all read `node.type` off the
same tree without knowing where any of it came from.

### Spans

`chars 492-497` is not computed by the checker. sqlglot stashes token offsets on the nodes it
positioned:

```
Round       7    11        # the name token: sql[7:12] == "round"
Column      None None      # composite nodes carry nothing
Identifier  13   18        # "status"
Literal     21   21        # "2"
```

Two helpers, because the two kinds of finding point at different things. F001/F002 blame *the
call* → `name_span`. F003 blames *one argument* → the hull.

An earlier run used the hull for F001 and underlined `zeroifnull(amount` — the closing paren is
not a positioned leaf, so the hull stops short. **That is exactly the unbalanced-hull problem
`source.py` already documents and repairs. Use `Positions`, not a local reimplementation.**

Also cheap: `check` currently calls `catalog_key` twice per node (128 calls for 64 `Func` nodes)
— once directly, once inside `pick_overload`. Pass it through.

### ❓Q9 — Diagnostic codes and which sink

New names in the existing kebab-case style, into `diagnostics/codes.py`:

```python
UNKNOWN_FUNCTION       = "unknown-function"        # F001
FUNCTION_ARITY         = "function-arity"          # F002
FUNCTION_ARGUMENT_TYPE = "function-argument-type"  # F003
```

But which path — the `diagnostics` sink or `validation`? F001/F002/F003 are pure SQL bugs with
no declaration involved. A declared-vs-inferred contradiction is a different thing: the user must
change one side or the other.

> **Recommendation:** F001–F003 → `diagnostics`. Declared-vs-inferred contradiction stays in
> `validation`, where `TYPE_MISMATCH` already lives.

**Logging.** INFO: `<path>: 3 findings (1 unknown-function, 1 function-arity,
1 function-argument-type)`. DEBUG: each finding with its span and the source snippet.

---

## The entrypoint

One function, steps numbered in comments, matching this document.

```python
def type_check_file(
    path: Path,
    declared: DeclaredSchemas,
    cfg: SqlrConfig,
) -> TypeCheckResult:
    """Type check one .sql file end to end. See type_check_plan.md for the why of each step."""
    source    = SourceDoc(path=path, text=path.read_text())
    positions = Positions(source.text)          # absolute offsets, valid across all statements
    dialect   = cfg.general.sql_dialect

    # Step 1 — parse. SQL text -> AST. No types exist yet; node.type is None everywhere.
    statements = sqlglot.parse(source.text, read=dialect)

    for statement in statements:
        scopes = scope_cache.get(statement)     # built once, reused by steps 2 and 5

        # Step 2 — gap-fill. qualify() rejects a schema that does not cover every column,
        # and none of its own flags avoid this. Undeclared columns become UNKNOWN, which is
        # a real sqlglot DType that every later pass already understands.
        schema = gapfilled(declared_for(path), statement, scopes)
        needs_inference = any(t == "UNKNOWN" for cols in schema.values() for t in cols.values())

        # Step 3 — qualify. Every column names its relation; `select *` becomes a real list.
        # Prerequisite for all type inference. 39% of runtime; nothing to optimise here.
        qualified = qualify(statement, schema=MappingSchema(schema, dialect=dialect),
                            dialect=dialect)

        # Step 3b — lineage and non-type facts. Unchanged modules, new (better) input.
        # Feeds constraint extraction, relationship inference and fixture generation.
        graph    = build_graph(qualified)
        resolver = Resolver(graph, star_over_join_behavior=cfg.general.star_over_join_behavior,
                            dialect=dialect, positions=positions)
        facts    = extract_facts(graph, resolver, positions)
        analysis = assemble(graph, resolver, facts, positions, source)

        # Step 4 — annotate, pass 1. Types every expression from what is currently known,
        # and — equally important — establishes what is ALREADY known so step 5 does not
        # over-claim about columns the user already declared.
        annotated = annotate_types(qualified, schema=ms, dialect=dialect)

        if needs_inference:
            # Step 5 — backward evidence. The catalog read in reverse: an argument position
            # constrains what feeds it. Only ever fills UNKNOWN slots, which is what makes
            # this converge in one round AND makes the skip below sound.
            evidence = backward_evidence(annotated, scopes)
            widened  = widen_schema(schema, evidence)

            # Step 6 — annotate, pass 2. Same tree, no re-parse, no re-qualify. After this,
            # declared and inferred types are indistinguishable downstream.
            annotated = annotate_types(annotated, schema=MappingSchema(widened, ...), ...)
        else:
            # Proven no-op when nothing is UNKNOWN: evidence is empty by construction, so
            # widen_schema returns its input and pass 2 cannot differ. Verified against both
            # a correct and a deliberately wrong complete schema. No config — see step 5.
            logger.info("schema fully declared; skipping inference passes 5-6")

        # Step 7 — check. The only step with an opinion. Reads node.type only, never the
        # SQL text: everything upstream has been flattened into one uniform annotated tree.
        findings = check(annotated, dialect)
```

---

## Deletion list

Delete before rewriting. Do not leave these behind "just in case".

| File | Lines | Symbol | Replaced by |
|---|---|---|---|
| `schema_resolution/__init__.py` | 78–105 | `FUNCTION_TYPE_MAP` | catalog, forward direction |
| `schema_resolution/__init__.py` | 109 | `TYPE_TRANSPARENT_FUNCTIONS` | `@arg0` / `@element` markers |
| `sql_analysis/facts.py` | 41 | `DATE_FUNCTION_CLASSES` | catalog, backward direction |
| `sql_analysis/facts.py` | 70 | `STRING_ARGUMENT_FUNCTIONS` | catalog, backward direction |
| `sql_analysis/facts.py` | 81 | `STRING_FIRST_ARGUMENT_FUNCTIONS` | catalog arity, per-parameter |
| `sql_analysis/facts.py` | 83 | `NUMERIC_ARGUMENT_FUNCTIONS` | catalog, backward direction |
| `sql_analysis/facts.py` | 95 | `NUMERIC_FIRST_ARGUMENT_FUNCTIONS` | catalog arity, per-parameter |

Their readers: `schema_resolution/__init__.py:216,229` and `facts.py:176,192-198`. The
`UsageKind` values `date_function`, `string_function`, `numeric_function` and
`function_argument` all become dead — remove them from the `Literal` in
`sql_analysis/types.py:30` too, or they will silently never be produced.

**Keep** everything in `classify_usage` that types columns from *literals* — comparisons, IN
lists, LIKE, boolean context, casts. No function catalog covers those.

---

## Tests worth writing before the code

1. **Catalog arity matches sqlglot.** For every entry with a class in `exp.FUNCTION_BY_NAME`,
   assert each `Sig`'s arity is reachable given that class's `arg_types`. Catches the argument
   -order class of bug.
2. **Coverage debt.** Parse a corpus; count `Func` nodes whose arguments are all typed but whose
   result is UNKNOWN. Assert it does not grow. This is the only thing that keeps step 4's four
   causes of UNKNOWN distinguishable as the catalog grows.
3. **Monotone widening.** `widen_schema(s, backward_evidence(t)) == s` when `s` has no UNKNOWN.
   Guards the skip.
4. **Skip equivalence.** Full pipeline with steps 5+6 forced on vs skipped, on a fully-declared
   schema — assert identical types *and* identical findings.
5. **Three-valued discipline.** An undeclared column flowing into a catalogued function produces
   zero findings. This is the false-positive regression test.
6. **Identifier case.** A declaration written `MySchema.Orders` against SQL written
   `myschema.orders`, both directions. Locks in whatever ❓Q7's answer turns out to be.

Per the existing rule in `tests/README.md`, all fixtures live with the tests. Nothing reads from
`examples/`.

---

## Open questions, collected

| # | Question | Step | Recommendation |
|---|---|---|---|
| Q1 | Replace `sql_analysis`, or typing layer only? | before 1 | typing layer only; `Resolver` survives |
| Q2 | Keep `ResolvedType` + evidence? | before 1 | keep; sqlglot type becomes one evidence kind |
| Q3 | Does `typemap`'s lattice stay the public type? | before 1 | sqlglot `DataType` internal, lattice at boundaries |
| Q4 | Multi-statement files | before 1 | per statement, merge findings, keep warning |
| Q5 | Catalog source / storage / dialects | 0 | generated from DuckDB, checked in, DuckDB only at launch |
| Q6 | Global dialect mutation | 0 | pass `expression_metadata=` per call instead |
| Q7 | Which tree does `build_graph` receive? | 3 | the fully-qualified one; **highest-risk integration point** |
| Q8 | Backward evidence through sqlr's weighting? | 5 | yes, not `setdefault` |
| Q9 | Diagnostic codes and which sink | 7 | F001–F003 → `diagnostics`; contradictions → `validation` |

---

# Revision — facts as the single mechanism

Status: **agreed, being implemented.** Everything above still describes the pipeline
correctly; this section replaces the *shape* of steps 5 and 7, and moves fact extraction.
Where the two disagree, this section wins.

## The one rule

> **A fact is a claim about a value. A claim contradicted by that value's actual type is an
> error; a claim about a value with no type at all is the inference.**

The checker does not care whether `node.type` came from a declaration, from a CTE's
computed projection, or from an earlier inference. That is what makes "declared is the
source of truth" a property of the *data* rather than a branch in the code.

### What this replaces

The plan above had two mechanisms doing one job. Step 5 read the catalog backwards to
produce evidence, and step 7 read the tree forwards to produce F003 — the same
argument-position knowledge, walked twice, reported in two vocabularies. `round(status, 2)`
with `status` declared `varchar` produced an F003 finding *and* a contradiction, at the same
span, in different words.

They are now one walk. Step 5's `backward_evidence` and F003 both disappear into
`extract_facts_from_annotated_tree`.

### Declared vs undeclared, stated as data

| Column | `node.type` after step 4 | What a fact does to it |
|---|---|---|
| declared in `sources.yml` | the declared type | contradiction → **error at the usage site** |
| produced by a CTE | computed bottom-up by sqlglot | contradiction → **error at the usage site** |
| undeclared source column | `UNKNOWN` | claims collected → **inference**, or conflict → error |

There is never an inferred-vs-declared comparison. A declaration is not a hypothesis to be
checked against the SQL; it is the type, and the SQL either agrees with it or is wrong.

## Two fact shapes

**Claim** — *this node must be family F.* `upper(x)` arg 0, `where x`, `x * 2`.

**Link** — *these two nodes share a domain.* No family named. When one end has a type and
the other is UNKNOWN, the type crosses. `a.x > b.y`, `coalesce(a, b)`, a UNION's arms.

A fact's subject is a **node**, with a column as an optional attachment:

```python
class ValueSite(NamedTuple):
    node: exp.Expr
    column: ColumnReference | None   # the case inference and generation care about
    span: SourceSpan | None
```

Keying a fact on a column instead would mean `round(upper(x))` — no column anywhere — is
reported by different code from `round(x)`, for the same defect. The column is what a fact
*lands on*, not what it is about.

## Operators are catalog entries

Comparisons, arithmetic and `coalesce` are not three hand-written tables. They are
overloaded signatures, and every rule falls out of `pick_overload`:

```
"+":  (NUMERIC, NUMERIC) -> @arg0 | (TEMPORAL, NUMERIC) -> @arg0 | (TEMPORAL, INTERVAL) -> @arg0
"*":  (NUMERIC, NUMERIC) -> @arg0
"=":  (@T, @T) -> BOOLEAN
```

`@T` is a type variable: it accepts anything, and two positions sharing one variable are a
link.

| Situation | Falls out as |
|---|---|
| a position no candidate overload leaves open | **claim**, carrying *every* family any candidate accepts there |
| every overload binds two positions to one type variable | **link** |
| any candidate accepts `ANY` at a position | **nothing** — a claim that cannot be contradicted is not a claim |

**A claim names a set of families, not one.** DuckDB's `*` takes `(NUMERIC, NUMERIC)` and
`(INTERVAL, NUMERIC)`, so `x * 2` says *numeric or interval* and nothing narrower. The two
readings of a claim respect that asymmetrically, and both have to:

- a **contradiction** needs the value to be in **no** family — so `interval '1 day' * 3` is
  silent and `'abc' * 3` is reported.
- an **inference** needs exactly **one** family to choose from — so `x * 2` types nothing.
  Picking one of two would be the coin-flip this whole design exists to avoid.

This is why there is no separate "no overload survived" branch, and no F003: `round('x', 2)`
contradicts `{NUMERIC}` and `'abc' + 5` contradicts `{NUMERIC, TEMPORAL, INTERVAL}` by the
same test. An earlier draft of this section claimed only where all candidates *agreed*,
which reported strictly less and needed a second code path to report the rest.

`a + b + c` needs no n-ary signature: it parses `Add(Add(a, b), c)` and annotation is
bottom-up, so the inner node's type feeds the outer one.

### ⚠️ Claims attach to the operator node, never to the expression

"This expression contains a `*`, so its operands are numeric" leaks across the `+`:
`order_date + (n * 2)` would claim `order_date` is NUMERIC. So would "this `+` has an
integer literal, so every operand is an integer" — and `date + integer` is a real DuckDB
overload, not an autocast. Both are the `date_trunc` gotcha in a new costume: a confident
error on valid SQL. The catalog encodes which of these the dialect actually has, per
operator node, which is strictly better than any hand rule.

### ⚠️ Operator signatures are for facts only — never installed as annotators

sqlglot already types `+`, `/` and `>` correctly, `TYPED_DIVISION` and DECIMAL precision
included. The plan's "keep the catalog a gap-filler" rule stands. So the catalog has two
consumers:

- `expression_metadata` — annotation. Functions with coverage gaps only. Operator keys are
  skipped for free: `exp.FUNCTION_BY_NAME` has no `"+"`, so no class is ever found for them.
- the fact walk — everything, operators included.

Operators have no `sql_name()` (they are `exp.Binary`, not `exp.Func`), so a small explicit
`{exp.Add: "+", ...}` map is the one place a class name appears. That is deliberate and
commented; these classes are core and older than any dialect module.

## Ordering — facts move out of step 3b

Fact extraction runs **after annotation pass 1**, not beside `build_graph`.

Step 5 above proves why: before pass 1 every `.type` is `None`, so `in_family` cannot tell
"undeclared" from "declared and fine", and every argument position generates a claim — the
spurious `order_ts -> TEMPORAL`. That was harmless while evidence was internal. Facts are
now shown to the user with a span attached, so the same over-claim would be a lie in a
report.

Facts hold **node references, not snapshotted types**. The checking pass therefore reads
post-step-6 types off the same mutated tree, and nothing has to be re-extracted after
widening.

```
step 4  annotate pass 1        establishes what is already known
facts   extract                claims, links, predicates, joins, nullability, cardinality
step 5  infer                  claims/links on UNKNOWN source columns -> types, or conflict
step 6  annotate pass 2        same tree, widened schema, in place
step 7  check                  F001, F002, and every contradicted claim
```

### Conflicting facts on an undeclared column

Error at **every** conflicting site, and the column **stays UNKNOWN**. Not a warning: sqlr's
position is that engine autocasting is never something to rely on, so `upper(x)` beside
`x > 5` is a defect, not a dialect feature. Staying UNKNOWN is what stops a guess from
cascading — UNKNOWN is absorbing, so everything downstream goes quiet instead of inheriting
a coin-flip.

## What survives as its own finding

F001 (unknown function) and F002 (arity) stay separate, and that is the right seam: they are
**structural**, need no types at all, and are about the *call*. F003 was never about the
call — it was always about the value flowing into it, which is what a fact is.

## Decisions settled in this revision

| Question | Decision | Reason |
|---|---|---|
| `cast(x as date)` — a claim about `x`? | **No claim.** The cast types its result only. | `cast(order_id as varchar)` does not make `order_id` a varchar, and `x::date` on a string column is the commonest cast written. v1 recorded it; as a user-visible claim it becomes a false-positive generator. |
| Column-to-column comparison | **Link, all six operators.** `!=` included. | Without autocasting, comparing two columns requires a common domain — `!=` demands it exactly as much as `=`. |
| A config for `!=` | **No.** | It would switch between a result and the same result — the argument already made for the steps 5+6 skip. Where `!=` genuinely differs is *relationship* inference (`=` suggests a foreign key, `!=` suggests nothing), so the join fact stays `=`-only and needs no knob. |
| Cross-scope links (set-op arms, `IN (subquery)`) | **In.** | ~25 lines given the node→scope index the fact walk builds anyway. |
| Transitive link propagation | **Out.** One round. | A link off a *just-inferred* column does not chain. Convergence in one round is what keeps widening monotone; a fixpoint needs its own proof. |
| `models:` declaration vs projected type | **Out, this round.** | Computed-vs-declared is a different comparison from anything here, and runs on `annotate_types`' returned data. |
| Facts as pydantic models | **No — frozen dataclasses.** | They hold live `exp.Expr` references on purpose, so step 7 reads current types. A serialisable view is a projection of them, later. |

## Modules

| Module | Contents |
|---|---|
| `catalog.py` | signatures, functions and operators; `@T` variables; variadic marker |
| `annotate.py` | step 0 wiring, families, `pick_overload` — unchanged in shape |
| `facts.py` | `extract_facts_from_annotated_tree` — claims, links, predicates, joins, nullability, cardinality |
| `infer.py` | verdicts per undeclared source column, `widen_schema_with_inferred_types` |
| `check.py` | `findings_for_unknown_functions`, `findings_for_calls_with_wrong_arity`, `findings_for_contradicted_claims` |
| `annotate_types.py` | steps 4-7 for one model and for a run; sqlglot's own imported as `annotate_types_with_sqlglot` |

## Tests

1. Catalog arity matches each class's `arg_types`.
2. Coverage debt does not grow.
3. Monotone widening — `widen(s, facts) == s` when `s` has no UNKNOWN.
4. Skip equivalence — steps 5+6 forced on vs skipped on a fully-declared schema.
5. Three-valued discipline — undeclared column into a catalogued function, zero findings.
6. `order_date + 7` produces **no** claim about `order_date`.
7. A position where matching-arity overloads disagree infers nothing, and still contradicts
   a value in none of their families.
8. Conflicting facts → error at every site, column stays UNKNOWN.
9. Link propagation across a join, and across a UNION's arms.
