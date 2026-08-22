# Fact extraction and schema inference — implementation plan

Supersedes `type_check_plan.md`'s revision section on **steps 5 and 7**, and the
`pipeline_architecture.md` description of **phases C, D and E**. Everything those documents
say about phases A and B — relation transparency, the closure walk, gap-filling, qualify,
annotation pass 1 — still stands and is not restated here. Where they disagree with this
document, this one wins.

Scope: goals **3** (usage contradicting a declaration) and **4** (inferring the types of
undeclared columns). Goal 5, whole-file checking against a `models:` declaration, stays out.

Status: **design agreed, not implemented.** Decisions are settled unless marked `❓`.

---

## What is already built, verified by running it

The previous plan's steps 4–7 exist and are wired: [facts.py](src/sqlr/sql_analysis2/facts.py),
[infer.py](src/sqlr/sql_analysis2/infer.py), [check.py](src/sqlr/sql_analysis2/check.py),
[catalog.py](src/sqlr/sql_analysis2/catalog.py), [annotate.py](src/sqlr/sql_analysis2/annotate.py),
[annotate_types.py](src/sqlr/sql_analysis2/annotate_types.py). 449 tests pass.

Forcing `sql_dialect: duckdb` on `examples/interconnected_sql_files` — the project declares
`snowflake`, which is gap 1 below — the pipeline already produces the two things this document
is about:

```
===== department            # upper(department_name), declared number(38,0)
  error contradicted-type
    UPPER argument 1 expects STRING, but 'department_name' is declared number(38,0) (DECIMAL(38, 0))
===== sales
  claims 6  links 13
  INFER mydatabase.myschema.customer       is_active  BOOLEAN  ['= with TRUE']
  INFER mydatabase.myschema.instore_order  amount     INT      ['> with 0']
===== employee
  INFER mydatabase.myschema.raw_employee   status     TEXT     ["in list with 'ACTIVE'", ...]
```

So this is not a rewrite. It is one new resolution model, one new catalog layer, and four
contained fixes. `facts.py`'s walk survives almost unchanged; what changes is what consumes it.

---

## The one rule, restated

`type_check_plan.md` put it as:

> A fact is a claim about a value. A claim contradicted by that value's actual type is an
> error; a claim about a value with no type at all is the inference.

That is still true and still the reason claims and inference share one mechanism. What it does
not say — and what the rest of this document exists to fix — is **how strong an answer is**, and
therefore which of the two readings applies when a value's type came from another fact rather
than from a declaration.

> **Strength is a property of the evidence, and it crosses a link intact.** A value linked to a
> stated type is stated. A value linked to an inferred type is inferred. Disagreement among
> stated values is an **error**; disagreement among inferred values is a **resolution failure**.

---

## Type strength

Two axes, deliberately separate. Origin says where an answer came from; strength says what to do
when answers disagree.

```python
type TypeStrength = Literal["stated", "inferred", "unresolved"]
type TypeOrigin = Literal["declared", "computed", "literal", "linked", "claimed"]
```

| origin | strength | example |
|---|---|---|
| `declared` | stated | `data_type: varchar(20)` in a yml |
| `computed` | stated | sqlglot annotated it bottom-up — a function's return type, a CTE projection |
| `literal` | stated | `'2024-01-01'`, `0`, `true` |
| `linked` from a stated member | **stated** | `x = some_declared_date_col` |
| `linked` from an inferred member | inferred | `x = y`, where `y` was itself inferred |
| `claimed` | inferred | `upper(x)` says x is STRING |
| — | unresolved | nothing could say, or evidence conflicted |

Two consequences, both intended:

- **A literal is a stated type.** `where order_date >= '2024-01-01'` types `order_date` as
  STRING at stated strength, so `date_trunc('day', order_date)` elsewhere in the project is an
  **error**, not a shrug. sqlr does not model engine autocasting: if the column is really a date,
  the SQL should say `date '2024-01-01'` or the yml should declare it. This is the whole
  "typescript for sql" position, and it is the single most opinionated decision in the design.
- **A claim can never contradict an inferred type.** An inferred type is *derived from* the claim
  set, so if the claims disagreed the column is unresolved and there is nothing to contradict.
  That is what makes `findings_for_contradicted_claims` sound without a special case: it reports
  only against `stated` types, and everything else was already reconciled.

### ⚠️ Strength does not decay with distance

`a = b`, `b = c`, `c` declared `DATE` puts all three at **stated** DATE. Not "stated, then
weaker, then weaker still". A link asserts the values share a domain; there is no half-sharing,
and a decay rule would need a decay constant nobody can justify.

---

## The type lattice

Families become a tree. The flat set in [annotate.py:41](src/sqlr/sql_analysis2/annotate.py#L41)
cannot express `round` accepting any numeric while `list_extract` accepts only an integer index,
and it cannot answer "are these two types the same kind of thing" at more than one granularity.

```
ANY
├─ NUMERIC ──┬─ INTEGER    TINYINT SMALLINT INT BIGINT HUGEINT ...
│            ├─ DECIMAL    DECIMAL NUMERIC(p,s) ...
│            └─ FLOAT      REAL DOUBLE ...
├─ STRING                  VARCHAR TEXT CHAR ...
├─ TEMPORAL ─┬─ DATE
│            ├─ TIME
│            └─ TIMESTAMP  TIMESTAMP TIMESTAMPTZ TIMESTAMPNTZ ...
├─ BOOLEAN
├─ INTERVAL
├─ BINARY
├─ ARRAY<T>
└─ STRUCT
```

Three operations, and every type decision in the system is one of them:

```python
def in_family(dtype: exp.DataType | None, family: FamilyName) -> bool | None:
    """Three-valued, unchanged in contract. Now a descendant-or-self test on the tree."""

def nearest_common_family(left: FamilyName, right: FamilyName) -> FamilyName:
    """`INTEGER`,`DECIMAL` -> `NUMERIC`. `DATE`,`TIMESTAMP` -> `TEMPORAL`.
    `STRING`,`NUMERIC` -> `ANY`, which is the conflict signal."""

def widest_type_in(family: FamilyName) -> ColumnTypeName:
    """What an inferred family becomes in the schema. `NUMERIC` -> `DECIMAL(38,9)`."""
```

**`in_family` keeps its three-valued contract exactly.** `None` means "cannot say, stay quiet";
`False` means "definitely wrong, report it". Conflating them is how a checker gets a reputation
for lying, and every false positive found while building the demo traced back to that
distinction. The tree changes what `True` means, not what `None` means.

### Compatibility is nearest-common-ancestor, not equality

Two stated types conflict **iff their nearest common family is `ANY`**.

| pair | nearest common | verdict |
|---|---|---|
| `INT`, `DECIMAL(10,2)` | NUMERIC | compatible, widen to NUMERIC |
| `DATE`, `TIMESTAMP` | TEMPORAL | compatible, widen to TEMPORAL |
| `VARCHAR(20)`, `TEXT` | STRING | compatible |
| `VARCHAR`, `BIGINT` | ANY | **conflict** |
| `BOOLEAN`, `VARCHAR` | ANY | **conflict** |

`int_col = decimal_col` is ordinary SQL and not a defect. Reporting it is how a checker gets
switched off, which costs more than the bug it would have caught. The cases worth reporting —
a string compared to a number, a boolean used as text — all reach `ANY`.

### ⚠️ Precision and scale are recorded, never checked

`DECIMAL(10,2)` vs `DECIMAL(38,9)` is not a finding and `VARCHAR(20)` vs `VARCHAR(100)` is not a
finding. Parameters exist so a report can print what the user wrote and so fixture generation can
size a column later. Nothing in this document reads them. Adding a check on them later is a
deliberate act, not a natural extension.

Signatures should therefore name the **shallowest** node that is really required. A signature
saying `DECIMAL` where the engine accepts any numeric turns `round(int_col)` into an error.

---

## The resolution model: union-find over value sites

The core of this plan. It replaces `evidence_per_undeclared_column` and
`infer_types_for_undeclared_columns` in [infer.py](src/sqlr/sql_analysis2/infer.py), which today
key evidence on a column and resolve each column alone.

### Step I1 — build components

*In:* `Facts.type_links`. *Out:* a disjoint-set forest over `ValueSite`s.

```
for each link fact:            union(left, right)
for each pair of sites that resolve to the same (RelationKey, ColumnName):
                               union(a, b)
```

**The second union is not optional.** `orders.amount` at line 4 and `orders.amount` at line 10
are two distinct `exp.Column` nodes. Without merging them by schema slot, a claim at line 4 and a
link at line 10 land in different components and neither sees the other. This is the easiest line
in the plan to leave out and the one whose absence is hardest to notice.

Sites in no link are singleton components, so there is one uniform path.

### Step I2 — resolve each component

*In:* one component. *Out:* a `ComponentVerdict`.

```python
@dataclass(frozen=True)
class ComponentVerdict:
    members: list[ValueSite]
    type_name: ColumnTypeName | None     # what goes into the widened schema
    family: FamilyName | None            # what a report says
    strength: TypeStrength
    evidence: list[TypeEvidence]         # every anchor and claim, in source order
    findings: list[TypeFinding]
```

```
anchors = members whose node.type is known after annotation pass 1
claims  = every TypeClaim landing on any member

if anchors:
    fold the anchors with nearest_common_family
    reaches ANY   -> ERROR at every disagreeing anchor site; component unresolved
    otherwise     -> strength = "stated"
                     type     = the fold  (see "what a link propagates" below)
                     any claim not satisfied by it -> ERROR (contradicted-type)
else:
    intersect the claims' family sets
    exactly one   -> strength = "inferred", type = widest_type_in(family)
    empty         -> WARNING at every claim site; component unresolved
    several       -> unresolved, silent   (`x * 2` alone says numeric-or-interval)
```

That table is your rule, mechanised. Disagreement among stated members is an error because
someone wrote both types down and one of them is wrong. Disagreement among inferred members is a
warning because sqlr guessed twice and the guesses fought — the user's SQL may be fine and the
honest answer is "I could not tell, here is what I saw".

### What a link propagates

| the stated anchor is | the component gets |
|---|---|
| a column with a declared or computed type | that **exact type** — `= order_date` gives `DATE`, not "some temporal" |
| a literal | its **family**, materialised as `widest_type_in(family)` |

`where amount > 0` therefore infers NUMERIC → `DECIMAL(38,9)`, not `INT`. The literal proves the
column is numeric; it proves nothing about its width, and a narrow guess would falsely contradict
`amount * 1.5` later. `where d >= '2024-01-01'` infers STRING → `VARCHAR`, at stated strength,
per the position above.

### ⚠️ Never overwrite a stated type

The monotonicity invariant survives intact and is still the thing most likely to be broken by a
reasonable-sounding feature later:

> Widening only ever fills an `UNKNOWN` slot. A component whose verdict disagrees with a member's
> declared type reports the disagreement; it does not rewrite the declaration.

This is what makes widening converge and what keeps the phase-D skip sound.

---

## New and changed facts

### Step F1 — projection-passthrough links

*In:* every scope. *Out:* one link per projection that is a bare column.

`select amount from online_order` means the CTE's output `amount` and the storage column are the
same value. That is a link like any other, and with it union-find carries a claim down through
any depth of CTE.

```python
# emitted when the projection, with its alias stripped, is an exp.Column
LinkReasonKind += "projection-passthrough"
```

Without it, `sales.sql` infers nothing at all for its source columns: `sum(o.amount)` claims
NUMERIC on `order_summary.amount`, which is a CTE column with no schema slot, and the claim dies
there. With it:

```
order_summary.amount = active_customer_orders.amount = all_orders.amount
                     = online_orders.amount  = online_order.amount    <- storage
                     = instore_orders.amount = instore_order.amount   <- storage
```

One component, one claim, both storage columns typed. Set-operation arms are already linked by
`_link_set_operation_arms`, so the union arms join the same component with no extra code.

`amount * 2 as amount` is **not** a passthrough: it is not the same value. It is handled by F2.

### Step F2 — return-marker links

*In:* the catalog. *Out:* a link between a call node and one of its arguments.

A signature whose `returns` is `@argN` is asserting *the result is argument N's type*. That is
already a link, written down, and nothing reads it that way today.

```
amount * 2   ->  Sig(("NUMERIC","NUMERIC"), returns="@arg0")  ->  link(Mul node, amount)
```

So `amount * 2 as amount` in a CTE reaches the source column: the passthrough link from the
downstream read hits the `Mul` node, and `@arg0` carries it to `amount`. Backward propagation
through arithmetic, for free, from data that was already written.

**Emit the link only when every *surviving* candidate overload agrees on the marker.** Candidates
are filtered by `in_family(...) is False` first, exactly as `pick_overload` does:

```
amount * 2   candidates: (NUMERIC,NUMERIC)->@arg0
                         (INTERVAL,NUMERIC)->@arg0
                         (NUMERIC,INTERVAL)->@arg1   <- dropped, `2` is not an INTERVAL
             survivors agree on @arg0  ->  link
```

Without the filter, `*`'s three overloads disagree and the link is never emitted.

⚠️ Candidates are filtered against **pass-1** types, because facts are extracted once. A type that
only becomes known in pass 2 cannot retroactively narrow the candidate set. Accepted: the
alternative is re-extracting facts, and the case it would buy is a link whose own component
supplied the missing type — which the component then resolves anyway.

### Step F3 — the `@family(argN)` marker

*In:* the catalog. *Out:* a family-strength link.

`sum(x)` is not `@arg0` — `SUM(INT)` is `HUGEINT` in DuckDB and `NUMBER(38,0)` in Snowflake. But
it is certainly *numeric because x is numeric*. A new return marker says exactly that:

```python
"SUM": [Sig(params=("NUMERIC",), returns="@family(arg0)")],
"MIN": [Sig(params=("@T",),      returns="@arg0")],          # identity, not family
"MAX": [Sig(params=("@T",),      returns="@arg0")],
"AVG": [Sig(params=("NUMERIC",), returns="DOUBLE")],         # concrete, no link at all
```

A family link puts both ends in one component but contributes only the family, never the concrete
type. In practice inference wants the family anyway, so this costs one flag on `TypeEvidence`.

### What does not become a link

| construct | why not |
|---|---|
| `cast(x as date)` | Settled in `type_check_plan.md` and unchanged. A cast is the one place a type change is explicit and intended. `cast(order_id as varchar)` does not make `order_id` a varchar. |
| `f(x)` with a concrete return | The result's type is the signature's, and says nothing back about `x` beyond the claim already made. |
| a projection that is any other expression | Not the same value, and no marker says otherwise. |

---

## The catalog

### Step C1 — a common catalog, plus per-dialect layers

`upper(number_column)` is an error in every engine. Today
[catalog.py:206](src/sqlr/sql_analysis2/catalog.py#L206) holds `{"duckdb": {...}}` and
`signatures_for_dialect("snowflake")` returns `{}` **in silence** — which is why the project's
own `sql_dialect: snowflake` produces zero claims, zero links, zero findings and zero
catalog-driven inference across every model.

```python
CATALOG_COMMON:   dict[CatalogKey, list[Sig]]                    # operators + standard SQL
CATALOG_DIALECT:  dict[DialectName, dict[CatalogKey, list[Sig]]] # duckdb, snowflake, spark
```

**Merge rule: a dialect entry replaces the common entry for that key outright, never appends.**
Appending would mean a dialect could only ever add overloads, so it could not express *narrower*
— and the reason a key appears in a dialect layer at all is usually that the engine differs.
Replacement is also the only rule a reader can predict from looking at one file.

Everything in `_OPERATORS` and the standard-SQL half of `_DUCKDB_FUNCTIONS` moves to common.
Dialect layers get a handful of entries each, enough to prove the mechanism:

| dialect | example entries |
|---|---|
| duckdb | `LIST_EXTRACT`, `SPLIT`, `TIMESTAMP_TRUNC` — the existing annotation gaps |
| snowflake | `ZEROIFNULL`, `IFF`, `TO_VARCHAR`, `ARRAY_CONSTRUCT` |
| spark | `CONCAT_WS` arity, `DATE_ADD`, `EXPLODE` |

Not exhaustive, and deliberately so — `duckdb_functions()` generation is deferred until a real
project needs the coverage.

### Step C2 — gate `unknown-function` on catalog completeness

`findings_for_unknown_functions` reports any `Anonymous` call with no catalog entry. With a
non-exhaustive catalog that is a false-positive generator: every real function sqlglot happens to
parse as `Anonymous` becomes a user-facing error.

```python
CATALOG_IS_COMPLETE: dict[DialectName, bool] = {}   # default False for every dialect
```

F001 fires only for a dialect marked complete. Nothing is marked complete in this round, so the
finding is dormant — correct, because the statement "this function does not exist" is only
truthful from an exhaustive list. `function-arity` and `contradicted-type` are unaffected: both
reason from an entry that **is** present, so a thin catalog makes them quiet, never wrong.

### ⚠️ Signatures follow sqlglot's node order

Unchanged and still the gotcha that costs a false positive. `date_trunc('day', ts)` parses to
`TimestampTrunc(this=ts, unit='day')`, so its signature reads `(TEMPORAL, ANY)`. Writing it the
way the SQL reads produces a confident error on valid SQL. Every new dialect entry needs the
arity test from `type_check_plan.md`.

---

## Contained fixes

### Step X1 — `RelationKey` leak in the annotation layer

[annotate_types.py:284](src/sqlr/sql_analysis2/annotate_types.py#L284) does
`table = source.name.lower()` — the bare table name — and looks it up in
`declared_types_per_relation`, which D1 keyed on the full dotted `RelationKey`. Every lookup
misses. Visible in today's output:

```
RANKED  (cte)
   PERSON_ID    VARCHAR(20)   unknown     <- has a declared type, reported as unknown
   STREET       VARCHAR(200)  unknown
```

Use `relation_key_of(source)` from [relations.py:62](src/sqlr/sql_analysis2/relations.py#L62).

While there: `SourceColumnType.table` and `provenance_of_projection`'s `inferred` dict are
annotated `TableName` but hold `RelationKey`. Per the convention in `CLAUDE.md`, a type alias that
says the wrong thing is worse than no alias.

### Step X2 — declared type names through `typemap` at the boundary

`get_declared_types_per_relation` passes the type name **as written** into `MappingSchema`.
`declared/` already computes `resolved_type_name` via `typemap.resolve_type_name` and v2 throws it
away. An unrecognised name builds a user-defined type whose `in_family` is `False` for every
family, so **a typo in `data_type:` becomes a `contradicted-type` error on correct SQL**:

```
data_type: frobnicate        ->  column annotated `frobnicate`
                             ->  in_family(..., STRING) is False
                             ->  "UPPER argument 1 expects STRING, got frobnicate"
```

The fix is `type_check_plan.md` ❓Q3's answer, finally applied: the lattice converts at the
boundaries, sqlglot's `DataType` is the internal currency. An unresolvable name gets a new
finding rather than a bogus type.

```python
FindingCode += "unrecognized-declared-type"    # warning; the column falls back to UNKNOWN
```

### Step X3 — provenance reaches the report

`SourceColumnType.provenance` becomes the `TypeStrength`/`TypeOrigin` pair, and the render gains
the three message shapes `pipeline_architecture.md` already specified:

| the value's type came from | message |
|---|---|
| `sources.yml` | *`status` is declared `varchar(20)`; `round` needs NUMERIC* |
| a CTE's computed projection | *`ranked.amount` is TEXT here; `round` needs NUMERIC* |
| inference | *`status` is inferred TEXT from its use at line 4; `round` needs NUMERIC* |

A component verdict carries every anchor and claim in `evidence`, so the third message can name
the site that decided it — which is design principle 5, and the reason components keep their
whole evidence list rather than a winner.

---

## Findings

| code | severity | source |
|---|---|---|
| `unknown-function` | error | an `Anonymous` call, dialect catalog marked complete | 
| `function-arity` | error | no signature of that arity |
| `contradicted-type` | error | a claim contradicted by a **stated** type |
| `conflicting-usage` | **error** | two stated anchors in one component reaching `ANY` |
| `conflicting-usage` | **warning** | claims alone disagreeing on an undeclared column |
| `unrecognized-declared-type` | warning | a `data_type:` the lattice cannot resolve |

`conflicting-usage` carries both severities because it is one situation read at two strengths, and
splitting it into two codes would make a suppression rule have to know which. Every conflict emits
one finding **per site**, because every site is a place the user has to look and which one to
change is theirs to decide.

---

## Order of work

1. **X1** — the `RelationKey` fix. One line, unblocks reading every other result correctly.
2. **C1** — common catalog plus the three dialect layers. Without it nothing else is observable
   on the example project.
3. **Lattice** — the family tree, `nearest_common_family`, `widest_type_in`, and `in_family`
   rewritten against it. Everything downstream depends on the shape.
4. **F1, F2, F3** — the three new link sources, in `facts.py`. Each is independently testable
   against a fixture before any of them is consumed.
5. **I1, I2** — union-find and component resolution, replacing the body of `infer.py`.
6. **X2** — the declared-type boundary, and its finding.
7. **C2, X3** — the F001 gate and the report messages.

Steps 1–3 are prerequisite to everything. Steps 4 and 5 are the substance. 6 and 7 are polish that
should not be deferred past the first real run, because both are visible to the user.

---

## Deliberately out of scope

| | |
|---|---|
| **Cross-model reconciliation** | Each model infers independently and nothing merges the results, so one source column can be TEXT in one file and unknown in another. Fixture generation needs one answer per `(RelationKey, ColumnName)` for the whole project, and design principle 3 says it gets emitted for review. Deferred by agreement; it is the natural home for the *next* layer of conflict reporting. |
| **Name-pattern fallback** | `*_id`, `*_at`, `is_*`. Listed Basic in the design doc, dropped by `type_check_plan.md` without saying so. Staying dropped for now — it is the one evidence source that can be confidently wrong. |
| **Catalog generation from `duckdb_functions()`** | ❓Q5's answer, still the right one, still deferred. C2's completeness flag is the seam it lands on. |
| **Precision and scale in checking** | Recorded, printed, never compared. |
| **Goal 5** | Whole-file checking against a `models:` declaration. `ScopeColumns.complete` is already computed and propagated for it. |

## Deprecated

`sql_analysis/` is dead. Nothing in this plan imports from it; anything worth keeping is copied
out. `schema_resolution/` is superseded by `infer.py` and is now dead for the same reason. Neither
is deleted yet.

`typemap.py` **survives where it is**, at the top level. `declared/` already imports it and X2
adds a second reader. It is not part of v1.

---

## Tests to write first

Fixtures live with the tests; nothing reads from `examples/`.

**Strength and conflict**

1. `where d >= '2024-01-01'` and `upper(d)` elsewhere → `d` is STRING, **no** finding. Both agree.
2. `where d >= '2024-01-01'` and `date_trunc('day', d)` → **error** at the `date_trunc` site.
   Stated-vs-stated. This is the headline behaviour of the whole design.
3. `upper(x)` and `x > 5`, `x` undeclared and unlinked to any literal → **warning**, `x` stays
   UNKNOWN. Inferred-vs-inferred.
4. A declared `varchar(20)` column passed to `round` → `contradicted-type`, message naming the
   declaration and its yml line.

**The lattice**

5. `int_col = decimal_col` → no finding, component widens to NUMERIC.
6. `date_col = timestamp_col` → no finding, widens to TEMPORAL.
7. `varchar_col = int_col` → `conflicting-usage` error at both sites.
8. `DECIMAL(10,2)` against `DECIMAL(38,9)` → no finding. Parameters are never compared.

**Components**

9. `a.x = b.y`, `b.y = c.z`, `c.z` declared `DATE` → `a.x` and `b.y` both DATE, both **stated**.
10. Two reads of `orders.amount` in different scopes, a claim on one and a link on the other →
    one component. Guards the merge-by-schema-slot union.
11. Passthrough through three CTEs to a storage column (the `sales.sql` shape) → the storage
    column is typed.
12. `amount * 2 as amount` in a CTE, a NUMERIC claim downstream → the source `amount` is typed,
    via `@arg0`.
13. `sum(x) as total` with a NUMERIC claim on `total` → `x` gets the family, not `HUGEINT`.
14. A component with no anchor and claims `{NUMERIC, INTERVAL}` from `x * 2` alone → unresolved,
    **silent**. Ambiguity is not a finding.

**Catalog**

15. `upper(number_col)` is an error under duckdb, snowflake **and** spark. The common catalog's
    reason for existing.
16. A dialect entry for a key also in common replaces it rather than adding an overload.
17. Catalog arity matches each class's `arg_types`, for every entry with a sqlglot class.
18. `unknown-function` produces nothing for a dialect not marked complete.

**Boundaries and invariants**

19. `data_type: frobnicate` → `unrecognized-declared-type` warning, and **no**
    `contradicted-type`.
20. Monotone widening: `widen(s, verdicts) == s` when `s` has no UNKNOWN.
21. Skip equivalence: inference forced on vs skipped on a fully-declared schema — identical types
    *and* identical findings.
22. Three-valued discipline: an undeclared, unconstrained column flowing into a catalogued
    function produces zero findings. The false-positive regression test.
23. A declared column's projection reports provenance `declared`, not `unknown`. Guards X1.
