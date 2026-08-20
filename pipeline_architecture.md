# Pipeline architecture

Supersedes the *shape* of `type_check_plan.md` steps 2–3. Everything that document says about
steps 4–7, the catalog, three-valued `in_family`, and facts-as-the-single-mechanism still
stands and is not restated here. Where the two disagree, this one wins.

Status: **implemented**, except the `validate-schema` half of D3 - see "What was built".
D1-D3 are settled; see the end.

---

## The bug that forced this

```
sqlr qualify-schema --project-dir examples/interconnected_sql_files/ --select address
project/address.sql
error: Column 'STREET' could not be resolved. Line: 11, Col: 10
```

`address.sql` reads `street` from the CTE `ranked`. `ranked` is `select *, row_number() … from
mydatabase.myschema.raw_address`. `raw_address` is undeclared, so its star cannot expand, so
`ranked`'s projection list is literally `['*', 'rn']`.

Step 2 ([resolve.py:324-333](src/sqlr/sql_analysis2/resolve.py#L324-L333)) attributes `street`
to `ranked`, sees a `Scope` rather than an `exp.Table`, and drops it:

> *Anything else is a CTE or derived table, whose columns are its own projections — nothing to
> fabricate a declaration for.*

That comment is true only when the CTE's projections can actually be enumerated. Here they
cannot. `street` is fabricated on nothing, `raw_address` gets a 3-column gap-filled schema
(`person_id, updated_at, country`), and step 3's real `qualify` raises.

Verified: gap-filling `raw_address` with the 7 columns the file names makes the whole statement
qualify, every star expand, every scope enumerate:

```
ranked          -> [person_id, updated_at, country, street, city, state, zip, rn]
current_address -> [person_id, street, city, state, zip, full_address]
<final>         -> [person_id, street, city, state, zip, full_address]
```

**So the fix is contained to step 2. Nothing downstream needs changing.** What needs changing is
the *concept* step 2 is missing.

## The missing concept: relation transparency

The pipeline conflates two questions that only coincide in the easy case:

| Question | Answer for `street` in `address.sql` |
|---|---|
| Which relation does this column **read from**? | `ranked` — lexical, scope-local, sqlglot answers it |
| Which storage **owns** it, so a schema slot can be fabricated? | `raw_address` — transitive, nobody answers it today |

A relation is **closed** when its column set can be enumerated, and **open** when it projects a
star over something that cannot be. An open relation is *transparent*: a name it does not
produce itself passes straight through to whatever its star reads.

Transparency is transitive. `a` selects `*` from `b`, `b` selects `*` from undeclared `t` — a
column read off `a` is owned by `t`. Attribution must therefore be a walk over the relation
graph, not a lookup in one scope.

This also subsumes `known_column_names_of_source`
([resolve.py:98-122](src/sqlr/sql_analysis2/resolve.py#L98-L122)), which returns `None` — "no
answer available" — the moment a star appears. `None` throws away `rn`, which the CTE certainly
does produce. The replacement returns *both* halves: what is known, and what is open.

---

## The pipeline

Five phases matching your five goals. Each numbered step names its input, its output type, and
the one thing it is responsible for. **A step never does two of these jobs.**

```
A. structure   1 parse ──▶ 2 probe ──▶ 3 close ──▶ 4 attribute ──▶ 5 gap-fill ──▶ 6 qualify
B. declared    7 annotate pass 1
C. facts       8 extract facts
D. inference   9 infer ──▶ 10 annotate pass 2      (skipped when nothing is UNKNOWN)
E. checking    11 check
```

Phases B and E both consume declared types. They are **the same walk over the same facts** — see
"Why goals 3 and 5 are one pass" below.

---

### Phase A — structure. No type is read or written anywhere in it.

#### Step 1 — parse

*In:* file text. *Out:* one `exp.Expr`, or a file-level error.

Unchanged. One statement per file; a second is an error, not a warning. `Positions` is built once
per document here and stays valid across every later rewrite.

#### Step 2 — probe qualification

*In:* statement. *Out:* a throwaway qualified copy.

Unchanged, and the docstring at
[resolve.py:188-213](src/sqlr/sql_analysis2/resolve.py#L188-L213) explains why the empty schema
is load-bearing. Its job is exactly one thing: **give every bare column a qualifier**, using
sqlglot's own `Resolver` for join context, USING expansion, lateral derivation and alias
shadowing.

What it does *not* do — and today's code assumes it does — is tell you whether that qualifier is
a table. Step 3 answers that.

#### Step 3 — close the relation graph  ⟵ **new**

*In:* the probe tree + its scopes + `declared_schema`.
*Out:* `dict[ScopeId, dict[RelationAlias, RelationColumnSet]]`.

```python
type RelationAlias = str

class RelationColumnSet(NamedTuple):
    """What one relation visible in one scope is known to project."""

    alias: RelationAlias
    kind: SourceKind                      # "table" | "cte" | "derived" | "branch"
    storage: RelationKey | None           # the real table behind it, when kind == "table"
    known: frozenset[ColumnName]
    """Names this relation certainly projects."""
    open_origins: tuple[RelationKey, ...]
    """Storage tables reachable through a star this relation could not expand. Empty means
    closed: a name not in `known` is a mistake, not a pass-through."""
```

Computed in `traverse_scope` order, which is dependency order — every source of a scope is
already closed by the time the scope itself is.

```
for each scope, for each selected source:
    exp.Table, declared complete -> known = declared names,  open_origins = ()
    exp.Table, declared partial  -> known = declared names,  open_origins = (this table,)
    exp.Table, undeclared        -> known = {},              open_origins = (this table,)
    Scope                        -> the entry already computed for it

then the scope's own set:
    known        = output names of every non-star projection
                 | the `known` of every closed source a star covers
    open_origins = union of open_origins of every source a star covers
```

Star coverage: `select *` covers every selected source; `select t.*` covers `t` only; no star
covers nothing. Set-operation arms intersect — a branch projects a name only if every arm does,
and is open if any arm is.

For `address.sql` this produces:

| relation | known | open_origins |
|---|---|---|
| `raw_address` | `{}` | `(raw_address,)` |
| `ranked` | `{rn}` | `(raw_address,)` |
| `current_address` | `{person_id, street, city, state, zip, full_address}` | `()` |

`ranked` keeps `rn` *and* stays transparent to `raw_address`. That pair is the whole fix.

#### Step 4 — attribute every column to its owner

*In:* the probe's columns + step 3's closure. *Out:* `ResolvedColumns` (existing type).

One rule per column read `alias.name` in scope `S`, against `rel = closure[S][alias]`:

| condition | verdict |
|---|---|
| `name in rel.known` and `rel.kind == "table"` | fabricate a slot on `rel.storage` |
| `name in rel.known` and `rel` is a CTE/derived | resolved; **no slot** — its type is computed |
| `name not in rel.known`, exactly one `open_origin` | fabricate a slot on that origin |
| `name not in rel.known`, several `open_origins` | `star_over_join_behavior`: `error` → `AmbiguousColumn`, `guess` → first origin + `GuessedColumn` |
| `name not in rel.known`, no `open_origins` | `UnresolvableColumn` |

Four things this buys that today's code cannot express:

- **`address.sql` works.** Row 3 fires for `street`.
- **`star_over_join_behavior` comes back to life.** It is currently declared in
  [config/types.py:35](src/sqlr/config/types.py#L35) and read by nothing in `sql_analysis2`.
  Row 4 is the only place it belongs.
- **The unresolvable message becomes actionable.** Today: `Column 'STREET' could not be resolved.
  Line: 11, Col: 10` — from `qualify`, describing the rewritten tree. New, from row 5:
  `current_address reads 'street' from ranked, which projects only [rn]`. The reader learns
  which relation is closed and what it does project.
- **The typo trap closes properly.** A misspelled column against a *closed* relation is row 5,
  an error. Against an *open* one it is row 3, silently invented — which is correct, because
  nobody can tell a typo from a real column of an undeclared table.

The existing certainty grading — `judge_a_column_written_without_a_source`,
`sources_owning_and_sources_open_for` — folds into this table. `open_sources` is now
`open_origins`, computed once in step 3 instead of re-derived per column.

#### Step 5 — gap-fill the schema

*In:* attribution + declarations. *Out:* `dict[RelationKey, dict[ColumnName, ColumnTypeName]]`,
declared types where written, `UNKNOWN` elsewhere.

`get_declared_types_per_table` unchanged in behaviour. Two things it now also emits:

```python
needs_inference   = any slot is UNKNOWN                          # gates phase D
projection_is_a_lower_bound = any relation has open_origins      # see below
```

**⚠️ A star over an undeclared table expands to what the file happens to mention, not to what
the table has.** `raw_address` gets 7 columns because `address.sql` names 7. This is an
*under-approximation* and it must be visible, because a model's projected column list is
validated against `schema.yml` — validating a lower bound against a complete declaration
produces false "declared column not projected" errors.

So `ScopeColumns` gains `complete: bool`, propagated from `open_origins` through the scope graph,
and `validate-schema` reports missing columns only when the projection is complete. Where it is
not, the finding is `projection-incomplete` at warning severity, naming the undeclared table.
For `address.sql` the final projection *is* complete — `current_address` enumerates — so this
costs nothing in the common case.

#### Step 6 — qualify

*In:* statement + gap-filled schema. *Out:* `QualifiedStatement`.

Unchanged. **Every column names its relation and every star is a real projection list** — the one
sentence the whole phase exists to make true. Still 39% of runtime; nothing here to optimise.

---

### Phase B — declared types enter

#### Step 7 — annotate, pass 1

*In:* qualified tree + gap-filled schema. *Out:* the same tree, every node carrying `.type`;
`UNKNOWN` where nothing is known.

Declared types are already in the schema, so this is where they reach the tree. sqlglot handles
cross-CTE propagation, so `full_address` gets a type from `street` without anything of ours
running.

Pass 1's second job is the one that is easy to miss and load-bearing: **it establishes what is
already known.** Before it, every `.type` is `None`, and `in_family` cannot distinguish
"undeclared" from "declared and fine" — so every argument position would generate a claim and
phase C would over-claim about columns the user already described. That over-claim was harmless
when evidence was internal; facts now carry a span and reach the user.

---

### Phase C — facts

#### Step 8 — extract facts

*In:* annotated tree. *Out:* `Facts` — claims, links, predicates, joins, nullability,
cardinality.

One walk, as `type_check_plan.md`'s revision section describes, and that section is right:

- **Claim** — *this node must be one of families F.* From `pick_overload` over every
  arity-matching signature, so `x * 2` claims `{NUMERIC, INTERVAL}` and nothing narrower.
- **Link** — *these two nodes share a domain.* From signatures binding two positions to one type
  variable: `=`, joins, `coalesce`, set-op arms.

Facts hold **live node references, not snapshotted types**. That is why step 11 can read
post-inference types off the same mutated tree with nothing re-extracted.

The asymmetry that makes one fact serve both goals 3/5 and goal 4:

- a **contradiction** needs the value in **no** claimed family → `'abc' * 3` reported,
  `interval '1 day' * 3` silent.
- an **inference** needs exactly **one** family to choose from → `x * 2` types nothing rather
  than coin-flipping between numeric and interval.

---

### Phase D — inference *(skipped when `needs_inference` is false)*

#### Step 9 — infer

*In:* facts + gap-filled schema. *Out:* a type per undeclared **source** column, or a conflict.

Only UNKNOWN slots on storage tables. A CTE column is never inferred — its type is computed by
step 7 and overwriting it would fight forward propagation.

**The monotonicity invariant: widening only ever fills an UNKNOWN slot, never overwrites a
type.** That single rule is what makes widening converge in one round *and* what makes the
phase-D skip sound. It is the coupling most likely to be broken later by a reasonable-sounding
feature ("the SQL says string, the yml says int, trust the SQL"); if that feature is ever built,
the skip predicate changes with it rather than getting a config bolted on top.

Conflicting facts on one undeclared column: **error at every conflicting site, and the column
stays UNKNOWN.** Not a warning — relying on engine autocasting is a defect, not a dialect
feature. Staying UNKNOWN is what stops a guess cascading: UNKNOWN is absorbing, so everything
downstream goes quiet instead of inheriting a coin-flip.

#### Step 10 — annotate, pass 2

*In:* same tree, widened schema. *Out:* same tree, re-annotated in place.

No re-parse, no re-qualify: qualification depends on which columns *exist*, and widening never
changes the column set.

**Sharpening over the old plan:** gate pass 2 on `widened != schema`, not on `needs_inference`.
An undeclared project where inference finds nothing — the common case for a file with no `where`
clause and no function calls — currently pays a second full annotation pass for a guaranteed
no-op.

After this step, declared and inferred types are indistinguishable to everything downstream.
That interchangeability is the property worth protecting: the checker behaves identically on a
fully-documented project and an undocumented one, differing only in confidence.

---

### Phase E — checking

#### Step 11 — check

*In:* annotated tree + facts + `Positions`. *Out:* `list[ColumnFinding]`.

The only step with an opinion. Reads `node.type` and never the SQL text — everything upstream has
been flattened into one uniform annotated tree.

| code | source | needs types? |
|---|---|---|
| `unknown-function` | an `Anonymous` call with no catalog entry | no |
| `function-arity` | no signature of that arity | no |
| `contradicted-claim` | a claim contradicted by the value's actual type | yes |
| `conflicting-evidence` | step 9's conflict, one finding per site | yes |

`function-arity` is checked before types and `continue`s, so a 1-arg call never also reports a
bogus type mismatch against a 2-arg signature.

### Why goals 3 and 5 are one pass

Your goal 3 is *"declared number, used as varchar"*. Your goal 5 is *"type check the file"*. They
are the same walk, and forcing them apart is what made the previous design carry two mechanisms
for one job — `backward_evidence` and F003 reporting the same argument-position knowledge in two
vocabularies, at the same span, in different words.

There is no inferred-vs-declared comparison anywhere. A declaration is not a hypothesis to be
checked against the SQL; **it is the type, and the SQL either agrees with it or is wrong.**

The only difference between goals 3 and 5 is the *message*, and the information for that is
already in the fact:

| the value's type came from | message |
|---|---|
| `sources.yml` | *`status` is declared `varchar(20)`; `round` needs NUMERIC* |
| a CTE's computed projection | *`ranked.amount` is TEXT here; `round` needs NUMERIC* |
| step 9's inference | *`status` is inferred TEXT from its use at line 4; `round` needs NUMERIC* |

`ClaimReason` and `_declared_type_of_claim` in
[check.py:134](src/sqlr/sql_analysis2/check.py#L134) already carry this. Three templates, one
branch, one pass.

---

## What changes in the code

| Module | Change |
|---|---|
| `relations.py` | **new.** `RelationColumnSet`, the closure walk, star coverage, set-op intersection. Steps 3 only. |
| `resolve.py` | Loses `known_column_names_of_source`, `sources_owning_and_sources_open_for`, `judge_a_column_written_without_a_source` — all three become the step-4 verdict table over step 3's closure. Keeps the probe and its docstring. |
| `qualify.py` | `ScopeColumns` gains `complete: bool`. Delete the `print()` calls at lines 323, 350-351, and `resolve.py:244-245, 267`. |
| `reporting.py` | New messages for rows 4 and 5 of the step-4 table; `projection-incomplete`. |
| `annotate_types.py` | Pass-2 gate becomes `widened != schema`. |
| `facts.py` / `infer.py` / `check.py` | Unchanged in shape. |
| `declared/types.py` | `declaration_is_partial: bool = False` on `DeclaredRelation`. |
| `declared/__init__.py` | Read `meta:` in `_source`, `_table` and `_models`; merge source-level into table-level, table wins. Warn on `declaration_is_partial: true` with no `columns:` — it says nothing that omitting the entry does not. |
| `config` | `star_over_join_behavior` gets its first v2 reader. No new knob: dbt's `meta:` inheritance already covers a whole source. |
| everywhere | `TableName` as a schema key becomes `RelationKey`, the full dotted name. |

---

## What was built

Everything above, plus four things the design did not anticipate and one it got wrong.

**Corrected in the design.** Set-operation arms do **not** intersect their names. A set
operation's schema is positional — the arms are matched by position and the left one supplies
the names — so `select p from t1 union all select r from t2` produces a column called `p`, and
`r` is not a column of the union at all. Openness still unions across arms.

**Three bugs the tests forced out, none of which the plan predicted:**

- **The probe erases "written bare".** `qualify` has already attributed every bare column by the
  time step 4 runs, landing it on whichever source `infer_schema` fell back to. Taking that
  qualifier at face value turns every fallback into a certainty. `_candidate_relations` reads
  `offsets_of_columns_written_without_a_source` instead — the pre-probe record of what the user
  actually typed.
- **`star_over_join_behavior` is narrower than it looked.** Several *relations* that might own a
  bare name is a different question from one relation whose star reads several *tables*. The
  first has a fix the reader can apply — qualify the column — so it stays `unresolvable`; only
  the second consults the config. Conflating them made `select a.x, mystery from a join b` a
  silent guess.
- **Star qualifiers must not be lowercased.** `scope.sources` is keyed the way the probe
  normalised identifiers, which follows the dialect: Snowflake folds up, DuckDB down.
  Lowercasing one side matched nothing under Snowflake, so every `select t.*` there projected an
  empty relation.

**One pre-existing bug fixed on the way.** A correlated subquery reads its outer query's aliases
— `where exists (select 1 from customer c where c.id = o.customer_id)` names `o`, which belongs
to the enclosing scope. Looking only at the innermost scope reported every such column as a
mistyped alias. `RelationClosure.relation_named` walks outward. This is why `sales.sql` never
worked, and it had nothing to do with transparency.

**Deferred, deliberately.** `ScopeColumns.complete` is computed and propagated, and
`resolved.relations_read_through_an_unexpandable_star` is recorded and logged — but nothing
reads the flag yet, because the `models:`-declaration-versus-projection comparison it guards
does not exist (`type_check_plan.md` put that out of scope for this round). The flag is the data
that comparison will need; the `projection-incomplete` finding lands with it.

## Decisions — settled

### D1 — relation identity is the full name

`columns_per_table` is keyed on the bare lowercased table name, so
`mydatabase.myschema.raw_address` becomes `raw_address`. Two sources with the same table name in
different schemas collide silently.

**Decided: `RelationKey` is the full normalised name, and `qualify` gets the nested schema form.**

```python
type RelationKey = str
"""A relation's identity: the parts the SQL has to write, lowercased and dotted -
`mydatabase.myschema.raw_address`, or bare `employee` when the yml declares neither part.
The same string `DeclaredSourceTable.key` already produces."""
```

Half of this exists: [`DeclaredSourceTable.key`](src/sqlr/declared/types.py) is already the
normalised dotted name, and `DeclaredSchemas.sources` is already keyed by it. It is
`declared_columns()` in [resolve.py:31-44](src/sqlr/sql_analysis2/resolve.py#L31-L44) that
flattens it back down to a bare name, and the comment there admits the simplification.

`RelationKey` comes off the `exp.Table` node: `.catalog`, `.db`, `.name`, the empty parts
dropped, joined, lowercased — matching how `parts` builds the declaration side. A reference
writing fewer parts than the declaration matches only in dbt mode, which
`matches_loosely` already implements; standalone mode requires an exact write, as its docstring
says.

Handing this to sqlglot means the nested `{db: {schema: {table: {col: type}}}}` form, which
`MappingSchema` accepts and `qualify` resolves against a table node's own parts. `ensure_schema`
takes it unchanged.

Every producer of these keys is being rewritten for step 3 anyway; retrofitting later means
touching all of it twice. Closes `type_check_plan.md` ❓Q7.

### D2 — `declaration_is_partial`, declared in dbt's `meta:`

A declaration carrying columns is read as the complete list **by default**, and a declaration may
say otherwise. Three states, all expressible in step 3 with no new machinery:

| declaration | `known` | `open_origins` | reading an unlisted column |
|---|---|---|---|
| none | `{}` | `(this table,)` | fabricate a slot, infer its type |
| **complete** (default) | declared names | `()` | **error**, pointing at the declaration |
| **partial** | declared names | `(this table,)` | fabricate a slot, infer its type |

Partial is not a third code path. Its declared columns are known *and typed*; everything else
falls through exactly as an undeclared table does. The flag chooses one thing only: whether
`open_origins` is empty.

```yaml
sources:
  - name: mysource
    database: mydatabase
    schema: myschema
    meta:
      declaration_is_partial: true       # applies to every table below
    tables:
      - name: raw_address
        meta:
          declaration_is_partial: true   # or just this one
        columns:
          - name: person_id
            data_type: varchar(20)
```

**Why `meta:` and not a top-level key.** The end goal is that a dbt project's own yml is read
unchanged. dbt validates the keys it knows and rejects unknown ones at the schema level, but
`meta:` is free-form by design and dbt carries any key through it untouched. A sqlr setting
written there leaves the file a valid dbt file, and `dbt parse` stays green. Every sqlr-specific
setting that ever attaches to a declaration goes in the same place, for the same reason.

**Inheritance is dbt's, not ours.** dbt already defines `meta:` on a source as inherited by its
tables, with the table's own `meta:` winning. Adopt exactly that — a whole partially-documented
source is then one line, and no project-wide sqlr config knob is needed to get there. Reading
these two levels and merging them is the loader's whole job for this feature.

**The field.** `declaration_is_partial: bool = False` on `DeclaredRelation`, so
`DeclaredSourceTable` and `DeclaredModel` both carry it. Default `False` — **completeness is the
entire payoff for declaring.** A complete declaration turns a misspelled column into row 5 of the
step-4 table, an error naming the yml entry and its line via `DeclaredRelation.where`, instead of
a silently fabricated slot nothing can distinguish from a real column. Defaulting to partial
would mean declaring a table buys types and nothing else.

Two consequences worth stating:

- **A partial table makes every star over it a lower bound**, same as an undeclared one. Correct,
  and it is what D3 reports.
- **The flag also gates the `models:` comparison.** A model declared partial is one where
  "projected column not declared" is not a finding, while "declared column not projected" still
  is. Same field, same reading — the declaration either enumerates the relation or does not.

### D3 — the fabricated column set is reported when it can mislead

*Fabricated* means the `UNKNOWN` slots step 5 invents on an open relation from the columns the
file reads — the 7 on `raw_address`. Nobody declared them; they exist so `qualify` does not
raise, and they are evidence of what this file **uses**, never a claim about what the table
**has**.

The risk is that `select *` expands against them: the star yields the 7 columns this file
mentions, and the real table may have 30.

**Decided:** INFO log at step 5, always — `raw_address: 0 declared, 7 inferred`.
A `projection-incomplete` warning only when the lower bound reaches a model's own projection,
which is the one place it can produce a wrong answer (`validate-schema` comparing an
under-approximated projection against a complete `models:` declaration). `address.sql` triggers
the log and not the warning: `current_address` enumerates, so the model's projection is exact.

---

## Tests to write first

Fixtures live with the tests; nothing reads from `examples/`.

1. **The `address.sql` shape.** CTE with `select *, f(x)` over an undeclared table; a downstream
   CTE names four of its columns. Assert all four land on the storage table, and `rn` does not.
2. **Two hops.** `a` selects `*` from `b`, `b` selects `*` from undeclared `t`. Column read off
   `a` lands on `t`.
3. **Closed relation, missing name.** CTE enumerating its projections; read a name it does not
   project. Assert an unresolvable finding naming the CTE and its projection list — not a
   `qualify` `OptimizeError`.
4. **Star over a join of two undeclared tables.** `error` → ambiguous; `guess` → first origin
   plus a `GuessedColumn`. Both settings asserted.
5. **Declared table is closed by default.** Reading an unlisted column of a declared table is an
   error naming the yml entry and its line — not a fabricated slot.
6. **`declaration_is_partial: true` reopens it.** Same fixture, flag flipped: the unlisted column
   fabricates a slot and gets a type from inference, while the listed columns keep their declared
   types. Assert both halves — a partial table that lost its declared types would pass a naive
   version of this test.
7. **`meta:` inheritance.** Source-level `declaration_is_partial: true` reaches every table; a
   table setting `false` overrides it. Both directions.
8. **The yml stays valid dbt.** A fixture carrying the setting parses under the dbt-mode reader
   with no unknown-key warning, because it is under `meta:`.
9. **Set-op arms intersect.** A name projected by one arm only is not in the branch's `known`.
10. **Lower-bound projection.** `select * from undeclared_t` as a model's whole body: assert the
    projection is marked incomplete and `validate-schema` reports no missing columns.
11. **Monotone widening** — `widen(s, facts) == s` when `s` has no UNKNOWN.
12. **Skip equivalence** — phase D forced on vs skipped on a fully-declared schema; identical
    types *and* identical findings.
13. **Three-valued discipline** — an undeclared column into a catalogued function produces zero
    findings.
14. **Conflicting facts** — error at every site, column stays UNKNOWN.
15. **Relation identity** — two sources named `raw_address` in different schemas, each declaring
    different columns; assert no collision. Plus `MySchema.Orders` declared against
    `myschema.orders` written, both directions. Locks in D1.
