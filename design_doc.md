# sqlr — High-Level Design

A CLI that runs an individual CTE (or a whole `.sql` file) locally in DuckDB against
automatically generated, deterministic, human-inspectable fixture data.

Later, this same core becomes the backend for a VS Code extension. Nothing in this
design should assume a CLI is the only caller.

---

## Design principles

These fall out of decisions already made, and should be used to settle ambiguity later.

1. **Legibility over realism.** Fixtures are 10–100 rows meant to be read by a human.
   Patterned, sequential, obviously-related values beat statistically faithful ones.
2. **Zero mandatory metadata.** The tool must produce something useful on a repo with
   no dbt tests, no documented types, and no config file. Everything is inferred first,
   then overridable.
3. **Inference is emitted, not hidden.** Anything guessed is written to a file the user
   can review and correct. Reviewing is cheap; authoring is not.
4. **Deterministic at column granularity.** Changing one column changes exactly one
   column's values.
5. **Everything explains itself.** Any resolved value — a type, a relationship, a
   generator choice — carries provenance and can be printed back.
6. **No ORM.** Fixtures are Arrow/parquet. DuckDB reads them directly.

---

## Pipeline overview

```
config ──┐
         ├─> project catalog ──┬─> sql analysis ──┬──> schema resolution ──┐
         │                     │                  ├──> relationship inference ──> constraint set
         │                     │                  └──> constraint extraction ──┘
         │                     │                                               │
         │                     └─> declared schemas ──┐                        │
         │                                            v                        │
         │                                        diagnostics <────────────────┤
         │                                                                     │
         │                                                                     v
         │                                              generation planner (DAG + fingerprints)
         │                                                                     │
         │                                          ┌──────────────────────────┤
         │                                          v                          v
         │                                     providers ──> scenarios ──> fixture store
         │                                                                     │
         │                                                                     v
         └──────────────────────────────────────────────────> execution ──> output
```

`diagnostics` sits deliberately off the main line. It is a sink, not a stage: everything
upstream of generation writes findings into it, and nothing downstream depends on it.

## Built so far

`config`, `catalog`, `source`, `typemap`, `sql_analysis`, `schema_resolution`, `declared`,
`diagnostics`, and a thin `cli`. Everything from `relationship_inference` onwards is still
design.

The **Basic / Extended** split below is a scoping tier, not a status: plenty of Basic
bullets in the unbuilt modules do not exist yet, and a few Extended ones in the built
modules already do.

---

# Layer 1 — Foundation

## Module: `source`

Character ranges, and the text they index into. Depends on nothing; everything that
reports anything depends on it.

Design principle 5 — *everything explains itself* — is only worth anything if an
explanation can point somewhere. A resolved type that says "numeric" is weak next to one
that says "numeric, because of `amount > 20`, at line 14, column 9". That pointer is a
`SourceSpan`, and it has to be attachable to any fact any module derives.

**Basic**
- `SourceSpan`: half-open character range plus 0-based line/column at both ends.
- `SourceDoc`: the text plus its path; slices snippets on demand.
- `Positions`: a per-document index resolving a sqlglot expression to a span.

**Design notes**
- sqlglot records positions only on **leaf tokens** — `Identifier`, `Literal`, and the
  name token of some functions. `Column`, `EQ`, `Where`, `Select` carry nothing. The span
  of a composite expression is therefore the hull of the positioned leaves beneath it.
- sqlglot's own `line`/`col` are ignored. `col` is 1-based *and points at the token's last
  character*, which no editor wants. Both are recomputed from character offsets, 0-based,
  so a span converts to an LSP `Range` without arithmetic.
- Offsets are absolute across a whole file, so one index per document stays valid across
  every statement in it. `.copy()` and `qualify_tables()` both preserve the metadata.
- **Two limits follow from the hull approach, and callers must expect them.** Bare
  keywords are invisible: `shipped_on is null` spans only `shipped_on`, because no token
  is emitted for `IS NULL`. Punctuation is invisible too, but that one is repaired —
  an unbalanced hull is grown back over the brackets its leaves left behind, so
  `x in ('a','b')` does not lose its closing paren.
- **Every span is optional.** Anything sqlglot synthesised rather than parsed — an alias
  invented by `qualify_tables` — has no location at all. A missing span is normal, not
  exceptional, and no renderer may assume one.

**Extended**
- UTF-16 column offsets, which is what LSP actually specifies; the current 0-based
  character columns diverge on non-BMP characters.
- Span arithmetic (contains, overlaps) for mapping an editor cursor back to a fact.

## Module: `typemap`

The type lattice, and the mapping of written type names onto it. Shared by
`schema_resolution` (which reads type names out of casts) and `declared` (which reads them
out of yml), because the two have to agree — a `cast(x as varchar)` and a
`data_type: varchar(50)` must land on the same type or every comparison between inference
and declaration is noise.

**Basic**
- `ResolvedType`: concrete types plus the families `number` and `numeric` (see
  `schema_resolution` for why families exist).
- `TYPE_COVER` / `widen`: the lattice, and the least upper bound of two types.
- `compatible`: whether any concrete type satisfies both — the predicate a divergence
  check is built on.
- `normalize_type_name` / `resolve_type_name`: strip parameters and dialect spelling from
  a written name, so `varchar(50)`, `NUMBER(38,0)` and `timestamp without time zone` all
  resolve.

**Extended**
- Precision and scale, which are currently discarded — a generator eventually wants them.
- Dialect-specific type sets rather than one union of every warehouse's spelling.

## Module: `config`

Loads and merges layered configuration; owns the project root.

**Basic**
- Resolve project root: explicit `--project` flag, else path from a config file, else
  nearest ancestor containing a marker (`.git`, `dbt_project.yml`, `sqlr.yml`).
- Load a single `sqlr.yml`, validate with Pydantic, produce a typed config object.
- Accept an explicit config path from the CLI.
- Basic settings: default row count, seed, fixture directory, sql file globs.

**Extended**
- Full layer chain: built-in defaults → project config → per-table → per-column →
  per-file overrides → CLI flags, with first-match-wins resolution.
- Name-pattern rule sets (`*_id`, `*_at`, `is_*`) as user-editable config.
- `${VAR}` environment interpolation so paths and credentials aren't per-developer.
- Config versioning + migration on schema changes.
- Emit JSON Schema for editor autocomplete in the yml files.
- Provenance on every resolved setting (which layer supplied it).

## Module: `catalog` (project discovery)

Knows what files exist and what kind they are. Pure filesystem knowledge, no parsing.

**Basic**
- Walk the project root, find `.sql` files honoring include/exclude globs and `.gitignore`.
- Detect a dbt project (`dbt_project.yml`) and locate `models/`, `seeds/`, `target/`.
- Find `*.yml` / `*.yaml` files anywhere in the project. Deciding which of them *mean*
  anything is not this module's job — see `declared`.
- Return a file inventory with mtimes and content hashes.

**Extended**
- Parse dbt artifacts (`manifest.json`, `catalog.json`) via `dbt-artifacts-parser`.
- Distinguish models / sources / seeds / snapshots.
- Multi-project or monorepo support (several roots).
- Incremental rescan driven by mtime + hash, so repeat runs skip unchanged files.
- Watch mode for the eventual extension backend.

---

# Layer 2 — Understanding the SQL

## Module: `sql_analysis`

The single source of truth for anything derived from SQL text. Wraps `sqlglot`.
Everything downstream consumes its output; nothing else parses SQL.

**Basic**
- Parse a `.sql` file into an AST; tolerate and report syntax errors without crashing.
- Extract the CTE list: name, body range, character offsets in the source.
- Build the intra-file CTE dependency graph.
- Classify every table reference as *internal* (another CTE) or *external* (a real source).
- Collect column references per table.

**Every fact is located.** Usage, predicates, joins, nullability and cardinality each
carry two spans:

- `span` — the column reference, for naming the column;
- `context_span` — the expression that *constitutes* the evidence.

The distinction matters more than it looks. `amount > 20` is the evidence; `amount` alone
is not. A diagnostic that underlines the bare column name has thrown away the reason it
had something to say. Predicates additionally locate each literal, and every source
column keeps the span of every place the statement mentions it.

**Extended**
- Column-level lineage (which source column feeds which output column).
- Extract join predicates — the raw material for relationship inference.
- Extract filter predicates, comparison literals, and `case`/`in` values — the raw
  material for constraint extraction.
- Dialect detection and transpilation (author in Snowflake, execute in DuckDB).
- Harvest `-- @directive` comments for inline configuration.
- Incremental / debounced reparse for the extension backend.
- Handle macros and Jinja: either compile via dbt, or stub `{{ ref() }}` / `{{ source() }}`
  well enough to parse.

## Module: `schema_resolution`

Produces a resolved schema per external table: the definitive column list and types.

**Basic**
- Union three inputs: dbt yml columns, dbt catalog types (if artifacts exist), and
  columns discovered by `sql_analysis`.
- Infer types from usage when undeclared (`amount > 100` → numeric,
  `date_trunc('day', x)` → temporal, `x in ('a','b')` → short string).
- Infer types from name patterns as a fallback (`*_id`, `*_at`, `is_*`).
- Emit a warning per inferred column rather than guessing silently.
- Handle raw sources with no metadata at all — SQL-derived schema only.
- Nullability inference from `is null` / `coalesce` usage.

### Nothing is discarded

The governing rule of this module. Resolution picks one winner among competing pieces of
type evidence, but **the losers survive**, each with the range that produced it.

Collection and choice are therefore separate concerns: collection gathers everything a
column's usage implies and throws none of it away; choice decides which of it wins. This
is what lets a divergence report say more than "expected numeric" — it can point at the
comparison that proves it, list the two other places that agree, and underline the yml
line that disagrees. A single winning type cannot do any of that.

Three consequences worth stating, because each was a real loss before:

- **Join-group unification appends, it does not overwrite.** When `address.person_id`
  inherits `person.id`'s type, the inherited type arrives as an *additional* piece of
  evidence. Erasing the member's own evidence would leave nothing able to explain why the
  group settled where it did.
- **Name patterns are collected even when they lose.** At the lowest weight they change no
  outcome, so this costs nothing — but a user reading a report gets to see that the
  column's name agreed, or did not.
- **Constraints keep their literals, their kinds and their positions.** `where col > 20`
  has to survive as *"numeric, and the generated data must straddle 20, and the 20 was
  written here"*. Collapsing it to `numeric` throws away two thirds of what it said, and
  the fixture layer needs the other two thirds.

Type evidence is weighted, and the heaviest wins. Evidence that does not name a type
resolves to a **family** rather than a guess: `x * 12` and `x > 5` are equally true of an
INT, a DECIMAL and a DOUBLE, so they yield `numeric` rather than a coin flip. Only
evidence that names a type — a cast, a function with a fixed return type — yields a
concrete one. Equally-weighted evidence that disagrees is widened together rather than
arbitrated, and what was widened is recorded.

**Extended**
- Confidence scoring per column, with a threshold that triggers user prompts.
- Emit a reviewable `schema.inferred.yml` so users can correct types once.
- Warehouse `information_schema` as an authoritative source when a connection exists.
- Struct/array/nested type support.
- Evidence weights as configuration rather than constants, once there is field data on
  which of them are actually load-bearing.

## Module: `declared`

Loads the types the user wrote down, as opposed to the types the analyser inferred.

The format is **dbt's property yml, deliberately**. A project that already has such files
gets checked with no extra authoring, and a project with no dbt at all can write the same
thing. Only two fields are required of a column — `name` and `data_type` — so the cost of
adopting it is close to zero.

**Which key is read depends on the project.** The presence of a `dbt_project.yml` at the
project root is the whole test.

*Standalone* — no `dbt_project.yml`. There are no models, only sources: a project with no
dbt has no `ref()`/`source()` distinction to inherit, so every relation the SQL reads is a
table someone has to describe, and the ones this project builds say so with `sql_file:`.

```yml
sources:
  - name: mysource
    database: mydatabase
    schema: myschema
    tables:
      - name: raw_department      # read as mydatabase.myschema.raw_department
        columns:
          - name: department_id
            data_type: string
      - name: employee
        sql_file: employee        # ...and this one is built by employee.sql
```

The parts a table writes down **are** the relation, exactly: `database` and `schema` appear
in the name only when they are given, so a table declaring neither is written bare and one
declaring both has to be written out in full — including by the project's own files, when
the table it names is one of theirs. Nothing is defaulted in, because the value dbt would
take from a target profile is not knowable here and a guess would silently fail to match
the name the user actually wrote.

*dbt* — `models:` comes back, matched to a `.sql` file by stem, and a source's missing
parts are the ones dbt would fill from a profile, so they match anything. Reading dbt's own
`ref()`/`source()` templating is not implemented; this mode is currently the earlier
behaviour, kept working.

**Basic**
- Discover every yml in the project (via `catalog`) and parse the ones with a top-level
  `sources:` or `models:` key. Those keys are the discriminator: a yml with neither is not
  ours and is skipped in silence, not warned about.
- Map each `data_type` onto the lattice via `typemap`; an unrecognised name resolves to
  `unknown` and is reported rather than quietly ignored.
- Tolerate malformed yml. A broken file is a problem with that file, not a reason to stop
  analysing the SQL it was meant to describe.

**A relation described twice is fatal, not a warning that picks a winner.** dbt rejects the
same thing, and when two entries give one column two types there is no answer to choose.
Sources collide on the *relation they resolve to* rather than on their `source.table`
names, so two sources may both have a `raw_department` as long as they land in different
schemas. The same goes for two entries claiming one `sql_file:`, and for a column described
twice inside one entry.

**An unmatched relation is not fatal** — its columns simply have nothing to check against,
which is what adopting sqlr on an existing project looks like on day one. But a reference
whose table name matches a declaration that *nothing else in the run uses* is almost always
that declaration written short, so `near_miss_warnings` says which name to write instead of
leaving the user to work out that the declaration they can see is not the one being
applied. `models:` outside a dbt project gets the same treatment: ignored, but warned about
by name when an entry would have described a real `.sql` file, with the reason it was
ignored.

**Discovery is decoupled from interpretation.** `DeclarationProvider` is the seam: this
module implements it over yml files, and a dbt `manifest.json` / `catalog.json` reader can
implement it later without a single consumer changing. That separation is the whole point
of the module existing rather than the yml parsing living inside `schema_resolution`.

**Declarations are located too.** Parsing goes through `yaml.compose` rather than
`yaml.safe_load`, because the composed node tree carries source marks and the plain loader
throws them away. Without them a report could say a declaration was contradicted but not
show where the declaration was written — which is half the information the user needs.

**Extended**
- dbt `manifest.json` / `catalog.json` as a second provider, ranked above hand-written yml.
- Real dbt-project support: `ref()`/`source()` templating, and a target profile for the
  parts a source leaves out.
- Column-level tests (`accepted_values`, `not_null`) as constraint input, feeding
  `constraints` rather than `diagnostics`.
- Per-column `description` surfaced into generated fixture documentation.

## Module: `relationship_inference`

Builds the FK graph with as little user input as possible. Critical, since dbt tests
cannot be assumed.

**Basic**
- Harvest equi-join predicates across *all* `.sql` files in the project.
- Aggregate by column pair; count occurrences as evidence.
- Determine parent/child direction via heuristics (name match against table name,
  presence of a uniqueness signal, `group by` usage).
- Name-convention fallback for tables never joined in visible SQL
  (`<singular>_id` → `<table>.<pk>`), using `inflect` for pluralization.
- Emit `.sqlr/relationships.yml` with each relationship, its confidence, and its
  evidence; merge user edits back on subsequent runs and never overwrite them.

**Extended**
- **Conflict arbitration submodule** — deliberately separate, because the rules will be
  refined empirically. Handles: one child column joined to multiple parents, name
  convention contradicting an observed join, cycles in the graph. Should be a pluggable
  strategy with a rule set, logging every decision so rules can be tuned against real repos.
- Consume dbt `relationships` tests when present, as highest-confidence evidence.
- Composite / multi-column keys.
- Warehouse constraint metadata where declared.
- Cardinality inference (one-to-many vs many-to-many) from join shape.
- Self-referential and hierarchical relationships.

## Module: `constraints`

Merges everything known about *valid values* for each column into one constraint set.
Separate from `schema_resolution` because it merges from more sources and feeds
generation directly.

**Basic**
- Enum recovery from SQL: literals compared against a column
  (`where status in ('shipped','pending')`, `case when tier = 'gold'`).
- Range constraints from comparison predicates on dates and numbers.
- Null requirements from `is null` / `is not null` filters.
- Merge with user-declared constraints from config; user wins.

**The raw material is already there.** `schema_resolution` emits a `ValueConstraint` per
predicate carrying the operator, the literal values, their kind, and the position of each
— so `where col > 20` arrives as *"numeric, straddle 20, written here"* rather than as
`numeric`. Two identical predicates written in two places stay two constraints, because
they are two hints about the data and two places to point at. What is missing is the
merging and the satisfiability reasoning, not the extraction.

**Extended**
- Ingest dbt tests when present (`accepted_values`, `not_null`, `unique`,
  `dbt_utils`/`dbt_expectations` variants).
- Cross-column constraints (`end_date > start_date`).
- Multi-predicate satisfiability — guarantee some rows satisfy a whole conjunction,
  not just each predicate independently.
- Regex / format constraints.
- Constraint conflict detection ("this filter can never match").

---

# Layer 3 — Fixture generation

## Module: `generation_planner`

Decides what to generate, in what order, and what is stale. The brain of the fixture layer.

**Basic**
- Build the generation DAG from the relationship graph; topologically sort so parents
  precede children.
- Assign row counts and pool cardinalities per table, with legibility-oriented defaults
  (small, deliberately uneven, so at least one parent has several children, one has
  exactly one, and one has zero).
- Compute per-column fingerprints:
  `hash(seed, table, column, type, column_config, generator_version, parent_fp, sibling_fps)`.
- Compare against the stored manifest; produce a work list of stale columns only.
- **Row count is not a fingerprint input** — column generators are deterministic infinite
  sequences and materialize the first N, so growing the sample appends rows and leaves
  existing values byte-identical.

**Extended**
- Downstream staleness cascade: regenerating a parent key invalidates every FK column
  drawing from it, and anything derived from those.
- Cardinality derived from observed join shape rather than defaults.
- `renamed_from:` hints to preserve values across column renames.
- Cross-file fixture sharing and a global fixture namespace.
- Parallel generation of independent DAG branches.

## Module: `providers` (generators)

Produces column values. Registry of named generators plus the RNG discipline.

**Basic**
- Per-column RNG seeded from that column's fingerprint — never a shared global stream.
- Built-in providers by type: integer, decimal, string, date, timestamp, boolean.
- **Legible defaults**: sequential keys (`cust_001`), consecutive dates from a fixed
  anchor, round-ish amounts in a narrow band, alternating booleans.
- FK provider: sample from the parent's realized key pool rather than generating fresh.
- Deliberate edge-case placement rather than random draws — one null per nullable column,
  one boundary value per range, one empty string, one max-length string.
- Constraint satisfaction: pull from enum sets, straddle range boundaries.

**Extended**
- Row-wise correlation pass after the column pass. Correlated columns must *declare*
  which siblings they read; the declaration feeds the fingerprint. Undeclared reads
  are not permitted.
- Python provider registry via decorator, loaded from a conventional
  `sqlr_providers.py`, receiving RNG, row count, and already-generated siblings.
- Faker / Mimesis providers as opt-in for columns where realism is visible.
- Semantic providers keyed on column name (email, address, currency code).
- Distribution controls (skew, lognormal amounts) for when realism matters more than legibility.
- Locale support.

## Module: `scenarios`

Named, composable perturbations applied on top of clean generated data. This is the
layer that makes the tool a testing harness rather than a fixture filler.

**Basic**
- `clean` (no-op) and `dupes` (duplicate rows on a key).
- Selectable per run via `--scenario`.
- Applied after base generation, before writing.

**Extended**
- Library: orphaned FKs, extra nulls, out-of-range values, late-arriving records,
  empty table, single-row table, unicode and quote injection in strings, type-edge values.
- Composable stacking (`--scenario dupes+nulls`).
- Scenario is a fingerprint input, so each scenario caches independently.
- User-defined scenarios in config.
- Scenario matrix runs — execute a CTE against every scenario and diff the outputs.

## Module: `fixture_store`

Persistence and caching. Column-granular.

**Basic**
- Write `<fixture_dir>/<table>.parquet` plus `<table>.manifest.json` holding per-column
  fingerprints and metadata.
- Read existing fixtures on startup; hand the planner the current manifest.
- Splice regenerated columns into an existing Arrow table and rewrite the parquet
  (cheap, columnar — this is why there is no ORM).
- Expose key pools to the providers module for FK sampling.

**Extended**
- A thin `FixtureSink` interface, so a future real-data sampler, or a Postgres/Snowflake
  target, plugs in without touching caching, relationships, or scenarios. FK pool
  sampling must be able to consume a pool it did not generate.
- Fixture set versioning; commit a set to git and share it across a team.
- Garbage collection of orphaned fixtures.
- Import/export of fixture bundles.
- Optional real-data sampling provider (`sample://warehouse/orders`).

---

# Layer 4 — Running

## Module: `execution`

Assembles and runs the query in DuckDB.

**Basic**
- Register a DuckDB view per source table over `read_parquet()` of its fixture.
- Given a target CTE, topologically sort its transitive CTE dependencies and emit
  `WITH <deps> SELECT * FROM <target> LIMIT n`.
- Run the whole file as an alternative target.
- Return results as an Arrow table plus timing and row count.

**Extended**
- Materialize upstream CTEs as temp tables, invalidated by hashing each CTE's text.
- Query cancellation and timeouts.
- Run against real sources instead of fixtures (`--live`), reusing the same planner.
- Load `httpfs` for direct cloud reads.
- Explain-plan capture.
- Run dbt tests against the fixtures as a self-check that generation honored constraints.

## Module: `output`

Presents results and fixtures.

**Basic**
- Pretty table to the terminal (`rich`), with column types and row count.
- `--format json|csv` for piping.
- Truncation notice when limited.

**Extended**
- Write results to a file (parquet/csv/json).
- Diff two runs (useful for scenario matrices and for regression checks).
- Column statistics summary.
- Arrow-table handoff for the extension's virtualized grid, with windowed row fetch so
  full results never cross the process boundary.

---

# Layer 5 — Surface

## Module: `diagnostics`

Provenance, warnings, and the explain surface. Cross-cutting; every other module
writes into it. One channel, one renderer, one shape.

A `Diagnostic` is deliberately more than a string. It carries a **primary location** to
underline and a list of **related locations**, because the findings worth reporting are
the ones with several sites: *"this is declared varchar but used as a number"* is only
actionable when it can show the declaration and every contradicting use at once. The shape
is chosen to convert to an LSP diagnostic without loss.

**Basic**
- Structured finding collection with severity, a stable code, a location and related
  locations.
- Adapters folding the analyser's loose `errors` / `warnings` / ambiguities into the same
  channel, so nothing formats findings by hand.
- Text rendering with a caret underline for the terminal.
- `to_lsp` — 0-based LSP `Diagnostic` objects with `relatedInformation`.
- Human-readable error messages for the common failures.

### Type reconciliation

The first real consumer, and the reason the module exists now rather than later. Two
comparisons share one rule set:

- the statement's **projection** against the declaration for the `.sql` file itself — the
  source table that claims it with `sql_file:`, or a dbt project's `models:` entry;
- each **relation it reads** against the declaration of that name — which catches the
  interesting case, where `orders.sql` declares `revenue` a varchar and
  `revenue_report.sql` writes `where revenue > 1000`.

Rules that keep it quiet enough to be trusted:

- A mismatch is reported **only when declared and inferred types have no concrete type in
  common**. Declaring `integer` where `number` was inferred is not a disagreement:
  inference widens to a family when the evidence does not distinguish, and the user's
  declaration is the more specific of the two. Getting this wrong makes the tool cry wolf
  on correct code, which is how a checker gets switched off.
- **Severity scales with the weight of the winning evidence.** A cast contradicting a
  declaration is an error — a cast names a type outright. A usage pattern is a warning. A
  name pattern is a hint, because `*_at` losing to an explicit declaration is unremarkable.
- A `*` suppresses "undeclared column" reports: under a star the attributed column list is
  a subset of the real one, so absence from it proves nothing.
- "Declared but not produced" applies only to a file's **own** model. An upstream table
  declaring columns this file does not select is normal, and reporting it would bury the
  real findings.

**Extended**
- `explain <table>.<column>` — print resolved type, constraints, generator, seed, and
  why each was chosen. The evidence to do this already exists; only the command is missing.
- `explain relationships` — the full inferred graph with evidence.
- Map DuckDB binder errors back to source ranges, for the extension's diagnostics.
- Suggested fixes attached to findings — `data_type: numeric` as a one-click edit is the
  obvious first one, and the mismatch diagnostic already knows the type it would suggest.
- Suppression: a `# noqa`-style directive keyed on the diagnostic code.
- Structured log output for the RPC transport.

## Module: `cli`

Command surface and dependency wiring. Deliberately thin.

**Basic** — `typer`-based.
- `run <file> [--cte NAME]` — the primary command.
- `--config`, `--project`, `--seed`, `--rows`, `--scenario`, `--format`.
- `--format text|json`, where `json` emits the LSP diagnostic shape.
- `--strict` — exit non-zero when any diagnostic is an error, so the check is usable in CI.
- `init` — write a starter config and run inference to produce `relationships.yml`.

**Extended**
- `inspect <table>` — show the generated fixture.
- `explain`, `regenerate [--table T] [--column C]`, `clean`.
- `scenarios list`.
- `test` — run dbt tests against fixtures.

## Module: `rpc` (later)

Not built initially, but the seam should exist from the start.

- JSON-RPC over stdio, one long-lived process per workspace.
- Methods mirroring the CLI plus document-oriented ones: analyze buffer → CTE ranges,
  run CTE → run id, fetch row window.
- Keeps parse state warm across calls.
- Result tables held server-side by run id.

---

## Suggested build order

A walking skeleton first, then depth:

1. ~~`config` + `catalog` — minimal: find the file, load a config.~~ **done**
2. ~~`sql_analysis` — CTE list and external table references only.~~ **done**, and rather
   more: full column lineage, located facts, predicate and join extraction.
3. ~~`schema_resolution` — SQL-derived columns with naive type inference.~~ **done**,
   with the evidence retained rather than collapsed.
4. `providers` + `fixture_store` — one legible generator per type, write parquet.
5. `execution` + `output` — register views, run the CTE, print a table.

That's end-to-end value with no relationships, no constraints, no scenarios, no caching.

`source`, `typemap`, `declared` and `diagnostics` were not in the original order. They
arrived together, pulled in by one requirement — checking inference against declared types
and reporting *where* they diverge — which turned out to need character ranges threaded
through every fact in the pipeline. Doing that early was the right trade: retrofitting
positions onto a fact model that had been built without them would have touched every
module twice.

Then, in rough order of payoff:

6. `relationship_inference` (join harvesting) + FK providers — the thing that makes joins
   return rows.
7. `constraints` (filter-aware generation) — the thing that stops results coming back empty.
8. `generation_planner` fingerprints + column-level caching.
9. `scenarios`.
10. `diagnostics` / `explain` — the collection surface exists; the `explain` commands do not.
11. `rpc` and the extension. `to_lsp` already produces the shape it needs, which is a
    deliberate hedge: it keeps the span work honest, because a range that does not slice
    back to the offending text shows up immediately there.

Steps 6 and 7 are the two that decide whether the tool is trusted or abandoned, and both
depend on predicate extraction in `sql_analysis` — worth building that extraction well
even though the walking skeleton doesn't need it. That extraction now also preserves the
literals and their positions, which is what step 7 will actually generate from.