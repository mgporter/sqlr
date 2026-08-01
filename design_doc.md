# sqlrunner — High-Level Design

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
         ├─> project catalog ──> sql analysis ──┬──> schema resolution ──┐
         │                                      ├──> relationship inference ──> constraint set
         │                                      └──> constraint extraction ──┘
         │                                                                    │
         │                                                                    v
         │                                              generation planner (DAG + fingerprints)
         │                                                                    │
         │                                          ┌─────────────────────────┤
         │                                          v                         v
         │                                     providers ──> scenarios ──> fixture store
         │                                                                    │
         │                                                                    v
         └──────────────────────────────────────────────────> execution ──> output
```

---

# Layer 1 — Foundation

## Module: `config`

Loads and merges layered configuration; owns the project root.

**Basic**
- Resolve project root: explicit `--project` flag, else path from a config file, else
  nearest ancestor containing a marker (`.git`, `dbt_project.yml`, `sqlrunner.yml`).
- Load a single `sqlrunner.yml`, validate with Pydantic, produce a typed config object.
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
- Find `schema.yml` / `*.yml` files adjacent to models.
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

**Extended**
- Confidence scoring per column, with a threshold that triggers user prompts.
- Type reconciliation when sources disagree (yml says string, usage implies date).
- Emit a reviewable `schema.inferred.yml` so users can correct types once.
- Warehouse `information_schema` as an authoritative source when a connection exists.
- Nullability inference from `is null` / `coalesce` usage.
- Struct/array/nested type support.

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
- Emit `.sqlrunner/relationships.yml` with each relationship, its confidence, and its
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
  `sqlrunner_providers.py`, receiving RNG, row count, and already-generated siblings.
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
writes into it.

**Basic**
- Structured warning collection (inferred type, inferred relationship, unresolved column,
  empty result).
- `explain <table>.<column>` — print resolved type, constraints, generator, seed, and
  why each was chosen.
- Human-readable error messages for the common failures.

**Extended**
- `explain relationships` — the full inferred graph with evidence.
- Map DuckDB binder errors back to source ranges (character offsets already exist from
  `sql_analysis`), for the extension's diagnostics.
- Suggested fixes attached to warnings.
- Structured log output for the RPC transport.

## Module: `cli`

Command surface and dependency wiring. Deliberately thin.

**Basic** — `typer`-based.
- `run <file> [--cte NAME]` — the primary command.
- `--config`, `--project`, `--seed`, `--rows`, `--scenario`, `--format`.
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

1. `config` + `catalog` — minimal: find the file, load a config.
2. `sql_analysis` — CTE list and external table references only.
3. `schema_resolution` — SQL-derived columns with naive type inference.
4. `providers` + `fixture_store` — one legible generator per type, write parquet.
5. `execution` + `output` — register views, run the CTE, print a table.

That's end-to-end value with no relationships, no constraints, no scenarios, no caching.

Then, in rough order of payoff:

6. `relationship_inference` (join harvesting) + FK providers — the thing that makes joins
   return rows.
7. `constraints` (filter-aware generation) — the thing that stops results coming back empty.
8. `generation_planner` fingerprints + column-level caching.
9. `scenarios`.
10. `diagnostics` / `explain`.
11. `rpc` and the extension.

Steps 6 and 7 are the two that decide whether the tool is trusted or abandoned, and both
depend on predicate extraction in `sql_analysis` — worth building that extraction well
even though the walking skeleton doesn't need it.