# Refactor plan — evidence-carrying schema resolution

Goal: `schema_resolution` stops being a lossy summary of `sql_analysis` and becomes an
*indexed, source-located* view of it. Three consumers drive the requirements:

1. **Inference** — pick a type from the highest-weighted evidence (what it does today).
2. **Divergence checking** — compare inference against user-declared types and emit
   diagnostics carrying file, line, column range, and snippet, for a red squiggle.
3. **Fixture generation** — `where col > 20` must survive as *"numeric, and the data needs
   values either side of 20"*, not collapse into `numeric`.

All three need the same thing: **every piece of evidence, never just the winner, each
pinned to a source range.**

---

## Current losses

| Information | Where it exists | Where it dies |
|---|---|---|
| Source positions | Only on sqlglot leaf `Identifier`/`Literal` nodes | Never read, except `Ambiguity.line` via `line_of()` |
| File path | `analyze_file` argument | Not stored on `SqlAnalysisResult` |
| Losing type evidence | `_initial_evidence` loops all usages | Keeps one `_TypeEvidence`, discards the rest |
| Member evidence in a join group | `_unify_join_groups` | **Overwrites** `evidence[node]` with a `join_group` entry |
| Predicate `literal_kind` | `PredicateFact.literal_kind` | Dropped by `_constraints()` |
| Losing nullability facts | `nullability_by_node` | `_nullability()` returns one bool |
| Projection evidence | `_build_projection` | First non-unknown origin wins, rest dropped |
| CTE body ranges | Nowhere — design doc Basic item, never built | — |

---

## Part 0 — `src/sqlr/source.py` (new, foundation)

Everything else depends on this. Ships alone, testable alone.

```python
class SourceSpan(BaseModel, frozen=True):
    start: int        # absolute char offset, inclusive
    end: int          # absolute char offset, EXCLUSIVE
    start_line: int   # 0-based
    start_col: int    # 0-based
    end_line: int
    end_col: int

class SourceDoc(BaseModel):
    path: Path | None
    text: str
    def slice(self, span) -> str
    def line_text(self, span) -> str      # full lines the span touches, for a CLI caret
```

`Positions` — built once per `analyze_sql` call, holds the text and a precomputed list of
line-start offsets:

```python
class Positions:
    def span_of(self, expression: exp.Expr) -> SourceSpan | None
    def span_covering(self, expressions: Iterable[exp.Expr]) -> SourceSpan | None
```

Implementation notes, all verified against sqlglot 30.14 in this repo:

- Only leaf `Identifier`, `Literal`, and some function tokens carry
  `meta = {line, col, start, end}`. `Column`, `EQ`, `Where`, `Select` carry **nothing**.
  So `span_of` walks descendants, collects every `meta["start"]`/`meta["end"]`, and returns
  `min(start)` / `max(end) + 1`. This is what makes `a.amount > 20` spannable.
- sqlglot's `end` is **inclusive**; convert to exclusive once, here.
- **Ignore sqlglot's `line`/`col`.** `col` is 1-based *and points at the token's last
  character*, which is not what any editor wants. Derive `line`/`col` from `start`/`end`
  by bisect over the line-start index, 0-based, so the output is already LSP-shaped.
- Offsets are absolute across the whole file even for multi-statement parses (verified),
  so one `Positions` per file is correct.
- `.copy()` and `qualify_tables()` preserve `meta` (verified). Analysis can keep running
  on the qualified copy.
- Nodes synthesized by sqlglot (aliases invented by `qualify_tables`) have no meta at all.
  **Every span field is `SourceSpan | None`**, and degradation must be graceful, not an
  assertion.
- **Do not put spans on `ColumnNode`.** It is `frozen=True` and used as a dict key
  throughout; a span would fragment the identity. Spans attach to *facts and origins*.
- Snippets are **not** stored on models — they are derived on demand from `SourceDoc`.
  Storing them would multiply the JSON size of a result by several times.

---

## Part 1 — `sql_analysis` carries spans and identity

### Signature changes

```python
def analyze_sql(sql, *, path: Path | None = None, dialect=None, ...) -> SqlAnalysisResult
def analyze_file(path, ...)   # passes path through
```

`SqlAnalysisResult` gains `source: SourceDoc`. `Positions` is constructed in
`sql_analysis/__init__.py` and threaded into `Resolver`, `extract_facts`, `assemble`.

### Model changes (`sql_analysis/types.py`)

| Model | Added |
|---|---|
| `ColumnOrigin` | `span: SourceSpan \| None` — where this reference was written |
| `UsageFact` | `span` (the column ref), `context_span` (the enclosing expression) |
| `PredicateFact` | `span`, `context_span`, `value_spans: list[SourceSpan \| None]` |
| `JoinFact` | `left_span`, `right_span`, `context_span` (the ON equality) |
| `NullabilityFact` | `span`, `context_span` |
| `CardinalityFact` | `spans: list[SourceSpan \| None]` parallel to `nodes` |
| `Ambiguity` | `span` **replacing** `line: int \| None` |
| `OutputColumn` / `ProjectedColumn` | `span` (whole select item), `alias_span` |
| `SourceColumn` | `references: list[SourceSpan]` — *every* place the column appears |
| `Relation` | `span` (body range), `name_span` — closes the design doc's unbuilt "CTE body range / character offsets" Basic item |

`context_span` is the payload requirement 2 actually squiggles: for `where amount > 20`
the squiggle wants `amount > 20`, not just `amount`.

### Code changes

- `facts.py`: `_FactExtractor` already holds the `exp.Column` at every fact site — pass it
  to `positions.span_of()` and its `column.parent` to get `context_span`. `_predicate()`
  already isolates the literal nodes; return them so `value_spans` can be filled.
- `resolver.py`: `Resolution.origins` currently synthesizes `ColumnOrigin` with no
  expression. Add an optional `span` to `Resolution`, set by `resolve_column_expr` from the
  `exp.Column` it was handed, so origins are locatable. Origins produced by
  `resolver.resolve(ref, name)` inside `assemble` take the output column's span instead.
  Replace `line_of()` and `_star_line()` with `Positions`; `StarOverJoinAbort` takes a span.
- `relations.py`: record `OutputCol.expression` spans; delete `line_of()`.
- `assemble.py`: `_build_sources` currently dedupes columns by name keeping max
  confidence — extend it to also **accumulate** every origin span into
  `SourceColumn.references`.
- `_merge()` in `__init__.py`: merge `references` lists, keep max confidence as today.
- Delete the stray `print(facts.model_dump_json(...))` at
  `sql_analysis/__init__.py:104`.

---

## Part 2 — `schema_resolution` keeps every piece of evidence

### New public model (`schema_resolution/types.py`)

```python
EvidenceKind = Literal[
    "usage", "predicate", "cast", "function",
    "literal_argument", "name_pattern", "join_group",
]

class TypeEvidence(BaseModel):
    node: ColumnNode
    resolved_type: ResolvedType
    weight: int
    source: TypeSource
    kind: EvidenceKind
    detail: str | None                  # "compared_to_number", "cast to DECIMAL", ...
    span: SourceSpan | None
    context_span: SourceSpan | None
    via: ColumnNode | None = None       # join_group: the member it was inherited from
```

`_TypeEvidence` (the private dataclass) is **deleted** — it becomes this.

```python
class ValueConstraint(BaseModel):
    operator: PredicateOperator
    values: list[str] = []
    literal_kind: LiteralKind | None      # restored
    span: SourceSpan | None
    context_span: SourceSpan | None
    value_spans: list[SourceSpan | None] = []

class NullabilityResolution(BaseModel):
    nullable: bool | None
    chosen: NullabilityFact | None
    facts: list[NullabilityFact] = []     # all of them, including outer_join_padded

class ColumnSchema(BaseModel):
    name: str
    resolved_type: ResolvedType
    chosen: TypeEvidence | None           # the winner
    evidence: list[TypeEvidence] = []     # ALL of it, weight-descending
    widened_from: list[ResolvedType] = [] # set when an equal-weight tie was widened
    confidence: Confidence = "explicit"
    nullability: NullabilityResolution
    constraints: list[ValueConstraint] = []
    references: list[SourceSpan] = []
    join_group: int | None

class JoinGroup(BaseModel):              # replaces list[list[str]]
    members: list[ColumnNode]
    unified_type: ResolvedType
    facts: list[JoinFact] = []           # the joins that linked them, with spans

class StatementSchema(BaseModel):
    source: SourceDoc
    tables: list[TableSchema] = []
    projection: list[ProjectedColumnSchema] = []
    join_groups: list[JoinGroup] = []
    def column(self, table: str, column: str) -> ColumnSchema | None
```

`ColumnSchema.source` / `.evidence: str | None` are gone — folded into `chosen`.
`ProjectedColumnSchema` gets the same `chosen` + `evidence` treatment plus a `span`.

### Algorithm changes (`schema_resolution/__init__.py`)

Split the current `_initial_evidence` in two, so collection never discards:

- `_collect_evidence(node, usages, derived_outputs) -> list[TypeEvidence]` — one entry per
  usage fact, plus expression/cast/literal-argument entries, plus the name-pattern entry
  **always** (not just as a fallback; it becomes low-weight evidence that simply loses).
- `_choose(evidence) -> tuple[ResolvedType, TypeEvidence | None, list[ResolvedType]]` —
  the existing max-weight rule, with the existing equal-weight `widen()` tie-break, now
  recording what it widened from instead of splicing `"a+b"` into a string.

`_unify_join_groups` **stops overwriting**. It appends a `join_group` `TypeEvidence` with
the unified type and the best member's weight, then re-runs `_choose`. Members keep their
own evidence, which is what lets a diagnostic say *"typed `numeric` because it joins to
`person.id`, which is `numeric` because of `id > 0` at line 12"*.

`_constraints()` stops discarding `literal_kind` and stops deduping by
`(operator, values)` — two identical predicates in different places are two squiggle sites.
Dedupe on `(operator, values, span)` instead.

`_nullability()` returns the full `NullabilityResolution`.

Requirement 3 falls out of this: `ValueConstraint` now carries operator + literal values +
literal kind + spans, which is everything a boundary-value generator needs. Actual
generation stays out of scope — it is the design doc's `constraints` module. This refactor
only guarantees the data reaches it losslessly.

---

## Part 3 — declared schemas (new `src/sqlr/declared/`)

Discovery and consumption are **decoupled**, per the requirement that dbt artifacts can
later feed the same consumer.

```python
class DeclarationProvider(Protocol):
    def models(self) -> Iterable[DeclaredModel]: ...
```

First (and for now only) implementation: `YamlModelSchemaProvider`. A dbt
`manifest.json`/`catalog.json` provider slots in later without touching `diagnostics`.

### Discovery — `catalog`

Refactor `find_sql_files` into a shared `_find_files(root, globs, exclude)` carrying the
existing gitignore + `.git` exclusion logic, then add
`find_yaml_files(root, globs=["**/*.yml", "**/*.yaml"])`. Same `FileInventory` shape.

### Parsing — `declared/__init__.py`

- Read each yml. **Skip any file without a top-level `models:` list** — that is the
  discriminator, so unrelated yml in the project is ignored silently, not warned about.
- Each model entry needs `name` and `columns`; each column needs `name` and `data_type`.
  Entries missing those are skipped with a warning naming the file and line.
- `name` matches the **sql file stem**, case-insensitively, exactly like dbt.
- Malformed yaml → collected as a warning, never an exception. A broken yml must not stop
  analysis of the sql.

### Positions inside the yml

Parse with `yaml.compose()` rather than `yaml.safe_load()`. Composed nodes expose
`start_mark` / `end_mark` (index, line, column), which convert directly to `SourceSpan`.
This gives the diagnostic a *"declared as `varchar` here"* related location pointing into
the yml, not just into the sql.

### Types

```python
class DeclaredColumn(BaseModel):
    name: str
    data_type: str                 # verbatim, e.g. "varchar(50)"
    resolved_type: ResolvedType    # mapped, "unknown" if unrecognized
    span: SourceSpan | None
    type_span: SourceSpan | None   # the data_type scalar specifically

class DeclaredModel(BaseModel):
    name: str
    path: Path
    source: SourceDoc
    columns: list[DeclaredColumn]
    span: SourceSpan | None

class DeclaredSchemas(BaseModel):
    models: dict[str, DeclaredModel]   # keyed by lowercased name
    warnings: list[str]
```

### Shared type map — `src/sqlr/typemap.py` (new)

`CAST_TYPE_MAP` currently lives in `schema_resolution/__init__.py` and is keyed on bare
uppercase names. Move it out and give it a normalizer, since declared types arrive as
`varchar(50)`, `decimal(10,2)`, `number(38,0)`, `int64`, `timestamp_ntz`:

```python
def normalize_type_name(raw: str) -> str          # strip params, uppercase, alias
def resolve_declared_type(raw: str) -> ResolvedType  # "unknown" when unrecognized
```

Both `schema_resolution` (for `cast_type`) and `declared` consume it. Unrecognized
declared types produce an `unknown-declared-type` diagnostic rather than being silently
treated as `unknown`.

---

## Part 4 — `src/sqlr/diagnostics/` (new)

The design doc's Layer 5 module, built now because requirement 2 needs it. Cross-cutting:
`constraints`, `relationship_inference`, and `execution` all get to write into it later.

### `diagnostics/types.py`

```python
Severity = Literal["error", "warning", "info", "hint"]

class Location(BaseModel):
    path: Path | None
    span: SourceSpan | None
    snippet: str | None        # resolved from the owning SourceDoc at construction

class Related(Location):
    message: str               # "declared as varchar here", "evidence: amount > 20"

class Diagnostic(BaseModel):
    code: str
    severity: Severity
    message: str
    location: Location         # primary squiggle
    related: list[Related] = []
    table: str | None = None
    column: str | None = None

class DiagnosticReport(BaseModel):
    diagnostics: list[Diagnostic] = []
    @property
    def has_errors(self) -> bool
```

### `diagnostics/check.py`

`check_schema(schema: StatementSchema, declared: DeclaredModel | None) -> list[Diagnostic]`

| Code | Condition | Severity |
|---|---|---|
| `type-mismatch` | declared and inferred `TYPE_COVER` sets are **disjoint** (`string` vs `numeric`) | by evidence weight, below |
| `type-narrower-than-declared` | declared cover ⊃ inferred cover — declared `varchar`, inferred `string`: fine. No diagnostic. | — |
| `type-wider-than-inferred` | declared `integer`, inferred `number` — declared is a legal member of the family | none (declaration wins, it is more specific) |
| `unknown-declared-type` | `data_type` unmapped | warning |
| `undeclared-column` | inferred column absent from the declaration; suppressed when `TableSchema.star_expanded` | info |
| `missing-column` | declared column never referenced in the sql | hint |
| `unresolved-type` | inferred `unknown` and undeclared — generation cannot proceed | info |

Severity of `type-mismatch` follows the winning evidence's weight, so a cast disagreeing
with the declaration is louder than a name pattern:

- winner weight ≥ `EXPRESSION_WEIGHT` (80: cast, typed function) → **error**
- winner weight ≥ `USAGE_TYPE_WEIGHT` floor (10) → **warning**
- `name_pattern` only (5) → **hint**

Diagnostic construction, and the reason all of Part 1 and Part 2 exist:

- **primary location** = `chosen.context_span` (falling back to `chosen.span`, then the
  first entry in `references`),
- **related** = one entry per *other* `TypeEvidence` (`"also: in_list_strings at ..."`) —
  this is the "handle multiple Evidence" requirement — plus one entry for the declaration's
  `type_span` in the yml.

Message text is generated from the evidence, e.g.:

```
error: revenue is declared varchar but the SQL uses it as a number
  models/orders.sql:14:9
     14 |   where revenue > 1000
        |         ^^^^^^^^^^^^^^ compared to a numeric literal (weight 50)
  also used as a number here:
     22 |   sum(revenue) as total
  declared here:
    models/schema.yml:31:7   data_type: varchar
```

### `diagnostics/render.py`

- `render_text(report) -> str` — the block above, for the CLI.
- `to_lsp(report) -> list[dict]` — 0-based `{range: {start:{line,character}, end:...},
  severity: 1..4, code, message, relatedInformation: [...]}`. This is the artifact the
  future vscode extension consumes; building it now proves the spans are actually correct.

### Adapters

`from_analysis(result) -> list[Diagnostic]` folds the existing untyped
`SqlAnalysisResult.warnings`/`errors` and `Ambiguity` list into diagnostics, so
`cli.py` stops hand-formatting ambiguities and everything flows through one renderer.

---

## Part 5 — wiring (`cli.py`)

Per sql file: `analyze_file` → `resolve_schema` → look up `DeclaredSchemas` by file stem →
`check_schema` → accumulate into one `DiagnosticReport` → `render_text`. Declared schemas
are loaded **once** for the project, before the file loop.

Add `--strict` (exit 1 when the report has errors) and `--format text|json`, where `json`
emits the LSP shape. Remove the commented-out debug echoes and the `logger.debug` dumps of
whole schemas.

---

## Part 6 — tests

Fixtures owned by tests, per `tests/README.md`; nothing reads `examples/`.

- **`test_source.py`** (new) — offsets round-trip: `doc.slice(span)` equals the source text
  of the reference; multi-line sql; multi-statement sql (offsets absolute); a node with no
  meta yields `None` rather than raising; 0-based line/col.
- **`test_sql_analysis.py`** — every fact kind carries a span; `context_span` of
  `amount > 20` covers the whole comparison; `SourceColumn.references` has one entry per
  textual mention; CTE `Relation.span` covers the CTE body.
- **`test_schema_resolution.py`** — existing assertions get rewritten against
  `chosen.resolved_type`; new: losing evidence is retained and ordered; a join-group
  unification does **not** erase member evidence; an equal-weight tie records
  `widened_from`; `ValueConstraint` keeps `literal_kind` and duplicate predicates at
  different spans are both kept.
- **`test_declared.py`** (new) — a yml without `models:` is skipped; a model matches a sql
  stem; `data_type: varchar(50)` maps to `string`; yml spans point at the right line;
  malformed yaml produces a warning, not an exception.
- **`test_diagnostics.py`** (new) — `varchar` declared vs `numeric` inferred produces one
  error whose related list holds every other evidence span plus the yml declaration;
  declared `integer` vs inferred `number` produces nothing; `to_lsp` ranges are 0-based and
  slice back to the offending text.

---

## Order, and what breaks

1. **Part 0** — standalone, no consumers yet.
2. **Part 1** — touches all five `sql_analysis` files. Largest diff; everything downstream
   waits on it. `test_sql_analysis.py` needs updating for `Ambiguity.line` → `.span`.
3. **Part 2** — `test_schema_resolution.py` rewrites, `cli.py` compiles again.
4. **Part 3** — additive, no existing consumers.
5. **Part 4** — additive.
6. **Part 5 + 6**.

Parts 3 and 4 are independent of 1 and 2 up to the point where `check_schema` reads
`ColumnSchema.evidence`, so they can be built in parallel if useful.

**Risks**

- *Missing spans.* Synthesized nodes have no meta. Everything is `| None` and every
  renderer must handle it; a test asserts a span-less analysis still resolves a schema.
- *Result size.* A span on every fact grows `model_dump_json` output. Mitigated by keeping
  snippets out of the models and deriving them from `SourceDoc`.
- *`ColumnNode` identity.* Spans must not touch it — it is the hash key for every index in
  both modules.
- *Breaking `StatementSchema`.* Sanctioned; the only consumers are `cli.py` and the tests.
