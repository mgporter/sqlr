# Tests

## Fixtures

**Tests must never read from `examples/`.** That directory is a scratch area for manual
experimentation — files there change freely, and a test that depends on one breaks the
moment a line is edited for an unrelated reason.

Every test owns its fixtures:

- Inline SQL strings for the common case — `analyze_sql("select ...")`.
- `tests/fixtures/` for SQL too long to read inline, referenced relative to the test
  file, never relative to the working directory:

  ```python
  FIXTURES = Path(__file__).parent / "fixtures"
  PERSON_SQL = FIXTURES / "person.sql"
  ```

If a fixture and an example need to stay in sync, copy the file. Duplication is the point.
