# Conventions

## Naming

Prefer verbose, descriptive function names over compressed ones. The name should say what
the function returns or does, in full.

```python
def get_declared_types_per_table(...):  # yes
def gapfilled(...):                     # no
```

## Types

Prefer descriptive type aliases over bare primitives, so a signature says what the strings
and ints *are*, not just their runtime type.

```python
type TableName = str
type ColumnName = str
type ColumnTypeName = str

def get_declared_types_per_table(
    declared: dict[TableName, dict[ColumnName, ColumnTypeName]],  # yes
    declared: dict[str, dict[str, str]],                          # no
) -> ...
```

Spelling the nested alias out at the call site is the preferred form - do not collapse it
into a single opaque alias just to shorten the line.
