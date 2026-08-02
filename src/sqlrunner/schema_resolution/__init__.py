from collections import defaultdict

from sqlrunner.schema_resolution.types import ColumnSchema, ResolvedType, TableSchema
from sqlrunner.sql_analysis.types import ColumnUsage, SqlAnalysisResult

CAST_TYPE_MAP: dict[str, ResolvedType] = {
    "INT": "integer",
    "INTEGER": "integer",
    "BIGINT": "integer",
    "SMALLINT": "integer",
    "TINYINT": "integer",
    "DECIMAL": "decimal",
    "NUMERIC": "decimal",
    "FLOAT": "decimal",
    "DOUBLE": "decimal",
    "REAL": "decimal",
    "VARCHAR": "string",
    "TEXT": "string",
    "CHAR": "string",
    "STRING": "string",
    "NVARCHAR": "string",
    "DATE": "date",
    "TIMESTAMP": "timestamp",
    "TIMESTAMPTZ": "timestamp",
    "DATETIME": "timestamp",
    "BOOLEAN": "boolean",
    "BOOL": "boolean",
}

# (resolved_type, confidence) — higher confidence wins when a column has conflicting evidence.
USAGE_TYPE_WEIGHT: dict[str, int] = {
    "cast": 100,
    "boolean_context": 90,
    "date_function": 70,
    "in_list_strings": 60,
    "like": 60,
    "compared_to_number": 50,
    "in_list_numbers": 50,
    "arithmetic": 10,
}


def _type_from_usage(usage: ColumnUsage) -> tuple[ResolvedType, int] | None:
    weight = USAGE_TYPE_WEIGHT[usage.kind]

    if usage.kind == "cast":
        resolved = CAST_TYPE_MAP.get(usage.detail or "")
        return (resolved, weight) if resolved else None

    if usage.kind == "boolean_context":
        return ("boolean", weight)

    if usage.kind == "date_function":
        return ("date", weight)

    if usage.kind in ("in_list_strings", "like"):
        return ("string", weight)

    if usage.kind in ("compared_to_number", "in_list_numbers"):
        return ("integer" if usage.detail == "int" else "decimal", weight)

    if usage.kind == "arithmetic":
        return ("decimal", weight)

    return None


def _type_from_name(column: str) -> tuple[ResolvedType, str] | None:
    lower = column.lower()
    if lower.startswith("is_"):
        return ("boolean", "is_*")
    if lower.endswith("_id"):
        return ("integer", "*_id")
    if lower.endswith("_at"):
        return ("timestamp", "*_at")
    return None


def _resolve_column(name: str, usages: list[ColumnUsage]) -> ColumnSchema:
    best: tuple[ResolvedType, int, str] | None = None
    for usage in usages:
        candidate = _type_from_usage(usage)
        if candidate is None:
            continue
        resolved, weight = candidate
        if best is None or weight > best[1]:
            best = (resolved, weight, usage.kind)

    if best is not None:
        return ColumnSchema(name=name, resolved_type=best[0], source="usage", evidence=best[2])

    name_match = _type_from_name(name)
    if name_match is not None:
        resolved, pattern = name_match
        return ColumnSchema(name=name, resolved_type=resolved, source="name_pattern", evidence=pattern)

    return ColumnSchema(name=name, resolved_type="unknown", source="unknown", evidence=None)


def resolve_schema(analysis: SqlAnalysisResult) -> list[TableSchema]:

    # print(analysis.column_references)

    columns_by_table: dict[str, set[str]] = defaultdict(set)
    for reference in analysis.column_references:
        columns_by_table[reference.table].add(reference.column)

    usages_by_table_column: dict[tuple[str, str], list[ColumnUsage]] = defaultdict(list)
    for usage in analysis.column_usages:
        usages_by_table_column[(usage.table, usage.column)].append(usage)

    tables: list[TableSchema] = []
    for table in sorted(analysis.external_sources):
        column_names = sorted(columns_by_table.get(table, set()))
        columns = [
            _resolve_column(name, usages_by_table_column.get((table, name), []))
            for name in column_names
        ]
        tables.append(TableSchema(name=table, columns=columns))

    return tables
