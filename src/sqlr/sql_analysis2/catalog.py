from dataclasses import dataclass

@dataclass(frozen=True)
class Sig:
    params: tuple[str, ...]   # "ARRAY", "NUMERIC", "STRING", "BOOLEAN", "ANY"
    returns: str              # type string or "@element" / "@arg0" / "@argN"

CATALOG = {
    "duckdb": {
        "LIST_EXTRACT": [
            Sig(params=("ARRAY", "NUMERIC"), returns="@element")
        ]
    }
}