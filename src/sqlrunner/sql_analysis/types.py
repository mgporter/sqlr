from pydantic import BaseModel


class Cte(BaseModel):
    name: str
    depends_on: list[str]


class SqlAnalysisResult(BaseModel):
    ctes: list[Cte]
    external_sources: list[str]
    errors: list[str]
