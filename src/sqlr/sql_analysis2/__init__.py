

from pathlib import Path

import sqlglot
from sqlglot import ParseError
from sqlglot.optimizer.qualify import qualify

from sqlr.config.types import SqlrConfig
from sqlr.declared.types import DeclaredSchemas
from sqlr.selection.types import Model
from sqlr.sql_analysis2.sourcedoc import Positions, SourceDoc


def validate_schema(cfg: SqlrConfig, declared: DeclaredSchemas, models: list[Model]) -> None:

    for model in models:
        sql = Path(model.path).read_text()
        source = SourceDoc(path=model.path, text=sql)

        # One index for the whole document: sqlglot's character offsets are absolute, so they
        # stay valid across every statement in the file.
        positions = Positions(sql)
    
        # Step 1: parse.
        try:
            statements = sqlglot.parse(sql, read=cfg.general.sql_dialect)

        except ParseError as e:
            print(f"error: {model.relative_path}: {str(e)}")
            # return SqlAnalysisResult(source=source, errors=[str(e)])
            return

        first_statement = statements[0]

        if first_statement is None:
            print(f"error: {model.relative_path}: No statements found in the SQL file.")
            return

        print(repr(first_statement))

        
    
        # sqlAnalysisResults: list[SqlAnalysisResult] = []
