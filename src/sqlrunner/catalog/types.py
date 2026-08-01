from pathlib import Path

from pydantic import BaseModel


class SqlFile(BaseModel):
    path: Path
    relative_path: str
    mtime: float
    content_hash: str


class FileInventory(BaseModel):
    project_root: Path
    files: list[SqlFile]
