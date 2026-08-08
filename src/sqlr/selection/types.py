"""What a model is, and what an index of them answers.

A *model* is one `.sql` file addressed by its bare filename, dbt-style: `orders`, never
`models/marts/orders.sql`. That naming is only usable if the name is unique across the
whole search space, which is why the index is built eagerly and rejects collisions rather
than resolving them - see `SelectionError`.
"""

from pathlib import Path

from pydantic import BaseModel

from sqlr.catalog.types import SqlFile

__all__ = ["Model", "ModelIndex", "SelectionError"]


class SelectionError(Exception):
    """A selector, or the project layout it was resolved against, was unusable.

    Carries a message already written for a terminal, so the CLI can print it as-is.
    """


class Model(BaseModel):
    name: str
    """The file's stem. `models/marts/orders.sql` is `orders`."""
    file: SqlFile

    @property
    def path(self) -> Path:
        return self.file.path

    @property
    def relative_path(self) -> str:
        return self.file.relative_path


class ModelIndex(BaseModel):
    """Every model a project exposes, by name.

    Built once per command run: duplicate detection needs the whole set, and every
    selector is answered from it.
    """

    project_root: Path
    search_paths: list[Path]
    """Where models were looked for. The project root itself when `model_paths` is unset."""
    models: list[Model] = []
    """Sorted by name, so unselected runs have a stable order."""

    def get(self, name: str) -> Model | None:
        return next((model for model in self.models if model.name == name), None)

    @property
    def names(self) -> list[str]:
        return [model.name for model in self.models]
