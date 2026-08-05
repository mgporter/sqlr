"""Load the types the user declared, from dbt-shaped model schema yml.

The format is dbt's, deliberately: a project that already has `schema.yml` files gets
checked with no extra authoring, and a project without dbt can write the same thing.
Only two fields are required of each column - `name` and `data_type`.

```yml
models:
  - name: orders          # matches orders.sql
    columns:
      - name: revenue
        data_type: numeric
```

Discovery is decoupled from interpretation. `DeclarationProvider` is the seam: this
module implements it over yml files, and a future dbt `manifest.json` reader can
implement it without any consumer changing.

Parsing goes through `yaml.compose` rather than `yaml.safe_load`, because the composed
node tree carries source marks. Without them a diagnostic could say a declaration was
contradicted but not where the declaration was written.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, cast

import yaml

from sqlrunner.catalog.types import FileInventory
from sqlrunner.declared.types import DeclaredColumn, DeclaredModel, DeclaredSchemas
from sqlrunner.source import Positions, SourceDoc, SourceSpan
from sqlrunner.typemap import resolve_type_name

MODELS_KEY = "models"
"""The discriminator. A yml without it is not ours and is skipped in silence."""


class DeclarationProvider(Protocol):
    """Anything that can produce declared model schemas."""

    def models(self) -> Iterable[DeclaredModel]: ...


class YamlModelSchemaProvider:
    """Reads dbt-shaped `models:` yml out of a file inventory."""

    def __init__(self, inventory: FileInventory) -> None:
        self.inventory = inventory
        self.warnings: list[str] = []

    def models(self) -> Iterable[DeclaredModel]:
        for file in self.inventory.files:
            yield from self._models_in(file.path, file.relative_path)

    def _models_in(self, path: Path, label: str) -> list[DeclaredModel]:
        try:
            text = path.read_text()
        except OSError as e:
            self.warnings.append(f"{label}: could not be read ({e})")
            return []

        try:
            root = _compose(text)
        except yaml.YAMLError as e:
            # A broken yml is a problem with that file, not a reason to stop analysing
            # the SQL it was meant to describe.
            self.warnings.append(f"{label}: invalid yaml ({_terse(e)})")
            return []

        if not isinstance(root, yaml.MappingNode):
            return []

        models_node = _entry(root, MODELS_KEY)
        if not isinstance(models_node, yaml.SequenceNode):
            return []

        source = SourceDoc(path=path, text=text)
        positions = Positions(text)

        models: list[DeclaredModel] = []
        for entry in models_node.value:
            model = self._model(entry, path, label, source, positions)
            if model is not None:
                models.append(model)
        return models

    def _model(
        self,
        node: yaml.Node,
        path: Path,
        label: str,
        source: SourceDoc,
        positions: Positions,
    ) -> DeclaredModel | None:
        if not isinstance(node, yaml.MappingNode):
            return None

        name_node = _entry(node, "name")
        name = _scalar(name_node)
        if not name:
            self.warnings.append(
                f"{label}:{_line(node)}: model entry has no name; skipped"
            )
            return None

        columns: list[DeclaredColumn] = []
        columns_node = _entry(node, "columns")
        if isinstance(columns_node, yaml.SequenceNode):
            for entry in columns_node.value:
                column = self._column(entry, label, positions)
                if column is not None:
                    columns.append(column)

        return DeclaredModel(
            name=name,
            path=path,
            source=source,
            columns=columns,
            span=_span(positions, node),
            name_span=_span(positions, name_node),
        )

    def _column(
        self, node: yaml.Node, label: str, positions: Positions
    ) -> DeclaredColumn | None:
        if not isinstance(node, yaml.MappingNode):
            return None

        name_node = _entry(node, "name")
        name = _scalar(name_node)
        if not name:
            self.warnings.append(
                f"{label}:{_line(node)}: column entry has no name; skipped"
            )
            return None

        type_node = _entry(node, "data_type")
        written_type = _scalar(type_node)
        if not written_type:
            # A documented column with no declared type is normal in dbt and simply has
            # nothing to check against. Not a warning.
            return None

        return DeclaredColumn(
            name=name,
            written_type=written_type,
            resolved_type_name=resolve_type_name(written_type),
            description=_scalar(_entry(node, "description")),
            span=_span(positions, node),
            name_span=_span(positions, name_node),
            type_span=_span(positions, type_node),
        )


def load_declared_schemas(
    inventory: FileInventory,
    providers: Iterable[DeclarationProvider] | None = None,
) -> DeclaredSchemas:
    """Collect declarations from every provider into one index.

    Later models with the same name lose to earlier ones, and say so.
    """
    yaml_provider = YamlModelSchemaProvider(inventory)
    sources: list[DeclarationProvider] = (
        [yaml_provider] if providers is None else list(providers)
    )

    models: dict[str, DeclaredModel] = {}
    warnings: list[str] = []

    for provider in sources:
        for model in provider.models():
            key = model.name.lower()
            existing = models.get(key)
            if existing is not None:
                warnings.append(
                    f"model {model.name!r} is declared in both {existing.path} and "
                    f"{model.path}; the first wins"
                )
                continue
            models[key] = model
        warnings.extend(getattr(provider, "warnings", []))

    return DeclaredSchemas(models=models, warnings=warnings)


# ---- yaml node helpers ---------------------------------------------------------------


def _compose(text: str) -> yaml.Node | None:
    """`yaml.compose`, with the return type its type stub omits.

    Typeshed declares `compose` without one, so its result is untyped and infects every
    expression downstream of it. Pinning it here keeps the ignore to a single line.
    """
    composed = yaml.compose(text)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return cast("yaml.Node | None", composed)


def _entry(node: yaml.MappingNode, key: str) -> yaml.Node | None:
    for key_node, value_node in node.value:
        if isinstance(key_node, yaml.ScalarNode) and key_node.value == key:
            return value_node
    return None


def _scalar(node: yaml.Node | None) -> str | None:
    if not isinstance(node, yaml.ScalarNode):
        return None
    value = str(node.value).strip()
    return value or None


def _span(positions: Positions, node: yaml.Node | None) -> SourceSpan | None:
    """Convert a composed node's marks into a span.

    PyYAML's end mark for a block scalar runs to the start of the next token, trailing
    newline included, so it is trimmed back to the value itself.
    """
    if node is None:
        return None
    start = node.start_mark.index
    end = node.end_mark.index
    text = positions.text
    while end > start and text[end - 1].isspace():
        end -= 1
    if end <= start:
        return None
    return positions.span(start, end)


def _line(node: yaml.Node) -> int:
    return node.start_mark.line + 1


def _terse(error: yaml.YAMLError) -> str:
    return " ".join(str(error).split())
