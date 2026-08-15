"""Source text and character ranges.

Every fact the analyser derives has to be pointable back at the SQL that produced it -
a diagnostic that says "this column is a number" is worth little next to one that says
"this column is a number *because of this comparison, right here*".

sqlglot records positions only on leaf tokens: `Identifier`, `Literal`, and the name token
of a few functions. `Column`, `EQ`, `Where` and `Select` carry nothing. So the span of any
composite expression is the hull of the leaves under it, which is what `Positions.span_of`
computes.

sqlglot's own `line`/`col` are deliberately ignored. `col` is 1-based *and points at the
token's last character*, which no editor wants. Both are recomputed from the character
offsets, 0-based, so a `SourceSpan` drops straight into an LSP `Range`.

Two limits follow from taking the hull of positioned leaves, and callers should expect
them:

- **Bare keywords are invisible.** `shipped_on is null` spans only `shipped_on`, because
  no token is emitted for `IS NULL`. The location is right, just narrower than a reader
  would draw it. `TRUE` and `FALSE` fall in the same class: sqlglot parses them into
  `exp.Boolean`, which carries no position, so `is_active = true` spans only `is_active`
  and the constraint's `value_spans` entry for the `true` is None.
- **Punctuation is invisible too**, but that one is repairable and is repaired: see
  `_balance_parens`, which grows a hull back over the brackets its leaves left behind.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from sqlglot import exp


def token_offsets_of(expression: exp.Expr | None) -> tuple[int, int] | None:
    """The half-open offset hull of every positioned token under `expression`.

    Split out of `Positions.span_of` because it needs no source text: a caller that only
    wants to *identify* a node - "is this the same column I saw before qualification?" -
    gets a stable key without building a line index for a document it may not have.

    The pair is half-open, unlike sqlglot's own inclusive `end`.
    """
    if expression is None:
        return None

    start: int | None = None
    end: int | None = None

    for node in expression.walk():
        meta = node.meta
        node_start = meta.get("start")
        node_end = meta.get("end")
        if not isinstance(node_start, int) or not isinstance(node_end, int):
            continue
        if start is None or node_start < start:
            start = node_start
        # sqlglot's `end` points at the last character; make it exclusive.
        if end is None or node_end + 1 > end:
            end = node_end + 1

    if start is None or end is None:
        return None
    return start, end


class SourceSpan(BaseModel, frozen=True):
    """A half-open character range in one source file, with line/column resolved.

    Offsets are absolute within the file, so they stay valid across a multi-statement
    parse. Lines and columns are 0-based to match LSP.
    """

    start: int
    end: int
    """Exclusive. sqlglot's own `end` is inclusive; the conversion happens once, here."""
    start_line: int
    start_col: int
    end_line: int
    end_col: int

    def __str__(self) -> str:
        return f"{self.start_line + 1}:{self.start_col + 1}"


class SourceDoc(BaseModel):
    """The text a result was derived from, plus the file it came from.

    Snippets are *not* stored on facts - there are thousands of them and they would
    multiply the size of a serialised result. They are cut from here on demand instead.
    """

    path: Path | None = None
    text: str = ""

    @property
    def name(self) -> str:
        return str(self.path) if self.path is not None else "<sql>"

    def slice(self, span: SourceSpan | None) -> str | None:
        """The exact text a span covers."""
        if span is None:
            return None
        return self.text[span.start : span.end]

    def lines_of(self, span: SourceSpan | None) -> list[str]:
        """Every whole line the span touches, for a caret-underlined CLI snippet."""
        if span is None:
            return []
        lines = self.text.splitlines()
        return lines[span.start_line : span.end_line + 1]


class Positions:
    """Resolves sqlglot expressions to spans. One per parsed document.

    Built once because the line index is O(text) to compute and O(log n) to query.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self._line_starts: list[int] = [0]
        for index, character in enumerate(text):
            if character == "\n":
                self._line_starts.append(index + 1)

    # ---- offsets --------------------------------------------------------------

    def line_col(self, offset: int) -> tuple[int, int]:
        """0-based (line, column) of an absolute character offset."""
        offset = max(0, min(offset, len(self.text)))
        line = bisect_right(self._line_starts, offset) - 1
        return line, offset - self._line_starts[line]

    def span(self, start: int, end: int) -> SourceSpan:
        """Build a span from a half-open offset pair."""
        start_line, start_col = self.line_col(start)
        end_line, end_col = self.line_col(end)
        return SourceSpan(
            start=start,
            end=end,
            start_line=start_line,
            start_col=start_col,
            end_line=end_line,
            end_col=end_col,
        )

    # ---- expressions ----------------------------------------------------------

    def span_of(self, expression: exp.Expr | None) -> SourceSpan | None:
        """The hull of every positioned token under `expression`.

        Returns None when nothing under it carries a position - which happens for nodes
        sqlglot synthesised rather than parsed, such as an alias invented by
        `qualify_tables`. Callers must treat a missing span as normal, not exceptional.
        """
        offsets = token_offsets_of(expression)
        if offsets is None:
            return None
        start, end = offsets
        if end > len(self.text):
            # The offsets belong to some other document. That means this index was built
            # from text the expression was not parsed from, and any span derived here
            # would point at the wrong place - which is worse than pointing nowhere.
            return None
        start, end = self._balance_parens(start, end)
        return self.span(start, end)

    def _balance_parens(self, start: int, end: int) -> tuple[int, int]:
        """Grow a hull outwards over the brackets its leaves left behind.

        Punctuation carries no position, so the hull of `x in ('a', 'b')` stops at the
        final quote and the hull of `(a + b)` starts at `a`. An underline that drops a
        bracket looks broken, so an unbalanced hull is extended over the adjacent
        brackets that would close it - and only over those, so a stray parenthesis inside
        a string literal can cost at most one character either side.
        """
        depth = self.text.count("(", start, end) - self.text.count(")", start, end)

        while depth > 0:
            cursor = end
            while cursor < len(self.text) and self.text[cursor].isspace():
                cursor += 1
            if cursor >= len(self.text) or self.text[cursor] != ")":
                break
            end = cursor + 1
            depth -= 1

        while depth < 0:
            cursor = start - 1
            while cursor >= 0 and self.text[cursor].isspace():
                cursor -= 1
            if cursor < 0 or self.text[cursor] != "(":
                break
            start = cursor
            depth += 1

        return start, end

    def span_covering(
        self, expressions: Iterable[exp.Expr | None]
    ) -> SourceSpan | None:
        """The hull of several expressions, skipping the ones with no position."""
        spans = [span for span in map(self.span_of, expressions) if span is not None]
        if not spans:
            return None
        return self.span(
            min(span.start for span in spans), max(span.end for span in spans)
        )


NO_POSITIONS = Positions("")
"""Placeholder for callers that have no source text. Yields no spans."""
