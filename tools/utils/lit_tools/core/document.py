# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Line-oriented document model for LLVM lit test files.

This module provides the core data structures for representing lit test files
as immutable, line-oriented documents. This enables accurate line number
preservation, exact newline handling, and efficient span-based operations.

Key classes:
- Line: Atomic line unit with text, newline, and metadata
- Span: Half-open range [start, end) over line indices
- FileDoc: Immutable document as tuple of Lines
- CheckLabelExtractor: Utility for extracting CHECK-LABEL names
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Module-level regex for CHECK-LABEL extraction
# Matches any *-LABEL: pattern followed by optional prefix and @name
# Group 1 captures the name after @
_CHECK_LABEL_PATTERN = re.compile(r"//\s*[\w-]+-LABEL:\s*.*?@([\w.$-]+)")


@dataclass(frozen=True)
class Line:
    r"""Represents a single line of text with preserved formatting.

    Attributes:
        text: Line content without trailing newline character
        newline: The newline character(s): "\n", "\r\n", or "" for EOF
        index: 0-based index of this line in the FileDoc
        source_line: 1-based original line number (immutable diagnostic reference)
        tags: Structural classification tags like "RUN_HEADER", "DELIMITER", "CHECK"
    """

    text: str
    newline: str
    index: int
    source_line: int
    tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        r"""Validate line invariants."""
        if self.newline not in ("", "\n", "\r\n"):
            raise ValueError(
                f"Invalid newline character: {repr(self.newline)}. "
                f"Must be '', '\\n', or '\\r\\n'"
            )
        if self.index < 0:
            raise ValueError(f"Line index must be non-negative, got {self.index}")
        if self.source_line < 1:
            raise ValueError(
                f"Source line must be >= 1 (1-based), got {self.source_line}"
            )

    def get_full_line(self) -> str:
        """Returns the complete line including its newline character."""
        return self.text + self.newline

    def with_tags(self, tags: frozenset[str]) -> "Line":
        r"""Returns a new Line with updated tags (immutable update)."""
        return Line(
            text=self.text,
            newline=self.newline,
            index=self.index,
            source_line=self.source_line,
            tags=tags,
        )


@dataclass(frozen=True)
class Span:
    """Half-open range [start, end) over line indices.

    Uses Python slice semantics: start is inclusive, end is exclusive.
    This makes slicing operations natural: lines[span.start:span.end]

    Attributes:
        start: Inclusive 0-based start index
        end: Exclusive 0-based end index
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        """Validate span invariants."""
        if self.start < 0:
            raise ValueError(f"Span start must be non-negative, got {self.start}")
        if self.end < self.start:
            raise ValueError(
                f"Invalid span [{self.start}, {self.end}): end must be >= start"
            )

    @property
    def length(self) -> int:
        """Number of lines in this span."""
        return self.end - self.start

    @property
    def is_empty(self) -> bool:
        """True if span contains no lines."""
        return self.start == self.end

    def contains(self, index: int) -> bool:
        """Check if a line index falls within this span."""
        return self.start <= index < self.end


@dataclass(frozen=True)
class FileDoc:
    """Immutable document representation as a tuple of Lines.

    This is the core data structure for representing lit test files.
    All lines are stored in a flat tuple, and cases are represented
    as Span indices over this tuple.

    Attributes:
        lines: Tuple of Line objects representing the file
        path: Original file path (for error messages)
    """

    lines: tuple[Line, ...]
    path: Path

    @classmethod
    def from_text(cls, text: str, path: Path = Path()) -> "FileDoc":
        r"""Parse text into FileDoc with exact fidelity.

        Uses splitlines(keepends=True) to preserve exact newline characters.
        Handles LF, CRLF, and files without trailing newlines correctly.

        Args:
            text: Raw file content
            path: File path for error messages

        Returns:
            FileDoc with all lines parsed

        Example:
            >>> doc = FileDoc.from_text("line1\\nline2\\n")
            >>> len(doc.lines)
            2
            >>> doc.lines[0].text
            'line1'
            >>> doc.lines[0].newline
            '\\n'
        """
        raw_lines = text.splitlines(keepends=True)
        lines = []

        for idx, raw in enumerate(raw_lines):
            if raw.endswith("\r\n"):
                lines.append(
                    Line(
                        text=raw[:-2],
                        newline="\r\n",
                        index=idx,
                        source_line=idx + 1,
                        tags=frozenset(),
                    )
                )
            elif raw.endswith("\n"):
                lines.append(
                    Line(
                        text=raw[:-1],
                        newline="\n",
                        index=idx,
                        source_line=idx + 1,
                        tags=frozenset(),
                    )
                )
            else:
                # Last line without newline
                lines.append(
                    Line(
                        text=raw,
                        newline="",
                        index=idx,
                        source_line=idx + 1,
                        tags=frozenset(),
                    )
                )

        return cls(lines=tuple(lines), path=path)

    def to_text(self) -> str:
        r"""Serialize document to string with exact fidelity.

        Preserves all newline characters and content exactly as parsed.

        Returns:
            String representation of the document

        Example:
            >>> doc = FileDoc.from_text("line1\\nline2")
            >>> doc.to_text()
            'line1\\nline2'
        """
        return "".join(line.get_full_line() for line in self.lines)

    def slice(self, span: Span) -> tuple[Line, ...]:
        r"""Extract lines for a given span.

        Args:
            span: Span defining the range to extract

        Returns:
            Tuple of lines in the span

        Raises:
            IndexError: If span extends beyond document bounds

        Example:
            >>> doc = FileDoc.from_text("line1\\nline2\\nline3\\n")
            >>> span = Span(start=1, end=3)
            >>> lines = doc.slice(span)
            >>> len(lines)
            2
        """
        if span.end > len(self.lines):
            raise IndexError(
                f"Span [{span.start}, {span.end}) extends beyond document "
                f"with {len(self.lines)} lines"
            )
        return self.lines[span.start : span.end]

    def with_line_tags(self, line_index: int, tags: frozenset[str]) -> "FileDoc":
        r"""Returns new FileDoc with tags updated for a specific line.

        This is an immutable update operation.

        Args:
            line_index: Index of line to update
            tags: New tags for the line

        Returns:
            New FileDoc with updated line
        """
        if not (0 <= line_index < len(self.lines)):
            raise IndexError(
                f"Line index {line_index} out of range [0, {len(self.lines)})"
            )

        updated_lines = list(self.lines)
        updated_lines[line_index] = self.lines[line_index].with_tags(tags)
        return FileDoc(lines=tuple(updated_lines), path=self.path)


class CheckLabelExtractor:
    """Utility for extracting CHECK-LABEL names from lit test lines.

    IREE lit tests use CHECK-LABEL directives to name test cases.
    This extractor handles all variations of CHECK-LABEL patterns.

    Patterns matched:
    - // CHECK-LABEL: @function_name
    - // CHECK-LABEL: stream.executable @name
    - // FOO-LABEL: @name (any prefix before -LABEL:)
    - // CHECK-LABEL: func.func @name(%args)

    The label is everything after '@' up to the next whitespace or '('.
    Allowed characters: letters, digits, '.', '$', '-', '_'
    """

    @staticmethod
    def extract_all_names(lines: Iterable[Line]) -> list[str]:
        r"""Extract all unique CHECK-LABEL names from lines.

        Returns list of unique names in order of first appearance.
        This is useful when a case has multiple CHECK-LABEL directives
        with different prefixes (CHECK, FOO, BAR).

        Args:
            lines: Iterable of Line objects to search

        Returns:
            List of unique label names found, in order

        Example:
            >>> lines = [
            ...     Line("// CHECK-LABEL: @foo", "\\n", 0, 1),
            ...     Line("// FOO-LABEL: @foo", "\\n", 1, 2),
            ...     Line("// BAR-LABEL: @bar", "\\n", 2, 3),
            ... ]
            >>> CheckLabelExtractor.extract_all_names(lines)
            ['foo', 'bar']
        """
        seen = set()
        names = []
        for line in lines:
            for match in _CHECK_LABEL_PATTERN.finditer(line.text):
                name = match.group(1)
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
        return names

    @staticmethod
    def extract_primary_name(
        lines: Iterable[Line],
    ) -> tuple[str | None, tuple[str, ...]]:
        r"""Extract primary name (first found) and all names from CHECK-LABELs.

        Args:
            lines: Iterable of Line objects to search

        Returns:
            Tuple of (primary_name, all_names_tuple)
            - primary_name: First CHECK-LABEL name found, or None
            - all_names_tuple: All unique names found, or empty tuple

        Example:
            >>> lines = [
            ...     Line("// CHECK-LABEL: @foo", "\\n", 0, 1),
            ...     Line("// FOO-LABEL: @foo", "\\n", 1, 2),
            ... ]
            >>> primary, all_names = CheckLabelExtractor.extract_primary_name(lines)
            >>> primary
            'foo'
            >>> all_names
            ('foo',)
        """
        all_names = CheckLabelExtractor.extract_all_names(lines)
        if not all_names:
            return None, ()
        return all_names[0], tuple(all_names)
