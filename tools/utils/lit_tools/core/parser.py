# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Parser for LLVM lit test files with line-oriented structure.

This module implements parsing of lit test files into the span-based document model.
The parser identifies test case boundaries (delimiters), classifies lines by type
(RUN, CHECK, DELIMITER, BODY), and extracts metadata for each case.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from .document import CheckLabelExtractor, FileDoc, Line, Span

# Tag constants for line classification
TAG_RUN_HEADER = "RUN_HEADER"  # File-level RUN line
TAG_RUN_CASE = "RUN_CASE"  # Case-local RUN line
TAG_DELIMITER = "DELIMITER"  # // ----- separator
TAG_CHECK = "CHECK"  # CHECK directive
TAG_BODY = "BODY"  # Everything else

# Pattern for identifying RUN lines
RUN_LINE_PATTERN = re.compile(r"^\s*//\s*RUN:\s*(.*)$")

# Pattern for identifying delimiters (exact match required)
DELIMITER_PATTERN = re.compile(r"^\s*//\s*-----\s*$")

# Pattern for identifying CHECK directives (CHECK, CHECK-LABEL, FOO-CHECK, etc.)
CHECK_PATTERN = re.compile(r"^\s*//\s*[\w-]*CHECK")


@dataclass(frozen=True)
class TestCase:
    """Represents a single test case with metadata.

    A test case is a span of lines in the document, typically separated
    by // ----- delimiters. Cases may have header RUN lines (file-level)
    and case-local RUN lines (within the case body).
    """

    doc: FileDoc  # Document containing this case
    number: int  # 1-based case number
    name: str | None  # From CHECK-LABEL or None
    all_names: tuple[str, ...]  # All CHECK-LABEL names in case
    span: Span  # Lines belonging to this case
    header_run_lines: frozenset[int]  # File-level RUN lines
    local_run_lines: frozenset[int]  # Case-local RUN lines
    check_count: int  # Number of CHECK directives

    @property
    def run_lines(self) -> frozenset[int]:
        """All RUN line indices (header + local)."""
        return self.header_run_lines | self.local_run_lines

    @property
    def start_line(self) -> int:
        """1-based start line number (for backward compatibility)."""
        return self.span.start + 1

    @property
    def end_line(self) -> int:
        """1-based end line number, inclusive (for backward compatibility)."""
        # Span uses exclusive end, but old API used inclusive
        return self.span.end

    @property
    def line_count(self) -> int:
        """Total number of lines in case."""
        return self.span.length

    def render_for_testing(self) -> str:
        """Render case for lit execution with line number preservation.

        This mode is used when extracting a case for testing with lit/FileCheck.
        RUN lines are replaced with blank lines to preserve line numbers for
        FileCheck diagnostics, ensuring CHECK directives reference the correct
        source locations.

        Returns:
            String with RUN lines blanked, preserving total line count

        Example:
            Input case (lines 5-8):
                // RUN: iree-opt %s
                // CHECK: foo
                func @test() { }
                (blank line)

            Output:
                (blank line)
                // CHECK: foo
                func @test() { }
                (blank line)
        """
        case_lines = self.doc.slice(self.span)
        result_lines = []

        for line in case_lines:
            if TAG_RUN_HEADER in line.tags or TAG_RUN_CASE in line.tags:
                # Replace RUN line content with blank, keep newline.
                result_lines.append(line.newline)
            else:
                # Keep all other lines as-is.
                result_lines.append(line.get_full_line())

        return "".join(result_lines)

    def render_normalized(self) -> str:
        """Render case body for normalized output.

        This mode is used when rebuilding test files with clean formatting.
        RUN lines are dropped entirely, leading/trailing newlines are removed,
        and internal consecutive blank lines are normalized to exactly one.

        Note: This function removes ALL trailing newlines. The caller
        (typically build_file_content) is responsible for adding a final
        newline if needed for the file format.

        Returns:
            String with RUN lines dropped, blanks trimmed, and internal
            blank lines normalized (sequences of 2+ newlines → 2 newlines)

        Example:
            Input case (lines 5-8):
                // RUN: iree-opt %s
                (blank line)
                // CHECK: foo


                func @test() { }


            Output (no trailing newline):
                // CHECK: foo

                func @test() { }
        """
        case_lines = self.doc.slice(self.span)
        result_lines = []

        for line in case_lines:
            # Skip RUN lines entirely in normalized mode.
            if TAG_RUN_HEADER in line.tags or TAG_RUN_CASE in line.tags:
                continue
            result_lines.append(line.get_full_line())

        # Join and trim leading/trailing blank lines.
        content = "".join(result_lines)

        # Strip leading blank lines.
        while content.startswith("\n") or content.startswith("\r\n"):
            content = content[2:] if content.startswith("\r\n") else content[1:]

        # Strip ALL trailing newlines (build_file_content will add exactly one).
        while content.endswith("\n") or content.endswith("\r\n"):
            content = content[:-2] if content.endswith("\r\n") else content[:-1]

        # Normalize internal consecutive blank lines (max 1 consecutive).
        # Replace runs of 2+ consecutive newlines with exactly 2 newlines.
        return re.sub(r"\n\n+", "\n\n", content)

    @property
    def content(self) -> str:
        """Get case content (normalized, without RUN lines).

        Returns:
            Case body with RUN lines removed and formatting normalized
        """
        return self.render_normalized()

    def extract_local_run_lines(self) -> list[tuple[int, str]]:
        """Extract case-local RUN commands with absolute line positions.

        Returns RUN commands that appear within this test case body,
        along with their absolute line indices for injection purposes.

        Returns:
            List of (absolute_line_index, command) tuples where absolute_line_index
            is the 0-based index in synthesized content (with blank line padding)

        Example:
            >>> case.extract_local_run_lines()
            [(15, 'iree-opt --pass-pipeline=... %s')]
        """
        local_indices = sorted(self.local_run_lines)
        result = []

        for idx in local_indices:
            line = self.doc.lines[idx]
            match = RUN_LINE_PATTERN.match(line.text)
            if not match:
                continue

            content = match.group(1).strip()

            # Handle continuation lines
            current_idx = idx
            cmd = content
            while current_idx < self.span.end - 1:
                current_line = self.doc.lines[current_idx]
                if not current_line.text.rstrip().endswith("\\"):
                    break

                current_idx += 1
                next_line = self.doc.lines[current_idx]
                next_match = RUN_LINE_PATTERN.match(next_line.text)
                if next_match:
                    cmd = (
                        cmd.rstrip("\\").strip()
                        + " "
                        + next_match.group(1).lstrip("\\").strip()
                    )

            # Calculate absolute index in synthesized content
            # Synthesized content has (case.start_line - 1) blank lines at top
            relative_idx = idx - self.span.start
            absolute_idx = (self.start_line - 1) + relative_idx

            result.append((absolute_idx, cmd.rstrip("\\").strip()))

        return result


@dataclass(frozen=True)
class TestFile:
    """Parsed lit test file with structure."""

    doc: FileDoc  # The document
    cases: tuple[TestCase, ...]  # All test cases
    header_span: Span | None  # Lines before first delimiter
    delimiter_lines: tuple[int, ...]  # Indices of // ----- lines

    def extract_run_lines(self, raw: bool = False) -> list[str]:
        """Extract header RUN commands from test file.

        Args:
            raw: If True, return each RUN line separately (for line preservation).
                If False, join continuation lines into complete commands.

        Returns:
            List of RUN command strings

        Example:
            >>> test_file.extract_run_lines()
            ['iree-opt --split-input-file %s | FileCheck %s']
        """
        if not self.cases:
            return []

        # Get header RUN line indices from first case
        header_indices = sorted(self.cases[0].header_run_lines)

        if raw:
            # Raw mode: return each RUN line separately
            commands = []
            for idx in header_indices:
                line = self.doc.lines[idx]
                match = RUN_LINE_PATTERN.match(line.text)
                if match:
                    commands.append(match.group(1).strip())
            return commands

        # Normal mode: join continuation lines
        commands = []
        current_cmd = None
        continuing = False

        for idx in header_indices:
            line = self.doc.lines[idx]
            match = RUN_LINE_PATTERN.match(line.text)
            if not match:
                continue

            content = match.group(1).strip()

            if continuing and current_cmd is not None:
                current_cmd = (
                    current_cmd.rstrip("\\").strip()
                    + " "
                    + content.lstrip("\\").strip()
                )
            else:
                current_cmd = content

            continuing = content.endswith("\\")

            if not continuing:
                commands.append(current_cmd.rstrip("\\").strip())
                current_cmd = None

        if current_cmd is not None:
            commands.append(current_cmd.rstrip("\\").strip())

        return commands

    def find_case_by_number(self, case_number: int) -> TestCase:
        """Find test case by number.

        Args:
            case_number: 1-based case number

        Returns:
            TestCase object

        Raises:
            ValueError: If case_number is out of range

        Example:
            >>> case = test_file.find_case_by_number(2)
            >>> case.name
            'second_function'
        """
        if case_number < 1 or case_number > len(self.cases):
            raise ValueError(
                f"Case number {case_number} out of range (1-{len(self.cases)})"
            )
        return self.cases[case_number - 1]

    def find_case_by_name(self, name: str) -> TestCase:
        """Find test case by CHECK-LABEL name.

        Args:
            name: Function name (with or without @ prefix)

        Returns:
            TestCase object

        Raises:
            ValueError: If no case with given name found

        Example:
            >>> case = test_file.find_case_by_name("my_function")
            >>> case.number
            2
        """
        search_name = name.lstrip("@")
        for case in self.cases:
            if case.name == search_name:
                return case
        raise ValueError(f"No test case found with name '{name}'")

    def find_case_by_line(self, line_number: int) -> TestCase:
        """Find test case containing a specific line number.

        Useful for locating the test case from error messages that reference
        line numbers.

        Args:
            line_number: 1-based line number

        Returns:
            TestCase containing the specified line

        Raises:
            ValueError: If line_number is out of range

        Example:
            >>> # Error at line 142 - which case is that?
            >>> case = test_file.find_case_by_line(142)
            >>> case.name
            'third_function'
        """
        if line_number < 1:
            raise ValueError(f"Line number must be >= 1, got {line_number}")

        for case in self.cases:
            if case.start_line <= line_number <= case.end_line:
                return case

        if self.cases:
            last_line = self.cases[-1].end_line
            raise ValueError(
                f"Line number {line_number} out of range (file has {last_line} lines)"
            )
        raise ValueError(f"Line number {line_number} out of range (empty file)")

    @property
    def uses_split_input_file(self) -> bool:
        """Check if file uses --split-input-file mode.

        Examines RUN lines to determine if --split-input-file flag is present.
        This affects how FileCheck interprets test boundaries.

        Returns:
            True if any RUN line contains --split-input-file

        Example:
            >>> if test_file.uses_split_input_file:
            ...     # CHECK-DAG can match across // ----- boundaries
            ...     validate_check_label_anchoring()
        """
        run_lines = self.extract_run_lines()
        return any("--split-input-file" in cmd for cmd in run_lines)


def _tag_lines(doc: FileDoc) -> FileDoc:
    """Classify lines by structural type.

    Tags lines as RUN_HEADER, RUN_CASE, DELIMITER, CHECK, or BODY.
    RUN lines before first delimiter are header RUNs.
    RUN lines after delimiter are case-local RUNs.

    Args:
        doc: Document to tag

    Returns:
        New FileDoc with lines tagged
    """
    delimiter_indices = _find_delimiter_indices(doc)
    first_delimiter_idx = delimiter_indices[0] if delimiter_indices else len(doc.lines)

    # Build new lines in single O(N) pass to avoid O(N²) behavior
    updated_lines = []
    for idx, line in enumerate(doc.lines):
        tags = set()

        # Check for delimiter
        if DELIMITER_PATTERN.match(line.text):
            tags.add(TAG_DELIMITER)

        # Check for RUN line
        elif RUN_LINE_PATTERN.match(line.text):
            if idx < first_delimiter_idx:
                tags.add(TAG_RUN_HEADER)
            else:
                tags.add(TAG_RUN_CASE)

        # Check for CHECK directive
        elif CHECK_PATTERN.match(line.text):
            tags.add(TAG_CHECK)

        # Everything else is body
        else:
            tags.add(TAG_BODY)

        # Create new Line with tags
        updated_lines.append(
            Line(
                text=line.text,
                newline=line.newline,
                index=line.index,
                source_line=line.source_line,
                tags=frozenset(tags),
            )
        )

    return FileDoc(lines=tuple(updated_lines), path=doc.path)


def _find_delimiter_indices(doc: FileDoc) -> list[int]:
    """Find all // ----- delimiter line indices.

    Args:
        doc: Document to search

    Returns:
        List of 0-based indices where delimiters appear
    """
    indices = []
    for idx, line in enumerate(doc.lines):
        if DELIMITER_PATTERN.match(line.text):
            indices.append(idx)
    return indices


def _build_case_spans(doc: FileDoc, delimiter_indices: list[int]) -> list[Span]:
    """Build span for each test case based on delimiter positions.

    Cases are regions between delimiters. If no delimiters, entire file
    is one case. Empty cases (consecutive delimiters) get zero-length spans.

    Args:
        doc: Document being parsed
        delimiter_indices: List of delimiter line indices

    Returns:
        List of Span objects, one per case
    """
    if not delimiter_indices:
        # No delimiters: entire file is one case
        return [Span(start=0, end=len(doc.lines))]

    spans = []

    # First case: start of file to first delimiter
    spans.append(Span(start=0, end=delimiter_indices[0]))

    # Middle cases: between consecutive delimiters
    for i in range(len(delimiter_indices) - 1):
        start = delimiter_indices[i] + 1
        end = delimiter_indices[i + 1]
        spans.append(Span(start=start, end=end))

    # Last case: after last delimiter to EOF
    last_start = delimiter_indices[-1] + 1
    if last_start < len(doc.lines):
        spans.append(Span(start=last_start, end=len(doc.lines)))

    return spans


def _extract_case_metadata(
    doc: FileDoc, span: Span, case_number: int, header_run_indices: frozenset[int]
) -> TestCase:
    """Extract metadata for a single test case.

    Extracts:
    - CHECK-LABEL names
    - Local RUN line indices
    - CHECK directive count

    Args:
        doc: Document containing the case
        span: Span defining case boundaries
        case_number: 1-based case number
        header_run_indices: File-level RUN line indices

    Returns:
        TestCase with extracted metadata
    """
    case_lines = doc.slice(span)

    # Extract CHECK-LABEL names
    primary_name, all_names = CheckLabelExtractor.extract_primary_name(case_lines)

    # Find local RUN lines within this case
    local_run_indices = set()
    for idx in range(span.start, span.end):
        if TAG_RUN_CASE in doc.lines[idx].tags:
            local_run_indices.add(idx)

    # Count CHECK directives
    check_count = sum(1 for line in case_lines if TAG_CHECK in line.tags)

    return TestCase(
        doc=doc,
        number=case_number,
        name=primary_name,
        all_names=all_names,
        span=span,
        header_run_lines=header_run_indices,
        local_run_lines=frozenset(local_run_indices),
        check_count=check_count,
    )


def parse_test_file(path: Path) -> TestFile:
    """Parse a lit test file into structured representation.

    This is the main entry point for parsing. It:
    1. Reads file into FileDoc
    2. Tags lines by type
    3. Finds delimiters
    4. Builds case spans
    5. Extracts metadata for each case

    Args:
        path: Path to lit test file

    Returns:
        TestFile with all structure and metadata

    Example:
        >>> test_file = parse_test_file(Path("test.mlir"))
        >>> for case in test_file.cases:
        ...     print(f"Case {case.number}: {case.name}")
    """
    # Read and parse file
    text = path.read_text(encoding="utf-8")
    doc = FileDoc.from_text(text, path=path)

    # Tag lines by type
    doc = _tag_lines(doc)

    # Find structure
    delimiter_indices = _find_delimiter_indices(doc)
    case_spans = _build_case_spans(doc, delimiter_indices)

    # Determine header span (lines before first delimiter)
    header_span = Span(start=0, end=delimiter_indices[0]) if delimiter_indices else None

    # Find header RUN lines
    header_run_indices = frozenset(
        idx for idx in range(len(doc.lines)) if TAG_RUN_HEADER in doc.lines[idx].tags
    )

    # Extract metadata for each case
    cases = []
    for case_num, span in enumerate(case_spans, start=1):
        case = _extract_case_metadata(doc, span, case_num, header_run_indices)
        cases.append(case)

    return TestFile(
        doc=doc,
        cases=tuple(cases),
        header_span=header_span,
        delimiter_lines=tuple(delimiter_indices),
    )
