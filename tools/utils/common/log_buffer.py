# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

r"""Log buffer for efficient log traversal and context extraction.

This module provides LogBuffer, a utility class for efficient access to log
content with support for:
- Line-based access with 0-indexed line numbers
- Context extraction (lines around a specific line)
- Byte offset to line number conversion
- Forward and backward pattern searching with configurable limits
- Automatic stripping of log format prefixes (GitHub Actions, ctest, cmake, etc.)

Example usage:
    # Basic usage without prefix stripping.
    log_buffer = LogBuffer(log_content)

    # Auto-detect and strip log prefixes (GitHub Actions, etc.).
    log_buffer = LogBuffer(log_content, auto_detect_format=True)

    # Explicitly strip specific formats.
    log_buffer = LogBuffer(log_content, strip_formats=['github_actions', 'ctest'])

    # Get specific line (from stripped content if stripping enabled).
    line = log_buffer.get_line(42)

    # Get context around error line.
    context = log_buffer.get_lines_around(42, context=5)

    # Search backward for RUN command.
    match = log_buffer.find_previous_match(offset, r'^// RUN:')

    # Access original unstripped content if needed.
    original = log_buffer.get_original_content()
"""

import bisect
import re
from re import Match, Pattern


class LogBuffer:
    """Provides efficient access and traversal for log content.

    This class wraps log content and provides efficient utilities for
    extractors to navigate and extract context. All line numbers are
    0-indexed to match Python list indexing.

    Supports automatic stripping of log format prefixes (GitHub Actions
    timestamps, ctest markers, etc.) to provide clean content for pattern
    matching.

    Attributes:
        content: Log content as string (preprocessed if stripping enabled).
    """

    # Compiled patterns for format detection (compiled once, reused for all logs).
    _GITHUB_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s")

    def __init__(
        self,
        log_content: str,
        strip_formats: list[str] | None = None,
        auto_detect_format: bool = False,
    ) -> None:
        """Initialize LogBuffer with optional prefix stripping.

        Args:
            log_content: Full log content as a string.
            strip_formats: List of format types to strip. Supported formats:
                - 'github_actions': GitHub Actions timestamp prefix
                - 'ctest': ctest [ctest] prefix
                - 'cmake': cmake [cmake] or [build] prefix
                None (default) means no stripping.
            auto_detect_format: If True, auto-detect format from content and
                strip automatically. Overrides strip_formats if True.
        """
        # Preserve original content for debugging/access.
        self._original_content = log_content
        self._strip_formats = strip_formats or []

        # Auto-detect if requested.
        if auto_detect_format:
            self._strip_formats = self._detect_formats(log_content)

        # Apply stripping if formats specified.
        if self._strip_formats:
            self.content = self._apply_stripping(log_content, self._strip_formats)
        else:
            self.content = log_content

        # Build internal structures from preprocessed content.
        self._lines = self.content.splitlines()
        self._line_offsets = self._compute_line_offsets()

    def _compute_line_offsets(self) -> list[int]:
        """Compute character offset for each line start.

        Note: These are character offsets (Python string indices), not byte
        offsets. For ASCII logs these are identical, but for UTF-8 logs with
        multi-byte characters they differ. Since we use these offsets only
        internally for line lookups, character offsets are sufficient.

        Returns:
            List of character offsets, one per line in the log.
        """
        offsets = []
        offset = 0
        for line in self._lines:
            offsets.append(offset)
            offset += len(line) + 1  # +1 for newline character.
        return offsets

    def _detect_formats(self, content: str) -> list[str]:
        """Auto-detect log format from content.

        Samples first few lines to determine format type. Currently detects:
        - GitHub Actions (ISO timestamp format)
        - Future: ctest, cmake, ninja formats

        Args:
            content: Log content to analyze.

        Returns:
            List of detected format identifiers.
        """
        # Sample first 10 lines for detection.
        sample_lines = content.split("\n", 10)[:10]

        # Single-pass format detection: check each line for all patterns.
        # Use set for efficient membership testing and to support multiple concurrent formats.
        detected_formats = set()

        for line in sample_lines:
            stripped = line.strip()

            # Check for each format type (logs can have multiple formats).
            if self._GITHUB_PATTERN.search(line):
                detected_formats.add("github_actions")

            if stripped.startswith("[ctest]"):
                detected_formats.add("ctest")

            if stripped.startswith(
                ("[cmake]", "[build]", "[main]", "[proc]", "[driver]")
            ):
                detected_formats.add("cmake")

        # Return as list for consistency with API.
        return list(detected_formats)

    def _apply_stripping(self, content: str, formats: list[str]) -> str:
        """Apply prefix stripping for specified formats.

        Processes content line-by-line, applying format-specific stripping
        functions in order. Preserves line structure (newlines).

        Args:
            content: Original log content.
            formats: List of format types to strip (e.g., ['github_actions']).

        Returns:
            Content with prefixes stripped.
        """
        lines = content.splitlines()
        stripped_lines = []

        for line in lines:
            stripped_line = line
            for fmt in formats:
                if fmt == "github_actions":
                    stripped_line = self._strip_github_actions_prefix(stripped_line)
                elif fmt == "ctest":
                    stripped_line = self._strip_ctest_prefix(stripped_line)
                elif fmt == "cmake":
                    stripped_line = self._strip_cmake_prefix(stripped_line)
            stripped_lines.append(stripped_line)

        return "\n".join(stripped_lines)

    @staticmethod
    def _strip_github_actions_prefix(line: str) -> str:
        r"""Strip GitHub Actions log prefix from a line.

        GitHub Actions logs have format:
        {job_name}\t{step_name}\t{timestamp}Z {content}

        Example:
        linux_x64_gcc\tUNKNOWN STEP\t2025-09-16T09:23:05.2141216Z Content here

        The timestamp is ISO 8601 format with fractional seconds ending in Z.
        BOM character (\\ufeff) may appear before timestamp on first line.

        Args:
            line: Log line potentially with GitHub Actions prefix.

        Returns:
            Line with prefix stripped, or original if no prefix found.
        """
        # Pattern: anything followed by ISO timestamp ending in Z, then space, then content.
        # The .+? non-greedy match captures job/step tabs before timestamp.
        match = re.match(r"^.+?\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s+(.*)$", line)
        if match:
            return match.group(1)
        return line

    @staticmethod
    def _strip_ctest_prefix(line: str) -> str:
        """Strip ctest log prefix from a line.

        ctest logs have format: [ctest] content

        Args:
            line: Log line potentially with ctest prefix.

        Returns:
            Line with prefix stripped, or original if no prefix found.
        """
        # Simple prefix removal.
        if line.startswith("[ctest] "):
            return line[8:]  # len('[ctest] ') == 8
        return line

    @staticmethod
    def _strip_cmake_prefix(line: str) -> str:
        """Strip cmake/build log prefix from a line.

        cmake/VSCode logs have format: [cmake] content or [build] content

        Args:
            line: Log line potentially with cmake prefix.

        Returns:
            Line with prefix stripped, or original if no prefix found.
        """
        if line.startswith("[cmake] "):
            return line[8:]  # len('[cmake] ') == 8
        if line.startswith("[build] "):
            return line[8:]  # len('[build] ') == 8
        return line

    def get_original_content(self) -> str:
        """Get original unprocessed log content.

        Returns:
            Original log content before any prefix stripping.
        """
        return self._original_content

    def get_line(self, line_num: int) -> str | None:
        """Get line by number (0-indexed).

        Args:
            line_num: Line number (0-indexed).

        Returns:
            Line content if line_num is valid, None otherwise.
        """
        if 0 <= line_num < len(self._lines):
            return self._lines[line_num]
        return None

    def get_lines(self) -> list[str]:
        """Get all lines in the log.

        Returns:
            List of all lines (after prefix stripping if enabled).
        """
        return self._lines

    def get_lines_around(self, line_num: int, context: int = 5) -> list[str]:
        """Get lines around a line number with context.

        Args:
            line_num: Center line number (0-indexed).
            context: Number of lines before and after to include.

        Returns:
            List of lines including context. Always includes the center line
            if it exists, plus up to `context` lines before and after.
        """
        start = max(0, line_num - context)
        end = min(len(self._lines), line_num + context + 1)
        return self._lines[start:end]

    def get_context(self, line_num: int, before: int = 5, after: int = 5) -> list[str]:
        """Get context lines around a line number.

        Args:
            line_num: Center line number (0-indexed).
            before: Number of lines before to include.
            after: Number of lines after to include.

        Returns:
            List of lines including context. Always includes the center line
            if it exists, plus up to `before` lines before and `after` lines after.
        """
        start = max(0, line_num - before)
        end = min(len(self._lines), line_num + after + 1)
        return self._lines[start:end]

    def get_line_number_from_offset(self, offset: int) -> int:
        """Convert character offset to line number using binary search.

        Args:
            offset: Character offset into the log content.

        Returns:
            Line number (0-indexed) containing the offset. Returns -1 for
            empty logs, or the last line number if offset is beyond the end.
        """
        if not self._line_offsets:
            return -1  # Empty log.

        # Binary search for the insertion point.
        # bisect_left returns index where offset would be inserted to maintain order.
        idx = bisect.bisect_left(self._line_offsets, offset)

        if idx == len(self._line_offsets):
            # Offset is beyond the start of the last line.
            return len(self._lines) - 1
        if self._line_offsets[idx] == offset:
            # Offset exactly matches a line start.
            return idx
        # Offset is within a line (before idx).
        return max(0, idx - 1)

    def _get_compiled_pattern(self, pattern: str | Pattern) -> Pattern:
        """Get compiled regex pattern from string or Pattern object.

        Args:
            pattern: Regex pattern as string or pre-compiled Pattern.

        Returns:
            Compiled re.Pattern object.
        """
        if isinstance(pattern, str):
            return re.compile(pattern)
        return pattern

    def find_previous_match(
        self, start_offset: int, pattern: str | Pattern, max_lines: int = 50
    ) -> Match | None:
        """Search backward from offset for regex pattern.

        Searches backward from the line containing start_offset (inclusive),
        up to max_lines away, for the first line matching the pattern.

        Args:
            start_offset: Character offset to start searching from.
            pattern: Regex pattern (string or compiled Pattern).
            max_lines: Maximum number of lines to search backward (including
                start line).

        Returns:
            First regex Match object found, or None if no match within range.
        """
        compiled_pattern = self._get_compiled_pattern(pattern)
        start_line = self.get_line_number_from_offset(start_offset)
        if start_line < 0:
            return None  # Empty log.

        search_start = max(0, start_line - max_lines + 1)

        # Search backwards through lines, including start_line.
        for line_num in range(start_line, search_start - 1, -1):
            line = self.get_line(line_num)
            if line:
                match = compiled_pattern.search(line)
                if match:
                    return match
        return None

    def find_next_match(
        self, start_offset: int, pattern: str | Pattern, max_lines: int = 50
    ) -> Match | None:
        """Search forward from offset for regex pattern.

        Searches forward from the line containing start_offset (inclusive),
        up to max_lines away, for the first line matching the pattern.

        Args:
            start_offset: Character offset to start searching from.
            pattern: Regex pattern (string or compiled Pattern).
            max_lines: Maximum number of lines to search forward (including
                start line).

        Returns:
            First regex Match object found, or None if no match within range.
        """
        compiled_pattern = self._get_compiled_pattern(pattern)
        start_line = self.get_line_number_from_offset(start_offset)
        if start_line < 0:
            return None  # Empty log.

        search_end = min(len(self._lines), start_line + max_lines)

        for line_num in range(start_line, search_end):
            line = self.get_line(line_num)
            if line:
                match = compiled_pattern.search(line)
                if match:
                    return match
        return None

    def find_all_matches(self, pattern: str | Pattern) -> list[tuple[int, Match]]:
        """Find all matches of a pattern in the entire log.

        Args:
            pattern: Regex pattern (string or compiled Pattern).

        Returns:
            List of (line_number, Match) tuples for all matches found.
            Line numbers are 0-indexed.
        """
        compiled_pattern = self._get_compiled_pattern(pattern)
        matches = []
        for line_num, line in enumerate(self._lines):
            match = compiled_pattern.search(line)
            if match:
                matches.append((line_num, match))
        return matches

    @property
    def line_count(self) -> int:
        """Get total number of lines in the log."""
        return len(self._lines)
