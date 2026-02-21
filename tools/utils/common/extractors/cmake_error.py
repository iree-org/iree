# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CMake configuration error extractor.

Extracts CMake configuration failures from build logs. Uses two-phase detection:
1. Find definitive failure marker: "-- Configuring incomplete, errors occurred!"
2. Search backward for "CMake Error at" lines to collect all errors

This approach prevents false positives from informational messages like:
- "Could NOT find Python module pygments" (optional dependency search)
- "-- Looking for pthread.h - not found" (feature probes)

Example usage:
    from common.extractors.cmake_error import CMakeErrorExtractor
    from common.log_buffer import LogBuffer

    extractor = CMakeErrorExtractor()
    log_buffer = LogBuffer(log_content, auto_detect_format=True)
    issues = extractor.extract(log_buffer)

    for issue in issues:
        print(f"CMake error at {issue.file_path}:{issue.line}: {issue.message}")
"""

import re

from common.extractors.base import Extractor
from common.issues import CMakeErrorIssue, Severity
from common.log_buffer import LogBuffer


class CMakeErrorExtractor(Extractor):
    """Extracts CMake configuration errors from logs.

    Only reports errors when CMake configuration definitively fails (indicated
    by "-- Configuring incomplete, errors occurred!" marker). Ignores
    informational messages about optional dependencies.
    """

    name = "cmake_error"
    activation_keywords = ["cmake", "CMake"]  # Only run on CMake configure logs.

    # Definitive failure marker - CMake configuration failed.
    _CONFIGURE_INCOMPLETE_RE = re.compile(
        r"--\s+Configuring incomplete, errors occurred!"
    )

    # CMake error pattern with file location.
    # Example: "CMake Error at compiler/CMakeLists.txt:123 (find_package):"
    _CMAKE_ERROR_RE = re.compile(
        r"CMake Error at (?P<file>[^:]+):(?P<line>\d+)\s+\((?P<function>[^)]+)\):"
    )

    # CMake fatal error pattern (no file location).
    # Example: "CMake Error: Could not find cmake module file: ..."
    _CMAKE_FATAL_RE = re.compile(r"CMake Error:\s+(?P<msg>.+)")

    # Patterns to IGNORE (false positives).
    _IGNORE_PATTERNS = [
        re.compile(r"Could NOT find", re.IGNORECASE),  # Optional dependency search.
        re.compile(r"--\s+Looking for .+ - not found"),  # Feature probe.
        re.compile(r"--\s+.+\s+disabled"),  # Optional feature disabled.
    ]

    def extract(self, log_buffer: LogBuffer) -> list[CMakeErrorIssue]:
        """Extract CMake configuration errors from log.

        Uses two-phase detection:
        1. Check if configuration failed (look for "Configuring incomplete")
        2. If failed, search backward for all "CMake Error" lines

        Args:
            log_buffer: LogBuffer with build log content.

        Returns:
            List of CMakeErrorIssue objects (empty if configuration succeeded).
        """
        issues = []

        # Phase 1: Find definitive failure marker.
        failure_line_idx = self._find_configuration_failure(log_buffer)
        if failure_line_idx is None:
            # Configuration succeeded - no errors to report.
            return []

        # Phase 2: Search backward from failure marker for all CMake errors.
        lines = log_buffer.get_lines()
        for line_idx in range(failure_line_idx, -1, -1):
            line = lines[line_idx]

            # Skip false positives.
            if self._should_ignore_line(line):
                continue

            # Check for CMake Error with file location.
            match = self._CMAKE_ERROR_RE.search(line)
            if match:
                issue = self._extract_cmake_error(log_buffer, line_idx, match)
                issues.append(issue)
                continue

            # Check for CMake Fatal Error (no file location).
            match = self._CMAKE_FATAL_RE.search(line)
            if match:
                issue = self._extract_cmake_fatal(log_buffer, line_idx, match)
                issues.append(issue)

        # Reverse to maintain chronological order (we searched backward).
        issues.reverse()
        return issues

    def _find_configuration_failure(self, log_buffer: LogBuffer) -> int | None:
        """Find the "Configuring incomplete, errors occurred!" marker.

        This is the definitive indicator that CMake configuration failed.

        Args:
            log_buffer: LogBuffer with log content.

        Returns:
            Line index of failure marker, or None if configuration succeeded.
        """
        for line_idx, line in enumerate(log_buffer.get_lines()):
            if self._CONFIGURE_INCOMPLETE_RE.search(line):
                return line_idx
        return None

    def _should_ignore_line(self, line: str) -> bool:
        """Check if line should be ignored (false positive).

        Args:
            line: Log line to check.

        Returns:
            True if line should be ignored.
        """
        return any(pattern.search(line) for pattern in self._IGNORE_PATTERNS)

    def _extract_cmake_error(
        self, log_buffer: LogBuffer, line_idx: int, match: re.Match
    ) -> CMakeErrorIssue:
        """Extract CMake error with file location.

        Error format:
            CMake Error at <file>:<line> (<function>):
              <indented error message>
              <more error lines>

        Args:
            log_buffer: LogBuffer with log content.
            line_idx: Line index of error.
            match: Regex match with file, line, function groups.

        Returns:
            CMakeErrorIssue instance.
        """
        file_path = match.group("file")
        line = int(match.group("line"))
        function = match.group("function")

        # Extract indented error message (next 1-10 lines that are indented).
        error_message_lines = []
        lines = log_buffer.get_lines()
        for i in range(line_idx + 1, min(line_idx + 11, len(lines))):
            if lines[i].startswith("  ") or lines[i].startswith("\t"):
                error_message_lines.append(lines[i].strip())
            elif lines[i].strip() == "":
                # Blank line continues message.
                continue
            else:
                # Non-indented line - end of error message.
                break

        error_message = " ".join(error_message_lines) if error_message_lines else ""

        return CMakeErrorIssue(
            severity=Severity.CRITICAL,
            actionable=True,
            message=f"CMake error in {function}: {error_message}"
            if error_message
            else f"CMake error in {function}",
            line_number=line_idx,
            error_type="configuration",
            cmake_file=file_path,
            cmake_line=line,
            cmake_command=function,
            context_lines=log_buffer.get_context(line_idx, before=2, after=10),
        )

    def _extract_cmake_fatal(
        self, log_buffer: LogBuffer, line_idx: int, match: re.Match
    ) -> CMakeErrorIssue:
        """Extract CMake fatal error (no file location).

        Args:
            log_buffer: LogBuffer with log content.
            line_idx: Line index of error.
            match: Regex match with msg group.

        Returns:
            CMakeErrorIssue instance.
        """
        error_message = match.group("msg").strip()

        return CMakeErrorIssue(
            severity=Severity.CRITICAL,
            actionable=True,
            message=f"CMake fatal error: {error_message}",
            line_number=line_idx,
            error_type="fatal",
            cmake_file="",
            cmake_line=0,
            cmake_command=None,
            context_lines=log_buffer.get_context(line_idx, before=5, after=10),
        )
