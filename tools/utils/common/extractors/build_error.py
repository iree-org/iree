# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

r"""Build error extractor (C/C++ compilation and linker errors).

Extracts C/C++ compilation errors, linker errors, and undefined references
from build logs. Handles GCC, Clang, and MSVC compiler output formats.

Uses failure-first approach:
1. Find "FAILED:" markers from build system (ninja/make)
2. Extract compiler errors from context around failure
3. Also catch linker errors (undefined reference, multiple definition)

PERFORMANCE CRITICAL - BANNED TECHNIQUES:
  ❌ DO NOT use complex regex patterns with negative character classes like:
     r"(?P<file>[^\\s:]+):(?P<line>\\d+):(?P<col>\\d+):\\s+error:\\s+(?P<msg>.+)"
     These cause catastrophic backtracking on non-matching lines (20x slower).

  ✅ INSTEAD use fast substring checks + simple string parsing:
     if ": error:" in line:
         parts = line.split(": error:", 1)
         location = parts[0].rsplit(":", 2)  # Simple, no backtracking.

  This extractor processes thousands of lines per log. Every line gets checked.
  Fast substring checks (O(n)) + simple parsing >>> regex backtracking (O(n²)).

Example usage:
    from common.extractors.build_error import BuildErrorExtractor
    from common.log_buffer import LogBuffer

    extractor = BuildErrorExtractor()
    log_buffer = LogBuffer(log_content, auto_detect_format=True)
    issues = extractor.extract(log_buffer)

    for issue in issues:
        print(f"{issue.file_path}:{issue.line}: {issue.compiler_message}")
"""

import re

from common.extractors.base import Extractor
from common.issues import BuildErrorIssue, Severity
from common.log_buffer import LogBuffer


class BuildErrorExtractor(Extractor):
    """Extracts C/C++ build errors from logs.

    Supports GCC, Clang, and MSVC compiler error formats.
    Only reports errors, not warnings or build progress.
    """

    name = "build_error"
    activation_keywords = [
        # Build errors only happen during compilation - use precise build output markers.
        "ninja:",  # Ninja build tool output (e.g., "ninja: build stopped").
        "make[",  # Make recursion (e.g., "make[1]: Entering directory").
        "FAILED:",  # Build target failure (capital F, with colon).
        "clang++",  # C++ compiler invocation.
        "g++",  # GCC C++ compiler.
        "/usr/bin/c++",  # Compiler path.
        "CMakeFiles",  # CMake build artifacts (only in build output).
        "[1/",  # Build progress marker (e.g., "[1/100] Building").
        "Compiling CXX",  # CMake build action.
    ]

    # Build system failure marker.
    # Example: "FAILED: compiler/src/CMakeFiles/iree_compiler.dir/main.cpp.o"
    _FAILED_TARGET_RE = re.compile(r"^FAILED:\s+(?P<target>.+)$")

    # Linker undefined reference pattern (GCC/Clang).
    # Example: "undefined reference to `symbol_name'"
    _UNDEFINED_REF_RE = re.compile(r"undefined reference to [`'](?P<symbol>[^'`]+)[`']")

    # MSVC linker unresolved external pattern.
    # Example: "error LNK2019: unresolved external symbol"
    _MSVC_UNRESOLVED_RE = re.compile(
        r"error LNK\d+:\s+unresolved external symbol\s+(?P<symbol>\S+)"
    )

    # Multiple definition error (linker).
    # Example: "multiple definition of `symbol_name'"
    _MULTIPLE_DEF_RE = re.compile(r"multiple definition of [`'](?P<symbol>[^'`]+)[`']")

    # Build stopped marker.
    _BUILD_STOPPED_RE = re.compile(
        r"ninja: build stopped|make.*Error \d+", re.IGNORECASE
    )

    def extract(self, log_buffer: LogBuffer) -> list[BuildErrorIssue]:
        """Extract build errors from log.

        Args:
            log_buffer: LogBuffer with build log content.

        Returns:
            List of BuildErrorIssue objects.
        """
        issues = []

        # Scan for errors.
        for line_idx, line in enumerate(log_buffer.get_lines()):
            # Check for build system failure marker.
            match = self._FAILED_TARGET_RE.match(line)
            if match:
                # Extract compiler error from context (next 20 lines).
                error_issue = self._extract_from_failed_target(
                    log_buffer, line_idx, match
                )
                if error_issue:
                    issues.append(error_issue)
                continue

            # Check for GCC/Clang compile errors (fast string check, then parse).
            # Format: "file.cpp:123:45: error: message"
            if ": error:" in line:
                error_issue = self._try_parse_gcc_error(log_buffer, line_idx, line)
                if error_issue:
                    issues.append(error_issue)
                    continue

            # Check for MSVC compile errors (fast string check, then parse).
            # Format: "file.cpp(123): error C2065: message" or "file.cpp(123,45): error ..."
            if ") : error " in line or "): error " in line:
                error_issue = self._try_parse_msvc_error(log_buffer, line_idx, line)
                if error_issue:
                    issues.append(error_issue)
                    continue

            # Check for linker errors (undefined reference).
            match = self._UNDEFINED_REF_RE.search(line)
            if match:
                issues.append(self._extract_undefined_ref(log_buffer, line_idx, match))
                continue

            # Check for MSVC linker unresolved external.
            match = self._MSVC_UNRESOLVED_RE.search(line)
            if match:
                issues.append(
                    self._extract_msvc_unresolved(log_buffer, line_idx, match)
                )
                continue

            # Check for multiple definition errors.
            match = self._MULTIPLE_DEF_RE.search(line)
            if match:
                issues.append(self._extract_multiple_def(log_buffer, line_idx, match))
                continue

        return issues

    def _extract_from_failed_target(
        self, log_buffer: LogBuffer, line_idx: int, match: re.Match
    ) -> BuildErrorIssue | None:
        """Extract compiler error from FAILED target context.

        Searches next 20 lines for compiler error (GCC/Clang/MSVC format).

        Args:
            log_buffer: LogBuffer with log content.
            line_idx: Line index of FAILED marker.
            match: Regex match with target group.

        Returns:
            BuildErrorIssue if found, None otherwise.
        """
        target = match.group("target")
        lines = log_buffer.get_lines()

        # Search next 20 lines for compiler error.
        for i in range(line_idx + 1, min(line_idx + 21, len(lines))):
            # Try GCC/Clang format.
            if ": error:" in lines[i]:
                gcc_error = self._try_parse_gcc_error(log_buffer, i, lines[i])
                if gcc_error:
                    return gcc_error

            # Try MSVC format.
            if ") : error " in lines[i] or "): error " in lines[i]:
                msvc_error = self._try_parse_msvc_error(log_buffer, i, lines[i])
                if msvc_error:
                    return msvc_error

        # No specific error found - return generic failure.
        return BuildErrorIssue(
            severity=Severity.CRITICAL,
            actionable=True,
            message=f"Build failed: {target}",
            line_number=line_idx,
            error_type="build_failure",
            file_path=target,
            line=0,
            column=0,
            compiler_message=f"Build target failed: {target}",
            context_lines=log_buffer.get_context(line_idx, before=2, after=10),
        )

    def _try_parse_gcc_error(
        self, log_buffer: LogBuffer, line_idx: int, line: str
    ) -> BuildErrorIssue | None:
        """Parse GCC/Clang compilation error using string operations.

        Format: "file.cpp:123:45: error: message"

        Args:
            log_buffer: LogBuffer with log content.
            line_idx: Line index of error.
            line: Line content.

        Returns:
            BuildErrorIssue if valid format, None otherwise.
        """
        # Split on ": error:" to separate location from message.
        parts = line.split(": error:", 1)
        if len(parts) != 2:
            return None

        location_str = parts[0]
        message = parts[1].strip()

        # Parse location: "file.cpp:123:45" -> ["file.cpp", "123", "45"]
        # Use rsplit to get last 2 colons (line:col), rest is file path.
        location_parts = location_str.rsplit(":", 2)
        if len(location_parts) != 3:
            return None

        file_path = location_parts[0]
        line_str = location_parts[1]
        col_str = location_parts[2]

        # Validate line and column are digits.
        if not line_str.isdigit() or not col_str.isdigit():
            return None

        return BuildErrorIssue(
            severity=Severity.CRITICAL,
            actionable=True,
            message=message,
            line_number=line_idx,
            error_type="compile",
            file_path=file_path,
            line=int(line_str),
            column=int(col_str),
            compiler_message=message,
            compiler="gcc/clang",
            context_lines=log_buffer.get_context(line_idx, before=3, after=5),
        )

    def _try_parse_msvc_error(
        self, log_buffer: LogBuffer, line_idx: int, line: str
    ) -> BuildErrorIssue | None:
        """Parse MSVC compilation error using string operations.

        Format: "file.cpp(123): error C2065: message"
        Format: "file.cpp(123,45): error C2065: message"

        Args:
            log_buffer: LogBuffer with log content.
            line_idx: Line index of error.
            line: Line content.

        Returns:
            BuildErrorIssue if valid format, None otherwise.
        """
        # Split on " error " or ": error " to separate location from error.
        if "): error " in line:
            parts = line.split("): error ", 1)
        elif ") : error " in line:
            parts = line.split(") : error ", 1)
        else:
            return None

        if len(parts) != 2:
            return None

        location_str = parts[0]  # "file.cpp(123,45" or "file.cpp(123"
        error_str = parts[1]  # "C2065: message"

        # Parse location: split on "(" to get file and line/col.
        if "(" not in location_str:
            return None

        file_path, line_col_str = location_str.rsplit("(", 1)

        # Parse line and optional column.
        if "," in line_col_str:
            # Format: "123,45"
            line_str, col_str = line_col_str.split(",", 1)
            if not line_str.isdigit() or not col_str.isdigit():
                return None
            line_num = int(line_str)
            col_num = int(col_str)
        else:
            # Format: "123"
            if not line_col_str.isdigit():
                return None
            line_num = int(line_col_str)
            col_num = 0

        # Extract error code if present (e.g., "C2065: message").
        error_code = ""
        message = error_str.strip()
        if message.startswith("C") and ":" in message:
            code_parts = message.split(":", 1)
            if code_parts[0][1:].isdigit():  # "C2065" -> "2065" is digits
                error_code = code_parts[0]
                message = code_parts[1].strip()

        full_message = f"{error_code}: {message}" if error_code else message

        return BuildErrorIssue(
            severity=Severity.CRITICAL,
            actionable=True,
            message=full_message,
            line_number=line_idx,
            error_type="compile",
            file_path=file_path,
            line=line_num,
            column=col_num,
            compiler_message=message,
            compiler="msvc",
            context_lines=log_buffer.get_context(line_idx, before=3, after=5),
        )

    def _extract_undefined_ref(
        self, log_buffer: LogBuffer, line_idx: int, match: re.Match
    ) -> BuildErrorIssue:
        """Extract undefined reference linker error.

        Args:
            log_buffer: LogBuffer with log content.
            line_idx: Line index of error.
            match: Regex match with symbol group.

        Returns:
            BuildErrorIssue instance.
        """
        symbol = match.group("symbol")

        return BuildErrorIssue(
            severity=Severity.CRITICAL,
            actionable=True,
            message=f"Undefined reference to '{symbol}'",
            line_number=line_idx,
            error_type="undefined_reference",
            file_path="",  # Linker errors don't have specific file.
            line=0,
            column=0,
            compiler_message=f"undefined reference to `{symbol}'",
            compiler="ld",
            context_lines=log_buffer.get_context(line_idx, before=5, after=3),
        )

    def _extract_msvc_unresolved(
        self, log_buffer: LogBuffer, line_idx: int, match: re.Match
    ) -> BuildErrorIssue:
        """Extract MSVC unresolved external symbol linker error.

        Args:
            log_buffer: LogBuffer with log content.
            line_idx: Line index of error.
            match: Regex match with symbol group.

        Returns:
            BuildErrorIssue instance.
        """
        symbol = match.group("symbol")

        return BuildErrorIssue(
            severity=Severity.CRITICAL,
            actionable=True,
            message=f"Unresolved external symbol '{symbol}'",
            line_number=line_idx,
            error_type="undefined_reference",
            file_path="",
            line=0,
            column=0,
            compiler_message=f"unresolved external symbol {symbol}",
            compiler="link.exe",
            context_lines=log_buffer.get_context(line_idx, before=5, after=3),
        )

    def _extract_multiple_def(
        self, log_buffer: LogBuffer, line_idx: int, match: re.Match
    ) -> BuildErrorIssue:
        """Extract multiple definition linker error.

        Args:
            log_buffer: LogBuffer with log content.
            line_idx: Line index of error.
            match: Regex match with symbol group.

        Returns:
            BuildErrorIssue instance.
        """
        symbol = match.group("symbol")

        return BuildErrorIssue(
            severity=Severity.CRITICAL,
            actionable=True,
            message=f"Multiple definition of '{symbol}'",
            line_number=line_idx,
            error_type="multiple_definition",
            file_path="",
            line=0,
            column=0,
            compiler_message=f"multiple definition of `{symbol}'",
            compiler="ld",
            context_lines=log_buffer.get_context(line_idx, before=5, after=5),
        )
