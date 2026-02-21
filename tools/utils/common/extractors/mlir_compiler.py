# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Extractor for MLIR compiler errors (iree-compile, mlir-opt, iree-opt).

This extractor detects and parses MLIR compiler diagnostic messages including:
- Compilation errors (type mismatches, invalid operations, etc.)
- Operation-specific failures (failed to tile, out-of-bounds, etc.)
- Multi-line diagnostics with notes and operation dumps

Key features:
- Multi-line error collection: source snippet + caret + notes
- Operation name extraction: from message or source code
- Graceful degradation: Extracts partial data when reports incomplete
- FileCheck exclusion: Skips test infrastructure errors (handled separately)
- Lit-tools compatible: File paths ready for iree-lit-test/iree-lit-extract

Design philosophy:
"Extract compiler errors that help LLMs debug MLIR code, not test infrastructure failures."
"""

import re

from common.extractors.base import Extractor
from common.issues import Issue, MLIRCompilerIssue, Severity
from common.log_buffer import LogBuffer


class MLIRCompilerExtractor(Extractor):
    """Extracts MLIR compiler error diagnostics from logs."""

    name = "mlir_compiler"
    activation_keywords = [".mlir"]  # Only run on logs compiling MLIR files.

    # Patterns to exclude (test infrastructure, not compiler errors).
    # Includes both CHECK prefixes and generic FileCheck error messages.
    FILECHECK_PATTERNS = [
        "CHECK:",
        "CHECK-LABEL:",
        "CHECK-SAME:",
        "CHECK-NOT:",
        "CHECK-DAG:",
        "CHECK-NEXT:",
        # Generic FileCheck failure messages (stable upstream LLVM patterns).
        "expected string not found in input",
        "is not on the same line as previous match",
        "possible intended match here",
    ]

    # Source snippet collection limit.
    SOURCE_SNIPPET_MAX_LINES = 3  # Max lines before caret or blank.

    # Compiled regex patterns (defined once, not per-line).
    _ERROR_LINE_RE = re.compile(r"^(.+\.mlir):(\d+):(\d+): error: (.+)$")
    _NOTE_LINE_RE = re.compile(r"^(.+\.mlir):(\d+):(\d+): note: (.+)$")
    _DIAGNOSTIC_LINE_RE = re.compile(r"^.+\.mlir:\d+:\d+:")
    _ERROR_WARNING_LINE_RE = re.compile(r"^.+\.mlir:\d+:\d+: (error|warning):")
    _CARET_LINE_RE = re.compile(r"^\s*\^")
    _OP_FROM_MESSAGE_RE = re.compile(r"'([^']+)' op")
    _OP_FROM_SOURCE_RE = re.compile(r"%\w+(?::\d+)?\s*=\s*([a-z_][a-z0-9_.]*)\s+")
    _TEST_TARGET_RE = re.compile(r"/(tests|compiler|artifacts)/(.+\.mlir)$")

    # Compile command extraction patterns.
    _COMPILED_WITH_RE = re.compile(r"^\s*Compiled with:\s*$")
    _INVOKED_WITH_RE = re.compile(r"^\s*Invoked with:\s*$")
    _CD_COMMAND_RE = re.compile(
        r"\bcd\s+.*?\s+&&\s+.*\b(iree-compile|iree-opt|mlir-opt)\b"
    )
    _SHELL_XTRACE_RE = re.compile(r"^\s*\+\s+.*\b(iree-compile|iree-opt|mlir-opt)\b")

    def extract(self, log_buffer: LogBuffer) -> list[Issue]:
        """Extract all MLIR compiler errors from log.

        Args:
            log_buffer: LogBuffer with log content (may be prefix-stripped).

        Returns:
            List of MLIRCompilerIssue objects, one per detected error.
            Returns empty list if no MLIR compiler errors found.
            Excludes FileCheck errors (test infrastructure).
            Never raises exceptions - all failures are graceful.
        """
        issues = []
        i = 0

        while i < log_buffer.line_count:
            line = log_buffer.get_line(i)

            # Match primary error line: FILE.mlir:LINE:COL: error: MESSAGE
            if match := self._ERROR_LINE_RE.match(line):
                file_path = match.group(1)
                # Validate line/column numbers are positive (1-indexed).
                error_line = max(1, int(match.group(2)))
                error_column = max(1, int(match.group(3)))
                error_message = match.group(4)

                # Skip FileCheck errors (test infrastructure, not compiler).
                if any(pattern in error_message for pattern in self.FILECHECK_PATTERNS):
                    i += 1
                    continue

                # Parse full error context (source snippet, notes, etc.).
                error_data = self._parse_mlir_error(
                    log_buffer, i, file_path, error_line, error_column, error_message
                )

                # Extract compile command for reproduction.
                compile_command = self._extract_compile_command(log_buffer, i)

                # Create issue.
                issue = MLIRCompilerIssue(
                    severity=Severity.HIGH,
                    actionable=True,
                    message=f"MLIR compiler error: {error_message[:80]}",
                    line_number=i,
                    source_extractor=self.name,
                    file_path=file_path,
                    error_line=error_line,
                    error_column=error_column,
                    error_message=error_message,
                    error_type=self._infer_error_type(error_message),
                    source_snippet=error_data.get("source_snippet", ""),
                    caret_line=error_data.get("caret_line", ""),
                    operation=error_data.get("operation"),
                    notes=error_data.get("notes", []),
                    full_diagnostic=error_data.get("full_diagnostic", ""),
                    test_target=error_data.get("test_target"),
                    compile_command=compile_command,
                )
                issues.append(issue)

                # Skip to end of this error.
                i = error_data.get("next_line", i + 1)
            else:
                i += 1

        return issues

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_mlir_error(
        self,
        log_buffer: LogBuffer,
        start_line: int,
        file_path: str,
        error_line: int,
        error_column: int,
        error_message: str,
    ) -> dict:
        """Parse complete MLIR error with source snippet and notes.

        Args:
            log_buffer: LogBuffer to read from.
            start_line: Line number where error starts (0-indexed).
            file_path: Path to .mlir file with error.
            error_line: Line number in .mlir file (1-indexed).
            error_column: Column number in .mlir file (1-indexed).
            error_message: Error message text.

        Returns:
            Dict with parsed error data:
            - 'operation': str or None - MLIR operation name
            - 'source_snippet': str - MLIR source code line(s)
            - 'caret_line': str - Line with caret (^) marker
            - 'notes': List[dict] - Related diagnostic notes
            - 'full_diagnostic': str - Complete multi-line error text
            - 'test_target': str or None - Inferred test target path
            - 'next_line': int - Line number after this error
        """
        data = {}
        i = start_line + 1

        # Extract operation name from error message: 'op_name' op ...
        if op_match := self._OP_FROM_MESSAGE_RE.search(error_message):
            data["operation"] = op_match.group(1)

        # Collect source snippet (next 1-3 lines until caret or blank).
        source_lines = []
        caret_line = ""
        while (
            i < log_buffer.line_count
            and i < start_line + self.SOURCE_SNIPPET_MAX_LINES + 1
        ):
            line = log_buffer.get_line(i)

            # Stop at blank line or next diagnostic.
            if not line.strip() or self._DIAGNOSTIC_LINE_RE.match(line):
                break

            # Check for caret line (^).
            if self._CARET_LINE_RE.match(line):
                caret_line = line
                i += 1
                break

            source_lines.append(line)
            i += 1

        data["source_snippet"] = "\n".join(source_lines)
        data["caret_line"] = caret_line

        # Extract operation name from source if not in message.
        # Pattern: %result = operation_name ...
        if (
            not data.get("operation")
            and source_lines
            and (op_match := self._OP_FROM_SOURCE_RE.search(source_lines[0]))
        ):
            data["operation"] = op_match.group(1)

        # Collect related notes (note: see current operation:, called from, etc.).
        notes = []
        while i < log_buffer.line_count:
            line = log_buffer.get_line(i)

            # Match note diagnostic: FILE.mlir:LINE:COL: note: MESSAGE
            if note_match := self._NOTE_LINE_RE.match(line):
                note_file = note_match.group(1)
                # Validate line/column numbers are positive (1-indexed).
                note_line = max(1, int(note_match.group(2)))
                note_col = max(1, int(note_match.group(3)))
                note_msg = note_match.group(4)

                # Collect note body (especially for "see current operation:").
                note_body, next_i = self._collect_note_body(log_buffer, i + 1, note_msg)

                notes.append(
                    {
                        "file": note_file,
                        "line": note_line,
                        "column": note_col,
                        "type": note_msg,
                        "body": note_body,
                    }
                )

                i = next_i

            # Stop at next error or warning.
            elif self._ERROR_WARNING_LINE_RE.match(line):
                break

            # Stop at blank line.
            elif not line.strip():
                i += 1
                break

            else:
                i += 1

        data["notes"] = notes
        data["next_line"] = i

        # Collect full diagnostic text (from error line to end).
        full_lines = []
        for line_num in range(start_line, i):
            full_lines.append(log_buffer.get_line(line_num))
        data["full_diagnostic"] = "\n".join(full_lines)

        # Try to infer test target from file path.
        data["test_target"] = self._infer_test_target(file_path)

        return data

    def _collect_note_body(
        self, log_buffer: LogBuffer, start_line: int, note_msg: str
    ) -> tuple[str, int]:
        """Collect multi-line note body (e.g., operation dump).

        Args:
            log_buffer: LogBuffer to read from.
            start_line: Line after "note:" line (0-indexed).
            note_msg: Note message text (e.g., "see current operation:").

        Returns:
            Tuple of (note_body_text, next_line_number):
            - note_body_text: Collected note body (may be multi-line)
            - next_line_number: Line number after note body
        """
        body_lines = []
        i = start_line

        # For "see current operation:", collect indented MLIR dump.
        if "see current operation" in note_msg:
            while i < log_buffer.line_count:
                line = log_buffer.get_line(i)

                # Stop at next diagnostic.
                if self._DIAGNOSTIC_LINE_RE.match(line):
                    break

                # Stop at blank line.
                if not line.strip():
                    i += 1
                    break

                # Collect indented lines (operation dump).
                body_lines.append(line)
                i += 1

        # For other notes, just collect until blank line or next diagnostic.
        else:
            while i < log_buffer.line_count:
                line = log_buffer.get_line(i)

                # Stop at next diagnostic or blank line.
                if self._DIAGNOSTIC_LINE_RE.match(line) or not line.strip():
                    break

                body_lines.append(line)
                i += 1

        return "\n".join(body_lines), i

    def _infer_error_type(self, error_message: str) -> str:
        """Infer error type category from error message.

        Args:
            error_message: Error diagnostic text.

        Returns:
            Error type string (e.g., "out-of-bounds", "failed-to-tile").
            Returns empty string if no clear category.
        """
        msg_lower = error_message.lower()

        # Common error patterns - specific checks before general checks.
        if "out-of-bounds" in msg_lower or "out of bounds" in msg_lower:
            return "out-of-bounds"
        # Specific operand dominance check before general dominance check.
        if "operand #" in msg_lower and "does not dominate" in msg_lower:
            return "operand-does-not-dominate"
        if "does not dominate" in msg_lower:
            return "does-not-dominate"
        if "failed to tile" in msg_lower:
            return "failed-to-tile"
        # Note: "anaysis" typo appears in real IREE compiler output.
        if "failed to analyze" in msg_lower or "failed to anaysis" in msg_lower:
            return "failed-to-analyze"
        if "type mismatch" in msg_lower:
            return "type-mismatch"
        if "invalid operation" in msg_lower:
            return "invalid-operation"
        if "expected" in msg_lower and "to have" in msg_lower:
            return "expectation-failed"
        # No clear category - return empty string.
        return ""

    def _infer_test_target(self, file_path: str) -> str | None:
        """Infer test target path from file path.

        Args:
            file_path: Path to .mlir file (may be absolute or relative).

        Returns:
            Test target path if detectable, None otherwise.

        Examples:
            - "/home/runner/_work/iree/iree/tests/e2e/linalg/argmax.mlir"
              → "tests/e2e/linalg/argmax.mlir"
            - "model.mlir" → None
        """
        # Try to extract repository-relative path.
        # Common patterns: /path/to/iree/tests/... or /path/to/iree/compiler/...
        if match := self._TEST_TARGET_RE.search(file_path):
            return f"{match.group(1)}/{match.group(2)}"

        # If file path contains test-related keywords, keep as-is.
        if any(keyword in file_path for keyword in ["test", "compiler", "artifacts"]):
            # Try to extract just the relevant portion.
            parts = file_path.split("/")
            for i, part in enumerate(parts):
                if part in ["tests", "compiler", "artifacts"]:
                    return "/".join(parts[i:])

        # Can't infer test target - return None.
        return None

    def _clean_command(self, cmd: str) -> str:
        r"""Normalize compile command for reproduction.

        Args:
            cmd: Raw command string from log.

        Returns:
            Cleaned command with normalized whitespace and unescaped quotes.

        Transformations:
            - Strip leading/trailing whitespace
            - Unescape quotes: \" → "
            - Remove shell xtrace prefix: "+ " → ""
        """
        return cmd.strip().replace(r"\"", '"').lstrip("+ ")

    def _is_in_test_framework_context(
        self, log_buffer: LogBuffer, error_line_num: int
    ) -> bool:
        """Detect if error is in ONNX/pytest/Python API test context.

        These contexts include large IR dumps between error and compile command,
        so we need to scan further ahead.

        Args:
            log_buffer: LogBuffer to search.
            error_line_num: Line number where error was found (0-indexed).

        Returns:
            True if in test framework context requiring extended forward search.
        """
        # Look backward up to 30 lines for test framework markers.
        search_start = max(0, error_line_num - 30)
        for i in range(error_line_num, search_start - 1, -1):
            line = log_buffer.get_line(i)
            # ONNX/pytest markers.
            if "Error invoking iree-compile" in line:
                return True
            if "Stderr diagnostics:" in line:
                return True
            if "IREE compile and run:" in line:
                return True
            # Python API markers.
            if "CompilerToolError" in line:
                return True
            # Pytest failure markers.
            if "_ IREE compile" in line or "test_onnx" in line:
                return True
        return False

    def _find_test_boundary(self, log_buffer: LogBuffer, start_line: int) -> int:
        """Find the end of current test section for bounded forward search.

        Args:
            log_buffer: LogBuffer to search.
            start_line: Line to start searching from (0-indexed).

        Returns:
            Line number of test boundary, or buffer end if no boundary found.
        """
        # Test boundary markers (indicate start of new test or section).
        # These should be specific enough to not appear within IR dumps.
        boundary_markers = [
            "======",  # Pytest section divider (row of equals).
            "------",  # Pytest test divider (row of dashes).
            "_ IREE compile and run:",  # Next ONNX test.
            "____ ",  # Pytest failure section (e.g., "____ test_name ____").
        ]

        # Scan forward for boundary markers (max 5000 lines to avoid infinite loops).
        max_scan = min(log_buffer.line_count, start_line + 5000)
        # Skip first few lines to avoid matching markers in current test's output.
        search_start = start_line + 10
        for i in range(search_start, max_scan):
            line = log_buffer.get_line(i)
            # Check if line starts with boundary marker (not just contains).
            for marker in boundary_markers:
                if line.startswith(marker):
                    return i
        return log_buffer.line_count

    def _extract_compile_command(
        self, log_buffer: LogBuffer, error_line_num: int
    ) -> str | None:
        """Extract compile command that triggered MLIR error.

        Uses 4-strategy search based on corpus analysis:
        1. FORWARD search (adaptive): "Compiled with:" marker (ONNX/pytest)
           - Test framework context: Scan until test boundary or command found
           - Other context: Scan 100 lines
        2. FORWARD search (adaptive): "Invoked with:" marker (Python API)
        3. BACKWARD search (20 lines): "cd ... && iree-" pattern (CMake e2e tests)
        4. BACKWARD search (20 lines): "+ iree-" shell xtrace (lit/bazel tests)

        Note: ONNX/pytest tests can have 1000+ line IR dumps between error and
        command marker, so we detect test framework context and scan to boundary.

        Args:
            log_buffer: LogBuffer to search.
            error_line_num: Line number where error was found (0-indexed).

        Returns:
            Compile command if found, None otherwise.

        Examples of extracted commands:
            - "cd /path && iree-compile model.mlir --target=vulkan -o output.vmfb"
            - "iree-opt '--pass-pipeline=...' /path/test.mlir"
            - "iree-compile /path/model.mlir --iree-hal-target-backends=llvm-cpu"
        """
        # Determine forward search limit based on context.
        if self._is_in_test_framework_context(log_buffer, error_line_num):
            # In test framework: scan until we find test boundary.
            forward_search_limit = self._find_test_boundary(log_buffer, error_line_num)
        else:
            # Not in test framework: use modest forward search.
            forward_search_limit = min(log_buffer.line_count, error_line_num + 100)

        # Strategy 1: Search FORWARD for "Compiled with:" marker (ONNX/pytest tests).
        for i in range(error_line_num, forward_search_limit):
            line = log_buffer.get_line(i)
            if self._COMPILED_WITH_RE.match(line):
                # Next non-empty line contains command.
                for j in range(i + 1, min(log_buffer.line_count, i + 5)):
                    next_line = log_buffer.get_line(j)
                    if next_line.strip():
                        return self._clean_command(next_line)

        # Strategy 2: Search FORWARD for "Invoked with:" marker (Python compiler API).
        for i in range(error_line_num, forward_search_limit):
            line = log_buffer.get_line(i)
            if self._INVOKED_WITH_RE.match(line):
                # Next non-empty line contains command (may start with " iree-").
                for j in range(i + 1, min(log_buffer.line_count, i + 5)):
                    next_line = log_buffer.get_line(j)
                    if next_line.strip():
                        return self._clean_command(next_line)

        # Strategy 3: Search BACKWARD for "cd ... && iree-" (CMake e2e tests).
        search_start = max(0, error_line_num - 20)
        for i in range(error_line_num - 1, search_start - 1, -1):
            line = log_buffer.get_line(i)
            if self._CD_COMMAND_RE.search(line):
                return self._clean_command(line)

        # Strategy 4: Search BACKWARD for "+ iree-" shell xtrace (lit/bazel tests).
        for i in range(error_line_num - 1, search_start - 1, -1):
            line = log_buffer.get_line(i)
            if self._SHELL_XTRACE_RE.match(line):
                return self._clean_command(line)

        # No compile command found.
        return None
