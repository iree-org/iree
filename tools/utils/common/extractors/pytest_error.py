# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pytest test failure extractor.

Extracts Python test failures from pytest output, including AssertionErrors,
ValueErrors, and other exceptions. Handles both pytest error markers (E prefix)
and test summary lines.

Pattern examples:
    E               AssertionError: Dispatch count mismatch: expected 12, got 794
    E           ValueError: NYI floating point type: f8E4M3FNUZ
    FAILED tests/path/test.py::test_name - AssertionError: message

Example usage:
    from common.extractors.pytest_error import PytestErrorExtractor
    from common.log_buffer import LogBuffer

    extractor = PytestErrorExtractor()
    log_buffer = LogBuffer(log_content, auto_detect_format=True)
    issues = extractor.extract(log_buffer)

    for issue in issues:
        print(f"Test '{issue.test_name}' failed: {issue.exception_message}")
"""


from common.extractors.base import Extractor
from common.issues import PythonTestIssue, Severity
from common.log_buffer import LogBuffer


class PytestErrorExtractor(Extractor):
    """Extracts pytest test failures from logs.

    Captures Python exceptions (AssertionError, ValueError, etc.) from pytest
    output and test summary lines.
    """

    name = "pytest"
    activation_keywords = ["pytest", "test session starts"]  # Only run on pytest logs.

    def extract(self, log_buffer: LogBuffer) -> list[PythonTestIssue]:
        """Extract pytest test failures from log.

        Args:
            log_buffer: LogBuffer with log content.

        Returns:
            List of PythonTestIssue objects.
        """
        issues = []
        lines = log_buffer.get_lines()

        # Track seen exceptions to avoid duplicates (FAILED summaries duplicate E errors).
        seen_errors = set()

        for line_idx, line in enumerate(lines):
            # Check for FAILED summary line (fast string check, then parse).
            # Format: "FAILED <test_path> - <exception_type>: <message>"
            # Works with ANY exception type, not just "Error" or "Exception" suffixes.
            if line.startswith("FAILED "):
                result = self._try_parse_pytest_failed(log_buffer, line_idx, line)
                if result:
                    test_path, exception_type, exception_message = result

                    # Use test path + exception type as key to avoid duplicates.
                    error_key = (test_path, exception_type, exception_message[:50])
                    if error_key in seen_errors:
                        continue
                    seen_errors.add(error_key)

                    # Extract test name from path (format: file.py::test_name).
                    test_name = (
                        test_path.split("::")[-1] if "::" in test_path else test_path
                    )

                    issues.append(
                        PythonTestIssue(
                            severity=Severity.HIGH,
                            actionable=True,
                            message=f"Pytest test failed: {exception_type}: {exception_message}",
                            line_number=line_idx,
                            test_name=test_name,
                            exception_type=exception_type,
                            exception_message=exception_message,
                            stack_trace=[],
                            context_lines=log_buffer.get_context(
                                line_idx, before=2, after=3
                            ),
                        )
                    )
                    continue

            # Check for pytest E-prefixed error (fast string check, then parse).
            # Format: "E               <ExceptionType>: <message>"
            if line.lstrip().startswith("E ") and ": " in line:
                result = self._try_parse_pytest_e_error(log_buffer, line_idx, line)
                if result:
                    exception_type, exception_message = result

                    # Try to find test name by looking backward for test indicators.
                    test_name = self._find_test_name(log_buffer, line_idx)

                    # Use exception info as key (no test path for E errors).
                    error_key = (
                        test_name or "unknown",
                        exception_type,
                        exception_message[:50],
                    )
                    if error_key in seen_errors:
                        continue
                    seen_errors.add(error_key)

                    issues.append(
                        PythonTestIssue(
                            severity=Severity.HIGH,
                            actionable=True,
                            message=f"Python test exception: {exception_type}: {exception_message}",
                            line_number=line_idx,
                            test_name=test_name or "unknown",
                            exception_type=exception_type,
                            exception_message=exception_message,
                            stack_trace=[],
                            context_lines=log_buffer.get_context(
                                line_idx, before=5, after=3
                            ),
                        )
                    )
                    continue

        return issues

    def _find_test_name(self, log_buffer: LogBuffer, line_idx: int) -> str | None:
        """Find test name by searching backward for test indicators.

        Args:
            log_buffer: LogBuffer with log content.
            line_idx: Line index of error.

        Returns:
            Test name if found, None otherwise.
        """
        lines = log_buffer.get_lines()

        # Search backward up to 50 lines for test indicators.
        for i in range(line_idx - 1, max(0, line_idx - 50), -1):
            line = lines[i]

            # Look for pytest test run line.
            # Pattern: "tests/path/test.py::test_name PASSED" or "FAILED"
            if "::" in line and any(
                marker in line for marker in ["PASSED", "FAILED", "SKIPPED"]
            ):
                parts = line.split("::")
                if len(parts) >= 2:
                    return parts[-1].split()[0]  # Extract test name before status.

            # Look for pytest test collection.
            # Pattern: "tests/path/test.py::test_name"
            if "::" in line and "test" in line.lower():
                parts = line.split("::")
                if len(parts) >= 2:
                    return parts[-1].strip()

        return None

    def _try_parse_pytest_failed(
        self, log_buffer: LogBuffer, line_idx: int, line: str
    ) -> tuple | None:
        """Parse pytest FAILED line using string operations.

        Format: "FAILED <test_path> - <exception_type>: <message>"
        Works with ANY exception type (not just Error/Exception suffixes).

        Args:
            log_buffer: LogBuffer with log content.
            line_idx: Line index.
            line: Line content.

        Returns:
            (test_path, exception_type, exception_message) if valid, None otherwise.
        """
        # Remove "FAILED " prefix.
        rest = line[7:]  # len("FAILED ") = 7

        # Split on " - " to separate test path from error.
        if " - " not in rest:
            return None

        test_path, error_part = rest.split(" - ", 1)

        # Split error part on ": " to get exception type and message.
        if ": " in error_part:
            exception_type, message = error_part.split(": ", 1)
        else:
            # No colon - just use the whole thing as message.
            exception_type = "UnknownError"
            message = error_part

        return (test_path.strip(), exception_type.strip(), message.strip())

    def _try_parse_pytest_e_error(
        self, log_buffer: LogBuffer, line_idx: int, line: str
    ) -> tuple | None:
        """Parse pytest E-prefixed error line using string operations.

        Format: "E               <ExceptionType>: <message>"
        Format: "E           ValueError: message"

        Args:
            log_buffer: LogBuffer with log content.
            line_idx: Line index.
            line: Line content.

        Returns:
            (exception_type, exception_message) if valid, None otherwise.
        """
        # Strip leading whitespace and "E ".
        rest = line.lstrip()[2:]  # Skip "E "

        # Split on ": " to get exception type and message.
        if ": " not in rest:
            return None

        exception_type, message = rest.split(": ", 1)

        # Exception type should be reasonable (not empty, not too long).
        exception_type = exception_type.strip()
        if not exception_type or len(exception_type) > 200:
            return None

        return (exception_type, message.strip())
