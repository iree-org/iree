# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CTest test failure extractor.

Extracts CTest test failures from CTest output logs using two-phase detection:
1. Find definitive failure markers: "The following tests FAILED:" or "Errors while running CTest"
2. Search backward for all "***Failed", "***Timeout", "***Exception" test result lines

This approach ensures we only report failures when CTest actually failed, not when
all tests passed successfully.

Example usage:
    from common.extractors.ctest_error import CTestErrorExtractor
    from common.log_buffer import LogBuffer

    extractor = CTestErrorExtractor()
    log_buffer = LogBuffer(log_content, auto_detect_format=True)
    issues = extractor.extract(log_buffer)

    for issue in issues:
        print(f"Test {issue.test_name} {issue.failure_reason}: {issue.message}")
"""

import re

from common.extractors.base import Extractor
from common.issues import CTestErrorIssue, Severity
from common.log_buffer import LogBuffer


class CTestErrorExtractor(Extractor):
    """Extracts CTest test failures from logs.

    Only reports test failures when CTest definitively failed (indicated by
    "The following tests FAILED:" or "Errors while running CTest"). This prevents
    false positives from successful test runs.
    """

    name = "ctest_error"
    activation_keywords = ["ctest", "Test #"]  # Only run on CTest logs.

    # Definitive failure markers - CTest failed.
    _FAILED_TESTS_SUMMARY_RE = re.compile(r"The following tests? FAILED:")
    _ERRORS_WHILE_RUNNING_RE = re.compile(r"Errors while running CTest")

    # CTest test result pattern with failure indicator.
    # Example: "123/456 Test #78: test_name ...***Failed    0.45 sec"
    # Note the "***" before status - this distinguishes from "...Passed"
    _TEST_FAILED_RE = re.compile(
        r"(?P<current>\d+)/(?P<total>\d+)\s+Test\s+#(?P<num>\d+):\s+(?P<name>\S+)\s+"
        r"\.+\*+(?P<status>Failed|Timeout|Exception)\s+(?P<time>[\d.]+)\s+sec"
    )

    # CTest summary line listing failed tests.
    # Example: "  78 - test_name (Failed)"
    _SUMMARY_TEST_RE = re.compile(
        r"^\s+(?P<num>\d+)\s+-\s+(?P<name>.+?)\s+\((?P<status>Failed|Timeout|Exception)\)"
    )

    def extract(self, log_buffer: LogBuffer) -> list[CTestErrorIssue]:
        """Extract CTest test failures from log.

        Uses two-phase detection:
        1. Check if CTest failed (look for failure summary markers)
        2. If failed, search backward for all test failure lines

        Args:
            log_buffer: LogBuffer with CTest log content.

        Returns:
            List of CTestErrorIssue objects (empty if all tests passed).
        """
        issues = []

        # Phase 1: Find definitive failure marker.
        failure_line_idx = self._find_ctest_failure(log_buffer)
        if failure_line_idx is None:
            # All tests passed - no failures to report.
            return []

        # Phase 2: Search backward from failure marker for all test failures.
        lines = log_buffer.get_lines()
        seen_tests = set()  # Deduplicate by test number.

        for line_idx in range(failure_line_idx, -1, -1):
            line = lines[line_idx]

            # Check for test result line with failure.
            match = self._TEST_FAILED_RE.search(line)
            if match:
                test_num = int(match.group("num"))
                if test_num not in seen_tests:
                    seen_tests.add(test_num)
                    issue = self._extract_test_failure(log_buffer, line_idx, match)
                    issues.append(issue)
                continue

            # Also check summary lines (in case result line is missing).
            match = self._SUMMARY_TEST_RE.match(line)
            if match:
                test_num = int(match.group("num"))
                if test_num not in seen_tests:
                    seen_tests.add(test_num)
                    issue = self._extract_summary_failure(log_buffer, line_idx, match)
                    issues.append(issue)

        # Reverse to maintain chronological order (we searched backward).
        issues.reverse()
        return issues

    def _find_ctest_failure(self, log_buffer: LogBuffer) -> int | None:
        """Find CTest failure summary marker.

        Looks for "The following tests FAILED:" or "Errors while running CTest".

        Args:
            log_buffer: LogBuffer with log content.

        Returns:
            Line index of failure marker, or None if all tests passed.
        """
        for line_idx, line in enumerate(log_buffer.get_lines()):
            if self._FAILED_TESTS_SUMMARY_RE.search(line):
                return line_idx
            if self._ERRORS_WHILE_RUNNING_RE.search(line):
                return line_idx
        return None

    def _extract_test_failure(
        self, log_buffer: LogBuffer, line_idx: int, match: re.Match
    ) -> CTestErrorIssue:
        """Extract test failure from CTest result line.

        Args:
            log_buffer: LogBuffer with log content.
            line_idx: Line index of test result.
            match: Regex match with test info.

        Returns:
            CTestErrorIssue instance.
        """
        status = match.group("status")
        test_name = match.group("name")
        test_num = int(match.group("num"))
        elapsed_time = float(match.group("time"))

        # Determine severity and actionability from status.
        if status == "Timeout":
            severity = Severity.MEDIUM  # Often infrastructure/resource issues.
            actionable = False
            reason = "timeout"
        elif status == "Exception":
            severity = Severity.HIGH  # Crashes, segfaults.
            actionable = True
            reason = "exception"
        else:  # Failed
            severity = Severity.HIGH  # Test assertion failures.
            actionable = True
            reason = "failed"

        # Extract test output (search backward for test start).
        test_output = self._extract_test_output(log_buffer, line_idx, test_num)

        return CTestErrorIssue(
            severity=severity,
            actionable=actionable,
            message=f"Test {test_name} {reason}",
            line_number=line_idx,
            test_name=test_name,
            test_number=test_num,
            failure_reason=reason,
            elapsed_time=elapsed_time,
            test_output_tail=test_output[-100:]
            if test_output
            else [],  # Last 100 lines.
            context_lines=log_buffer.get_context(line_idx, before=5, after=5),
        )

    def _extract_summary_failure(
        self, log_buffer: LogBuffer, line_idx: int, match: re.Match
    ) -> CTestErrorIssue:
        """Extract test failure from CTest summary line.

        Used when test result line is missing but summary line exists.

        Args:
            log_buffer: LogBuffer with log content.
            line_idx: Line index of summary line.
            match: Regex match with test info.

        Returns:
            CTestErrorIssue instance.
        """
        status = match.group("status")
        test_name = match.group("name")
        test_num = int(match.group("num"))

        # Determine severity and actionability from status.
        if status == "Timeout":
            severity = Severity.MEDIUM
            actionable = False
            reason = "timeout"
        elif status == "Exception":
            severity = Severity.HIGH
            actionable = True
            reason = "exception"
        else:  # Failed
            severity = Severity.HIGH
            actionable = True
            reason = "failed"

        # Try to find full test result line by searching backward.
        test_output = self._extract_test_output(log_buffer, line_idx, test_num)

        return CTestErrorIssue(
            severity=severity,
            actionable=actionable,
            message=f"Test {test_name} {reason}",
            line_number=line_idx,
            test_name=test_name,
            test_number=test_num,
            failure_reason=reason,
            elapsed_time=0.0,  # Not available from summary line.
            test_output_tail=test_output[-100:] if test_output else [],
            context_lines=log_buffer.get_context(line_idx, before=10, after=2),
        )

    def _extract_test_output(
        self, log_buffer: LogBuffer, result_line_idx: int, test_num: int
    ) -> list[str]:
        """Extract test output by searching backward for test start.

        Args:
            log_buffer: LogBuffer with log content.
            result_line_idx: Line index of test result.
            test_num: Test number.

        Returns:
            List of output lines.
        """
        lines = log_buffer.get_lines()
        output = []

        # Search backward up to 1000 lines for test start marker.
        # Look for "Test #N:" or "Start #N:" patterns.
        test_start_re = re.compile(rf"(?:Test|Start)\s+#{test_num}:")

        for i in range(result_line_idx - 1, max(0, result_line_idx - 1000), -1):
            if test_start_re.search(lines[i]):
                # Found test start - collect forward to result.
                for j in range(i, result_line_idx):
                    output.append(lines[j])
                break

            # Stop if we hit another test result (different test).
            if self._TEST_FAILED_RE.search(lines[i]):
                other_match = self._TEST_FAILED_RE.search(lines[i])
                if other_match and int(other_match.group("num")) != test_num:
                    break

        # Fallback: if no start found, grab context.
        if not output:
            output = log_buffer.get_context(result_line_idx, before=50, after=0)

        return output
