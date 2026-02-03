# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bazel build system error extractor.

Extracts Bazel-specific build failures including missing targets and build
summary errors. Uses two-phase detection to avoid false positives.

Note: Compiler errors from Bazel builds (GCC/Clang format) are handled by
BuildErrorExtractor. This extractor only handles Bazel-specific errors.

Two-phase detection:
1. Find "ERROR: Build did NOT complete successfully" marker
2. Search backward for specific Bazel error patterns

Example usage:
    from common.extractors.bazel_error import BazelErrorExtractor
    from common.log_buffer import LogBuffer

    extractor = BazelErrorExtractor()
    log_buffer = LogBuffer(log_content, auto_detect_format=True)
    issues = extractor.extract(log_buffer)

    for issue in issues:
        print(f"Bazel error: {issue.error_type} in {issue.target}")
"""

import re

from common.extractors.base import Extractor
from common.issues import BazelErrorIssue, Severity
from common.log_buffer import LogBuffer


class BazelErrorExtractor(Extractor):
    """Extracts Bazel build system errors from logs.

    Only reports Bazel-specific errors when build definitively fails
    (indicated by "ERROR: Build did NOT complete successfully"). This
    prevents false positives from informational messages.
    """

    name = "bazel_error"
    activation_keywords = [
        "--bazelrc=",
        "--noworkspace_rc",
    ]  # Only run on Bazel logs.

    # Failure markers - Bazel build or test failed.
    # Handles ANSI escape codes: [31m[1mERROR: [0mBuild did NOT complete successfully
    _BUILD_FAILED_RE = re.compile(
        r"(?:\x1b\[\d+m|\s)*ERROR:(?:\x1b\[\d+m|\s)+Build did NOT complete successfully"
    )

    # Test failure patterns (can occur after successful build).
    _TEST_NO_TARGETS_RE = re.compile(
        r"(?:\x1b\[\d+m|\s)*ERROR:(?:\x1b\[\d+m|\s)+No test targets were found"
    )

    # Package evaluation errors (occur before build).
    _PACKAGE_ERROR_RE = re.compile(
        r"(?:\x1b\[\d+m|\s)*ERROR:(?:\x1b\[\d+m|\s)+package contains errors:\s+(?P<package>[^\s]+)"
    )

    # General evaluation errors.
    _EVAL_ERROR_RE = re.compile(
        r"(?:\x1b\[\d+m|\s)*ERROR:(?:\x1b\[\d+m|\s)+Error evaluating"
    )

    # Missing target error pattern.
    # Example: "ERROR: /path/to/BUILD.bazel: no such target '@@workspace//package:target':"
    # Example: "ERROR: /path/to/BUILD.bazel:123:45: no such target '@@workspace//package:target':"
    # Handles ANSI escape codes: [31m[1mERROR: [0m/path/to/BUILD.bazel: no such target...
    _MISSING_TARGET_RE = re.compile(
        r"(?:\x1b\[\d+m|\s)*ERROR:(?:\x1b\[\d+m|\s)+(?P<file>[^:]+?)(?::(?P<line>\d+))?(?::(?P<col>\d+))?:\s+"
        r"no such target\s+'(?P<target_full>@@?(?P<workspace>[^/]+)//(?P<package>[^:]+):(?P<target>[^']+))'",
        re.IGNORECASE,
    )

    # Failed target summary line.
    # Example: "//compiler/bindings/c:loader_test                               FAILED TO BUILD"
    _FAILED_TARGET_SUMMARY_RE = re.compile(r"^(?P<target>//[^\s]+)\s+FAILED TO BUILD")

    def extract(self, log_buffer: LogBuffer) -> list[BazelErrorIssue]:
        """Extract Bazel build and test errors from log.

        Uses multi-phase detection:
        1. Check if Bazel build/test failed (multiple failure patterns)
        2a. If failed, search backward for missing target errors
        2b. Search forward for failed target summaries
        3. Check for test-specific failures (no targets, package errors)

        Args:
            log_buffer: LogBuffer with build log content.

        Returns:
            List of BazelErrorIssue objects (empty if build/test succeeded).
        """
        issues = []

        # Phase 1: Find any failure marker.
        failure_info = self._find_any_failure(log_buffer)
        if failure_info is None:
            # Build and tests succeeded - no errors to report.
            return []

        failure_line_idx, failure_type = failure_info

        # Phase 2a: Search backward from failure marker for missing target errors.
        lines = log_buffer.get_lines()
        seen_targets = set()  # Deduplicate by target name.

        for line_idx in range(failure_line_idx, -1, -1):
            line = lines[line_idx]

            # Check for missing target error.
            match = self._MISSING_TARGET_RE.search(line)
            if match:
                target_full = match.group("target_full")
                if target_full not in seen_targets:
                    seen_targets.add(target_full)
                    issue = self._extract_missing_target(log_buffer, line_idx, match)
                    issues.append(issue)

        # Phase 2b: Search forward from failure marker for failed target summaries.
        failed_targets = []
        for line_idx in range(
            failure_line_idx + 1, min(failure_line_idx + 50, len(lines))
        ):
            line = lines[line_idx]

            # Check for failed target summary line.
            match = self._FAILED_TARGET_SUMMARY_RE.match(line)
            if match:
                target = match.group("target")
                failed_targets.append(target)

        # If we found no specific errors, create a generic failure summary.
        if not issues:
            # Determine message and error type based on failure_type.
            if failure_type == "test_no_targets":
                message = "Bazel test failed: No test targets were found"
                error_type = "test_no_targets"
            elif failure_type == "package_error":
                message = "Bazel package evaluation failed"
                error_type = "package_error"
            elif failure_type == "eval_error":
                message = "Bazel evaluation failed"
                error_type = "eval_error"
            elif failed_targets:
                # We have failed target summaries.
                message = f"Bazel build failed: {len(failed_targets)} target(s)"
                error_type = "build_failed"
            else:
                # Generic failure (no details available).
                message = "Bazel build failed"
                error_type = "build_failed"

            issues.append(
                BazelErrorIssue(
                    severity=Severity.CRITICAL,
                    actionable=True,
                    message=message,
                    line_number=failure_line_idx,
                    error_type=error_type,
                    failed_targets=failed_targets,
                    context_lines=log_buffer.get_context(
                        failure_line_idx, before=20, after=5
                    ),
                )
            )

        # Reverse to maintain chronological order (we searched backward).
        issues.reverse()
        return issues

    def _find_any_failure(self, log_buffer: LogBuffer) -> tuple[int, str] | None:
        """Find any Bazel failure marker (build or test).

        Checks for multiple failure patterns in priority order:
        1. Build did NOT complete successfully (definitive build failure)
        2. No test targets were found (test failure)
        3. Package contains errors (package evaluation failure)
        4. Error evaluating (general evaluation failure)

        Args:
            log_buffer: LogBuffer with log content.

        Returns:
            Tuple of (line_index, failure_type) or None if no failures found.
        """
        for line_idx, line in enumerate(log_buffer.get_lines()):
            if self._BUILD_FAILED_RE.search(line):
                return (line_idx, "build_failed")
            if self._TEST_NO_TARGETS_RE.search(line):
                return (line_idx, "test_no_targets")
            if self._PACKAGE_ERROR_RE.search(line):
                return (line_idx, "package_error")
            if self._EVAL_ERROR_RE.search(line):
                return (line_idx, "eval_error")
        return None

    def _extract_missing_target(
        self, log_buffer: LogBuffer, line_idx: int, match: re.Match
    ) -> BazelErrorIssue:
        """Extract missing target error.

        Error format:
            ERROR: /path/to/BUILD.bazel:123:45: no such target '@@workspace//package:target':
              target 'TargetName' not declared in package 'package'

        Args:
            log_buffer: LogBuffer with log content.
            line_idx: Line index of error.
            match: Regex match with file, line, workspace, package, target groups.

        Returns:
            BazelErrorIssue instance.
        """
        bazel_file = match.group("file")
        bazel_line = int(match.group("line")) if match.group("line") else 0
        workspace = match.group("workspace")
        package = match.group("package")
        target = match.group("target")
        target_full = match.group("target_full")

        # Extract additional error context from next few lines.
        error_details = []
        lines = log_buffer.get_lines()
        for i in range(line_idx + 1, min(line_idx + 6, len(lines))):
            line = lines[i].strip()
            if line.startswith("target") or line.startswith("and referenced by"):
                error_details.append(line)
            elif not line or line.startswith("ERROR:"):
                # End of this error's context.
                break

        error_context = " ".join(error_details) if error_details else ""

        return BazelErrorIssue(
            severity=Severity.CRITICAL,
            actionable=True,
            message=f"Missing Bazel target: {target_full}"
            + (f" ({error_context})" if error_context else ""),
            line_number=line_idx,
            error_type="missing_target",
            bazel_file=bazel_file,
            bazel_line=bazel_line,
            workspace=workspace,
            package=package,
            target=target,
            context_lines=log_buffer.get_context(line_idx, before=2, after=10),
        )
