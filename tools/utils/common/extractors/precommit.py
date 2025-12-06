# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pre-commit hook failure extractor.

Extracts failures from pre-commit hooks (formatting, linting, etc.). These are
typically infrastructure issues that can be fixed by running pre-commit locally.

Pattern examples:
    Run bazel_to_cmake.py on BUILD.bazel files......Failed
    - hook id: bazel_to_cmake_1
    - files were modified by this hook

    Trim Trailing Whitespace.......................Failed
    - hook id: trailing-whitespace

Example usage:
    from common.extractors.precommit import PrecommitErrorExtractor
    from common.log_buffer import LogBuffer

    extractor = PrecommitErrorExtractor()
    log_buffer = LogBuffer(log_content, auto_detect_format=True)
    issues = extractor.extract(log_buffer)

    for issue in issues:
        print(f"Pre-commit hook '{issue.hook_id}' failed: {issue.message}")
"""

import re

from common.extractors.base import Extractor
from common.issues import PrecommitErrorIssue, Severity
from common.log_buffer import LogBuffer


class PrecommitErrorExtractor(Extractor):
    """Extracts pre-commit hook failures from logs.

    Pre-commit failures are typically low-severity infrastructure issues
    that can be fixed by running `pre-commit run --all-files` locally.
    """

    name = "precommit"
    activation_keywords = ["pre-commit run"]  # Only run on logs with pre-commit.

    # Pre-commit hook failure line.
    # Pattern: "Run bazel_to_cmake.py on BUILD.bazel files......Failed"
    # or: "Trim Trailing Whitespace.......................Failed"
    # or: "Run Black to format Python filesFailed" (ANSI codes replace dots)
    # Handles both actual ANSI escape codes (\x1b[41m) and literal bracket codes ([41m).
    # GitHub Actions logs contain literal [41m, not escape sequences.
    _HOOK_FAILED_RE = re.compile(
        r"(?:(?:\x1b)?\[\d+m)*(?P<description>.+?)(?:\.{3,}|(?:(?:\x1b)?\[\d+m)+)(?:(?:\x1b)?\[\d+m)*Failed(?:(?:\x1b)?\[m)*"
    )

    # Hook ID line (follows failure line).
    # Pattern: "- hook id: bazel_to_cmake_1"
    # Handles both ANSI escape codes and literal brackets: [2m- hook id: bazel_to_cmake_1[m.
    # Exclude escape chars (\x1b), brackets ([), and whitespace from hook_id.
    _HOOK_ID_RE = re.compile(
        r"(?:(?:\x1b)?\[\d+m)*-\s+hook id:\s+(?P<hook_id>[^\x1b\[\s]+)(?:(?:\x1b)?\[m)*"
    )

    # Files modified indicator (optional).
    _FILES_MODIFIED_RE = re.compile(r"-\s+files were modified by this hook")

    # Exit code indicator (optional).
    _EXIT_CODE_RE = re.compile(r"-\s+exit code:\s+(?P<code>\d+)")

    def extract(self, log_buffer: LogBuffer) -> list[PrecommitErrorIssue]:
        """Extract pre-commit hook failures from log.

        Args:
            log_buffer: LogBuffer with log content.

        Returns:
            List of PrecommitErrorIssue objects.
        """
        issues = []
        lines = log_buffer.get_lines()

        for line_idx, line in enumerate(lines):
            # Check for hook failure line.
            match = self._HOOK_FAILED_RE.search(line)
            if not match:
                continue

            hook_description = match.group("description").strip()

            # Look for hook ID in next few lines.
            hook_id = None
            files_modified = False
            error_details_lines = []

            for i in range(line_idx + 1, min(line_idx + 10, len(lines))):
                detail_line = lines[i]

                # Check for hook ID.
                id_match = self._HOOK_ID_RE.search(detail_line)
                if id_match:
                    hook_id = id_match.group("hook_id")
                    continue

                # Check for files modified.
                if self._FILES_MODIFIED_RE.search(detail_line):
                    files_modified = True
                    continue

                # Check for exit code (captured for potential future use).
                if self._EXIT_CODE_RE.search(detail_line):
                    continue

                # Collect other detail lines (indented with -).
                if detail_line.strip().startswith("-"):
                    error_details_lines.append(detail_line.strip())
                elif not detail_line.strip():
                    # Blank line continues.
                    continue
                else:
                    # Non-detail line - end of hook failure block.
                    break

            # Determine error type and severity.
            if files_modified:
                error_type = "modified_files"
                severity = (
                    Severity.LOW
                )  # Just needs git commit after running pre-commit.
                message = (
                    f"Pre-commit hook '{hook_id or hook_description}' modified files"
                )
            else:
                error_type = "check_failed"
                severity = Severity.MEDIUM  # Actual check failure, needs investigation.
                message = f"Pre-commit hook '{hook_id or hook_description}' failed"

            error_details = (
                " ".join(error_details_lines) if error_details_lines else None
            )

            issues.append(
                PrecommitErrorIssue(
                    severity=severity,
                    actionable=True,  # User can fix by running pre-commit locally.
                    message=message,
                    line_number=line_idx,
                    hook_id=hook_id or "",
                    hook_description=hook_description,
                    error_type=error_type,
                    files_modified=files_modified,
                    error_details=error_details,
                    context_lines=log_buffer.get_context(line_idx, before=2, after=8),
                )
            )

        return issues
