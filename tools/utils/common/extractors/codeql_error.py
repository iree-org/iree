# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CodeQL code scanning failure extractor.

Extracts failures from CodeQL/code scanning analysis, including language
detection failures, analysis upload failures, and fatal errors.

Pattern examples:
    CodeQL detected code written in JavaScript/TypeScript, but not any written in Python.
    ##[error]Encountered a fatal error while running "/opt/hostedtoolcache/CodeQL/..."
    Analysis upload status is failed.
    CodeQL job status was configuration error.

Example usage:
    from common.extractors.codeql_error import CodeQLErrorExtractor
    from common.log_buffer import LogBuffer

    extractor = CodeQLErrorExtractor()
    log_buffer = LogBuffer(log_content, auto_detect_format=True)
    issues = extractor.extract(log_buffer)

    for issue in issues:
        print(f"CodeQL error: {issue.message}")
"""

import re

from common.extractors.base import Extractor
from common.issues import CodeQLIssue, Severity
from common.log_buffer import LogBuffer


class CodeQLErrorExtractor(Extractor):
    """Extracts CodeQL code scanning failures from logs.

    CodeQL failures are typically infrastructure/configuration issues that
    indicate analysis setup problems, not code bugs.
    """

    name = "codeql"
    activation_keywords = ["codeql", "CodeQL"]  # Only run on CodeQL logs.

    # CodeQL language detection failure.
    # Pattern: "CodeQL detected code written in JavaScript/TypeScript, but not any written in Python."
    _LANGUAGE_DETECTION_RE = re.compile(
        r"CodeQL detected code written in (?P<found>\w+(?:/\w+)?), but not any written in (?P<expected>\w+)\."
    )

    # CodeQL fatal error in GitHub Actions.
    # Pattern: "##[error]Encountered a fatal error while running "/opt/hostedtoolcache/CodeQL/...". Exit code was 32..."
    _FATAL_ERROR_RE = re.compile(
        r"##\[error\]Encountered a fatal error while running \"(?P<command>[^\"]+)\"\.\s+Exit code was (?P<exit_code>\d+)"
    )

    # Analysis upload status failure.
    # Pattern: "Analysis upload status is failed."
    _UPLOAD_FAILED_RE = re.compile(
        r"Analysis upload status is (?P<status>failed|error)"
    )

    # CodeQL job status.
    # Pattern: "CodeQL job status was configuration error."
    _JOB_STATUS_RE = re.compile(
        r"CodeQL job status was (?P<status>configuration error|fatal error|[\w\s]+)"
    )

    def extract(self, log_buffer: LogBuffer) -> list[CodeQLIssue]:
        """Extract CodeQL failures from log.

        Args:
            log_buffer: LogBuffer with log content.

        Returns:
            List of CodeQLIssue objects (infrastructure issues).
        """
        issues = []
        lines = log_buffer.get_lines()

        # Track seen errors to avoid duplicates.
        seen_errors = set()

        for line_idx, line in enumerate(lines):
            # Check for fatal error first (most specific - has ##[error] marker).
            # This must come before language detection because the ##[error] line
            # may contain quoted language detection text.
            match = self._FATAL_ERROR_RE.search(line)
            if match:
                command = match.group("command")
                exit_code = match.group("exit_code")

                # Extract command name for deduplication.
                # Command format: "/path/to/codeql/codeql database finalize ..."
                # We want "codeql" not the subcommand or path fragments.
                command_parts = command.split()
                if len(command_parts) > 0:
                    # Get basename of first part (the executable).
                    command_name = command_parts[0].split("/")[-1]
                else:
                    command_name = "unknown"
                error_key = f"fatal_{command_name}_{exit_code}"
                if error_key in seen_errors:
                    continue
                seen_errors.add(error_key)

                issues.append(
                    CodeQLIssue(
                        severity=Severity.HIGH,
                        actionable=False,  # Infrastructure issue.
                        message=f"CodeQL fatal error: {command_name} exited with code {exit_code}",
                        line_number=line_idx,
                        error_category="fatal",
                        command=command_name,
                        exit_code=int(exit_code),
                        context_lines=log_buffer.get_context(
                            line_idx, before=5, after=3
                        ),
                    )
                )
                continue

            # Check for upload failure.
            match = self._UPLOAD_FAILED_RE.search(line)
            if match:
                status = match.group("status")

                error_key = f"upload_{status}"
                if error_key in seen_errors:
                    continue
                seen_errors.add(error_key)

                issues.append(
                    CodeQLIssue(
                        severity=Severity.MEDIUM,
                        actionable=False,  # Infrastructure issue.
                        message=f"CodeQL analysis upload {status}",
                        line_number=line_idx,
                        error_category="upload",
                        status=status,
                        context_lines=log_buffer.get_context(
                            line_idx, before=3, after=3
                        ),
                    )
                )
                continue

            # Check for job status.
            match = self._JOB_STATUS_RE.search(line)
            if match:
                status = match.group("status")

                error_key = f"status_{status}"
                if error_key in seen_errors:
                    continue
                seen_errors.add(error_key)

                issues.append(
                    CodeQLIssue(
                        severity=Severity.LOW,
                        actionable=False,  # Infrastructure issue.
                        message=f"CodeQL job status: {status}",
                        line_number=line_idx,
                        error_category="status",
                        status=status,
                        context_lines=log_buffer.get_context(
                            line_idx, before=2, after=2
                        ),
                    )
                )
                continue

            # Check for language detection failure (least specific - matches quoted text).
            # This comes last to avoid matching language text quoted in ##[error] lines.
            match = self._LANGUAGE_DETECTION_RE.search(line)
            if match:
                found_lang = match.group("found")
                expected_lang = match.group("expected")

                error_key = f"lang_{expected_lang}_{found_lang}"
                if error_key in seen_errors:
                    continue
                seen_errors.add(error_key)

                issues.append(
                    CodeQLIssue(
                        severity=Severity.MEDIUM,
                        actionable=False,  # Infrastructure/config issue.
                        message=f"CodeQL language detection failure: expected {expected_lang}, found {found_lang}",
                        line_number=line_idx,
                        error_category="language",
                        expected_language=expected_lang,
                        found_language=found_lang,
                        context_lines=log_buffer.get_context(
                            line_idx, before=2, after=5
                        ),
                    )
                )

        return issues
