# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Base extractor class and utilities for log analysis.

This module defines the Extractor abstract base class that all log extractors
must implement. Extractors use a single-phase approach:

- **extract()** - Pattern matching and issue extraction. Returns empty list when nothing found.

Example extractor implementation:
    class SanitizerExtractor(Extractor):
        name = "sanitizer"

        def extract(self, log_buffer: LogBuffer) -> List[Issue]:
            issues = []

            # Find all TSAN warnings.
            for line_num, match in log_buffer.find_all_matches(
                r"WARNING: ThreadSanitizer: (.+)"
            ):
                # Extract detailed context, stack traces, etc.
                issue = SanitizerIssue(
                    severity=Severity.CRITICAL,
                    actionable=True,
                    message=match.group(1),
                    sanitizer_type="TSAN",
                    line_number=line_num,
                    source_extractor="sanitizer",
                )
                issues.append(issue)

            return issues  # Returns empty list if no sanitizer errors found.
"""

from abc import ABC, abstractmethod

from common.issues import Issue
from common.log_buffer import LogBuffer


class Extractor(ABC):
    """Abstract base class for all log extractors.

    Each extractor implements pattern matching and diagnostic extraction for
    a specific type of log failure (sanitizers, LIT tests, build errors, etc.).

    Extractors use a two-level filtering architecture:
    1. **Log-level prefilter**: TriageEngine checks activation_keywords to
       determine if this extractor should run on a given log.
    2. **Line-level extraction**: extract() performs pattern matching on
       relevant logs only.

    This approach avoids O(N×M) scaling where all M extractors scan all N
    lines. Instead, only relevant extractors run on each log.

    Subclasses must define:
        - name: Unique identifier for the extractor
        - activation_keywords: List of strings that indicate this extractor
          should run (e.g., ["pre-commit run"]). Empty list means always run.
        - extract(): Pattern matching and issue extraction
    """

    name: str = "unknown"  # Subclasses must override.
    activation_keywords: list[str] = []  # Subclasses override for log-level filtering.

    @abstractmethod
    def extract(self, log_buffer: LogBuffer) -> list[Issue]:
        """Extract diagnostic issues from log.

        Performs pattern matching, context traversal, and diagnostic extraction.
        Returns empty list when no relevant errors are found.

        Args:
            log_buffer: Log content to analyze.

        Returns:
            List of Issue objects found in the log. Empty list if nothing found.

        Example:
            def extract(self, log_buffer: LogBuffer) -> List[Issue]:
                issues = []
                for line_num, match in log_buffer.find_all_matches(pattern):
                    # Parse context, extract diagnostics.
                    issue = SanitizerIssue(...)
                    issues.append(issue)
                return issues  # Empty list if no matches.
        """
        pass
