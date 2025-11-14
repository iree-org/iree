# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TriageEngine orchestrates extractors for comprehensive log analysis.

The TriageEngine runs multiple extractors in sequence, aggregates their results,
deduplicates issues, and sorts them by severity. It provides a unified interface
for analyzing logs regardless of the specific failure types present.

Example usage:
    from common.triage_engine import TriageEngine
    from common.extractors.sanitizer import SanitizerExtractor
    from common.extractors.mlir_compiler import MLIRCompilerExtractor
    from common.log_buffer import LogBuffer

    # Create engine with desired extractors.
    engine = TriageEngine([
        SanitizerExtractor(),
        MLIRCompilerExtractor(),
    ])

    # Analyze log.
    log_buffer = LogBuffer(log_content, auto_detect_format=True)
    result = engine.analyze(log_buffer)

    # Access results.
    print(f"Found {len(result.issues)} issues")
    for issue in result.get_actionable():
        print(f"  {issue.severity.name}: {issue.message}")
"""

import logging

from common.extractors.base import Extractor
from common.log_buffer import LogBuffer
from common.triage_result import TriageResult

logger = logging.getLogger(__name__)


class TriageEngine:
    """Orchestrates extractors for comprehensive log triage.

    The engine runs all registered extractors on a log and aggregates their
    results into a single TriageResult. Issues are deduplicated (by extractor
    name and line number) and sorted by severity, then line number.

    No root cause analysis is performed - extractors return structured Issues
    with rich diagnostic data, and the engine simply aggregates them. This
    approach is simpler and more transparent than pattern-based co-occurrence
    rules.

    Attributes:
        extractors: List of Extractor instances to run.
    """

    def __init__(self, extractors: list[Extractor]) -> None:
        """Initialize TriageEngine with list of extractors.

        Args:
            extractors: List of Extractor instances to run on logs.
        """
        self.extractors = extractors

    def _identify_relevant_extractors(self, log_buffer: LogBuffer) -> list[Extractor]:
        """Identify which extractors should run on this log.

        Performs ONE scan through log content to check for activation keywords.
        Extractors with no activation keywords always run (e.g., sanitizer,
        gpu_driver, build_error).

        This log-level prefilter avoids O(N×M) scaling where all M extractors
        scan all N lines. Instead, only relevant extractors run on each log.

        Args:
            log_buffer: LogBuffer containing log content.

        Returns:
            List of extractors that should run on this log.
        """
        log_content = log_buffer.content  # Single string for fast checking.
        relevant = []

        for extractor in self.extractors:
            # No activation keywords = always run (e.g., sanitizer, gpu_driver).
            if not extractor.activation_keywords:
                relevant.append(extractor)
                continue

            # Check if ANY activation keyword present.
            if any(keyword in log_content for keyword in extractor.activation_keywords):
                relevant.append(extractor)

        return relevant

    def analyze(self, log_buffer: LogBuffer) -> TriageResult:
        """Run relevant extractors and aggregate results.

        Uses two-level filtering architecture:
        1. Log-level prefilter: Identify which extractors are relevant
        2. Line-level extraction: Only relevant extractors run

        Each extractor is run independently. If an extractor raises an exception,
        it is logged and the engine continues with other extractors. This ensures
        that one broken extractor doesn't prevent analysis by other extractors.

        Args:
            log_buffer: LogBuffer containing log content with auto-detected format.

        Returns:
            TriageResult with deduplicated and sorted issues from all extractors.
        """
        all_issues = []
        extractors_run = []
        extractor_errors = []

        # Log-level prefilter - identify relevant extractors.
        relevant_extractors = self._identify_relevant_extractors(log_buffer)

        # Run only relevant extractors.
        for extractor in relevant_extractors:
            try:
                issues = extractor.extract(log_buffer)

                # Tag each issue with its source extractor.
                for issue in issues:
                    issue.source_extractor = extractor.name

                all_issues.extend(issues)
                extractors_run.append(extractor.name)

            except Exception as e:  # noqa: BLE001
                # Log error but continue with other extractors.
                # We catch all exceptions to ensure one broken extractor
                # doesn't prevent analysis by other extractors.
                error_msg = f"Extractor {extractor.name} failed: {e}"
                logger.error(error_msg)
                extractor_errors.append({"extractor": extractor.name, "error": str(e)})
                # Still mark as run (attempted).
                extractors_run.append(extractor.name)

        # Deduplicate: same extractor + same line_number = duplicate.
        # This prevents double-counting when multiple patterns match the same line.
        seen = set()
        unique_issues = []
        for issue in all_issues:
            key = (issue.source_extractor, issue.line_number or -1)
            if key not in seen:
                seen.add(key)
                unique_issues.append(issue)

        # Sort by: severity (descending) → line_number (ascending) → extractor (stable).
        # This ensures CRITICAL issues appear first, earliest issues first within
        # each severity level, and consistent ordering for issues at the same line.
        sorted_issues = sorted(
            unique_issues,
            key=lambda issue: (
                -issue.severity.value,  # Descending severity (CRITICAL=4 first).
                (
                    issue.line_number if issue.line_number is not None else float("inf")
                ),  # Issues without line numbers at end.
                issue.source_extractor,  # Stable sort by extractor name.
            ),
        )

        return TriageResult(
            issues=sorted_issues,
            extractors_run=extractors_run,
            extractor_errors=extractor_errors,
            log_line_count=log_buffer.line_count,
        )
