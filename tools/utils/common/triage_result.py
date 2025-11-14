# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TriageResult container for aggregated triage analysis results.

This module provides the TriageResult class which holds all issues found by
the TriageEngine, along with filtering and serialization methods for different
output formats (markdown, JSON, etc.).

Example usage:
    result = engine.analyze(log_buffer)

    # Filter by severity.
    critical = result.get_by_severity(Severity.CRITICAL)

    # Filter by actionable.
    actionable = result.get_actionable()

    # Serialize to JSON.
    json_data = result.to_dict()
"""

from dataclasses import dataclass, field
from typing import Any

from common.issues import Issue, Severity


@dataclass
class TriageResult:
    """Aggregated results from running TriageEngine on a log.

    Contains all issues found by all extractors, deduplicated and sorted by
    severity and line number. Provides filtering methods for different views
    of the issues (by severity, by extractor, actionable vs infrastructure).

    Attributes:
        issues: All issues found, deduplicated and sorted.
        extractors_run: Names of extractors that were executed.
        extractor_errors: Errors encountered during extraction.
        log_line_count: Total number of lines in the analyzed log.
    """

    issues: list[Issue]
    extractors_run: list[str]
    extractor_errors: list[dict[str, str]] = field(default_factory=list)
    log_line_count: int = 0

    def get_by_severity(self, severity: Severity) -> list[Issue]:
        """Filter issues by severity level.

        Args:
            severity: Severity level to filter by.

        Returns:
            List of issues matching severity.
        """
        return [issue for issue in self.issues if issue.severity == severity]

    def get_by_extractor(self, extractor_name: str) -> list[Issue]:
        """Filter issues by source extractor.

        Args:
            extractor_name: Name of extractor (e.g., "sanitizer", "mlir_compiler").

        Returns:
            List of issues from that extractor.
        """
        return [
            issue for issue in self.issues if issue.source_extractor == extractor_name
        ]

    def get_actionable(self) -> list[Issue]:
        """Get only actionable issues (not infrastructure).

        Actionable issues are code bugs that require fixes. Infrastructure
        issues are flakes, driver crashes, timeouts, etc. that are outside
        the developer's control.

        Returns:
            List of issues where actionable=True.
        """
        return [issue for issue in self.issues if issue.actionable]

    def get_infrastructure(self) -> list[Issue]:
        """Get only infrastructure issues (non-actionable).

        Infrastructure issues are driver crashes, network flakes, timeouts,
        and other failures not caused by code bugs.

        Returns:
            List of issues where actionable=False.
        """
        return [issue for issue in self.issues if not issue.actionable]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with issues, extractors_run, and summary statistics.
        """
        # Convert issues to dicts (if they have to_dict method).
        issues_dicts = []
        for issue in self.issues:
            if hasattr(issue, "to_dict"):
                issues_dicts.append(issue.to_dict())
            else:
                # Fallback for Issue types without to_dict.
                issues_dicts.append(
                    {
                        "severity": issue.severity.name,
                        "actionable": issue.actionable,
                        "message": issue.message,
                        "line_number": issue.line_number,
                        "source_extractor": issue.source_extractor,
                    }
                )

        return {
            "issues": issues_dicts,
            "extractors_run": self.extractors_run,
            "extractor_errors": self.extractor_errors,
            "summary": {
                "total_issues": len(self.issues),
                "actionable_issues": len(self.get_actionable()),
                "infrastructure_issues": len(self.get_infrastructure()),
                "by_severity": {
                    "critical": len(self.get_by_severity(Severity.CRITICAL)),
                    "high": len(self.get_by_severity(Severity.HIGH)),
                    "medium": len(self.get_by_severity(Severity.MEDIUM)),
                    "low": len(self.get_by_severity(Severity.LOW)),
                },
                "log_line_count": self.log_line_count,
            },
        }
