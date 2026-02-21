# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Formatters for triage results (markdown, JSON, checklist).

This module provides formatters that work with TriageResult containing
structured Issue objects (replacing the old pattern-based root cause system).

Example usage:
    from triage_common.formatters import MarkdownFormatter

    formatter = MarkdownFormatter()
    markdown = formatter.format(result)
    print(markdown)
"""

from typing import Any

from common.issues import Issue
from common.triage_result import TriageResult


class MarkdownFormatter:
    """Formats triage results as human-readable markdown."""

    def format(self, result: TriageResult, include_context: bool = True) -> str:
        """Generate markdown triage report.

        Args:
            result: TriageResult to format.
            include_context: Whether to include error context lines.

        Returns:
            Formatted markdown string.
        """
        lines = []

        # Header with summary.
        lines.append("# CI Failure Triage Report")
        lines.append("")
        lines.append(f"**Total Issues**: {len(result.issues)}")
        lines.append(f"**Actionable**: {len(result.get_actionable())}")
        lines.append(f"**Infrastructure**: {len(result.get_infrastructure())}")
        lines.append("")

        # Show actionable issues first (fix checklist).
        actionable = result.get_actionable()
        if actionable:
            lines.extend(self._format_actionable_section(actionable, include_context))
            lines.append("")

        # Show infrastructure issues.
        infrastructure = result.get_infrastructure()
        if infrastructure:
            lines.extend(self._format_infrastructure_section(infrastructure))
            lines.append("")

        # Summary line.
        lines.extend(self._format_summary(actionable, infrastructure))

        return "\n".join(lines)

    def _format_actionable_section(
        self, issues: list[Issue], include_context: bool
    ) -> list[str]:
        """Format actionable issues as fix checklist."""
        lines = ["## Fix Checklist", ""]

        for _i, issue in enumerate(issues, 1):
            lines.extend(self._format_issue_item(issue, include_context))
            lines.append("")

        return lines

    def _format_issue_item(self, issue: Issue, include_context: bool) -> list[str]:
        """Format a single issue item."""
        # Use issue type name as title.
        issue_type = type(issue).__name__.replace("Issue", "")

        lines = [
            f"- [ ] **{issue_type}** ({issue.severity.name})",
            f"  - {issue.message}",
            f"  - **Source**: {issue.source_extractor}",
        ]

        if issue.line_number is not None:
            lines.append(f"  - **Line**: {issue.line_number}")

        # Show type-specific fields (polymorphic).
        if hasattr(issue, "file_path") and issue.file_path:
            lines.append(f"  - **File**: {issue.file_path}")
        if hasattr(issue, "error_line") and issue.error_line:
            lines.append(
                f"  - **Location**: {issue.file_path}:{issue.error_line}:{issue.error_column}"
            )
        if hasattr(issue, "operation") and issue.operation:
            lines.append(f"  - **Operation**: {issue.operation}")
        if hasattr(issue, "test_name") and issue.test_name:
            lines.append(f"  - **Test**: {issue.test_name}")
        if hasattr(issue, "cmake_file") and issue.cmake_file:
            lines.append(f"  - **CMake**: {issue.cmake_file}:{issue.cmake_line}")

        # Show context.
        if include_context and issue.context_lines:
            lines.extend(self._format_context(issue.context_lines))

        return lines

    def _format_context(self, context_lines: list[str]) -> list[str]:
        """Format error context lines."""
        lines = ["  - **Context**:", "    ```"]

        # Show ALL context lines - no truncation.
        # The error line must always be included in the output.
        for ctx_line in context_lines:
            stripped = ctx_line.strip()
            if stripped:  # Skip empty lines.
                lines.append(f"    {stripped}")

        lines.append("    ```")
        return lines

    def _format_infrastructure_section(self, infrastructure: list[Issue]) -> list[str]:
        """Format infrastructure issues section."""
        lines = [
            "## Infrastructure Issues (Non-Actionable)",
            "",
            "These failures are not code bugs:",
            "- Driver crashes or GPU hangs",
            "- Network timeouts or download failures",
            "- Resource exhaustion (OOM, disk full)",
            "",
        ]

        for issue in infrastructure:
            issue_type = type(issue).__name__.replace("Issue", "")
            lines.extend(
                [
                    f"### {issue_type}",
                    f"- {issue.message}",
                    f"- **Source**: {issue.source_extractor}",
                    f"- **Severity**: {issue.severity.name}",
                    "",
                ]
            )

        return lines

    def _format_summary(
        self, actionable: list[Issue], infrastructure: list[Issue]
    ) -> list[str]:
        """Format the TL;DR summary line."""
        lines = ["---"]

        if not actionable and infrastructure:
            issue_names = [
                type(issue).__name__.replace("Issue", "") for issue in infrastructure
            ]
            lines.append(
                f"**TL;DR**: Infrastructure flake ({', '.join(issue_names[:3])}). No code changes needed."
            )
        elif actionable and not infrastructure:
            lines.append(
                f"**TL;DR**: {len(actionable)} code bug(s) to fix. See checklist above."
            )
        elif actionable and infrastructure:
            lines.append(
                f"**TL;DR**: {len(actionable)} code bug(s) + {len(infrastructure)} infrastructure issue(s)."
            )
        else:
            lines.append("**TL;DR**: No issues found.")

        return lines


class JSONFormatter:
    """Formats triage results as JSON."""

    def format(self, result: TriageResult) -> dict[str, Any]:
        """Generate JSON triage report.

        Args:
            result: TriageResult to format.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        return result.to_dict()


class ChecklistFormatter:
    """Formats triage results as a simple checklist for LLMs."""

    def format(self, result: TriageResult) -> str:
        """Generate simple checklist for LLM consumption.

        Args:
            result: TriageResult to format.

        Returns:
            Formatted checklist string.
        """
        lines = []
        lines.append("# Fix Checklist")
        lines.append("")

        # Only include actionable items.
        actionable = result.get_actionable()

        if not actionable:
            lines.append("No actionable issues found.")
            return "\n".join(lines)

        for i, issue in enumerate(actionable, 1):
            # Simplified checkbox item.
            issue_type = type(issue).__name__.replace("Issue", "")
            identifier = f"{issue_type}-{i}"
            lines.append(f"- [ ] {identifier}: {issue.message}")

            # Show location if available.
            if hasattr(issue, "file_path") and issue.file_path:
                lines.append(f"     File: {issue.file_path}")
            if issue.line_number is not None:
                lines.append(f"     Line: {issue.line_number}")

        return "\n".join(lines)
