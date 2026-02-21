# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Output formatters for CI triage results.

This module provides formatters for:
- Human-readable markdown output with checklists
- JSON output for automation and LLM consumption
- Summary statistics
"""

from dataclasses import dataclass
from typing import Any

from ci.core.patterns import PatternMatch, RootCause


@dataclass
class TriageResult:
    """Complete triage analysis result."""

    run_id: str
    job_id: str
    job_name: str
    root_causes: list[RootCause]
    unmatched_patterns: list[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "job_id": self.job_id,
            "job_name": self.job_name,
            "root_causes": [
                {
                    "name": rc.rule.name,
                    "category": rc.rule.category,
                    "priority": rc.rule.priority,
                    "actionable": rc.rule.actionable,
                    "description": rc.rule.description,
                    "primary_pattern": rc.rule.primary_pattern,
                    "secondary_patterns": rc.rule.secondary_patterns,
                    "matches": [
                        {
                            "pattern_name": m.pattern_name,
                            "match_text": m.match_text,
                            "line_number": m.line_number,
                            "context_before": m.context_before,
                            "context_after": m.context_after,
                            "extracted_fields": m.extracted_fields,
                        }
                        for m in rc.all_matches
                    ],
                }
                for rc in self.root_causes
            ],
            "unmatched_patterns": self.unmatched_patterns or [],
        }


class MarkdownFormatter:
    """Formats triage results as human-readable markdown."""

    def format(self, result: TriageResult, include_context: bool = True) -> str:
        """Generate markdown triage report.

        Args:
            result: TriageResult to format
            include_context: Whether to include error context lines

        Returns:
            Formatted markdown string
        """
        lines = []

        # Header.
        lines.append("# CI Failure Triage Report")
        lines.append("")
        lines.append(f"**Run ID**: {result.run_id}")
        lines.append(f"**Job ID**: {result.job_id}")
        lines.append(f"**Job**: {result.job_name}")
        lines.append("")

        # Brief summary.
        actionable = [rc for rc in result.root_causes if rc.rule.actionable]
        infrastructure = self._filter_infrastructure(result.root_causes)

        lines.extend(self._format_summary(actionable, infrastructure))
        lines.extend(self._format_actionable_checklist(actionable, include_context))
        lines.extend(self._format_infrastructure_section(infrastructure))
        lines.extend(self._format_tldr_summary(actionable, infrastructure))

        return "\n".join(lines)

    def _filter_infrastructure(self, root_causes: list[RootCause]) -> list[RootCause]:
        """Filter infrastructure issues (high/critical severity only)."""
        return [
            rc
            for rc in root_causes
            if not rc.rule.actionable and self._get_severity(rc) in ["high", "critical"]
        ]

    def _format_summary(
        self, actionable: list[RootCause], infrastructure: list[RootCause]
    ) -> list[str]:
        """Format the brief summary section."""
        summary_parts = []
        if actionable:
            summary_parts.append(f"{len(actionable)} actionable")
        if infrastructure:
            summary_parts.append(f"{len(infrastructure)} infrastructure")

        return [
            f"**Summary**: {', '.join(summary_parts) if summary_parts else '0 issues found'}",
            "",
        ]

    def _format_actionable_checklist(
        self, actionable: list[RootCause], include_context: bool
    ) -> list[str]:
        """Format the actionable issues checklist."""
        if not actionable:
            return []

        lines = ["## Fix Checklist", ""]

        for _i, rc in enumerate(actionable, 1):
            lines.extend(self._format_single_actionable_item(rc, include_context))
            lines.append("")

        return lines

    def _format_single_actionable_item(
        self, rc: RootCause, include_context: bool
    ) -> list[str]:
        """Format a single actionable issue item."""
        lines = [
            f"- [ ] **{rc.rule.name}** ({rc.rule.category})",
            f"  - {rc.rule.description}",
            f"  - **Severity**: {self._get_severity_icon(rc)} {self._get_severity(rc)}",
            f"  - **Occurrences**: {len(rc.all_matches)} error(s)",
        ]

        # Show first match details.
        if rc.primary_matches:
            first_match = rc.primary_matches[0]
            lines.append(f"  - **Location**: Line {first_match.line_number}")

            # Show extracted fields.
            if first_match.extracted_fields:
                lines.append("  - **Details**:")
                for field_name, field_value in first_match.extracted_fields.items():
                    lines.append(f"    - {field_name}: `{field_value}`")

        # Show error context.
        if include_context and rc.primary_matches:
            lines.extend(self._format_error_context(rc.primary_matches[0]))

        return lines

    def _format_error_context(self, match: PatternMatch) -> list[str]:
        """Format error context with 2 lines before/after.

        Note: Expects content to be pre-stripped by LogBuffer. Pattern matching
        should use LogBuffer(content, auto_detect_format=True) to remove log
        prefixes before creating PatternMatch objects.
        """
        lines = ["  - **Error Context**:", "    ```"]

        # Show reduced context: 2 lines before, match, 2 lines after.
        # Content should already be stripped by LogBuffer.
        for ctx_line in match.context_before[-2:]:
            stripped = ctx_line.strip()
            if stripped:  # Skip empty lines.
                lines.append(f"    {stripped}")

        match_stripped = match.match_text.strip()
        lines.append(f"    >>> {match_stripped}")

        for ctx_line in match.context_after[:2]:
            stripped = ctx_line.strip()
            if stripped:  # Skip empty lines.
                lines.append(f"    {stripped}")

        lines.append("    ```")
        return lines

    def _format_infrastructure_section(
        self, infrastructure: list[RootCause]
    ) -> list[str]:
        """Format infrastructure issues section."""
        if not infrastructure:
            return []

        lines = [
            "## Infrastructure Issues (Non-Actionable)",
            "",
            "These failures are not code bugs. They may be:",
            "- Driver crashes or GPU hangs",
            "- Network timeouts or download failures",
            "- Resource exhaustion (OOM, disk full)",
            "",
        ]

        for rc in infrastructure:
            lines.extend(
                [
                    f"### {rc.rule.name}",
                    f"- {rc.rule.description}",
                    f"- **Category**: {rc.rule.category}",
                    f"- **Occurrences**: {len(rc.all_matches)} match(es)",
                    "",
                ]
            )

        return lines

    def _format_tldr_summary(
        self, actionable: list[RootCause], infrastructure: list[RootCause]
    ) -> list[str]:
        """Format the TL;DR summary line."""
        lines = ["---"]

        if not actionable and infrastructure:
            issue_names = [rc.rule.name for rc in infrastructure]
            lines.append(
                f"**TL;DR**: Infrastructure flake ({', '.join(issue_names)}). No code changes needed."
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

    def _get_severity(self, root_cause: RootCause) -> str:
        """Get severity from first primary match."""
        if not root_cause.primary_matches:
            return "medium"

        # Severity is stored in the Pattern, not PatternMatch.
        # We'd need to pass patterns to the formatter to access this.
        # For now, infer from priority.
        priority = root_cause.rule.priority
        if priority >= 90:
            return "critical"
        if priority >= 75:
            return "high"
        if priority >= 50:
            return "medium"
        return "low"

    def _get_severity_icon(self, root_cause: RootCause) -> str:
        """Get severity icon emoji."""
        severity = self._get_severity(root_cause)
        icons = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🔵",
        }
        return icons.get(severity, "⚪")


class JSONFormatter:
    """Formats triage results as JSON."""

    def format(self, result: TriageResult) -> dict[str, Any]:
        """Generate JSON triage report.

        Args:
            result: TriageResult to format

        Returns:
            Dictionary suitable for JSON serialization
        """
        return result.to_dict()


class ChecklistFormatter:
    """Formats triage results as a simple checklist for LLMs."""

    def format(self, result: TriageResult) -> str:
        """Generate simple checklist for LLM consumption.

        Args:
            result: TriageResult to format

        Returns:
            Formatted checklist string
        """
        lines = []
        lines.append(f"# Fix Checklist for {result.job_name}")
        lines.append("")

        # Only include actionable items.
        actionable = [rc for rc in result.root_causes if rc.rule.actionable]

        if not actionable:
            lines.append("No actionable issues found.")
            return "\n".join(lines)

        for i, rc in enumerate(actionable, 1):
            # Simplified checkbox item.
            identifier = f"RC-{i}"
            lines.append(f"- [ ] {identifier}: {rc.rule.name}")

            # Show first error details.
            if rc.primary_matches:
                first_match = rc.primary_matches[0]
                lines.append(
                    f"     Line {first_match.line_number}: {first_match.match_text[:100]}"
                )

                # Show fix hint from extracted fields.
                if first_match.extracted_fields:
                    hints = []
                    for field_name, field_value in first_match.extracted_fields.items():
                        if field_name == "file_path":
                            hints.append(f"File: {field_value[0]}")
                        elif field_name == "error_message":
                            hints.append(f"Error: {field_value[0]}")
                    if hints:
                        lines.append(f"     → Fix: {', '.join(hints)}")

        return "\n".join(lines)
