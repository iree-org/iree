# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pattern matching and root cause analysis for CI failures.

This module provides functionality to:
- Load error patterns from YAML configuration
- Match patterns against log content
- Extract structured data from matches
- Group co-occurring patterns by root cause
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PatternMatch:
    """A single pattern match in a log file."""

    pattern_name: str
    match_text: str
    line_number: int
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)
    extracted_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class Pattern:
    """Error pattern definition."""

    name: str
    regex: re.Pattern
    severity: str
    actionable: bool
    context_lines: int
    description: str
    extractors: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, name: str, config: dict[str, Any]) -> "Pattern":
        """Create a Pattern from YAML configuration."""
        return cls(
            name=name,
            regex=re.compile(config["pattern"], re.IGNORECASE | re.MULTILINE),
            severity=config.get("severity", "medium"),
            actionable=config.get("actionable", True),
            context_lines=config.get("context_lines", 3),
            description=config.get("description", ""),
            extractors=config.get("extract", []),
        )

    def match(self, content: str, lines: list[str]) -> list[PatternMatch]:
        """Find all matches of this pattern in the content.

        Args:
            content: Full log content as a single string
            lines: Log content split into lines

        Returns:
            List of PatternMatch objects for each match found
        """
        matches = []

        for match_obj in self.regex.finditer(content):
            # Find line number.
            line_num = content[: match_obj.start()].count("\n") + 1

            # Extract context.
            context_start = max(0, line_num - self.context_lines - 1)
            context_end = min(len(lines), line_num + self.context_lines)

            # Build full context for field extraction (fields may be on adjacent lines).
            context_before = lines[context_start : line_num - 1]
            context_after = lines[line_num:context_end]
            # Use full line containing match, not just matched substring.
            matched_line = lines[line_num - 1]
            full_context = "\n".join(context_before + [matched_line] + context_after)

            # Extract fields if extractors are defined.
            extracted = {}
            for extractor in self.extractors:
                field_name = extractor["name"]
                field_regex = re.compile(extractor["regex"])
                field_match = field_regex.search(full_context)
                if field_match:
                    # If regex has groups, extract them.
                    if field_match.groups():
                        extracted[field_name] = field_match.groups()
                    else:
                        extracted[field_name] = field_match.group(0)

            matches.append(
                PatternMatch(
                    pattern_name=self.name,
                    match_text=match_obj.group(0),
                    line_number=line_num,
                    context_before=context_before,
                    context_after=context_after,
                    extracted_fields=extracted,
                )
            )

        return matches


@dataclass
class RootCauseRule:
    """Rule for grouping co-occurring patterns into a root cause."""

    name: str
    primary_pattern: str
    secondary_patterns: list[str]
    description: str
    priority: int
    actionable: bool
    category: str

    @classmethod
    def from_yaml(cls, config: dict[str, Any]) -> "RootCauseRule":
        """Create a RootCauseRule from YAML configuration."""
        return cls(
            name=config["name"],
            primary_pattern=config["primary_pattern"],
            secondary_patterns=config.get("secondary_patterns", []),
            description=config.get("description", ""),
            priority=config.get("priority", 50),
            actionable=config.get("actionable", True),
            category=config.get("category", "unknown"),
        )

    def matches(self, pattern_names: list[str]) -> bool:
        """Check if this rule matches the given set of patterns.

        Args:
            pattern_names: List of pattern names found in a log

        Returns:
            True if the primary pattern is present and at least one
            secondary pattern (if defined) is also present
        """
        if self.primary_pattern not in pattern_names:
            return False

        # If no secondary patterns defined, primary is enough.
        if not self.secondary_patterns:
            return True

        # At least one secondary pattern must be present.
        return any(pattern in pattern_names for pattern in self.secondary_patterns)


@dataclass
class RootCause:
    """A root cause grouping multiple related error patterns."""

    rule: RootCauseRule
    primary_matches: list[PatternMatch]
    secondary_matches: list[PatternMatch]

    @property
    def all_matches(self) -> list[PatternMatch]:
        """Get all matches (primary + secondary) for this root cause."""
        return self.primary_matches + self.secondary_matches


class PatternLoader:
    """Loads error patterns and root cause rules from YAML files."""

    def __init__(self, patterns_file: Path, rules_file: Path) -> None:
        """Initialize the pattern loader.

        Args:
            patterns_file: Path to patterns.yaml
            rules_file: Path to cooccurrence_rules.yaml
        """
        self.patterns_file = patterns_file
        self.rules_file = rules_file
        self.patterns: dict[str, Pattern] = {}
        self.rules: list[RootCauseRule] = []
        self.pattern_priorities: dict[str, int] = {}

    def load(self) -> None:
        """Load all patterns and rules from YAML files."""
        self._load_patterns()
        self._load_rules()

    def _load_patterns(self) -> None:
        """Load error patterns from YAML."""
        with open(self.patterns_file) as f:
            data = yaml.safe_load(f)

        patterns_config = data.get("patterns", {})
        for name, config in patterns_config.items():
            self.patterns[name] = Pattern.from_yaml(name, config)

    def _load_rules(self) -> None:
        """Load root cause rules from YAML."""
        with open(self.rules_file) as f:
            data = yaml.safe_load(f)

        # Load root cause grouping rules.
        rules_config = data.get("root_cause_rules", [])
        for rule_config in rules_config:
            self.rules.append(RootCauseRule.from_yaml(rule_config))

        # Sort rules by priority (highest first).
        self.rules.sort(key=lambda r: r.priority, reverse=True)

        # Load pattern priorities for tie-breaking.
        self.pattern_priorities = data.get("pattern_priorities", {})


class PatternMatcher:
    """Matches error patterns against log content."""

    def __init__(self, loader: PatternLoader) -> None:
        """Initialize the pattern matcher.

        Args:
            loader: PatternLoader with loaded patterns and rules
        """
        self.loader = loader

    def analyze_log(self, content: str) -> dict[str, list[PatternMatch]]:
        """Analyze a log file and find all pattern matches.

        Args:
            content: Full log content as a string

        Returns:
            Dictionary mapping pattern names to lists of matches
        """
        lines = content.splitlines()
        results = {}

        for pattern_name, pattern in self.loader.patterns.items():
            matches = pattern.match(content, lines)
            if matches:
                results[pattern_name] = matches

        return results


class RootCauseAnalyzer:
    """Analyzes pattern matches to identify root causes."""

    def __init__(self, loader: PatternLoader) -> None:
        """Initialize the root cause analyzer.

        Args:
            loader: PatternLoader with loaded patterns and rules
        """
        self.loader = loader

    def identify_root_causes(
        self, pattern_matches: dict[str, list[PatternMatch]]
    ) -> list[RootCause]:
        """Group co-occurring patterns into root causes.

        Args:
            pattern_matches: Dictionary of pattern matches from PatternMatcher

        Returns:
            List of RootCause objects, sorted by priority
        """
        pattern_names = list(pattern_matches.keys())
        root_causes = []

        # Track which patterns have been assigned to a root cause.
        assigned_patterns = set()

        # Try to match each rule.
        for rule in self.loader.rules:
            # Skip rules whose patterns have already been assigned to higher-priority rules.
            if rule.primary_pattern in assigned_patterns:
                continue
            if any(p in assigned_patterns for p in rule.secondary_patterns):
                continue

            if rule.matches(pattern_names):
                # Collect primary matches.
                primary_matches = pattern_matches.get(rule.primary_pattern, [])

                # Collect secondary matches.
                secondary_matches = []
                for secondary_pattern in rule.secondary_patterns:
                    if secondary_pattern in pattern_matches:
                        secondary_matches.extend(pattern_matches[secondary_pattern])

                root_cause = RootCause(
                    rule=rule,
                    primary_matches=primary_matches,
                    secondary_matches=secondary_matches,
                )
                root_causes.append(root_cause)

                # Mark these patterns as assigned.
                assigned_patterns.add(rule.primary_pattern)
                assigned_patterns.update(rule.secondary_patterns)

        # Handle any unassigned patterns as standalone issues.
        for pattern_name in pattern_names:
            if pattern_name not in assigned_patterns:
                # Create a synthetic root cause for this pattern.
                pattern = self.loader.patterns[pattern_name]
                synthetic_rule = RootCauseRule(
                    name=f"{pattern_name}_standalone",
                    primary_pattern=pattern_name,
                    secondary_patterns=[],
                    description=pattern.description,
                    priority=self.loader.pattern_priorities.get(pattern_name, 50),
                    actionable=pattern.actionable,
                    category="uncategorized",
                )
                root_cause = RootCause(
                    rule=synthetic_rule,
                    primary_matches=pattern_matches[pattern_name],
                    secondary_matches=[],
                )
                root_causes.append(root_cause)

        return root_causes


def load_default_patterns() -> PatternLoader:
    """Load patterns from default locations in tools/utils/ci/.

    Returns:
        PatternLoader with patterns and rules loaded
    """
    # Find the ci directory.
    ci_dir = Path(__file__).parent.parent

    patterns_file = ci_dir / "patterns.yaml"
    rules_file = ci_dir / "cooccurrence_rules.yaml"

    if not patterns_file.exists():
        raise FileNotFoundError(f"Patterns file not found: {patterns_file}")
    if not rules_file.exists():
        raise FileNotFoundError(f"Rules file not found: {rules_file}")

    loader = PatternLoader(patterns_file, rules_file)
    loader.load()
    return loader
