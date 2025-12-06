# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for root cause analysis."""

import sys
import unittest
from pathlib import Path

# Add project tools/utils to path for imports.
sys.path.insert(0, str(Path(__file__).parents[2]))

from ci.core import patterns

# Module-level fixture directory (absolute path for CWD-independence).
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class TestRootCauseAnalyzer(unittest.TestCase):
    """Tests for root cause identification and grouping."""

    def setUp(self):
        """Load test patterns and rules."""
        self.patterns_file = _FIXTURES_DIR / "test_patterns.yaml"
        self.rules_file = _FIXTURES_DIR / "test_cooccurrence_rules.yaml"
        self.loader = patterns.PatternLoader(self.patterns_file, self.rules_file)
        self.loader.load()
        self.analyzer = patterns.RootCauseAnalyzer(self.loader)

    def test_group_cooccurring_patterns(self):
        """Test grouping ROCm cleanup crash with abort."""
        # Create pattern matches for rocclr_memobj and aborted.
        pattern_matches = {
            "rocclr_memobj": [
                patterns.PatternMatch(
                    pattern_name="rocclr_memobj",
                    match_text="rocclr/device/device.cpp:2891",
                    line_number=10,
                    context_before=[],
                    context_after=[],
                )
            ],
            "aborted": [
                patterns.PatternMatch(
                    pattern_name="aborted",
                    match_text="Aborted (core dumped)",
                    line_number=15,
                    context_before=[],
                    context_after=[],
                )
            ],
        }

        root_causes = self.analyzer.identify_root_causes(pattern_matches)

        # Should group into single "rocm_cleanup_crash" root cause.
        self.assertEqual(len(root_causes), 1)
        self.assertEqual(root_causes[0].rule.name, "rocm_cleanup_crash")
        self.assertFalse(root_causes[0].rule.actionable)

        # Should include both matches.
        self.assertEqual(len(root_causes[0].primary_matches), 1)
        self.assertEqual(len(root_causes[0].secondary_matches), 1)

    def test_single_pattern_not_grouped(self):
        """Test single pattern creates individual root cause."""
        pattern_matches = {
            "compile_error": [
                patterns.PatternMatch(
                    pattern_name="compile_error",
                    match_text="fatal error: file not found",
                    line_number=42,
                    context_before=[],
                    context_after=[],
                )
            ]
        }

        root_causes = self.analyzer.identify_root_causes(pattern_matches)

        # Should create one root cause for compile error.
        self.assertEqual(len(root_causes), 1)
        self.assertEqual(root_causes[0].rule.name, "compilation_failure")
        self.assertTrue(root_causes[0].rule.actionable)

    def test_priority_based_rule_selection(self):
        """Test higher priority rules are selected first."""
        # Create matches for timeout and lit_test_failed.
        pattern_matches = {
            "timeout": [
                patterns.PatternMatch(
                    pattern_name="timeout",
                    match_text="TIMEOUT: exceeded limit",
                    line_number=5,
                    context_before=[],
                    context_after=[],
                )
            ],
            "lit_test_failed": [
                patterns.PatternMatch(
                    pattern_name="lit_test_failed",
                    match_text="TEST 'test.mlir' FAILED",
                    line_number=10,
                    context_before=[],
                    context_after=[],
                )
            ],
        }

        root_causes = self.analyzer.identify_root_causes(pattern_matches)

        # Should group into test_timeout rule (has both as patterns).
        self.assertEqual(len(root_causes), 1)
        self.assertEqual(root_causes[0].rule.name, "test_timeout")

    def test_multiple_matches_same_pattern(self):
        """Test multiple matches of same pattern."""
        pattern_matches = {
            "compile_error": [
                patterns.PatternMatch(
                    pattern_name="compile_error",
                    match_text="error: undefined reference to 'foo'",
                    line_number=10,
                    context_before=[],
                    context_after=[],
                ),
                patterns.PatternMatch(
                    pattern_name="compile_error",
                    match_text="error: undefined reference to 'bar'",
                    line_number=20,
                    context_before=[],
                    context_after=[],
                ),
            ]
        }

        root_causes = self.analyzer.identify_root_causes(pattern_matches)

        # Should create one root cause with multiple matches.
        self.assertEqual(len(root_causes), 1)
        self.assertEqual(len(root_causes[0].primary_matches), 2)

    def test_no_matches_returns_empty(self):
        """Test empty pattern matches returns empty root causes."""
        pattern_matches = {}

        root_causes = self.analyzer.identify_root_causes(pattern_matches)

        self.assertEqual(len(root_causes), 0)

    def test_unassigned_patterns_create_individual_causes(self):
        """Test patterns not in rules create individual root causes."""
        # Use filecheck_failed which is not in our test rules.
        pattern_matches = {
            "filecheck_failed": [
                patterns.PatternMatch(
                    pattern_name="filecheck_failed",
                    match_text="CHECK: expected string not found",
                    line_number=5,
                    context_before=[],
                    context_after=[],
                )
            ]
        }

        root_causes = self.analyzer.identify_root_causes(pattern_matches)

        # Should create a synthetic root cause for unassigned pattern.
        self.assertEqual(len(root_causes), 1)

        # Root cause should use pattern as name.
        self.assertEqual(root_causes[0].rule.primary_pattern, "filecheck_failed")


class TestRootCauseDataStructure(unittest.TestCase):
    """Tests for RootCause data structure."""

    def test_all_matches_property(self):
        """Test all_matches combines primary and secondary."""
        rule = patterns.RootCauseRule(
            name="test_rule",
            primary_pattern="p1",
            secondary_patterns=["p2"],
            description="Test",
            priority=50,
            actionable=True,
            category="test",
        )

        primary = patterns.PatternMatch(
            pattern_name="p1",
            match_text="primary",
            line_number=1,
            context_before=[],
            context_after=[],
        )
        secondary = patterns.PatternMatch(
            pattern_name="p2",
            match_text="secondary",
            line_number=2,
            context_before=[],
            context_after=[],
        )

        rc = patterns.RootCause(
            rule=rule, primary_matches=[primary], secondary_matches=[secondary]
        )

        # all_matches should combine both.
        self.assertEqual(len(rc.all_matches), 2)
        self.assertIn(primary, rc.all_matches)
        self.assertIn(secondary, rc.all_matches)


if __name__ == "__main__":
    unittest.main()
