# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for CHECK pattern matching module."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from lit_tools.core.check_matcher import CheckMatcher
from lit_tools.core.check_pattern import CheckPattern, CheckPatternParser
from lit_tools.core.ir_index import IRIndex


class TestBasicMatching(unittest.TestCase):
    """Tests for basic CHECK to IR matching."""

    def test_single_check_single_ir(self):
        """Test matching single CHECK to single IR line."""
        # IR starts at line 10.
        ir_lines = ["%x = arith.constant 0 : index"]
        ir_index = IRIndex(ir_lines, start_line=10)

        # CHECK at line 5 (before IR, common pattern).
        check = CheckPattern(
            check_type="CHECK",
            line_num=5,
            operation="arith.constant",
            captures=[],
            raw_pattern="%x = arith.constant 0",
        )

        matcher = CheckMatcher(ir_index, [check])
        matches = matcher.match_all()

        self.assertEqual(len(matches), 1)
        self.assertIsNotNone(matches[0].ir_line)
        self.assertEqual(matches[0].ir_line.operation, "arith.constant")
        self.assertEqual(matches[0].ir_line.line_num, 10)
        self.assertEqual(matches[0].confidence, 1.0)

    def test_no_operation_in_check(self):
        """Test CHECK without operation returns None match."""
        ir_lines = ["%x = arith.constant 0 : index"]
        ir_index = IRIndex(ir_lines)

        check = CheckPattern(
            check_type="CHECK",
            line_num=0,
            operation=None,  # No operation.
            captures=[],
            raw_pattern="%[[A]], %[[B]]",
        )

        matcher = CheckMatcher(ir_index, [check])
        matches = matcher.match_all()

        self.assertEqual(len(matches), 1)
        self.assertIsNone(matches[0].ir_line)
        self.assertEqual(matches[0].confidence, 0.0)
        self.assertIn("No operation", matches[0].match_reason)

    def test_no_matching_ir(self):
        """Test CHECK with operation not found in IR."""
        ir_lines = ["%x = arith.constant 0 : index"]
        ir_index = IRIndex(ir_lines)

        check = CheckPattern(
            check_type="CHECK",
            line_num=0,
            operation="scf.for",  # Not in IR.
            captures=[],
            raw_pattern="scf.for %i = %c0",
        )

        matcher = CheckMatcher(ir_index, [check])
        matches = matcher.match_all()

        self.assertEqual(len(matches), 1)
        self.assertIsNone(matches[0].ir_line)
        self.assertEqual(matches[0].confidence, 0.0)
        self.assertIn("No IR line", matches[0].match_reason)


class TestProximityMatching(unittest.TestCase):
    """Tests for proximity-based matching."""

    def test_prefer_closer_match(self):
        """Test that closer IR line is preferred."""
        ir_lines = [
            "%c0 = arith.constant 0 : index",  # line 0
            "",
            "",
            "",
            "",
            "%c1 = arith.constant 1 : index",  # line 5
        ]
        ir_index = IRIndex(ir_lines)

        # CHECK near line 1 should match line 0 (distance 1).
        check = CheckPattern(
            check_type="CHECK",
            line_num=1,
            operation="arith.constant",
            captures=[],
            raw_pattern="%c0 = arith.constant",
        )

        matcher = CheckMatcher(ir_index, [check])
        matches = matcher.match_all()

        self.assertEqual(len(matches), 1)
        self.assertIsNotNone(matches[0].ir_line)
        self.assertEqual(matches[0].ir_line.line_num, 0)  # Closest.

    def test_outside_window(self):
        """Test IR line outside proximity window is not matched."""
        ir_lines = [
            "%c0 = arith.constant 0 : index",  # line 0
        ]
        ir_index = IRIndex(ir_lines)

        # CHECK at line 100, with default window=50, won't find line 0.
        check = CheckPattern(
            check_type="CHECK",
            line_num=100,
            operation="arith.constant",
            captures=[],
            raw_pattern="%c0 = arith.constant",
        )

        matcher = CheckMatcher(ir_index, [check], proximity_window=50)
        matches = matcher.match_all()

        self.assertEqual(len(matches), 1)
        self.assertIsNone(matches[0].ir_line)


class TestSequentialMatching(unittest.TestCase):
    """Tests for multiple CHECKs with same operation."""

    def test_multiple_checks_multiple_ir(self):
        """Test multiple CHECKs match to multiple IR lines sequentially."""
        ir_lines = [
            "%c0 = arith.constant 0 : index",  # line 0
            "%c1 = arith.constant 1 : index",  # line 1
            "%c2 = arith.constant 2 : index",  # line 2
        ]
        ir_index = IRIndex(ir_lines)

        checks = [
            CheckPattern(
                check_type="CHECK",
                line_num=0,
                operation="arith.constant",
                captures=[],
                raw_pattern="%c0",
            ),
            CheckPattern(
                check_type="CHECK",
                line_num=1,
                operation="arith.constant",
                captures=[],
                raw_pattern="%c1",
            ),
        ]

        matcher = CheckMatcher(ir_index, checks)
        matches = matcher.match_all()

        self.assertEqual(len(matches), 2)
        # First CHECK should match line 0 (closest).
        self.assertEqual(matches[0].ir_line.line_num, 0)
        # Second CHECK should match line 1 (closest to CHECK at line 1).
        self.assertEqual(matches[1].ir_line.line_num, 1)


class TestCheckNextSame(unittest.TestCase):
    """Tests for CHECK-NEXT and CHECK-SAME semantics."""

    def test_check_next_success(self):
        """Test CHECK-NEXT matches next IR line."""
        ir_lines = [
            "%c0 = arith.constant 0 : index",  # line 0
            "%c1 = arith.constant 1 : index",  # line 1
        ]
        ir_index = IRIndex(ir_lines)

        checks = [
            CheckPattern(
                check_type="CHECK",
                line_num=0,
                operation="arith.constant",
                captures=[],
                raw_pattern="%c0",
            ),
            CheckPattern(
                check_type="CHECK-NEXT",
                line_num=1,
                operation="arith.constant",
                captures=[],
                raw_pattern="%c1",
            ),
        ]

        matcher = CheckMatcher(ir_index, checks)
        matches = matcher.match_all()

        self.assertEqual(len(matches), 2)
        # First CHECK matches line 0.
        self.assertEqual(matches[0].ir_line.line_num, 0)
        # CHECK-NEXT matches line 1 (next line after previous match).
        self.assertEqual(matches[1].ir_line.line_num, 1)
        self.assertEqual(matches[1].confidence, 1.0)

    def test_check_next_no_previous(self):
        """Test CHECK-NEXT fails without previous match."""
        ir_lines = ["%c0 = arith.constant 0 : index"]
        ir_index = IRIndex(ir_lines)

        check = CheckPattern(
            check_type="CHECK-NEXT",
            line_num=0,
            operation="arith.constant",
            captures=[],
            raw_pattern="%c0",
        )

        matcher = CheckMatcher(ir_index, [check])
        matches = matcher.match_all()

        self.assertEqual(len(matches), 1)
        self.assertIsNone(matches[0].ir_line)
        self.assertEqual(matches[0].confidence, 0.0)
        self.assertIn("requires previous match", matches[0].match_reason)

    def test_check_same_success(self):
        """Test CHECK-SAME matches same IR line as previous."""
        ir_lines = ["%c0 = arith.constant 0 : index"]
        ir_index = IRIndex(ir_lines)

        checks = [
            CheckPattern(
                check_type="CHECK",
                line_num=0,
                operation="arith.constant",
                captures=[],
                raw_pattern="%c0 = arith.constant 0",
            ),
            CheckPattern(
                check_type="CHECK-SAME",
                line_num=1,
                operation=None,  # Doesn't matter for SAME.
                captures=[],
                raw_pattern=": index",
            ),
        ]

        matcher = CheckMatcher(ir_index, checks)
        matches = matcher.match_all()

        self.assertEqual(len(matches), 2)
        # Both should match the same line.
        self.assertEqual(matches[0].ir_line.line_num, 0)
        self.assertEqual(matches[1].ir_line.line_num, 0)
        self.assertEqual(matches[1].confidence, 1.0)

    def test_check_same_no_previous(self):
        """Test CHECK-SAME fails without previous match."""
        ir_lines = ["%c0 = arith.constant 0 : index"]
        ir_index = IRIndex(ir_lines)

        check = CheckPattern(
            check_type="CHECK-SAME",
            line_num=0,
            operation=None,
            captures=[],
            raw_pattern=": index",
        )

        matcher = CheckMatcher(ir_index, [check])
        matches = matcher.match_all()

        self.assertEqual(len(matches), 1)
        self.assertIsNone(matches[0].ir_line)
        self.assertEqual(matches[0].confidence, 0.0)


class TestRealWorldScenario(unittest.TestCase):
    """Test realistic scenario from bug report."""

    def test_bug_report_case(self):
        """Test the actual bug scenario: CHECK before IR it verifies."""
        # Simplified version of the bug report scenario.
        ir_lines = [
            "func @test() {",
            "  %awaited = stream.timepoint.await %tp => %clone",  # line 1
            "  %loop_result = scf.for %i = %c0 to %c10",  # line 2
            "    iter_args(%iter = %awaited) -> (!stream.resource<external>)",
            "}",
        ]
        ir_index = IRIndex(ir_lines, start_line=0)

        # Parse CHECK patterns (simplified).
        parser = CheckPatternParser()
        check_lines = [
            "// CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[TP]] => %[[CLONE]]",
            "// CHECK: %[[LOOP_RESULT:.+]] = scf.for %{{.+}} = %c0 to %c10",
        ]

        checks = [
            parser.parse_check_line(line, i) for i, line in enumerate(check_lines)
        ]
        checks = [c for c in checks if c is not None]

        matcher = CheckMatcher(ir_index, checks)
        matches = matcher.match_all()

        # First CHECK should match line 1 (stream.timepoint.await).
        self.assertEqual(len(matches), 2)
        self.assertIsNotNone(matches[0].ir_line)
        self.assertEqual(matches[0].ir_line.operation, "stream.timepoint.await")
        self.assertEqual(matches[0].ir_line.line_num, 1)

        # Second CHECK should match line 2 (scf.for).
        self.assertIsNotNone(matches[1].ir_line)
        self.assertEqual(matches[1].ir_line.operation, "scf.for")
        self.assertEqual(matches[1].ir_line.line_num, 2)


if __name__ == "__main__":
    unittest.main()
