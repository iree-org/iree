# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for CHECK pattern parsing module."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from lit_tools.core.check_pattern import CheckPatternParser


class TestCheckDirectiveRecognition(unittest.TestCase):
    """Tests for recognizing different CHECK directive types."""

    def setUp(self):
        self.parser = CheckPatternParser()

    def test_basic_check(self):
        """Test parsing basic CHECK directive."""
        line = "// CHECK: %x = arith.constant 0"
        pattern = self.parser.parse_check_line(line, 42)

        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.check_type, "CHECK")
        self.assertEqual(pattern.line_num, 42)
        self.assertEqual(pattern.raw_pattern, "%x = arith.constant 0")

    def test_check_next(self):
        """Test parsing CHECK-NEXT directive."""
        line = "// CHECK-NEXT: scf.yield %x"
        pattern = self.parser.parse_check_line(line, 10)

        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.check_type, "CHECK-NEXT")

    def test_check_same(self):
        """Test parsing CHECK-SAME directive."""
        line = "// CHECK-SAME: iter_args(%y)"
        pattern = self.parser.parse_check_line(line, 10)

        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.check_type, "CHECK-SAME")

    def test_check_label(self):
        """Test parsing CHECK-LABEL directive."""
        line = "// CHECK-LABEL: func @test"
        pattern = self.parser.parse_check_line(line, 10)

        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.check_type, "CHECK-LABEL")

    def test_check_not(self):
        """Test parsing CHECK-NOT directive."""
        line = "// CHECK-NOT: stream.timepoint.await"
        pattern = self.parser.parse_check_line(line, 10)

        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.check_type, "CHECK-NOT")

    def test_non_check_line(self):
        """Test that non-CHECK lines return None."""
        test_cases = [
            "// This is a comment",
            "%x = arith.constant 0",
            "  // CHECK without colon",
            "",
        ]

        for line in test_cases:
            with self.subTest(line=line):
                pattern = self.parser.parse_check_line(line, 0)
                self.assertIsNone(pattern)

    def test_check_with_whitespace(self):
        """Test CHECK with various whitespace."""
        test_cases = [
            "//CHECK: %x = arith.constant",
            "  // CHECK: %x = arith.constant",
            "//  CHECK  :  %x = arith.constant",
        ]

        for line in test_cases:
            with self.subTest(line=line):
                pattern = self.parser.parse_check_line(line, 0)
                self.assertIsNotNone(pattern)
                self.assertEqual(pattern.check_type, "CHECK")

    def test_custom_check_prefixes(self):
        """Test custom check prefixes via --check-prefix."""
        test_cases = [
            ("// AMDGPU: %x = arith.constant", "AMDGPU"),
            ("// VULKAN-NEXT: scf.yield", "VULKAN-NEXT"),
            ("// DEBUG-SAME: iter_args", "DEBUG-SAME"),
            ("// SPIRV-LABEL: func @test", "SPIRV-LABEL"),
            ("// CPU-NOT: stream.timepoint", "CPU-NOT"),
        ]

        for line, expected_type in test_cases:
            with self.subTest(line=line):
                pattern = self.parser.parse_check_line(line, 0)
                self.assertIsNotNone(pattern)
                self.assertEqual(pattern.check_type, expected_type)


class TestCaptureExtraction(unittest.TestCase):
    """Tests for extracting FileCheck captures."""

    def setUp(self):
        self.parser = CheckPatternParser()

    def test_single_capture_definition(self):
        """Test extracting a single capture definition."""
        line = "// CHECK: %[[X:.+]] = arith.constant"
        pattern = self.parser.parse_check_line(line, 0)

        self.assertEqual(len(pattern.captures), 1)
        cap = pattern.captures[0]
        self.assertEqual(cap.name, "X")
        self.assertEqual(cap.pattern, ".+")
        self.assertTrue(cap.is_definition)

    def test_single_capture_reference(self):
        """Test extracting a single capture reference."""
        line = "// CHECK: scf.yield %[[X]]"
        pattern = self.parser.parse_check_line(line, 0)

        self.assertEqual(len(pattern.captures), 1)
        cap = pattern.captures[0]
        self.assertEqual(cap.name, "X")
        self.assertIsNone(cap.pattern)
        self.assertFalse(cap.is_definition)

    def test_multiple_captures(self):
        """Test extracting multiple captures."""
        line = "// CHECK: %[[A:.+]], %[[B:.+]] = some.op(%[[C]])"
        pattern = self.parser.parse_check_line(line, 0)

        self.assertEqual(len(pattern.captures), 3)

        # First two are definitions.
        self.assertEqual(pattern.captures[0].name, "A")
        self.assertTrue(pattern.captures[0].is_definition)

        self.assertEqual(pattern.captures[1].name, "B")
        self.assertTrue(pattern.captures[1].is_definition)

        # Third is a reference.
        self.assertEqual(pattern.captures[2].name, "C")
        self.assertFalse(pattern.captures[2].is_definition)

    def test_capture_with_complex_pattern(self):
        """Test capture with complex regex pattern."""
        line = "// CHECK: %[[RESULT:[a-z0-9_]+]] = arith.constant"
        pattern = self.parser.parse_check_line(line, 0)

        self.assertEqual(len(pattern.captures), 1)
        self.assertEqual(pattern.captures[0].pattern, "[a-z0-9_]+")

    def test_no_captures(self):
        """Test line with no captures."""
        line = "// CHECK: scf.yield"
        pattern = self.parser.parse_check_line(line, 0)

        self.assertEqual(len(pattern.captures), 0)

    def test_upper_snake_case_enforcement(self):
        """Test that capture names must be UPPER_SNAKE_CASE."""
        # Valid names.
        valid_lines = [
            "// CHECK: %[[X]]",
            "// CHECK: %[[AWAITED]]",
            "// CHECK: %[[LOOP_RESULT]]",
            "// CHECK: %[[_PRIVATE]]",
            "// CHECK: %[[VALUE123]]",
        ]

        for line in valid_lines:
            with self.subTest(line=line):
                pattern = self.parser.parse_check_line(line, 0)
                self.assertGreaterEqual(len(pattern.captures), 1)

        # Invalid names (should not be recognized as captures).
        invalid_lines = [
            "// CHECK: %[[x]]",  # Lowercase.
            "// CHECK: %[[awaited]]",  # Lowercase.
            "// CHECK: %[[123]]",  # Starts with digit.
        ]

        for line in invalid_lines:
            with self.subTest(line=line):
                pattern = self.parser.parse_check_line(line, 0)
                # These won't be recognized as captures.
                self.assertEqual(len(pattern.captures), 0)


class TestOperationExtraction(unittest.TestCase):
    """Tests for extracting operation names from CHECK patterns."""

    def setUp(self):
        self.parser = CheckPatternParser()

    def test_operation_with_capture(self):
        """Test extracting operation when capture is present."""
        line = "// CHECK: %[[X:.+]] = arith.constant 0"
        pattern = self.parser.parse_check_line(line, 0)

        self.assertEqual(pattern.operation, "arith.constant")

    def test_operation_without_capture(self):
        """Test extracting operation without captures."""
        line = "// CHECK: scf.yield %x"
        pattern = self.parser.parse_check_line(line, 0)

        self.assertEqual(pattern.operation, "scf.yield")

    def test_namespaced_operations(self):
        """Test extracting various namespaced operations."""
        test_cases = [
            ("// CHECK: stream.timepoint.await %tp", "stream.timepoint.await"),
            ("// CHECK: scf.for %i = %c0 to %c10", "scf.for"),
            ("// CHECK: util.func @test", "util.func"),
            ("// CHECK: stream.async.execute with()", "stream.async.execute"),
        ]

        for line, expected_op in test_cases:
            with self.subTest(line=line):
                pattern = self.parser.parse_check_line(line, 0)
                self.assertEqual(pattern.operation, expected_op)

    def test_single_word_operations(self):
        """Test extracting single-word operations."""
        test_cases = [
            ("// CHECK: call @function()", "call"),
            ("// CHECK: return %x", "return"),
            ("// CHECK: yield %value", "yield"),
        ]

        for line, expected_op in test_cases:
            with self.subTest(line=line):
                pattern = self.parser.parse_check_line(line, 0)
                self.assertEqual(pattern.operation, expected_op)

    def test_no_operation(self):
        """Test line with no operation."""
        test_cases = [
            "// CHECK: %[[A]], %[[B]]",  # Just captures.
            "// CHECK: iter_args(%[[X]])",  # Structural syntax.
        ]

        for line in test_cases:
            with self.subTest(line=line):
                pattern = self.parser.parse_check_line(line, 0)
                self.assertIsNone(pattern.operation)

    def test_capture_sanitization(self):
        """Test that captures are properly sanitized before operation extraction."""
        # The capture %[[AWAITED:.+]] should not confuse operation extraction.
        line = "// CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[TP]]"
        pattern = self.parser.parse_check_line(line, 0)

        # Should extract "stream.timepoint.await", not be confused by ".+" in capture.
        self.assertEqual(pattern.operation, "stream.timepoint.await")

    def test_filecheck_regex_ignored(self):
        """Test that FileCheck regex syntax {{...}} is ignored."""
        line = "// CHECK: %[[X:.+]] = arith.constant {{.+}} : index"
        pattern = self.parser.parse_check_line(line, 0)

        # Should extract "arith.constant", not be confused by {{.+}}.
        self.assertEqual(pattern.operation, "arith.constant")


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and corner scenarios."""

    def setUp(self):
        self.parser = CheckPatternParser()

    def test_check_same_without_operation(self):
        """Test CHECK-SAME with no operation (continues previous line)."""
        line = "// CHECK-SAME: iter_args(%{{.+}} = %[[AWAITED]])"
        pattern = self.parser.parse_check_line(line, 0)

        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.check_type, "CHECK-SAME")
        self.assertIsNone(pattern.operation)
        self.assertEqual(len(pattern.captures), 1)
        self.assertEqual(pattern.captures[0].name, "AWAITED")

    def test_multiple_operations_in_check(self):
        """Test CHECK with multiple operations (extracts first)."""
        # This is a rare case but can occur.
        line = "// CHECK: %[[X:.+]] = scf.for %{{.+}} = %c0"
        pattern = self.parser.parse_check_line(line, 0)

        # Should extract first operation found.
        self.assertEqual(pattern.operation, "scf.for")

    def test_empty_pattern(self):
        """Test CHECK with empty pattern."""
        line = "// CHECK:"
        pattern = self.parser.parse_check_line(line, 0)

        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.raw_pattern, "")
        self.assertIsNone(pattern.operation)
        self.assertEqual(len(pattern.captures), 0)

    def test_parse_file(self):
        """Test parsing multiple CHECK directives from file."""
        lines = [
            "func @test() {",
            "  // CHECK-LABEL: func @test",
            "  %c0 = arith.constant 0 : index",
            "  // CHECK: %[[C0:.+]] = arith.constant 0",
            "  %c1 = arith.constant 1 : index",
            "  // CHECK: %[[C1:.+]] = arith.constant 1",
            "  scf.yield %c0",
            "  // CHECK: scf.yield %[[C0]]",
            "}",
        ]

        patterns = self.parser.parse_file(lines)

        # Should find 4 CHECK directives.
        self.assertEqual(len(patterns), 4)

        # Verify line numbers are correct.
        self.assertEqual(patterns[0].line_num, 1)  # CHECK-LABEL on line 1.
        self.assertEqual(patterns[1].line_num, 3)  # First CHECK on line 3.
        self.assertEqual(patterns[2].line_num, 5)  # Second CHECK on line 5.
        self.assertEqual(patterns[3].line_num, 7)  # Third CHECK on line 7.

        # Verify check types.
        self.assertEqual(patterns[0].check_type, "CHECK-LABEL")
        self.assertEqual(patterns[1].check_type, "CHECK")
        self.assertEqual(patterns[2].check_type, "CHECK")
        self.assertEqual(patterns[3].check_type, "CHECK")


if __name__ == "__main__":
    unittest.main()
