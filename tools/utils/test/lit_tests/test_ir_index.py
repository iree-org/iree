# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for IR indexing module."""

import unittest

from lit_tools.core.ir_index import IRIndex


class TestIRLineParsing(unittest.TestCase):
    """Tests for parsing individual IR lines into IRLine objects."""

    def test_single_assignment(self):
        """Test parsing single SSA value assignment."""
        lines = ["%x = arith.constant 0 : index"]
        index = IRIndex(lines)

        self.assertEqual(len(index.ir_lines), 1)
        ir = index.ir_lines[0]
        self.assertEqual(ir.operation, "arith.constant")
        self.assertEqual(ir.ssa_results, ["x"])
        self.assertTrue(ir.is_assignment)

    def test_tuple_assignment(self):
        """Test parsing tuple assignment with multiple SSA results."""
        lines = ["%a, %b = some.op : () -> (i32, i32)"]
        index = IRIndex(lines)

        self.assertEqual(len(index.ir_lines), 1)
        ir = index.ir_lines[0]
        self.assertEqual(ir.ssa_results, ["a", "b"])
        self.assertTrue(ir.is_assignment)

    def test_no_assignment(self):
        """Test parsing operation without assignment (e.g., scf.yield)."""
        lines = ["scf.yield %x : i32"]
        index = IRIndex(lines)

        self.assertEqual(len(index.ir_lines), 1)
        ir = index.ir_lines[0]
        self.assertEqual(ir.operation, "scf.yield")
        self.assertEqual(ir.ssa_results, [])
        self.assertFalse(ir.is_assignment)

    def test_semantic_ssa_names_not_filtered(self):
        """Test that ALL SSA names are kept, including %0, %c0, %arg0."""
        lines = [
            "%0 = arith.constant 0 : index",
            "%c0 = arith.constant 0 : index",
            "%arg0 = some.op : () -> i32",
        ]
        index = IRIndex(lines)

        # All three lines should be indexed.
        self.assertEqual(len(index.ir_lines), 3)

        # Check each SSA name is preserved.
        self.assertEqual(index.ir_lines[0].ssa_results, ["0"])
        self.assertEqual(index.ir_lines[1].ssa_results, ["c0"])
        self.assertEqual(index.ir_lines[2].ssa_results, ["arg0"])

    def test_empty_lines_skipped(self):
        """Test that empty lines are skipped."""
        lines = ["", "  ", "%x = arith.constant 0"]
        index = IRIndex(lines)

        self.assertEqual(len(index.ir_lines), 1)

    def test_comment_lines_skipped(self):
        """Test that comment lines are skipped."""
        lines = [
            "// This is a comment",
            "  // Another comment",
            "%x = arith.constant 0",
        ]
        index = IRIndex(lines)

        self.assertEqual(len(index.ir_lines), 1)


class TestOperationExtraction(unittest.TestCase):
    """Tests for extracting operation names from IR lines."""

    def test_standard_operations(self):
        """Test extracting standard namespaced operations."""
        test_cases = [
            ("stream.timepoint.await %tp => %x", "stream.timepoint.await"),
            ("scf.for %i = %c0 to %c10", "scf.for"),
            ("util.func @test() {", "util.func"),
            ("arith.addf %x, %y : f32", "arith.addf"),
        ]

        for line, expected_op in test_cases:
            with self.subTest(line=line):
                index = IRIndex([line])
                if index.ir_lines:
                    self.assertEqual(index.ir_lines[0].operation, expected_op)

    def test_nested_namespaces(self):
        """Test operations with deeply nested namespaces."""
        lines = ["%x = stream.async.execute with() {"]
        index = IRIndex(lines)

        self.assertEqual(len(index.ir_lines), 1)
        self.assertEqual(index.ir_lines[0].operation, "stream.async.execute")

    def test_operation_without_assignment(self):
        """Test extracting operations from non-assignment lines."""
        test_cases = [
            "scf.yield %x : i32",
            "func.return %result : tensor<4xf32>",
            "cf.br ^bb1(%x : i32)",
        ]

        for line in test_cases:
            with self.subTest(line=line):
                index = IRIndex([line])
                self.assertEqual(len(index.ir_lines), 1)
                self.assertIsNotNone(index.ir_lines[0].operation)

    def test_single_word_operations(self):
        """Test extracting single-word operations (call, return, etc.)."""
        test_cases = [
            ("call @function(%x) : (i32) -> i32", "call"),
            ("return %result : tensor<4xf32>", "return"),
            ("yield %value : i32", "yield"),
        ]

        for line, expected_op in test_cases:
            with self.subTest(line=line):
                index = IRIndex([line])
                self.assertEqual(len(index.ir_lines), 1)
                self.assertEqual(index.ir_lines[0].operation, expected_op)


class TestIndexing(unittest.TestCase):
    """Tests for index building and lookup."""

    def test_by_operation_single(self):
        """Test looking up single operation type."""
        lines = [
            "%x = arith.constant 0 : index",
            "%y = arith.addf %x, %x : f32",
        ]
        index = IRIndex(lines)

        constants = index.find_by_operation("arith.constant")
        self.assertEqual(len(constants), 1)
        self.assertEqual(constants[0].ssa_results, ["x"])

    def test_by_operation_multiple(self):
        """Test looking up multiple instances of same operation."""
        lines = [
            "%c0 = arith.constant 0 : index",
            "%c1 = arith.constant 1 : index",
            "%c2 = arith.constant 2 : index",
        ]
        index = IRIndex(lines)

        constants = index.find_by_operation("arith.constant")
        self.assertEqual(len(constants), 3)

    def test_by_operation_not_found(self):
        """Test looking up operation that doesn't exist."""
        lines = ["%x = arith.constant 0 : index"]
        index = IRIndex(lines)

        result = index.find_by_operation("scf.for")
        self.assertEqual(result, [])

    def test_by_line_num(self):
        """Test looking up IR line by line number."""
        lines = ["%x = arith.constant 0 : index"]
        index = IRIndex(lines, start_line=100)

        ir_line = index.get_line(100)
        self.assertIsNotNone(ir_line)
        self.assertEqual(ir_line.ssa_results, ["x"])

        # Line that doesn't exist.
        self.assertIsNone(index.get_line(999))


class TestProximitySearch(unittest.TestCase):
    """Tests for proximity-based operation search."""

    def test_proximity_filter_basic(self):
        """Test filtering by proximity window."""
        lines = [
            "%c0 = arith.constant 0 : index",  # line 0
            "%c1 = arith.constant 1 : index",  # line 1
            "",
            "",
            "",
            "%c2 = arith.constant 2 : index",  # line 5
        ]
        index = IRIndex(lines)

        # Search near line 0 with window=2.
        result = index.find_by_operation("arith.constant", near_line=0, window=2)
        # Should find lines 0, 1 (within 2 lines of line 0).
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].line_num, 0)  # Closest
        self.assertEqual(result[1].line_num, 1)

    def test_proximity_prefer_closer(self):
        """Test that closer matches are returned first."""
        lines = [
            "%c0 = arith.constant 0 : index",  # line 0
            "",
            "%c1 = arith.constant 1 : index",  # line 2
            "",
            "",
            "%c2 = arith.constant 2 : index",  # line 5
        ]
        index = IRIndex(lines)

        # Search near line 3.
        result = index.find_by_operation("arith.constant", near_line=3, window=10)

        # All three should be found, sorted by distance from line 3.
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].line_num, 2)  # Distance 1
        self.assertEqual(result[1].line_num, 5)  # Distance 2
        self.assertEqual(result[2].line_num, 0)  # Distance 3

    def test_proximity_return_all_ties(self):
        """Test that when multiple lines are equidistant, all are returned."""
        lines = [
            "%c0 = arith.constant 0 : index",  # line 0
            "",
            "%c1 = arith.constant 1 : index",  # line 2
        ]
        index = IRIndex(lines)

        # Search near line 1 - both line 0 and line 2 are distance 1.
        result = index.find_by_operation("arith.constant", near_line=1, window=10)

        # Both should be returned.
        self.assertEqual(len(result), 2)
        # When tied, preserve original line order.
        self.assertEqual(result[0].line_num, 0)
        self.assertEqual(result[1].line_num, 2)

    def test_proximity_window_boundary(self):
        """Test behavior at exact window boundary."""
        lines = [
            "%c0 = arith.constant 0 : index",  # line 0
            "",
            "",
            "",
            "",
            "%c1 = arith.constant 1 : index",  # line 5 (exactly 5 away from line 0)
        ]
        index = IRIndex(lines)

        # Search near line 0 with window=5.
        result = index.find_by_operation("arith.constant", near_line=0, window=5)

        # Line 5 is exactly at the boundary (5 lines away), should be included.
        self.assertEqual(len(result), 2)

    def test_proximity_beyond_window(self):
        """Test that lines beyond window are excluded."""
        lines = [
            "%c0 = arith.constant 0 : index",  # line 0
            "",
            "",
            "",
            "",
            "",
            "%c1 = arith.constant 1 : index",  # line 6 (beyond window of 5)
        ]
        index = IRIndex(lines)

        # Search near line 0 with window=5.
        result = index.find_by_operation("arith.constant", near_line=0, window=5)

        # Only line 0 should be found.
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].line_num, 0)

    def test_proximity_no_near_line(self):
        """Test that omitting near_line returns all candidates."""
        lines = [
            "%c0 = arith.constant 0 : index",
            "%c1 = arith.constant 1 : index",
            "%c2 = arith.constant 2 : index",
        ]
        index = IRIndex(lines)

        # No proximity filter.
        result = index.find_by_operation("arith.constant")

        # All three should be returned.
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
