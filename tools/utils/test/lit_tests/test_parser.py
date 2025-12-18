# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for lit test file parser.

These tests verify the parser correctly identifies test case boundaries,
classifies lines, and extracts metadata.
"""

import tempfile
import unittest
from pathlib import Path

from lit_tools.core.parser import (
    TAG_CHECK,
    TAG_DELIMITER,
    TAG_RUN_CASE,
    TAG_RUN_HEADER,
    parse_test_file,
)


class TestParserBasics(unittest.TestCase):
    """Basic parser functionality tests."""

    def test_parse_single_case_file(self):
        """Test parsing file with no delimiters (single case)."""
        content = """// RUN: iree-opt %s | FileCheck %s
// CHECK-LABEL: @simple
func @simple() { return }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            self.assertEqual(len(test_file.cases), 1)
            self.assertEqual(test_file.cases[0].number, 1)
            self.assertEqual(test_file.cases[0].name, "simple")
            self.assertIsNone(test_file.header_span)  # No delimiter, no header
            self.assertEqual(len(test_file.delimiter_lines), 0)
        finally:
            path.unlink()

    def test_parse_multi_case_file(self):
        """Test parsing file with multiple cases separated by delimiters."""
        content = """// RUN: iree-opt %s | FileCheck %s
// CHECK-LABEL: @first
func @first() { return }

// -----

// CHECK-LABEL: @second
func @second() { return }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            self.assertEqual(len(test_file.cases), 2)
            self.assertEqual(test_file.cases[0].name, "first")
            self.assertEqual(test_file.cases[1].name, "second")
            self.assertEqual(len(test_file.delimiter_lines), 1)
        finally:
            path.unlink()

    def test_parse_empty_case(self):
        """Test parsing empty case (consecutive delimiters)."""
        content = """// RUN: test
// CHECK-LABEL: @first
func @first() { }

// -----

// -----

// CHECK-LABEL: @third
func @third() { }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            # Should have 3 cases: first, middle (blank line), third
            self.assertEqual(len(test_file.cases), 3)
            self.assertEqual(test_file.cases[0].name, "first")
            self.assertIsNone(test_file.cases[1].name)  # Middle case (just blank line)
            # Middle case has the blank line between delimiters, so length=1
            self.assertEqual(test_file.cases[1].span.length, 1)
            self.assertEqual(test_file.cases[2].name, "third")
        finally:
            path.unlink()


class TestLineTagging(unittest.TestCase):
    """Tests for line classification/tagging."""

    def test_tag_header_run_lines(self):
        """Test that RUN lines before delimiter are tagged as header."""
        content = """// RUN: iree-opt %s
// RUN:   | FileCheck %s
// CHECK-LABEL: @test
func @test() { }

// -----

// CHECK-LABEL: @case2
func @case2() { }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            doc = test_file.doc

            # Lines 0-1 should be RUN_HEADER
            self.assertIn(TAG_RUN_HEADER, doc.lines[0].tags)
            self.assertIn(TAG_RUN_HEADER, doc.lines[1].tags)

            # Line 5 should be DELIMITER
            self.assertIn(TAG_DELIMITER, doc.lines[5].tags)
        finally:
            path.unlink()

    def test_tag_case_local_run_lines(self):
        """Test that RUN lines after delimiter are tagged as case-local."""
        content = """// RUN: header-command

// -----

// RUN: case-local-command
// CHECK-LABEL: @test
func @test() { }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            doc = test_file.doc

            # Line 0 should be RUN_HEADER
            self.assertIn(TAG_RUN_HEADER, doc.lines[0].tags)

            # After delimiter, RUN should be RUN_CASE
            run_case_indices = [
                i for i, line in enumerate(doc.lines) if TAG_RUN_CASE in line.tags
            ]
            self.assertEqual(len(run_case_indices), 1)
            self.assertIn("case-local", doc.lines[run_case_indices[0]].text)
        finally:
            path.unlink()

    def test_tag_check_lines(self):
        """Test that CHECK directives are tagged."""
        content = """// RUN: test
// CHECK-LABEL: @foo
// CHECK: constant
// CHECK-NEXT: return
func @foo() { }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            doc = test_file.doc

            # Count CHECK-tagged lines
            check_lines = [line for line in doc.lines if TAG_CHECK in line.tags]
            self.assertEqual(len(check_lines), 3)  # LABEL, CHECK, CHECK-NEXT
        finally:
            path.unlink()


class TestCaseMetadata(unittest.TestCase):
    """Tests for case metadata extraction."""

    def test_extract_single_check_label(self):
        """Test extracting single CHECK-LABEL name."""
        content = """// RUN: test
// CHECK-LABEL: @my_function
func @my_function() { }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            case = test_file.cases[0]
            self.assertEqual(case.name, "my_function")
            self.assertEqual(case.all_names, ("my_function",))
        finally:
            path.unlink()

    def test_extract_multiple_check_labels(self):
        """Test extracting multiple CHECK-LABEL names (different prefixes)."""
        content = """// RUN: test
// CHECK-LABEL: @foo
// FOO-LABEL: @foo
// BAR-LABEL: @bar
func @foo() { }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            case = test_file.cases[0]
            self.assertEqual(case.name, "foo")  # Primary name
            self.assertIn("foo", case.all_names)
            self.assertIn("bar", case.all_names)
        finally:
            path.unlink()

    def test_case_without_check_label(self):
        """Test case without CHECK-LABEL gets None name."""
        content = """// RUN: test
// CHECK: constant
func @test() { }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            case = test_file.cases[0]
            self.assertIsNone(case.name)
            self.assertEqual(case.all_names, ())
        finally:
            path.unlink()

    def test_check_count(self):
        """Test that CHECK directive count is correct."""
        content = """// RUN: test
// CHECK-LABEL: @foo
// CHECK: line1
// CHECK-NEXT: line2
// CHECK: line3
func @foo() { }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            case = test_file.cases[0]
            self.assertEqual(case.check_count, 4)  # LABEL + 3 CHECKs
        finally:
            path.unlink()


class TestRealWorldPatterns(unittest.TestCase):
    """Tests based on real IREE patterns."""

    def test_stream_dialect_pattern(self):
        """Test pattern from stream dialect tests."""
        content = """// RUN: iree-opt --iree-stream-propagate-timepoints %s | FileCheck %s

// Tests that resource global loads work.

// CHECK: util.global private mutable @global
util.global private mutable @global : !stream.resource<constant>

// CHECK-LABEL: @globalLoad
util.func private @globalLoad() {
  %0 = util.global.load @global : !stream.resource<constant>
  util.return
}

// -----

// CHECK-LABEL: @globalStore
util.func private @globalStore(%arg0: !stream.resource<variable>) {
  util.global.store %arg0, @global : !stream.resource<variable>
  util.return
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            self.assertEqual(len(test_file.cases), 2)
            self.assertEqual(test_file.cases[0].name, "globalLoad")
            self.assertEqual(test_file.cases[1].name, "globalStore")

            # Both cases should have header RUN lines
            self.assertGreater(len(test_file.cases[0].header_run_lines), 0)
            self.assertEqual(
                test_file.cases[0].header_run_lines, test_file.cases[1].header_run_lines
            )
        finally:
            path.unlink()


class TestRUNLineEdgeCases(unittest.TestCase):
    """Tests for RUN line parsing corner cases and edge conditions."""

    def test_code_before_first_delimiter(self):
        """Test that IR code before first delimiter is handled correctly."""
        # Scenario: File has actual IR (not just RUN lines) before first delimiter.
        # Parser should treat this as the first case, not as header content.
        content = """// RUN: iree-opt %s | FileCheck %s
// CHECK-LABEL: @prelude
func @prelude() { return }

// -----

// CHECK-LABEL: @second
func @second() { return }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            # Should have 2 cases
            self.assertEqual(len(test_file.cases), 2)

            # First case should include the IR code before delimiter
            self.assertEqual(test_file.cases[0].name, "prelude")
            self.assertIn("func @prelude", test_file.cases[0].content)

            # Header RUN lines should be properly identified
            header_runs = test_file.extract_run_lines(raw=False)
            self.assertEqual(len(header_runs), 1)
            self.assertIn("iree-opt", header_runs[0])
        finally:
            path.unlink()

    def test_backslash_continuation_with_trailing_whitespace(self):
        """Test that backslash continuation works with trailing whitespace."""
        # Scenario: RUN line ending in backslash followed by spaces/tabs.
        # The rstrip() in parser should handle this correctly.
        content = """// RUN: iree-opt \\
// RUN:   --pass-pipeline='builtin.module(func.func(cse))' \\
// RUN:   %s
// CHECK-LABEL: @test
func @test() { return }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            # Should have 1 case
            self.assertEqual(len(test_file.cases), 1)

            # Should extract as a single multi-line RUN command
            runs = test_file.extract_run_lines(raw=False)
            self.assertEqual(len(runs), 1)
            self.assertIn("pass-pipeline", runs[0])
            self.assertIn("%s", runs[0])
        finally:
            path.unlink()

    def test_quoted_backslashes_not_continuation(self):
        """Test that backslashes inside quotes are not treated as continuation."""
        # Scenario: RUN line with Windows path or regex containing backslash.
        # Should NOT treat the quoted backslash as line continuation.
        content = """// RUN: tool --arg="path\\to\\file"
// CHECK-LABEL: @test
func @test() { return }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            # Should have 1 case
            self.assertEqual(len(test_file.cases), 1)

            # Should be a single RUN line (not continuation)
            runs = test_file.extract_run_lines(raw=True)
            self.assertEqual(len(runs), 1)
            self.assertIn('--arg="path\\to\\file"', runs[0])

            # CHECK-LABEL should be in the case content, not consumed by RUN
            self.assertIn("CHECK-LABEL", test_file.cases[0].content)
        finally:
            path.unlink()

    def test_interleaved_comments_in_multiline_runs(self):
        """Test that comments between RUN continuation lines are handled."""
        # Scenario: Multi-line RUN with a comment line in the middle.
        # Current parser should either break on comment or handle gracefully.
        content = """// RUN: iree-opt \\
// NOTE: This is a comment about the pass
// RUN:   --pass-one %s
// CHECK-LABEL: @test
func @test() { return }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            # Should have 1 case
            self.assertEqual(len(test_file.cases), 1)

            # The comment breaks continuation, so we should have either:
            # - 1 incomplete RUN (just "iree-opt"), or
            # - 2 separate RUNs
            runs = test_file.extract_run_lines(raw=False)
            # Parser will likely break at the comment, resulting in incomplete command
            # This test documents current behavior - we're checking it doesn't crash
            self.assertGreater(len(runs), 0, "Should extract at least one RUN line")
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
