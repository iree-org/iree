# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for lit test file rendering.

These tests verify that test cases and complete files can be rendered
correctly in both line-preserving and normalized modes.
"""

import tempfile
import unittest
from pathlib import Path

from lit_tools.core.parser import parse_test_file
from lit_tools.core.rendering import build_file_content


class TestCaseRenderForTesting(unittest.TestCase):
    """Tests for TestCase.render_for_testing() method."""

    def test_preserves_line_count(self):
        """Test that render_for_testing preserves total line count."""
        content = """// RUN: iree-opt %s | FileCheck %s
// CHECK-LABEL: @foo
func @foo() {
  return
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            case = test_file.cases[0]
            rendered = case.render_for_testing()

            # Count lines in original vs rendered.
            original_lines = content.count("\n")
            rendered_lines = rendered.count("\n")
            self.assertEqual(original_lines, rendered_lines)
        finally:
            path.unlink()

    def test_blanks_run_lines(self):
        """Test that RUN lines are replaced with blank lines."""
        content = """// RUN: iree-opt %s
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
            rendered = case.render_for_testing()

            # First line should be blank (RUN replaced).
            lines = rendered.split("\n")
            self.assertEqual(lines[0], "")
            # CHECK line should remain.
            self.assertIn("// CHECK: constant", lines[1])
        finally:
            path.unlink()

    def test_preserves_check_positions(self):
        """Test that CHECK directives stay at original line numbers."""
        content = """// RUN: test
// First comment
// CHECK-LABEL: @foo
// CHECK: line1
// CHECK: line2
func @foo() { }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            case = test_file.cases[0]
            rendered = case.render_for_testing()

            lines = rendered.split("\n")
            # Line 0: blank (was RUN).
            self.assertEqual(lines[0], "")
            # Line 1: comment.
            self.assertIn("First comment", lines[1])
            # Line 2: CHECK-LABEL.
            self.assertIn("CHECK-LABEL", lines[2])
            # Line 3: CHECK.
            self.assertIn("CHECK: line1", lines[3])
        finally:
            path.unlink()

    def test_preserves_blank_lines(self):
        """Test that existing blank lines are preserved."""
        content = """// RUN: test

// CHECK: foo

func @test() { }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            case = test_file.cases[0]
            rendered = case.render_for_testing()

            lines = rendered.split("\n")
            # Blank line after RUN preserved.
            self.assertEqual(lines[1], "")
            # Blank line before func preserved.
            self.assertEqual(lines[3], "")
        finally:
            path.unlink()


class TestCaseRenderNormalized(unittest.TestCase):
    """Tests for TestCase.render_normalized() method."""

    def test_drops_run_lines(self):
        """Test that RUN lines are dropped entirely."""
        content = """// RUN: iree-opt %s
// CHECK: foo
func @test() { }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            case = test_file.cases[0]
            rendered = case.render_normalized()

            # RUN line should not appear.
            self.assertNotIn("RUN", rendered)
            # CHECK and body should remain.
            self.assertIn("// CHECK: foo", rendered)
            self.assertIn("func @test()", rendered)
        finally:
            path.unlink()

    def test_trims_leading_blank_lines(self):
        """Test that leading blank lines are removed."""
        content = """// RUN: test


// CHECK: foo
func @test() { }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            case = test_file.cases[0]
            rendered = case.render_normalized()

            # Should not start with blank lines.
            self.assertFalse(rendered.startswith("\n"))
            # Should start with CHECK.
            self.assertTrue(rendered.startswith("// CHECK:"))
        finally:
            path.unlink()

    def test_trims_trailing_blank_lines(self):
        """Test that trailing blank lines are removed."""
        content = """// RUN: test
// CHECK: foo
func @test() { }


"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            case = test_file.cases[0]
            rendered = case.render_normalized()

            # render_normalized removes ALL trailing newlines.
            self.assertTrue(rendered.endswith("}"))
            self.assertFalse(rendered.endswith("\n"))
        finally:
            path.unlink()

    def test_preserves_internal_content(self):
        """Test that internal structure is preserved."""
        content = """// RUN: test
// CHECK-LABEL: @foo

util.func @foo(%arg0: i32) {
  %c = arith.constant 1 : i32
  util.return
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            case = test_file.cases[0]
            rendered = case.render_normalized()

            # All body content should be present.
            self.assertIn("CHECK-LABEL", rendered)
            self.assertIn("util.func @foo", rendered)
            self.assertIn("arith.constant", rendered)
            self.assertIn("util.return", rendered)
        finally:
            path.unlink()


class TestBuildFileContent(unittest.TestCase):
    """Tests for build_file_content() function."""

    def test_preserving_mode_single_case(self):
        """Test preserving mode with single case file."""
        content = """// RUN: iree-opt %s | FileCheck %s
// CHECK-LABEL: @foo
func @foo() { }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            rebuilt = build_file_content(test_file, normalize=False)

            # Should have header RUN.
            self.assertIn("// RUN:", rebuilt)
            # Should have CHECK.
            self.assertIn("CHECK-LABEL", rebuilt)
            # Should have body.
            self.assertIn("func @foo", rebuilt)
        finally:
            path.unlink()

    def test_preserving_mode_multi_case(self):
        """Test preserving mode with multiple cases."""
        content = """// RUN: iree-opt %s | FileCheck %s
// CHECK-LABEL: @first
func @first() { }

// -----

// CHECK-LABEL: @second
func @second() { }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            rebuilt = build_file_content(test_file, normalize=False)

            # Should have header RUN.
            lines = rebuilt.split("\n")
            self.assertIn("RUN", lines[0])

            # Should have delimiter.
            self.assertIn("// -----", rebuilt)

            # Should have both cases.
            self.assertIn("@first", rebuilt)
            self.assertIn("@second", rebuilt)
        finally:
            path.unlink()

    def test_normalized_mode_drops_runs(self):
        """Test that normalized mode drops case-local RUN lines."""
        content = """// RUN: iree-opt %s | FileCheck %s
// CHECK-LABEL: @foo
func @foo() { }

// -----

// RUN: case-local-command
// CHECK-LABEL: @bar
func @bar() { }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            rebuilt = build_file_content(test_file, normalize=True)

            # Should have header RUN.
            self.assertIn("// RUN: iree-opt", rebuilt)

            # Should NOT have case-local RUN.
            self.assertNotIn("case-local-command", rebuilt)

            # Should have both cases.
            self.assertIn("@foo", rebuilt)
            self.assertIn("@bar", rebuilt)
        finally:
            path.unlink()

    def test_normalized_mode_trims_blanks(self):
        """Test that normalized mode trims excess blank lines."""
        content = """// RUN: test

// CHECK-LABEL: @foo


func @foo() { }


// -----


// CHECK-LABEL: @bar
func @bar() { }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)
            rebuilt = build_file_content(test_file, normalize=True)

            # Should not have consecutive blank lines within cases.
            self.assertNotIn("\n\n\n", rebuilt)
        finally:
            path.unlink()

    def test_idempotency(self):
        """Test that normalize(normalize(x)) == normalize(x)."""
        content = """// RUN: test


// CHECK-LABEL: @foo


func @foo() { }


// -----


// CHECK-LABEL: @bar


func @bar() { }


"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)

            # First normalization.
            normalized1 = build_file_content(test_file, normalize=True)

            # Parse normalized output and normalize again.
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".mlir", delete=False
            ) as f2:
                f2.write(normalized1)
                f2.flush()
                path2 = Path(f2.name)

            try:
                test_file2 = parse_test_file(path2)
                normalized2 = build_file_content(test_file2, normalize=True)

                # Should be identical.
                self.assertEqual(normalized1, normalized2)
            finally:
                path2.unlink()
        finally:
            path.unlink()

    def test_handles_empty_cases(self):
        """Test that empty cases are handled correctly."""
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
            rebuilt = build_file_content(test_file, normalize=True)

            # Should have both non-empty cases.
            self.assertIn("@first", rebuilt)
            self.assertIn("@third", rebuilt)

            # Should have delimiters.
            self.assertIn("// -----", rebuilt)
        finally:
            path.unlink()

    def test_ends_with_newline(self):
        """Test that built content always ends with newline."""
        content = """// RUN: test
// CHECK: foo
func @test() { }"""  # No trailing newline.

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            test_file = parse_test_file(path)

            # Both modes should end with newline.
            preserved = build_file_content(test_file, normalize=False)
            self.assertTrue(preserved.endswith("\n"))

            normalized = build_file_content(test_file, normalize=True)
            self.assertTrue(normalized.endswith("\n"))
        finally:
            path.unlink()


class TestRealWorldPatterns(unittest.TestCase):
    """Tests with real IREE patterns."""

    def test_stream_dialect_file(self):
        """Test rendering file with stream dialect patterns."""
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

            # Test preserving mode.
            preserved = build_file_content(test_file, normalize=False)
            self.assertIn("// RUN:", preserved)
            self.assertIn("globalLoad", preserved)
            self.assertIn("globalStore", preserved)

            # Test normalized mode.
            normalized = build_file_content(test_file, normalize=True)
            self.assertIn("// RUN:", normalized)
            self.assertIn("globalLoad", normalized)
            self.assertIn("globalStore", normalized)
            self.assertIn("// -----", normalized)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
