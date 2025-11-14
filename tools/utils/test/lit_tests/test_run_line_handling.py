# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for RUN line stripping and injection."""

import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lit_tools.core.parser import parse_test_file
from lit_tools.core.rendering import inject_run_lines_with_case
from lit_tools.core.text_utils import (
    _strip_run_lines_preserve_line_numbers,
    inject_run_lines,
)


class TestStripRunLines(unittest.TestCase):
    """Tests for _strip_run_lines_preserve_line_numbers()."""

    def test_single_run_line(self):
        """Test stripping single RUN line."""
        content = "// RUN: iree-opt %s\n// CHECK: foo\nbar"
        result = _strip_run_lines_preserve_line_numbers(content)
        lines = result.split("\n")

        self.assertEqual(lines[0], "", "Line 1 should be blank (was RUN)")
        self.assertEqual(lines[1], "// CHECK: foo", "Line 2 should be preserved")
        self.assertEqual(lines[2], "bar", "Line 3 should be preserved")

    def test_multiline_run_with_backslash(self):
        """Test stripping multi-line RUN with backslash continuation."""
        content = (
            "// RUN: iree-opt --pass1 \\\n"
            "// RUN:   --pass2 \\\n"
            "// RUN:   %s | FileCheck %s\n"
            "// CHECK: foo"
        )
        result = _strip_run_lines_preserve_line_numbers(content)
        lines = result.split("\n")

        self.assertEqual(lines[0], "", "Line 1 should be blank")
        self.assertEqual(lines[1], "", "Line 2 should be blank")
        self.assertEqual(lines[2], "", "Line 3 should be blank")
        self.assertEqual(lines[3], "// CHECK: foo", "Line 4 should be preserved")

    def test_run_line_in_middle_of_file(self):
        """Test that RUN lines anywhere in file are stripped."""
        content = (
            "// Some comment\n"
            "// RUN: iree-opt %s\n"
            "// CHECK: foo\n"
            "// RUN: another-tool\n"
            "bar"
        )
        result = _strip_run_lines_preserve_line_numbers(content)
        lines = result.split("\n")

        self.assertEqual(lines[0], "// Some comment", "Line 1 preserved")
        self.assertEqual(lines[1], "", "Line 2 should be blank (was RUN)")
        self.assertEqual(lines[2], "// CHECK: foo", "Line 3 preserved")
        self.assertEqual(lines[3], "", "Line 4 should be blank (was RUN)")
        self.assertEqual(lines[4], "bar", "Line 5 preserved")

    def test_no_run_lines(self):
        """Test no-op when no RUN lines present."""
        content = "// CHECK: foo\nbar\nbaz"
        result = _strip_run_lines_preserve_line_numbers(content)

        self.assertEqual(result, content, "Content should be unchanged")

    def test_indented_run_line(self):
        """Test stripping indented RUN line."""
        content = "  // RUN: iree-opt %s\n// CHECK: foo"
        result = _strip_run_lines_preserve_line_numbers(content)
        lines = result.split("\n")

        self.assertEqual(lines[0], "", "Line 1 should be blank")
        self.assertEqual(lines[1], "// CHECK: foo", "Line 2 preserved")

    def test_preserves_empty_lines(self):
        """Test that existing blank lines are preserved."""
        content = "// RUN: tool\n\n// CHECK: foo\n\nbar"
        result = _strip_run_lines_preserve_line_numbers(content)
        lines = result.split("\n")

        self.assertEqual(lines[0], "", "Line 1 blank (was RUN)")
        self.assertEqual(lines[1], "", "Line 2 blank (original)")
        self.assertEqual(lines[2], "// CHECK: foo", "Line 3 preserved")
        self.assertEqual(lines[3], "", "Line 4 blank (original)")
        self.assertEqual(lines[4], "bar", "Line 5 preserved")


class TestInjectRunLines(unittest.TestCase):
    """Tests for inject_run_lines()."""

    def test_case1_scenario(self):
        """Test case 1: start_line=1, content has blanks from stripped RUNs."""
        # Simulate case 1: RUN lines were at lines 1-3, now blanks
        content = "\n\n\n\n// CHECK-LABEL: @first_case\nutil.func..."
        run_lines = ["iree-opt %s | FileCheck %s"]

        # No blanks prepended (start_line = 1)
        result = inject_run_lines(content, run_lines)
        lines = result.split("\n")

        self.assertTrue(lines[0].startswith("// RUN:"), "Line 1 should be RUN")
        self.assertEqual(lines[1], "", "Line 2 should be blank")
        self.assertTrue("CHECK-LABEL" in lines[4], "Line 5 should be CHECK-LABEL")

    def test_case2_scenario(self):
        """Test case 2: start_line=14, prepended blanks + content blank."""
        # Simulate case 2: 13 prepended blanks + content with leading blank
        blank_prefix = "\n" * 13
        content = "\n// CHECK-LABEL: @second_case\nutil.func..."
        content_with_blanks = blank_prefix + content
        run_lines = ["iree-opt %s | FileCheck %s"]

        result = inject_run_lines(content_with_blanks, run_lines)
        lines = result.split("\n")

        self.assertTrue(lines[0].startswith("// RUN:"), "Line 1 should be RUN")
        self.assertEqual(lines[1], "", "Line 2 should be blank")
        self.assertTrue("CHECK-LABEL" in lines[14], "Line 15 should be CHECK-LABEL")

    def test_case3_scenario(self):
        """Test case 3: start_line=26, more prepended blanks."""
        blank_prefix = "\n" * 25
        content = "\n// CHECK-LABEL: @third_case\nutil.func..."
        content_with_blanks = blank_prefix + content
        run_lines = ["iree-opt %s | FileCheck %s"]

        result = inject_run_lines(content_with_blanks, run_lines)
        lines = result.split("\n")

        self.assertTrue(lines[0].startswith("// RUN:"), "Line 1 should be RUN")
        self.assertTrue("CHECK-LABEL" in lines[26], "Line 27 should be CHECK-LABEL")

    def test_multiple_run_lines(self):
        """Test injecting multiple RUN lines."""
        content = "\n\n\n\n\n// CHECK: foo"
        run_lines = ["cmd1", "cmd2", "cmd3"]

        result = inject_run_lines(content, run_lines)
        lines = result.split("\n")

        self.assertEqual(lines[0], "// RUN: cmd1", "Line 1 should be RUN 1")
        self.assertEqual(lines[1], "// RUN: cmd2", "Line 2 should be RUN 2")
        self.assertEqual(lines[2], "// RUN: cmd3", "Line 3 should be RUN 3")
        self.assertEqual(lines[3], "", "Line 4 should be blank")
        self.assertTrue("CHECK" in lines[5], "Line 6 should be CHECK")

    def test_no_run_lines(self):
        """Test no-op when no RUN lines to inject."""
        content = "\n\n// CHECK: foo"
        result = inject_run_lines(content, [])

        self.assertEqual(result, content, "Content should be unchanged")

    def test_no_run_fixture_header_and_body(self):
        """Ensure extract_run_lines returns empty and reinjection keeps content."""
        fixture = Path(__file__).parent / "fixtures" / "no_run_test.mlir"
        test_file_obj = parse_test_file(fixture)
        header_runs = test_file_obj.extract_run_lines(raw=False)
        self.assertEqual(header_runs, [])
        cases = list(test_file_obj.cases)
        self.assertEqual(len(cases), 1)
        case = cases[0]
        # Build synthesized content (blank prefix + stripped case)
        stripped = _strip_run_lines_preserve_line_numbers(case.content)
        synthesized = ("\n" * (case.start_line - 1)) + stripped
        rebuilt = inject_run_lines_with_case(
            synthesized, header_runs, case.extract_local_run_lines()
        )
        self.assertEqual(rebuilt, synthesized)

    def test_immediate_ir_no_leading_blanks(self):
        """Test when content has no leading blanks (shouldn't happen but handle it)."""
        content = "// CHECK-LABEL: @foo\nutil.func..."
        run_lines = ["iree-opt %s"]

        result = inject_run_lines(content, run_lines)
        lines = result.split("\n")

        # RUN line replaces first line (CHECK-LABEL)
        self.assertTrue(lines[0].startswith("// RUN:"), "Line 1 should be RUN")
        self.assertTrue("util.func" in lines[1], "Line 2 should be util.func")


class TestIntegrationWithParse(unittest.TestCase):
    """Test integration of strip and inject with parse_test_file()."""

    def setUp(self):
        """Set up test fixture path."""
        self.fixture_path = Path(__file__).parent / "fixtures" / "split_test.mlir"

    def test_all_cases_preserve_line_numbers(self):
        """Test that all 3 cases preserve line numbers through strip/inject cycle."""
        test_file_obj = parse_test_file(self.fixture_path)
        cases = list(test_file_obj.cases)
        run_lines = test_file_obj.extract_run_lines(raw=True)

        # inject_run_lines preserves original line numbers by replacing leading blanks.
        # The test verifies CHECK-LABELs remain at their original file line numbers.
        expected_check_lines = {
            1: 4,  # CHECK-LABEL for first_case at line 4
            2: 13,  # CHECK-LABEL for second_case at line 13
            3: 24,  # CHECK-LABEL for third_case at line 24
        }

        for case in cases:
            # Simulate lit_wrapper processing
            blank_prefix = "\n" * (case.start_line - 1)
            content_with_blanks = blank_prefix + case.render_for_testing()
            result = inject_run_lines(content_with_blanks, run_lines)

            # Verify CHECK-LABEL is at expected line
            lines = result.split("\n")
            expected_line = expected_check_lines[case.number]

            self.assertTrue(
                len(lines) >= expected_line, f"Case {case.number}: not enough lines"
            )
            self.assertTrue(
                "CHECK-LABEL" in lines[expected_line - 1],
                f"Case {case.number}: CHECK-LABEL not at line {expected_line}, "
                f"found: {lines[expected_line - 1][:40]}",
            )


if __name__ == "__main__":
    unittest.main()
