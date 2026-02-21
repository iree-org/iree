# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for lit.core.test_file module."""

# Add project tools/utils to path for imports
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from lit_tools.core.parser import parse_test_file
from lit_tools.core.rendering import inject_run_lines_with_case

# Module-level fixture directory (absolute path for CWD-independence).
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class TestParseTestFile(unittest.TestCase):
    """Tests for parse_test_file function."""

    def setUp(self):
        """Set up fixture paths."""
        self.simple_test = _FIXTURES_DIR / "simple_test.mlir"
        self.split_test = _FIXTURES_DIR / "split_test.mlir"
        self.names_test = _FIXTURES_DIR / "names_test.mlir"

    def test_parse_simple_file(self):
        """Test parsing file with single test case (no delimiters)."""
        test_file_obj = parse_test_file(self.simple_test)
        cases = list(test_file_obj.cases)

        self.assertEqual(len(cases), 1)

        case = cases[0]
        self.assertEqual(case.number, 1)
        self.assertEqual(case.name, "simple_function")
        self.assertEqual(case.start_line, 1)
        self.assertGreater(case.line_count, 0)
        self.assertEqual(case.check_count, 3)  # CHECK-LABEL + 2 CHECK lines

    def test_parse_split_file(self):
        """Test parsing file with multiple test cases separated by delimiters."""
        test_file_obj = parse_test_file(self.split_test)
        cases = list(test_file_obj.cases)

        self.assertEqual(len(cases), 3)

        # First case
        self.assertEqual(cases[0].number, 1)
        self.assertEqual(cases[0].name, "first_case")
        self.assertEqual(cases[0].check_count, 3)

        # Second case
        self.assertEqual(cases[1].number, 2)
        self.assertEqual(cases[1].name, "second_case")
        self.assertEqual(cases[1].check_count, 4)  # CHECK-LABEL + 3 CHECKs

        # Third case
        self.assertEqual(cases[2].number, 3)
        self.assertEqual(cases[2].name, "third_case")
        self.assertEqual(cases[2].check_count, 3)

    def test_parse_names_with_punctuation(self):
        """Test CHECK-LABEL name extraction with punctuation characters."""
        test_file_obj = parse_test_file(self.names_test)
        cases = list(test_file_obj.cases)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].name, "foo.bar$baz-1")

    def test_line_ranges(self):
        """Test that line ranges are correctly calculated."""
        test_file_obj = parse_test_file(self.split_test)
        cases = list(test_file_obj.cases)

        # First case starts at line 1
        self.assertEqual(cases[0].start_line, 1)

        # Subsequent cases start after delimiters
        self.assertGreater(cases[1].start_line, cases[0].end_line)
        self.assertGreater(cases[2].start_line, cases[1].end_line)

        # Line counts reflect original file structure (may include trailing blank lines).
        # Content is normalized (trailing blank lines stripped).
        for case in cases:
            lines_in_content = case.content.count("\n") + 1
            # line_count >= lines in content (trailing blank lines may be stripped from content)
            self.assertGreaterEqual(case.line_count, lines_in_content)

    def test_content_extraction(self):
        """Test that full content is extracted for each case."""
        test_file_obj = parse_test_file(self.split_test)
        cases = list(test_file_obj.cases)

        # Each case should contain its function
        self.assertIn("@first_case", cases[0].content)
        self.assertIn("@second_case", cases[1].content)
        self.assertIn("@third_case", cases[2].content)

        # Cases should not contain delimiter lines
        for case in cases:
            self.assertNotIn("// -----", case.content)


class TestExtractCaseByNumber(unittest.TestCase):
    """Tests for extract_case_by_number function."""

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_extract_first_case(self):
        """Test extracting first test case."""
        test_file_obj = parse_test_file(self.split_test)
        case = test_file_obj.find_case_by_number(1)
        self.assertEqual(case.number, 1)
        self.assertEqual(case.name, "first_case")

    def test_extract_middle_case(self):
        """Test extracting middle test case."""
        test_file_obj = parse_test_file(self.split_test)
        case = test_file_obj.find_case_by_number(2)
        self.assertEqual(case.number, 2)
        self.assertEqual(case.name, "second_case")

    def test_extract_last_case(self):
        """Test extracting last test case."""
        test_file_obj = parse_test_file(self.split_test)
        case = test_file_obj.find_case_by_number(3)
        self.assertEqual(case.number, 3)
        self.assertEqual(case.name, "third_case")

    def test_invalid_case_number_too_low(self):
        """Test error when case number is too low."""
        with self.assertRaises(ValueError) as cm:
            test_file_obj = parse_test_file(self.split_test)
            test_file_obj.find_case_by_number(0)
        self.assertIn("out of range", str(cm.exception))

    def test_invalid_case_number_too_high(self):
        """Test error when case number is too high."""
        with self.assertRaises(ValueError) as cm:
            test_file_obj = parse_test_file(self.split_test)
            test_file_obj.find_case_by_number(10)
        self.assertIn("out of range", str(cm.exception))


class TestExtractCaseByName(unittest.TestCase):
    """Tests for extract_case_by_name function."""

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_extract_by_name(self):
        """Test extracting test case by function name."""
        test_file_obj = parse_test_file(self.split_test)
        case = test_file_obj.find_case_by_name("second_case")
        self.assertEqual(case.number, 2)
        self.assertEqual(case.name, "second_case")

    def test_extract_by_name_with_at_prefix(self):
        """Test extracting with @ prefix in name."""
        test_file_obj = parse_test_file(self.split_test)
        case = test_file_obj.find_case_by_name("@second_case")
        self.assertEqual(case.number, 2)
        self.assertEqual(case.name, "second_case")

    def test_invalid_name(self):
        """Test error when name doesn't exist."""
        with self.assertRaises(ValueError) as cm:
            test_file_obj = parse_test_file(self.split_test)
            test_file_obj.find_case_by_name("nonexistent")
        self.assertIn("No test case found", str(cm.exception))


class TestExtractCaseByLineNumber(unittest.TestCase):
    """Tests for extract_case_by_line_number function."""

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_extract_first_case_by_line(self):
        """Test extracting first case by line number."""
        # Line 5 is in first case
        test_file_obj = parse_test_file(self.split_test)
        case = test_file_obj.find_case_by_line(5)
        self.assertEqual(case.number, 1)
        self.assertEqual(case.name, "first_case")

    def test_extract_second_case_by_line(self):
        """Test extracting second case by line number."""
        # Line 18 is in second case
        test_file_obj = parse_test_file(self.split_test)
        case = test_file_obj.find_case_by_line(18)
        self.assertEqual(case.number, 2)
        self.assertEqual(case.name, "second_case")

    def test_extract_third_case_by_line(self):
        """Test extracting third case by line number."""
        # Line 30 is in third case
        test_file_obj = parse_test_file(self.split_test)
        case = test_file_obj.find_case_by_line(30)
        self.assertEqual(case.number, 3)
        self.assertEqual(case.name, "third_case")

    def test_line_at_case_boundary_start(self):
        """Test line number at start of case."""
        # First line of second case
        test_file_obj = parse_test_file(self.split_test)
        case = test_file_obj.find_case_by_line(14)
        self.assertEqual(case.number, 2)

    def test_line_at_case_boundary_end(self):
        """Test line number at end of case."""
        # Last line of first case
        test_file_obj = parse_test_file(self.split_test)
        case = test_file_obj.find_case_by_line(11)
        self.assertEqual(case.number, 1)

    def test_invalid_line_number_zero(self):
        """Test error when line number is zero."""
        with self.assertRaises(ValueError) as cm:
            test_file_obj = parse_test_file(self.split_test)
            test_file_obj.find_case_by_line(0)
        self.assertIn("must be >= 1", str(cm.exception))

    def test_invalid_line_number_negative(self):
        """Test error when line number is negative."""
        with self.assertRaises(ValueError) as cm:
            test_file_obj = parse_test_file(self.split_test)
            test_file_obj.find_case_by_line(-5)
        self.assertIn("must be >= 1", str(cm.exception))

    def test_invalid_line_number_too_high(self):
        """Test error when line number exceeds file length."""
        with self.assertRaises(ValueError) as cm:
            test_file_obj = parse_test_file(self.split_test)
            test_file_obj.find_case_by_line(1000)
        self.assertIn("out of range", str(cm.exception))


class TestExtractRunLines(unittest.TestCase):
    """Tests for extract_run_lines function."""

    def setUp(self):
        """Set up fixture paths."""
        self.simple_test = _FIXTURES_DIR / "simple_test.mlir"
        self.split_test = _FIXTURES_DIR / "split_test.mlir"
        self.run_variants = _FIXTURES_DIR / "run_variants.mlir"

    def test_extract_single_run_line(self):
        """Test extracting single RUN line."""
        test_file_obj = parse_test_file(self.simple_test)
        run_lines = test_file_obj.extract_run_lines()
        self.assertEqual(len(run_lines), 1)
        self.assertIn("iree-opt", run_lines[0])
        self.assertIn("FileCheck", run_lines[0])

    def test_extract_multiline_run(self):
        """Test extracting multi-line RUN command with continuations."""
        test_file_obj = parse_test_file(self.split_test)
        run_lines = test_file_obj.extract_run_lines()
        self.assertEqual(len(run_lines), 1)

        # Should be joined into single line
        run_line = run_lines[0]
        self.assertIn("--split-input-file", run_line)
        self.assertIn("--pass-pipeline", run_line)
        self.assertIn("FileCheck", run_line)

        # Should not contain backslashes (continuations removed)
        self.assertNotIn("\\", run_line)

    def test_extract_run_without_space_after_slashes(self):
        """Test extracting RUN lines with //RUN: (no space) and indented comments."""
        test_file_obj = parse_test_file(self.run_variants)
        run_lines = test_file_obj.extract_run_lines()
        # Should still parse into a single combined command
        self.assertEqual(len(run_lines), 1)
        self.assertIn("--pass-pipeline", run_lines[0])
        self.assertNotIn("\\", run_lines[0])

    def test_inject_run_lines_with_case_out_of_range_error(self):
        """Test that inject_run_lines_with_case errors with clear message when indices go out of range."""
        # Content with only 3 lines
        content = "line1\nline2\nline3"
        header_runs = []
        # Try to inject at index 5 (out of range)
        case_runs = [(5, "echo test")]

        with self.assertRaises(ValueError) as ctx:
            inject_run_lines_with_case(content, header_runs, case_runs)

        error_msg = str(ctx.exception)
        # Verify error message contains actionable guidance
        self.assertIn("Cannot inject RUN line at index 5", error_msg)
        self.assertIn("content only has 3 lines", error_msg)
        self.assertIn("--replace-run-lines", error_msg)
        self.assertIn("Solutions:", error_msg)


if __name__ == "__main__":
    unittest.main()
