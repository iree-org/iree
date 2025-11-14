# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for iree_lit_extract tool."""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add project tools/utils to path for imports
sys.path.insert(0, str(Path(__file__).parents[2]))

from lit_tools import iree_lit_extract
from lit_tools.core import test_file

# Module-level fixture directory (absolute path for CWD-independence).
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class TestExtractByNumber(unittest.TestCase):
    """Tests for extracting test cases by number."""

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    @patch("sys.argv", ["iree-lit-extract", "dummy.mlir", "--case", "2"])
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_extract_second_case_to_stdout(self, mock_stdout):
        """Test extracting second case to stdout."""
        args = iree_lit_extract.parse_arguments()
        args.file = str(self.split_test)

        result = iree_lit_extract.main(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()

        # Should contain the actual function
        self.assertIn("util.func @second_case", output)

    def test_extract_to_file(self):
        """Test extracting test case to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with patch(
                "sys.argv",
                [
                    "iree-lit-extract",
                    str(self.split_test),
                    "--case",
                    "2",
                    "-o",
                    str(tmp_path),
                ],
            ):
                args = iree_lit_extract.parse_arguments()

            result = iree_lit_extract.main(args)

            self.assertEqual(result, 0)
            self.assertTrue(tmp_path.exists())

            # Verify content
            content = tmp_path.read_text()
            self.assertIn("@second_case", content)
            self.assertIn("util.func @second_case", content)

        finally:
            tmp_path.unlink()


class TestExtractByName(unittest.TestCase):
    """Tests for extracting test cases by name."""

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_extract_by_name(self, mock_stdout):
        """Test extracting by function name."""
        with patch(
            "sys.argv",
            ["iree-lit-extract", str(self.split_test), "--name", "third_case"],
        ):
            args = iree_lit_extract.parse_arguments()

        result = iree_lit_extract.main(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()

        self.assertIn("util.func @third_case", output)


class TestExtractByLineNumber(unittest.TestCase):
    """Tests for extracting test cases by line number."""

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_extract_by_containing_line(self, mock_stdout):
        """Test extracting test case containing specific line."""
        with patch(
            "sys.argv", ["iree-lit-extract", str(self.split_test), "--containing", "18"]
        ):
            args = iree_lit_extract.parse_arguments()

        result = iree_lit_extract.main(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()

        # Should extract second case body
        self.assertIn("util.func @second_case", output)


class TestIncludeRunLines(unittest.TestCase):
    """Tests for including RUN lines."""

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_include_run_lines(self, mock_stdout):
        """Test including RUN lines in output."""
        with patch(
            "sys.argv",
            [
                "iree-lit-extract",
                str(self.split_test),
                "--case",
                "1",
                "--include-run-lines",
            ],
        ):
            args = iree_lit_extract.parse_arguments()

        result = iree_lit_extract.main(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()

        # Should contain RUN lines
        self.assertIn("// RUN:", output)
        self.assertIn("iree-opt", output)
        self.assertIn("--split-input-file", output)
        self.assertIn("FileCheck", output)

        # RUN header should not be duplicated from case content
        self.assertEqual(output.count("// RUN:"), 1)

    def test_stdout_does_not_include_runs_by_default(self):
        with patch(
            "sys.argv", ["iree-lit-extract", str(self.split_test), "--case", "1"]
        ):
            args = iree_lit_extract.parse_arguments()
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            rc = iree_lit_extract.main(args)
            self.assertEqual(rc, 0)
            self.assertNotIn("// RUN:", mock_stdout.getvalue())

    def test_file_includes_runs_by_default(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            out = Path(tmp.name)
        try:
            with patch(
                "sys.argv",
                [
                    "iree-lit-extract",
                    str(self.split_test),
                    "--case",
                    "1,3",
                    "-o",
                    str(out),
                ],
            ):
                args = iree_lit_extract.parse_arguments()
            rc = iree_lit_extract.main(args)
            self.assertEqual(rc, 0)
            text = out.read_text()
            # RUN lines present once at the top
            self.assertIn("// RUN:", text)
            # Multiple cases separated by delimiter
            self.assertIn("// -----", text)
            # No synthetic banners
            self.assertNotIn("Test case ", text)
        finally:
            out.unlink(missing_ok=True)


class TestListMode(unittest.TestCase):
    """Tests for --list mode."""

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_list_mode(self, mock_stdout):
        """Test listing test cases."""
        with patch("sys.argv", ["iree-lit-extract", str(self.split_test), "--list"]):
            args = iree_lit_extract.parse_arguments()

        result = iree_lit_extract.main(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()

        # Should list all cases
        self.assertIn("3 test cases", output)
        self.assertIn("@first_case", output)
        self.assertIn("@second_case", output)
        self.assertIn("@third_case", output)


class TestErrorCases(unittest.TestCase):
    """Tests for error handling."""

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_file_not_found(self, mock_stderr):
        """Test error when file doesn't exist."""
        with patch(
            "sys.argv", ["iree-lit-extract", "/nonexistent/file.mlir", "--case", "1"]
        ):
            args = iree_lit_extract.parse_arguments()

        result = iree_lit_extract.main(args)

        self.assertEqual(result, 2)
        error = mock_stderr.getvalue()
        self.assertIn("File not found", error)

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_invalid_case_number(self, mock_stderr):
        """Test error when case number is out of range."""
        with patch(
            "sys.argv", ["iree-lit-extract", str(self.split_test), "--case", "99"]
        ):
            args = iree_lit_extract.parse_arguments()

        result = iree_lit_extract.main(args)

        self.assertEqual(result, 2)
        error = mock_stderr.getvalue()
        self.assertIn("not found", error)

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_invalid_name(self, mock_stderr):
        """Test error when function name doesn't exist."""
        with patch(
            "sys.argv",
            [
                "iree-lit-extract",
                str(self.split_test),
                "--name",
                "nonexistent_function",
            ],
        ):
            args = iree_lit_extract.parse_arguments()

        result = iree_lit_extract.main(args)

        self.assertEqual(result, 2)
        error = mock_stderr.getvalue()
        self.assertIn("Case with name", error)
        self.assertIn("not found", error)

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_invalid_line_number(self, mock_stderr):
        """Test error when line number is out of range."""
        with patch(
            "sys.argv",
            ["iree-lit-extract", str(self.split_test), "--containing", "9999"],
        ):
            args = iree_lit_extract.parse_arguments()

        result = iree_lit_extract.main(args)

        self.assertEqual(result, 2)
        error = mock_stderr.getvalue()
        self.assertIn("No case contains line", error)


class TestArgumentExclusivity(unittest.TestCase):
    """Tests for mutually exclusive argument handling."""

    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_selector_and_list_conflict(self, mock_stderr):
        with patch(
            "sys.argv",
            ["iree-lit-extract", str(self.split_test), "--case", "1", "--list"],
        ):
            args = iree_lit_extract.parse_arguments()

        result = iree_lit_extract.main(args)
        self.assertEqual(result, 2)
        self.assertIn("specify exactly one", mock_stderr.getvalue())

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_missing_all_selection(self, mock_stderr):
        with patch("sys.argv", ["iree-lit-extract", str(self.split_test)]):
            args = iree_lit_extract.parse_arguments()

        result = iree_lit_extract.main(args)
        self.assertEqual(result, 2)
        self.assertIn("specify exactly one", mock_stderr.getvalue())


class TestValidationStrict(unittest.TestCase):
    """Tests for strict validation when iree-opt is unavailable."""

    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    @patch("sys.stderr", new_callable=io.StringIO)
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_error_on_missing_iree_opt(self, mock_stdout, mock_stderr):
        # Mock build_detection.find_tool to raise FileNotFoundError
        with patch(
            "lit_tools.core.verification.build_detection.find_tool",
            side_effect=FileNotFoundError("iree-opt not found"),
        ):
            with patch(
                "sys.argv",
                ["iree-lit-extract", str(self.split_test), "--case", "1", "--verify"],
            ):
                args = iree_lit_extract.parse_arguments()

            result = iree_lit_extract.main(args)
            self.assertEqual(result, 1)
            err = mock_stderr.getvalue()
            self.assertIn("iree-opt", err)
            self.assertIn("not found", err)


class TestJSON(unittest.TestCase):
    """Tests for JSON output from iree-lit-extract."""

    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_list_json(self, mock_stdout):
        with patch(
            "sys.argv", ["iree-lit-extract", str(self.split_test), "--list", "--json"]
        ):
            args = iree_lit_extract.parse_arguments()

        rc = iree_lit_extract.main(args)
        self.assertEqual(rc, 0)
        payload = json.loads(mock_stdout.getvalue())
        self.assertEqual(payload["count"], 3)
        self.assertEqual(len(payload["cases"]), 3)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_extract_json_to_stdout_array(self, mock_stdout):
        with patch(
            "sys.argv",
            ["iree-lit-extract", str(self.split_test), "--case", "2", "--json"],
        ):
            args = iree_lit_extract.parse_arguments()

        rc = iree_lit_extract.main(args)
        self.assertEqual(rc, 0)
        payload = json.loads(mock_stdout.getvalue())
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["number"], 2)
        # Content included
        self.assertIn("util.func @second_case", payload[0]["content"])

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_extract_multiple_json_array(self, mock_stdout):
        with patch(
            "sys.argv",
            [
                "iree-lit-extract",
                str(self.split_test),
                "--case",
                "1,3",
                "--json",
            ],
        ):
            args = iree_lit_extract.parse_arguments()
        rc = iree_lit_extract.main(args)
        self.assertEqual(rc, 0)
        arr = json.loads(mock_stdout.getvalue())
        self.assertIsInstance(arr, list)
        self.assertEqual(len(arr), 2)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_extract_json_success(self, mock_stdout):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            out = Path(tmp.name)

        try:
            with patch(
                "sys.argv",
                [
                    "iree-lit-extract",
                    str(self.split_test),
                    "--case",
                    "2",
                    "-o",
                    str(out),
                    "--include-run-lines",
                    "--json",
                ],
            ):
                args = iree_lit_extract.parse_arguments()

            rc = iree_lit_extract.main(args)
            self.assertEqual(rc, 0)
            # With -o and --json, JSON should be written to the file, not stdout
            self.assertEqual(mock_stdout.getvalue(), "")
            self.assertTrue(out.exists())
            data = json.loads(out.read_text())
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["number"], 2)
            self.assertIn("util.func @second_case", data[0]["content"])
        finally:
            out.unlink(missing_ok=True)

    def test_filter_extraction(self):
        ten_cases = Path(_FIXTURES_DIR / "ten_cases_test.mlir")
        with patch(
            "sys.argv",
            [
                "iree-lit-extract",
                str(ten_cases),
                "--filter",
                "five",
            ],
        ):
            args = iree_lit_extract.parse_arguments()
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            rc = iree_lit_extract.main(args)
            self.assertEqual(rc, 0)
            out = mock_stdout.getvalue()
            self.assertIn("@case_five", out)
        # no temp files to clean up in this test


class TestFormatCaseInfo(unittest.TestCase):
    """Tests for format_case_info helper."""

    def test_format_named_case(self):
        """Test formatting named test case."""
        case = test_file.TestCase(
            number=2,
            name="my_function",
            content="",
            start_line=10,
            end_line=20,
            line_count=11,
            check_count=3,
        )

        result = iree_lit_extract.format_case_info(case)

        self.assertIn("Test case 2", result)
        self.assertIn("@my_function", result)
        self.assertIn("lines 10-20", result)

    def test_format_unnamed_case(self):
        """Test formatting unnamed test case."""
        case = test_file.TestCase(
            number=3,
            name=None,
            content="",
            start_line=30,
            end_line=40,
            line_count=11,
            check_count=0,
        )

        result = iree_lit_extract.format_case_info(case)

        self.assertIn("Test case 3", result)
        self.assertIn("(unnamed)", result)
        self.assertIn("lines 30-40", result)


if __name__ == "__main__":
    unittest.main()
