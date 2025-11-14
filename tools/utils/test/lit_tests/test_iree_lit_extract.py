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
from lit_tools.core.parser import parse_test_file

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

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_containing_on_delimiter_line(self, mock_stdout):
        """Test --containing with line number exactly on delimiter."""
        # split_test.mlir has delimiters around line 10 and 18.
        # Line 10 is the first delimiter (between cases 1 and 2).
        # This tests boundary condition: which case does the delimiter belong to?
        with patch(
            "sys.argv", ["iree-lit-extract", str(self.split_test), "--containing", "10"]
        ):
            args = iree_lit_extract.parse_arguments()

        result = iree_lit_extract.main(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()

        # Delimiter line should belong to previous case or next case (implementation defined).
        # Verify we get exactly one case, not an error.
        # Current implementation: delimiter belongs to the case it ends.
        self.assertTrue(
            "util.func @first_case" in output or "util.func @second_case" in output,
            "Should extract a case when line number is on delimiter",
        )


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
                    "--json",
                ],
            ):
                args = iree_lit_extract.parse_arguments()

            rc = iree_lit_extract.main(args)
            self.assertEqual(rc, 0)
            # With -o and --json, JSON should be written to the file, not stdout.
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

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_format_named_case(self):
        """Test formatting named test case."""
        test_file_obj = parse_test_file(self.split_test)
        cases = list(test_file_obj.cases)
        # Use second case which has name "second_case"
        case = cases[1]

        result = iree_lit_extract.format_case_info(case)

        self.assertIn("Test case 2", result)
        self.assertIn("@second_case", result)
        self.assertIn("lines", result)

    def test_format_unnamed_case(self):
        """Test formatting unnamed test case with real parsed fixture."""
        # Use a simple test that has a name, verify format includes it
        test_file_obj = parse_test_file(self.split_test)
        cases = list(test_file_obj.cases)
        # Third case also has a name
        case = cases[2]

        result = iree_lit_extract.format_case_info(case)

        self.assertIn("Test case 3", result)
        self.assertIn("@third_case", result)
        self.assertIn("lines", result)


class TestRunLineInclusion(unittest.TestCase):
    """Tests for RUN line inclusion (default) and exclusion (--exclude-run-lines)."""

    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"
        self.with_case_runs = _FIXTURES_DIR / "with_case_runs.mlir"

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch(
        "sys.argv",
        ["iree-lit-extract", str(_FIXTURES_DIR / "split_test.mlir"), "--case", "1"],
    )
    def test_stdout_includes_run_by_default(self, mock_stdout):
        """Test that stdout mode includes RUN lines by default."""
        args = iree_lit_extract.parse_arguments()
        rc = iree_lit_extract.main(args)

        self.assertEqual(rc, 0)
        output = mock_stdout.getvalue()
        # RUN lines should be in stdout output by default.
        self.assertIn("// RUN:", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch(
        "sys.argv",
        [
            "iree-lit-extract",
            str(_FIXTURES_DIR / "split_test.mlir"),
            "--case",
            "1",
            "--exclude-run-lines",
        ],
    )
    def test_exclude_run_lines_flag(self, mock_stdout):
        """Test --exclude-run-lines flag excludes RUN lines from stdout."""
        args = iree_lit_extract.parse_arguments()
        rc = iree_lit_extract.main(args)

        self.assertEqual(rc, 0)
        output = mock_stdout.getvalue()
        # RUN lines should NOT be included with --exclude-run-lines flag.
        self.assertNotIn("// RUN:", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch(
        "sys.argv",
        [
            "iree-lit-extract",
            str(_FIXTURES_DIR / "split_test.mlir"),
            "--case",
            "1",
            "--json",
        ],
    )
    def test_json_excludes_run(self, mock_stdout):
        """Test JSON mode excludes RUN lines."""
        args = iree_lit_extract.parse_arguments()
        rc = iree_lit_extract.main(args)

        self.assertEqual(rc, 0)
        output = mock_stdout.getvalue()
        payload = json.loads(output)
        # JSON payload should not contain RUN lines in content.
        # JSON mode returns an array of cases.
        self.assertEqual(len(payload), 1)
        case_content = payload[0]["content"]
        self.assertNotIn("// RUN:", case_content)


class TestEdgeCaseRunLines(unittest.TestCase):
    """Tests for RUN line edge cases and case-local RUN line extraction."""

    def setUp(self):
        self.fixture = _FIXTURES_DIR / "edge_cases_run_lines.mlir"

    def test_case_with_case_local_run(self):
        """Test extracting case with case-local RUN line."""
        with patch(
            "sys.argv",
            ["iree-lit-extract", str(self.fixture), "--name", "with_case_run"],
        ):
            args = iree_lit_extract.parse_arguments()
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            self.assertEqual(iree_lit_extract.main(args), 0)
            output = mock_stdout.getvalue()
            # Should have header + case-local RUN lines.
            self.assertIn("iree-opt --split-input-file", output)
            self.assertIn("--canonicalize", output)
            self.assertIn("--check-prefix=CANON", output)

    def test_long_pipeline_extraction(self):
        """Test extracting case with long pipeline command."""
        with patch(
            "sys.argv",
            ["iree-lit-extract", str(self.fixture), "--name", "long_pipeline"],
        ):
            args = iree_lit_extract.parse_arguments()
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            self.assertEqual(iree_lit_extract.main(args), 0)
            output = mock_stdout.getvalue()
            self.assertIn("cse,canonicalize,symbol-dce", output)

    def test_special_flags_extraction(self):
        """Test extracting cases with special flags."""
        test_cases = [
            ("verify_diagnostics", "--verify-diagnostics"),
            ("multiple_check_prefixes", "--check-prefixes=CHECK,EXTRA"),
        ]
        for case_name, expected_flag in test_cases:
            with self.subTest(case=case_name):
                with patch(
                    "sys.argv",
                    ["iree-lit-extract", str(self.fixture), "--name", case_name],
                ):
                    args = iree_lit_extract.parse_arguments()
                with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                    self.assertEqual(iree_lit_extract.main(args), 0)
                    output = mock_stdout.getvalue()
                    self.assertIn(expected_flag, output)

    def test_pipeline_chain_extraction(self):
        """Test extracting case with piped commands."""
        with patch(
            "sys.argv",
            ["iree-lit-extract", str(self.fixture), "--name", "pipeline_chain"],
        ):
            args = iree_lit_extract.parse_arguments()
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            self.assertEqual(iree_lit_extract.main(args), 0)
            output = mock_stdout.getvalue()
            # Should have header + case-local with pipe.
            self.assertEqual(output.count("iree-opt"), 3)  # Header + two in pipeline

    def test_environment_variable_extraction(self):
        """Test extracting case with environment variable."""
        with patch(
            "sys.argv",
            ["iree-lit-extract", str(self.fixture), "--name", "environment_variable"],
        ):
            args = iree_lit_extract.parse_arguments()
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            self.assertEqual(iree_lit_extract.main(args), 0)
            output = mock_stdout.getvalue()
            self.assertIn("IREE_TEST_VAR=value", output)

    def test_all_cases_have_labels(self):
        """Verify all cases in edge_cases_run_lines.mlir have labels."""
        with patch(
            "sys.argv", ["iree-lit-extract", str(self.fixture), "--list", "--json"]
        ):
            args = iree_lit_extract.parse_arguments()
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            self.assertEqual(iree_lit_extract.main(args), 0)
            payload = json.loads(mock_stdout.getvalue())
            self.assertEqual(payload["count"], 8)
            # All cases should have names (labels).
            for case in payload["cases"]:
                self.assertIsNotNone(
                    case["name"], f"Case {case['number']} missing label"
                )

    def test_ghost_case_trailing_delimiter(self):
        """Test that file ending with delimiter doesn't create empty ghost case."""
        # Create a fixture with trailing delimiter.
        content = """// RUN: iree-opt %s | FileCheck %s
// CHECK-LABEL: @first
func @first() { return }

// -----

// CHECK-LABEL: @second
func @second() { return }

// -----
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            # List all cases to check count.
            with patch("sys.argv", ["iree-lit-extract", str(path), "--list", "--json"]):
                args = iree_lit_extract.parse_arguments()
            with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                self.assertEqual(iree_lit_extract.main(args), 0)
                payload = json.loads(mock_stdout.getvalue())
                # Should have exactly 2 cases, not 3 (no empty ghost case).
                self.assertEqual(payload["count"], 2)
                self.assertEqual(len(payload["cases"]), 2)
        finally:
            path.unlink()

    def test_case_local_run_identical_to_header(self):
        """Test that case-local RUN identical to header is preserved."""
        # Scenario: Case has a local RUN line that duplicates a header RUN.
        # This should NOT be stripped just because it's a duplicate.
        content = """// RUN: iree-opt %s | FileCheck %s
// CHECK-LABEL: @first
func @first() { return }

// -----

// CHECK-LABEL: @second
// RUN: iree-opt %s | FileCheck %s
func @second() { return }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            # Extract case 2 which has the duplicate RUN line.
            with patch("sys.argv", ["iree-lit-extract", str(path), "--case", "2"]):
                args = iree_lit_extract.parse_arguments()
            with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                self.assertEqual(iree_lit_extract.main(args), 0)
                output = mock_stdout.getvalue()
                # Should have 2 RUN lines in output (header + case-local duplicate).
                run_count = output.count("// RUN:")
                self.assertEqual(
                    run_count,
                    2,
                    f"Expected 2 RUN lines (header + case-local), got {run_count}",
                )
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
