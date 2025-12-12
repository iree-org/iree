# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CLI tests for iree_lit_test with mocked execution.

We avoid invoking real tools by stubbing lit_wrapper.run_lit_on_case.
"""

import io
import json
import sys
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lit_tools import iree_lit_test
from lit_tools.core.lit_wrapper import LitResult

# Module-level fixture directory (absolute path for CWD-independence).
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class TestCLIJSON(unittest.TestCase):
    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_json_shape_minimal(self, mock_stdout, mock_run):
        # Two cases: 1 pass, 1 fail
        mock_run.side_effect = [
            LitResult(True, 1, "first_case", 0.12, "out1", "", None, []),
            LitResult(False, 2, "second_case", 0.34, "out2", "", "fail", ["cmd"]),
        ]

        with patch("sys.argv", ["iree-lit-test", str(self.split_test), "--json"]):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 1)
        payload = json.loads(mock_stdout.getvalue())
        self.assertEqual(payload["total_cases"], 2)
        self.assertEqual(payload["passed"], 1)
        self.assertEqual(payload["failed"], 1)
        self.assertIn("results", payload)
        self.assertIsNone(payload["results"][0].get("output"))
        self.assertEqual(payload["results"][1]["run_commands"], ["cmd"])

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_json_full_includes_output(self, mock_stdout, mock_run):
        mock_run.return_value = LitResult(
            False, 1, "first_case", 0.12, "out1", "", "fail", []
        )

        with patch(
            "sys.argv",
            [
                "iree-lit-test",
                str(self.split_test),
                "--case",
                "1",
                "--json",
                "--full-json",
            ],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 1)
        payload = json.loads(mock_stdout.getvalue())
        self.assertIn("output", payload["results"][0])

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    def test_json_verbose_purity(self, mock_run):
        # Even with verbose=True, when --json is set, stdout must be pure JSON.
        mock_run.return_value = LitResult(
            True, 1, "first_case", 0.12, "VERBOSE OUT", "", None, []
        )

        with patch(
            "sys.argv",
            [
                "iree-lit-test",
                str(self.split_test),
                "--case",
                "1",
                "--verbose",
                "--json",
            ],
        ):
            args = iree_lit_test.parse_arguments()

        out = io.StringIO()
        with patch("sys.stdout", out):
            rc = iree_lit_test.main(args)
        self.assertIn('"results"', out.getvalue())
        self.assertNotIn("VERBOSE OUT", out.getvalue())
        self.assertEqual(rc, 0)

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    def test_filter_and_workers(self, mock_run):
        # Three cases, only second matches filter; with workers=2 ensure only one run occurs
        mock_run.return_value = LitResult(
            True, 2, "second_case", 0.01, "out", "", None, []
        )

        with patch(
            "sys.argv",
            [
                "iree-lit-test",
                str(self.split_test),
                "--filter",
                "second",
                "--workers",
                "2",
                "--quiet",
            ],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 0)
        self.assertEqual(mock_run.call_count, 1)

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    def test_workers_stop_on_first_failure(self, mock_run):
        # side effects for all 3 cases (even though we break after first failure)
        mock_run.side_effect = [
            LitResult(False, 1, "first", 0.01, "out", "", "fail", []),
            LitResult(True, 2, "second", 0.01, "out", "", None, []),
            LitResult(True, 3, "third", 0.01, "out", "", None, []),
        ]

        with patch(
            "sys.argv",
            ["iree-lit-test", str(self.split_test), "--workers", "2", "--quiet"],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 1)

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    def test_json_output_file(self, mock_run):
        mock_run.return_value = LitResult(
            True, 1, "first_case", 0.12, "out1", "", None, []
        )

        with NamedTemporaryFile("r+", suffix=".json", delete=False) as tmp:
            out = Path(tmp.name)
        try:
            with patch(
                "sys.argv",
                [
                    "iree-lit-test",
                    str(self.split_test),
                    "--case",
                    "1",
                    "--json",
                    "--json-output",
                    str(out),
                    "--quiet",
                ],
            ):
                args = iree_lit_test.parse_arguments()

            rc = iree_lit_test.main(args)
            self.assertEqual(rc, 0)
            data = json.loads(out.read_text())
            self.assertEqual(data["passed"], 1)
        finally:
            out.unlink()


class TestMultipleCaseSelection(unittest.TestCase):
    """Tests for multiple --case selection (comma, ranges, multiple flags)."""

    def setUp(self):
        self.ten_cases = _FIXTURES_DIR / "ten_cases_test.mlir"

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    def test_multiple_case_comma_separated(self, mock_run):
        """Test --case with comma-separated values."""
        mock_run.side_effect = [
            LitResult(True, 1, "case_one", 0.1, "", "", None, []),
            LitResult(True, 3, "case_three", 0.1, "", "", None, []),
            LitResult(True, 5, "case_five", 0.1, "", "", None, []),
        ]

        with patch(
            "sys.argv",
            ["iree-lit-test", str(self.ten_cases), "--case", "1,3,5", "--quiet"],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 0)
        # Verify run_lit_on_case was called 3 times.
        self.assertEqual(mock_run.call_count, 3)

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    def test_multiple_case_flags(self, mock_run):
        """Test multiple --case flags."""
        mock_run.side_effect = [
            LitResult(True, 2, "case_two", 0.1, "", "", None, []),
            LitResult(True, 4, "case_four", 0.1, "", "", None, []),
        ]

        with patch(
            "sys.argv",
            [
                "iree-lit-test",
                str(self.ten_cases),
                "--case",
                "2",
                "--case",
                "4",
                "--quiet",
            ],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 0)
        self.assertEqual(mock_run.call_count, 2)

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    def test_case_range(self, mock_run):
        """Test --case with range syntax."""
        mock_run.side_effect = [
            LitResult(True, 1, "case_one", 0.1, "", "", None, []),
            LitResult(True, 2, "case_two", 0.1, "", "", None, []),
            LitResult(True, 3, "case_three", 0.1, "", "", None, []),
        ]

        with patch(
            "sys.argv",
            ["iree-lit-test", str(self.ten_cases), "--case", "1-3", "--quiet"],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 0)
        self.assertEqual(mock_run.call_count, 3)

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    def test_case_mixed_comma_and_range(self, mock_run):
        """Test --case with mixed comma and range."""
        mock_run.side_effect = [
            LitResult(True, 1, "case_one", 0.1, "", "", None, []),
            LitResult(True, 3, "case_three", 0.1, "", "", None, []),
            LitResult(True, 4, "case_four", 0.1, "", "", None, []),
            LitResult(True, 5, "case_five", 0.1, "", "", None, []),
            LitResult(True, 7, "case_seven", 0.1, "", "", None, []),
        ]

        with patch(
            "sys.argv",
            ["iree-lit-test", str(self.ten_cases), "--case", "1,3-5,7", "--quiet"],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 0)
        self.assertEqual(mock_run.call_count, 5)

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_case_not_found_in_multi(self, mock_stderr):
        """Test that error is reported when one case in multi-case doesn't exist."""
        with patch(
            "sys.argv",
            ["iree-lit-test", str(self.ten_cases), "--case", "1,99"],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 2)  # NOT_FOUND exit code.
        self.assertIn("Case 99 not found", mock_stderr.getvalue())

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_case_invalid_format(self, mock_stderr):
        """Test that error is reported for invalid --case format."""
        with patch(
            "sys.argv",
            ["iree-lit-test", str(self.ten_cases), "--case", "abc"],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 2)  # NOT_FOUND exit code.
        self.assertIn("Invalid case number", mock_stderr.getvalue())


class TestListMode(unittest.TestCase):
    """Tests for --list mode."""

    def setUp(self):
        self.ten_cases = _FIXTURES_DIR / "ten_cases_test.mlir"

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_list_shows_all_cases(self, mock_stdout):
        """Test that --list shows all cases."""
        with patch("sys.argv", ["iree-lit-test", str(self.ten_cases), "--list"]):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 0)
        output = mock_stdout.getvalue()
        # Verify header and all 10 cases are listed.
        self.assertIn("ten_cases_test.mlir: 10 test cases", output)
        for i in range(1, 11):
            self.assertIn(f"{i}:", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_list_ignores_case_filter(self, mock_stdout):
        """Test that --list ignores --case filter and shows all cases."""
        with patch(
            "sys.argv",
            [
                "iree-lit-test",
                str(self.ten_cases),
                "--case",
                "1",
                "--case",
                "3",
                "--list",
            ],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 0)
        output = mock_stdout.getvalue()
        # All 10 cases should still be listed.
        self.assertIn("10 test cases", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_list_json_output(self, mock_stdout):
        """Test --list with --json outputs JSON."""
        with patch(
            "sys.argv",
            ["iree-lit-test", str(self.ten_cases), "--list", "--json"],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 0)
        payload = json.loads(mock_stdout.getvalue())
        self.assertEqual(payload["count"], 10)
        self.assertEqual(len(payload["cases"]), 10)
        self.assertEqual(payload["cases"][0]["number"], 1)
        self.assertEqual(payload["cases"][0]["name"], "case_one")

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_list_quiet_suppresses_header(self, mock_stdout):
        """Test --list with --quiet suppresses header."""
        with patch(
            "sys.argv",
            ["iree-lit-test", str(self.ten_cases), "--list", "--quiet"],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 0)
        output = mock_stdout.getvalue()
        # Header should be suppressed.
        self.assertNotIn("ten_cases_test.mlir:", output)
        # But cases should still be listed.
        self.assertIn("1:", output)


class TestDryRunMode(unittest.TestCase):
    """Tests for --dry-run mode."""

    def setUp(self):
        self.ten_cases = _FIXTURES_DIR / "ten_cases_test.mlir"

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_dry_run_shows_all_cases(self, mock_stderr):
        """Test that --dry-run shows all cases without executing."""
        with patch(
            "sys.argv",
            ["iree-lit-test", str(self.ten_cases), "--dry-run"],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 0)
        output = mock_stderr.getvalue()
        # Verify all 10 cases are listed (note goes to stderr).
        for i in range(1, 11):
            self.assertIn(f"Case {i}", output)
            self.assertIn(f"[{i}]", output)

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_dry_run_with_case_filter(self, mock_stderr):
        """Test --dry-run with --case filter."""
        with patch(
            "sys.argv",
            [
                "iree-lit-test",
                str(self.ten_cases),
                "--case",
                "1",
                "--case",
                "3",
                "--case",
                "5",
                "--dry-run",
            ],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 0)
        output = mock_stderr.getvalue()
        # Verify selected cases are listed.
        self.assertIn("Case 1", output)
        self.assertIn("Case 3", output)
        self.assertIn("Case 5", output)
        # Should not include other cases.
        self.assertNotIn("Case 2", output)
        self.assertNotIn("Case 4", output)
        # Verify numbering is correct ([1], [2], [3] not [1], [3], [5]).
        self.assertIn("[1] Case 1", output)
        self.assertIn("[2] Case 3", output)
        self.assertIn("[3] Case 5", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_dry_run_json_output(self, mock_stdout):
        """Test --dry-run with --json."""
        with patch(
            "sys.argv",
            [
                "iree-lit-test",
                str(self.ten_cases),
                "--case",
                "1-3",
                "--dry-run",
                "--json",
            ],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 0)
        payload = json.loads(mock_stdout.getvalue())
        self.assertEqual(payload["total_cases"], 3)
        self.assertEqual(len(payload["selected_cases"]), 3)
        self.assertEqual(payload["selected_cases"][0]["number"], 1)
        self.assertEqual(payload["selected_cases"][1]["number"], 2)
        self.assertEqual(payload["selected_cases"][2]["number"], 3)

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_dry_run_quiet_suppresses_note(self, mock_stderr):
        """Test --dry-run with --quiet suppresses note but shows cases."""
        with patch(
            "sys.argv",
            [
                "iree-lit-test",
                str(self.ten_cases),
                "--case",
                "1",
                "--case",
                "2",
                "--dry-run",
                "--quiet",
            ],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 0)
        output = mock_stderr.getvalue()
        # With --quiet, all note() calls are suppressed, so output should be empty.
        self.assertEqual(output, "")

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_dry_run_with_filter(self, mock_stderr):
        """Test --dry-run with --filter."""
        with patch(
            "sys.argv",
            [
                "iree-lit-test",
                str(self.ten_cases),
                "--filter",
                "five|seven",
                "--dry-run",
            ],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 0)
        output = mock_stderr.getvalue()
        # Verify filtered cases are listed.
        self.assertIn("@case_five", output)
        self.assertIn("@case_seven", output)
        # Should only have 2 cases (count lines with "[N]" pattern).
        lines = [line for line in output.split("\n") if "  [" in line and "]" in line]
        self.assertEqual(len(lines), 2)


class TestFilterAndFilterOut(unittest.TestCase):
    """Tests for --filter and --filter-out regex filtering."""

    def setUp(self):
        self.ten_cases = _FIXTURES_DIR / "ten_cases_test.mlir"

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    def test_filter_only(self, mock_run):
        """Test --filter to include only matching cases."""
        # Mock for cases matching "five|seven".
        mock_run.side_effect = [
            LitResult(True, 5, "case_five", 0.1, "", "", None, []),
            LitResult(True, 7, "case_seven", 0.1, "", "", None, []),
        ]

        with patch(
            "sys.argv",
            ["iree-lit-test", str(self.ten_cases), "--filter", "five|seven", "--quiet"],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 0)
        self.assertEqual(mock_run.call_count, 2)

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    def test_filter_out_only(self, mock_run):
        """Test --filter-out to exclude matching cases."""
        # All 10 cases except those with "five|seven" should run (8 cases).
        mock_run.return_value = LitResult(True, 1, "case_one", 0.1, "", "", None, [])

        with patch(
            "sys.argv",
            [
                "iree-lit-test",
                str(self.ten_cases),
                "--filter-out",
                "five|seven",
                "--quiet",
            ],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 0)
        # Should run 8 cases (10 total - 2 excluded).
        self.assertEqual(mock_run.call_count, 8)

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    def test_filter_and_filter_out_combined(self, mock_run):
        """Test --filter and --filter-out combined."""
        # All cases contain "e" in "case_" (10 total), then exclude "five|seven" (2 cases).
        mock_run.return_value = LitResult(True, 1, "case_one", 0.1, "", "", None, [])

        with patch(
            "sys.argv",
            [
                "iree-lit-test",
                str(self.ten_cases),
                "--filter",
                "e",
                "--filter-out",
                "five|seven",
                "--quiet",
            ],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 0)
        # Should run: all except five and seven (8 cases).
        self.assertEqual(mock_run.call_count, 8)

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_filter_no_matches(self, mock_stderr):
        """Test --filter with no matches returns error."""
        with patch(
            "sys.argv",
            ["iree-lit-test", str(self.ten_cases), "--filter", "nonexistent_pattern"],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 2)
        self.assertIn("No cases matched --filter", mock_stderr.getvalue())

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_filter_out_excludes_all(self, mock_stderr):
        """Test --filter-out that excludes all cases returns error."""
        with patch(
            "sys.argv",
            ["iree-lit-test", str(self.ten_cases), "--filter-out", "case_"],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 2)
        self.assertIn("All cases excluded by --filter-out", mock_stderr.getvalue())

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_filter_out_with_dry_run(self, mock_stderr):
        """Test --filter-out with --dry-run."""
        with patch(
            "sys.argv",
            [
                "iree-lit-test",
                str(self.ten_cases),
                "--filter-out",
                "five|seven",
                "--dry-run",
            ],
        ):
            args = iree_lit_test.parse_arguments()

        rc = iree_lit_test.main(args)
        self.assertEqual(rc, 0)
        output = mock_stderr.getvalue()
        # Should list 8 cases (all except five and seven).
        self.assertNotIn("@case_five", output)
        self.assertNotIn("@case_seven", output)
        self.assertIn("@case_one", output)
        self.assertIn("@case_two", output)


class MockStdin:
    """Mock stdin object with buffer attribute for testing."""

    def __init__(self, content: bytes, is_tty: bool = False):
        """Initialize with binary content."""
        self.buffer = io.BytesIO(content)
        self._is_tty = is_tty
        self._content = content

    def isatty(self) -> bool:
        """Return whether this is a TTY (default: False for piped input)."""
        return self._is_tty

    def read(self) -> str:
        """Read all content and return as string."""
        return self._content.decode("utf-8")


class TestStdinMode(unittest.TestCase):
    """Tests for stdin mode with mocked lit execution."""

    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"
        self.simple_test = _FIXTURES_DIR / "simple_test.mlir"
        self.edge_cases = _FIXTURES_DIR / "edge_cases_test.mlir"
        self.with_case_runs = _FIXTURES_DIR / "with_case_runs.mlir"
        self.no_run = _FIXTURES_DIR / "no_run_test.mlir"

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    @patch("sys.stdin", MockStdin(b"func.func @test() { return }"))
    @patch("sys.argv", ["iree-lit-test"])
    def test_stdin_basic_echo_pipe(self, mock_run):
        """Test basic stdin input with simple IR."""
        mock_run.return_value = LitResult(True, 1, None, 0.1, "out", "", None, [])

        args = iree_lit_test.parse_arguments()
        rc = iree_lit_test.main(args)

        self.assertEqual(rc, 0)
        self.assertTrue(mock_run.called)

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    @patch("sys.argv", ["iree-lit-test"])
    def test_stdin_with_existing_run_lines(self, mock_run):
        """Test stdin with RUN lines already present in input."""
        ir_with_run = b"// RUN: iree-opt %s\nfunc.func @test() { return }"
        mock_run.return_value = LitResult(True, 1, None, 0.1, "out", "", None, [])

        with patch("sys.stdin", MockStdin(ir_with_run)):
            args = iree_lit_test.parse_arguments()
            rc = iree_lit_test.main(args)

        self.assertEqual(rc, 0)

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    @patch("sys.stderr", new_callable=io.StringIO)
    @patch("sys.argv", ["iree-lit-test"])
    def test_stdin_no_run_defaults_to_ireeopt(self, mock_stderr, mock_run):
        """Test stdin without RUN lines defaults to iree-opt %s."""
        ir_no_run = b"func.func @test() { return }"
        mock_run.return_value = LitResult(True, 1, None, 0.1, "out", "", None, [])

        with patch("sys.stdin", MockStdin(ir_no_run)):
            args = iree_lit_test.parse_arguments()
            rc = iree_lit_test.main(args)

        self.assertEqual(rc, 0)
        self.assertIn("defaulting to: iree-opt %s", mock_stderr.getvalue())

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    @patch("sys.argv", ["iree-lit-test", "--run", "iree-opt --canonicalize %s"])
    def test_stdin_run_flag_override(self, mock_run):
        """Test --run flag overrides RUN lines in stdin."""
        ir_with_run = b"// RUN: iree-opt %s\nfunc.func @test() { return }"
        mock_run.return_value = LitResult(True, 1, None, 0.1, "out", "", None, [])

        with patch("sys.stdin", MockStdin(ir_with_run)):
            args = iree_lit_test.parse_arguments()
            rc = iree_lit_test.main(args)

        self.assertEqual(rc, 0)
        # Verify --run was used (RUN override worked).
        self.assertTrue(mock_run.called)

    @patch("sys.stderr", new_callable=io.StringIO)
    @patch("sys.argv", ["iree-lit-test"])
    def test_stdin_empty_input_errors(self, mock_stderr):
        """Test empty stdin input produces error."""
        with patch("sys.stdin", MockStdin(b"")):
            args = iree_lit_test.parse_arguments()
            rc = iree_lit_test.main(args)

        self.assertEqual(rc, 1)
        self.assertIn("No input provided", mock_stderr.getvalue())

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    @patch("sys.argv", ["iree-lit-test"])
    def test_stdin_special_chars_single_quotes(self, mock_run):
        """Test stdin with single quotes in attributes."""
        # Case 1 from edge_cases_test.mlir
        ir_content = b"func.func @test() attributes {foo = 'bar'} { return }"
        mock_run.return_value = LitResult(True, 1, None, 0.1, "out", "", None, [])

        with patch("sys.stdin", MockStdin(ir_content)):
            args = iree_lit_test.parse_arguments()
            rc = iree_lit_test.main(args)

        self.assertEqual(rc, 0)

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    @patch("sys.argv", ["iree-lit-test"])
    def test_stdin_special_chars_double_quotes(self, mock_run):
        """Test stdin with double quotes in string literals."""
        ir_content = b'util.global @str = "escaped\\"quote" : !util.buffer'
        mock_run.return_value = LitResult(True, 1, None, 0.1, "out", "", None, [])

        with patch("sys.stdin", MockStdin(ir_content)):
            args = iree_lit_test.parse_arguments()
            rc = iree_lit_test.main(args)

        self.assertEqual(rc, 0)

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    @patch("sys.argv", ["iree-lit-test"])
    def test_stdin_special_chars_unicode(self, mock_run):
        """Test stdin with unicode characters."""
        ir_content = "// 日本語テスト\nfunc.func @test() { return }".encode()
        mock_run.return_value = LitResult(True, 1, None, 0.1, "out", "", None, [])

        with patch("sys.stdin", MockStdin(ir_content)):
            args = iree_lit_test.parse_arguments()
            rc = iree_lit_test.main(args)

        self.assertEqual(rc, 0)

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    @patch("sys.stderr", new_callable=io.StringIO)
    @patch("sys.argv", ["iree-lit-test"])
    def test_stdin_tty_detection_shows_prompt(self, mock_stderr, mock_run):
        """Test TTY detection shows prompt message."""
        ir_content = b"func.func @test() { return }"
        mock_run.return_value = LitResult(True, 1, None, 0.1, "out", "", None, [])

        with patch("sys.stdin", MockStdin(ir_content, is_tty=True)):
            args = iree_lit_test.parse_arguments()
            rc = iree_lit_test.main(args)

        self.assertEqual(rc, 0)
        output = mock_stderr.getvalue()
        self.assertIn("Reading test input from stdin", output)
        self.assertIn("Ctrl-D to finish", output)

    @patch("lit_tools.iree_lit_test.lit_wrapper.run_lit_on_case")
    @patch("sys.stderr", new_callable=io.StringIO)
    @patch("sys.argv", ["iree-lit-test", "--quiet"])
    def test_stdin_quiet_suppresses_prompt(self, mock_stderr, mock_run):
        """Test --quiet suppresses TTY prompt."""
        ir_content = b"func.func @test() { return }"
        mock_run.return_value = LitResult(True, 1, None, 0.1, "out", "", None, [])

        with patch("sys.stdin", MockStdin(ir_content, is_tty=True)):
            args = iree_lit_test.parse_arguments()
            rc = iree_lit_test.main(args)

        self.assertEqual(rc, 0)
        output = mock_stderr.getvalue()
        self.assertNotIn("Reading test input from stdin", output)


if __name__ == "__main__":
    unittest.main()
