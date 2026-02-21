# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Integration tests for iree-lit-test that actually run lit (not mocked).

These tests require:
- LLVM lit to be importable
- IREE build directory with iree-opt and FileCheck

Tests are automatically skipped if dependencies are unavailable.
"""

import io
import json
import os
import re
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout, suppress
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[2]))

from common import build_detection
from lit_tools import iree_lit_test
from lit_tools.core import lit_wrapper
from lit_tools.core.parser import parse_test_file

from test.test_helpers import run_python_module

# Module-level fixture directory (absolute path for CWD-independence).
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class TestIntegrationBasic(unittest.TestCase):
    """Basic end-to-end integration tests."""

    def setUp(self):
        """Set up test fixtures and skip if dependencies unavailable."""
        # Check if lit is importable.
        try:
            lit_wrapper._ensure_lit_importable()
        except Exception:
            self.skipTest("lit not importable in this environment")

        # Check if build directory is available.
        try:
            self.build_dir = build_detection.detect_build_dir()
        except FileNotFoundError:
            self.skipTest("Build directory not available (set IREE_BUILD_DIR)")

        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_basic_passing_case(self):
        """Run a simple passing test case end-to-end through lit pipeline."""
        # Parse test file to get cases.
        test_file_obj = parse_test_file(self.split_test)
        cases = list(test_file_obj.cases)
        self.assertGreater(len(cases), 0, "split_test.mlir should have test cases")

        # Run first case (known to pass).
        result = lit_wrapper.run_lit_on_case(
            case=cases[0],
            test_file_obj=test_file_obj,
            build_dir=self.build_dir,
            timeout=60,
            extra_flags=None,
            verbose=False,
            keep_temps=False,
        )

        # Verify results.
        self.assertTrue(result.passed, f"Test should pass but failed: {result.stderr}")
        self.assertEqual(result.case_number, 1)
        self.assertGreater(result.duration, 0, "Duration should be positive")
        self.assertIsNotNone(result.case_name)

    def test_keep_temps_preserves_file(self):
        """Verify --keep-temps actually keeps temp files."""
        test_file_obj = parse_test_file(self.split_test)
        cases = list(test_file_obj.cases)

        # Capture stdout to get "Temp file kept:" message.
        stdout_capture = io.StringIO()

        with redirect_stdout(stdout_capture):
            # Run with keep_temps=True and verbose to see temp file message.
            result = lit_wrapper.run_lit_on_case(
                case=cases[0],
                test_file_obj=test_file_obj,
                build_dir=self.build_dir,
                timeout=60,
                extra_flags=None,
                verbose=True,
                keep_temps=True,
            )

        # Verify test passed.
        self.assertTrue(result.passed, f"Test should pass: {result.stderr}")

        captured_output = stdout_capture.getvalue()

        # Extract temp file path from captured stdout.
        # Message format: "Temp file kept: <path>"
        temp_file_match = re.search(r"Temp file kept: (.+\.mlir)", captured_output)

        if not temp_file_match:
            # Fallback: construct expected temp directory using same logic as lit_wrapper.
            temp_dir = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
            temp_root = temp_dir / f"iree_lit_test_{os.getpid()}"

            # Find .mlir files in temp directory.
            if temp_root.exists():
                mlir_files = list(temp_root.glob("case*.mlir"))
                self.assertGreater(
                    len(mlir_files),
                    0,
                    f"Should find temp file in {temp_root}",
                )
                temp_file = mlir_files[0]
            else:
                self.fail(
                    f"Could not find temp file. Captured output:\n{captured_output}\n"
                    f"Expected temp dir: {temp_root}"
                )
        else:
            temp_file = Path(temp_file_match.group(1))

        self.assertTrue(
            temp_file.exists(),
            f"Temp file should exist with keep_temps=True: {temp_file}",
        )

        # Verify file contains test content.
        content = temp_file.read_text()
        self.assertIn("util.func", content, "Temp file should contain test content")

        # Clean up.
        temp_file.unlink()
        # Try to clean up parent directory if empty.
        with suppress(OSError):  # Directory not empty, that's fine.
            temp_file.parent.rmdir()

    def test_multiline_run_with_extra_flags(self):
        """Verify --extra-flags works with multi-line RUN directives."""
        # split_test.mlir has a 3-line RUN directive.
        test_file_obj = parse_test_file(self.split_test)
        cases = list(test_file_obj.cases)

        # Run with extra flags (use a harmless flag that shouldn't break test).
        result = lit_wrapper.run_lit_on_case(
            case=cases[0],
            test_file_obj=test_file_obj,
            build_dir=self.build_dir,
            timeout=60,
            extra_flags="--mlir-print-debuginfo",
            verbose=False,
            keep_temps=False,
        )

        # Should still pass (flag doesn't break the test).
        self.assertTrue(
            result.passed,
            f"Test should pass with extra flags, but failed: {result.stderr}",
        )

    def test_filecheck_line_numbers_match_lit(self):
        """Verify FileCheck error line numbers match between lit and iree-lit-test.

        This tests the core line number preservation feature - errors reported
        by FileCheck should show the same line numbers whether running via
        raw lit or iree-lit-test (which extracts cases to temp files with
        blank line padding).
        """
        # Use filecheck_error_test.mlir which is designed to fail.
        error_test = _FIXTURES_DIR / "filecheck_error_test.mlir"

        # Helper to extract first error line number from output.
        def first_error_line(text):
            m = re.search(r"\.mlir:(\d+):\d+:\s+error:", text)
            return int(m.group(1)) if m else None

        # 1) Run raw LLVM lit on the original file.
        try:
            from lit import (  # noqa: PLC0415 (deferred import for availability check)
                main as lit_main,
            )
        except Exception:
            self.skipTest("lit main not importable")

        saved_argv = sys.argv
        out_io, err_io = io.StringIO(), io.StringIO()
        try:
            sys.argv = ["lit", str(error_test), "-v"]
            with redirect_stdout(out_io), redirect_stderr(err_io), suppress(SystemExit):
                lit_main.main({})
        finally:
            sys.argv = saved_argv

        lit_output = out_io.getvalue() + "\n" + err_io.getvalue()
        lit_line = first_error_line(lit_output)
        self.assertIsNotNone(lit_line, "lit should report an error line number")

        # 2) Run iree-lit-test on case 2 with full JSON output.
        with patch(
            "sys.argv",
            [
                "iree-lit-test",
                str(error_test),
                "--case",
                "2",
                "--json",
                "--full-json",
            ],
        ):
            args = iree_lit_test.parse_arguments()

        out_json = io.StringIO()
        with redirect_stdout(out_json):
            rc = iree_lit_test.main(args)

        # Test should fail.
        self.assertNotEqual(rc, 0, "filecheck_error_test.mlir should fail")

        # Parse JSON and extract error line from failing case.
        payload = json.loads(out_json.getvalue())
        iree_lit_line = None
        for r in payload["results"]:
            if not r["passed"]:
                iree_lit_line = first_error_line(r.get("output", ""))
                break

        self.assertIsNotNone(
            iree_lit_line,
            "iree-lit-test should report an error line number",
        )

        # 3) Compare - line numbers should match.
        self.assertEqual(
            iree_lit_line,
            lit_line,
            f"Line numbers should match between lit ({lit_line}) "
            f"and iree-lit-test ({iree_lit_line})",
        )


class TestIntegrationParallel(unittest.TestCase):
    """Parallel execution integration tests."""

    def setUp(self):
        """Set up test fixtures and skip if dependencies unavailable."""
        # Check if lit is importable.
        try:
            lit_wrapper._ensure_lit_importable()
        except Exception:
            self.skipTest("lit not importable in this environment")

        # Check if build directory is available.
        try:
            self.build_dir = build_detection.detect_build_dir()
        except FileNotFoundError:
            self.skipTest("Build directory not available (set IREE_BUILD_DIR)")

        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_workers_actually_run_parallel(self):
        """Verify --workers N runs faster than sequential execution.

        This is a timing-based test with conservative threshold to avoid flakiness.
        """
        test_file_obj = parse_test_file(self.split_test)
        cases = list(test_file_obj.cases)
        if len(cases) < 3:
            self.skipTest("Need at least 3 test cases for meaningful parallel test")

        # Run first 3 cases sequentially (workers=1).
        start_time = time.time()
        sequential_results = []
        for case in cases[:3]:
            result = lit_wrapper.run_lit_on_case(
                case=case,
                test_file_obj=test_file_obj,
                build_dir=self.build_dir,
                timeout=60,
                extra_flags=None,
                verbose=False,
                keep_temps=False,
            )
            sequential_results.append(result)
        sequential_duration = time.time() - start_time

        # All should pass.
        for result in sequential_results:
            self.assertTrue(result.passed, f"Sequential test failed: {result.stderr}")

        # Run same 3 cases in parallel (workers=3).
        start_time = time.time()
        parallel_results = []

        def run_case(case):
            return lit_wrapper.run_lit_on_case(
                case=case,
                test_file_obj=test_file_obj,
                build_dir=self.build_dir,
                timeout=60,
                extra_flags=None,
                verbose=False,
                keep_temps=False,
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_case, case) for case in cases[:3]]
            parallel_results = [f.result() for f in futures]

        parallel_duration = time.time() - start_time

        # All should pass.
        for result in parallel_results:
            self.assertTrue(result.passed, f"Parallel test failed: {result.stderr}")

        # Verify parallel execution is faster than sequential.
        # Use conservative threshold (1.5x speedup) to avoid flaky failures.
        # This accounts for overhead, scheduling variance, and single-core systems.
        speedup_threshold = 0.67  # Parallel should take <=67% of sequential time.

        # Only assert if both runs took measurable time (>0.1s).
        if sequential_duration > 0.1 and parallel_duration > 0.1:
            speedup_ratio = parallel_duration / sequential_duration
            self.assertLess(
                speedup_ratio,
                speedup_threshold,
                f"Parallel execution should be faster. "
                f"Sequential: {sequential_duration:.2f}s, "
                f"Parallel: {parallel_duration:.2f}s, "
                f"Ratio: {speedup_ratio:.2f} (expected <{speedup_threshold})",
            )
        else:
            # Tests ran too fast to measure speedup reliably.
            # Just verify parallel execution completed without error.
            self.assertGreater(
                len(parallel_results),
                0,
                "Parallel execution completed successfully",
            )


class TestStdinIntegration(unittest.TestCase):
    """Integration tests for stdin mode with real lit execution."""

    def setUp(self):
        self.edge_cases = _FIXTURES_DIR / "edge_cases_test.mlir"
        self.ten_cases = _FIXTURES_DIR / "ten_cases_test.mlir"

    def test_stdin_from_subprocess_echo(self):
        """Test stdin input from echo via subprocess."""
        ir_content = "func.func @test() { return }"

        result = run_python_module(
            "lit_tools.iree_lit_test",
            [],
            input=ir_content,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("1 test case(s) passed", result.stderr)

    def test_stdin_from_subprocess_cat(self):
        """Test stdin input from file content via subprocess."""
        # Read edge case 1 (single quotes).
        with open(self.edge_cases) as f:
            lines = f.readlines()
        # Extract case 1: lines up to first separator.
        case_1_lines = []
        for line in lines:
            if "// -----" in line:
                break
            case_1_lines.append(line)
        ir_content = "".join(case_1_lines)

        result = run_python_module(
            "lit_tools.iree_lit_test",
            [],
            input=ir_content,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)

    def test_stdin_with_run_override(self):
        """Test stdin with --run flag override."""
        ir_content = "func.func @test() { return }"

        result = run_python_module(
            "lit_tools.iree_lit_test",
            ["--run", "iree-opt --split-input-file %s"],
            input=ir_content,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)

    def test_stdin_unicode_handling(self):
        """Test stdin with unicode characters."""
        ir_content = "// 日本語テスト\nfunc.func @test() { return }"

        result = run_python_module(
            "lit_tools.iree_lit_test",
            [],
            input=ir_content,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)

    def test_stdin_long_input(self):
        """Test stdin with large multi-case file content."""
        # Use entire ten_cases_test.mlir as stdin.
        with open(self.ten_cases) as f:
            ir_content = f.read()

        result = run_python_module(
            "lit_tools.iree_lit_test",
            [],
            input=ir_content,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
        # Should process all 10 cases.
        self.assertIn("10 test case(s) passed", result.stderr)

    def test_stdin_edge_cases_all_special_chars(self):
        """Test all edge cases from edge_cases_test.mlir via stdin."""
        for case_num in range(1, 6):
            with self.subTest(case=case_num):
                # Extract each case individually.
                extract_result = run_python_module(
                    "lit_tools.iree_lit_extract",
                    [str(self.edge_cases), "--case", str(case_num)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(extract_result.returncode, 0)

                # Pass extracted case to stdin of iree-lit-test.
                test_result = run_python_module(
                    "lit_tools.iree_lit_test",
                    [],
                    input=extract_result.stdout,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(
                    test_result.returncode,
                    0,
                    f"Case {case_num} failed: {test_result.stderr}",
                )


if __name__ == "__main__":
    unittest.main()
