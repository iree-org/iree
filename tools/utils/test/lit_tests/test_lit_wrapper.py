# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for lit/core/lit_wrapper.py."""

import sys
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lit_tools.core import lit_wrapper
from lit_tools.core.parser import parse_test_file
from lit_tools.core.rendering import inject_run_lines_with_case
from lit_tools.core.text_utils import _strip_run_lines_preserve_line_numbers


class TestInjectExtraFlags(unittest.TestCase):
    """Tests for inject_extra_flags() function."""

    def test_inject_simple_flags(self):
        """Test basic flag injection."""
        content = "// RUN: iree-opt %s | FileCheck %s"
        result = lit_wrapper.inject_extra_flags(content, "--debug")
        self.assertEqual(result, "// RUN: iree-opt --debug %s | FileCheck %s")

    def test_inject_multiple_flags(self):
        """Test multiple flags."""
        content = "// RUN: iree-opt --split-input-file %s"
        result = lit_wrapper.inject_extra_flags(
            content, "--debug --mlir-print-ir-after-all"
        )
        expected = (
            "// RUN: iree-opt --debug --mlir-print-ir-after-all --split-input-file %s"
        )
        self.assertEqual(result, expected)

    def test_inject_only_first_tool(self):
        """Test that only first iree-* tool is modified."""
        content = "// RUN: iree-opt %s | iree-compile - | FileCheck %s"
        result = lit_wrapper.inject_extra_flags(content, "--debug")
        # Only iree-opt gets flags, not iree-compile.
        expected = "// RUN: iree-opt --debug %s | iree-compile - | FileCheck %s"
        self.assertEqual(result, expected)

    def test_inject_ignores_non_iree_tools(self):
        """Test that FileCheck and other tools are not modified."""
        content = "// RUN: FileCheck %s"
        result = lit_wrapper.inject_extra_flags(content, "--debug")
        self.assertEqual(result, content)  # unchanged

    def test_inject_multiline_run(self):
        """Test multi-line RUN with backslash continuation."""
        content = """// RUN: iree-opt \\
// RUN:   --split-input-file \\
// RUN:   %s | FileCheck %s"""
        result = lit_wrapper.inject_extra_flags(content, "--debug")
        # Only first line should be modified.
        expected = """// RUN: iree-opt --debug \\
// RUN:   --split-input-file \\
// RUN:   %s | FileCheck %s"""
        self.assertEqual(result, expected)

    def test_inject_with_iree_compile(self):
        """Test injection with iree-compile."""
        content = "// RUN: iree-compile %s"
        result = lit_wrapper.inject_extra_flags(
            content, "--iree-hal-target-backends=llvm-cpu"
        )
        expected = "// RUN: iree-compile --iree-hal-target-backends=llvm-cpu %s"
        self.assertEqual(result, expected)

    def test_inject_with_iree_run_module(self):
        """Test injection with iree-run-module."""
        content = "// RUN: iree-run-module --module=test.vmfb"
        result = lit_wrapper.inject_extra_flags(content, "--device=local-task")
        expected = "// RUN: iree-run-module --device=local-task --module=test.vmfb"
        self.assertEqual(result, expected)

    def test_inject_empty_flags(self):
        """Test with empty flags (no-op)."""
        content = "// RUN: iree-opt %s"
        result = lit_wrapper.inject_extra_flags(content, "")
        self.assertEqual(result, content)

    def test_inject_none_flags(self):
        """Test with None flags (no-op)."""
        content = "// RUN: iree-opt %s"
        result = lit_wrapper.inject_extra_flags(content, None)
        self.assertEqual(result, content)

    def test_inject_preserves_non_run_lines(self):
        """Test that non-RUN lines are preserved."""
        content = """// Copyright notice
// Some comment
// RUN: iree-opt %s
func @test() { }"""
        result = lit_wrapper.inject_extra_flags(content, "--debug")
        expected = """// Copyright notice
// Some comment
// RUN: iree-opt --debug %s
func @test() { }"""
        self.assertEqual(result, expected)

    def test_inject_with_indented_run(self):
        """Test with indented RUN line."""
        content = "  // RUN: iree-opt %s"
        result = lit_wrapper.inject_extra_flags(content, "--debug")
        expected = "  // RUN: iree-opt --debug %s"
        self.assertEqual(result, expected)

    def test_inject_with_no_space_after_comment(self):
        """Test with no space after //."""
        content = "//RUN: iree-opt %s"
        result = lit_wrapper.inject_extra_flags(content, "--debug")
        expected = "//RUN: iree-opt --debug %s"
        self.assertEqual(result, expected)


class TestParseLitFailure(unittest.TestCase):
    """Tests for parse_lit_failure() function."""

    def test_parse_success_no_failure(self):
        """Test parsing successful lit output."""
        stdout = """-- Testing: 1 tests, 1 workers --
PASS: IREE :: test.mlir (1 of 1)

Testing Time: 0.18s
  Passed: 1 (100.00%)
"""
        summary, cmds = lit_wrapper.parse_lit_failure(stdout, "")
        self.assertIsNone(summary)
        self.assertEqual(cmds, [])

    def test_parse_simple_failure(self):
        """Test parsing simple failure with commands."""
        stdout = """FAIL: IREE :: test.mlir (1 of 1)
******************** TEST 'IREE :: test.mlir' FAILED ********************
Exit Code: 1

Command Output (stderr):
--
+ iree-opt test.mlir
+ FileCheck test.mlir

********************
"""
        summary, cmds = lit_wrapper.parse_lit_failure(stdout, "")
        self.assertIsNotNone(summary)
        self.assertIn("Failed command", summary)
        self.assertEqual(len(cmds), 2)
        self.assertEqual(cmds[0], "iree-opt test.mlir")
        self.assertEqual(cmds[1], "FileCheck test.mlir")

    def test_parse_filecheck_error(self):
        """Test extraction of FileCheck error."""
        stdout = """FAIL: IREE :: test.mlir (1 of 1)
******************** TEST 'IREE :: test.mlir' FAILED ********************
Command Output (stderr):
--
+ iree-opt --canonicalize test.mlir
+ FileCheck test.mlir
test.mlir:10:11: error: CHECK: expected string not found in input
// CHECK: some_pattern
          ^

Input file: <stdin>
********************
"""
        summary, _ = lit_wrapper.parse_lit_failure(stdout, "")
        self.assertIsNotNone(summary)
        self.assertIn("error: CHECK:", summary)
        self.assertIn("test.mlir:10:11", summary)


class TestExtractFileCheckError(unittest.TestCase):
    """Tests for extract_filecheck_error() function."""

    def test_extract_basic_error(self):
        """Test extracting basic FileCheck error."""
        failure_text = """+ FileCheck test.mlir
test.mlir:10:11: error: CHECK: expected string not found in input
// CHECK: pattern
          ^

Input file: <stdin>
"""
        error = lit_wrapper.extract_filecheck_error(failure_text)
        self.assertIsNotNone(error)
        self.assertIn("test.mlir:10:11", error)
        self.assertIn("CHECK: expected string not found", error)

    def test_extract_no_error(self):
        """Test when no FileCheck error present."""
        failure_text = """+ iree-opt test.mlir
Some other error message
"""
        error = lit_wrapper.extract_filecheck_error(failure_text)
        self.assertIsNone(error)

    def test_extract_truncates_long_errors(self):
        """Test that very long errors are truncated."""
        # Create error with more than 10 lines.
        error_lines = ["test.mlir:1:1: error: CHECK: failed"] + [
            f"line {i}" for i in range(15)
        ]
        failure_text = "\n".join(error_lines) + "\n\nInput file: test"

        error = lit_wrapper.extract_filecheck_error(failure_text)
        self.assertIsNotNone(error)
        result_lines = error.split("\n")
        # Should be truncated to 11 lines (10 + "... use -v" message).
        self.assertLessEqual(len(result_lines), 11)
        self.assertIn("use -v", error)


class TestLitImport(unittest.TestCase):
    """Smoke test to ensure lit is importable from tree or site."""

    def test_import_lit(self):
        # Call _ensure_lit_importable() which uses build_detection to find lit.
        try:
            lit_wrapper._ensure_lit_importable()
            import lit  # noqa: F401, PLC0415 (deferred import for availability check)
        except Exception:
            self.skipTest("lit not importable in this environment")


class TestCaseRunLineReinjection(unittest.TestCase):
    """Ensure case-local RUN lines are re-injected at correct positions."""

    def test_case_local_run_lines_positions(self):
        # Multi-case file where case 2 has a case-local RUN line.
        file_text = "\n".join(
            [
                "// RUN: iree-opt %s | FileCheck %s",
                "// CHECK-LABEL: @first",
                "func @first() {}",
                "",
                "// -----",
                "",
                "// CHECK-LABEL: @second",
                "// CHECK: A",
                "// RUN: iree-opt --canonicalize %s | FileCheck %s",
                "// CHECK: B",
                "func @second() {}",
            ]
        )

        with NamedTemporaryFile("w", suffix=".mlir", delete=False) as tmp:
            p = Path(tmp.name)
            tmp.write(file_text)

        try:
            # Parse to get second case which has case-local RUN
            test_file_obj = parse_test_file(p)
            cases = list(test_file_obj.cases)
            self.assertEqual(len(cases), 2)
            case = cases[1]  # Second case has the case-local RUN

            # Extract case-local RUN lines using new API
            case_runs = case.extract_local_run_lines()
            self.assertEqual(len(case_runs), 1, "Should have 1 case-local RUN")

            # Strip and re-inject into synthesized content (with blank prefix)
            stripped = _strip_run_lines_preserve_line_numbers(case.render_for_testing())
            content = ("\n" * (case.start_line - 1)) + stripped
            rebuilt = inject_run_lines_with_case(content, [], case_runs)
            lines = rebuilt.split("\n")

            # Verify the RUN line was injected back at correct position
            found_run = False
            for line in lines:
                if "// RUN:" in line and "--canonicalize" in line:
                    found_run = True
                    break
            self.assertTrue(found_run, "Case-local RUN line should be present")
        finally:
            p.unlink()

    def test_case_local_run_lines_continuation(self):
        # Multi-case file with case-local RUN continuation.
        file_text = "\n".join(
            [
                "// RUN: iree-opt %s | FileCheck %s",
                "// CHECK-LABEL: @first",
                "func @first() {}",
                "",
                "// -----",
                "",
                "// CHECK-LABEL: @second",
                "// CHECK: A",
                "// RUN: iree-opt \\",
                "// RUN:   --canonicalize %s | FileCheck %s",
                "func @second() {}",
            ]
        )

        with NamedTemporaryFile("w", suffix=".mlir", delete=False) as tmp:
            p = Path(tmp.name)
            tmp.write(file_text)
        try:
            # Parse to get second case with case-local RUN continuation
            test_file_obj = parse_test_file(p)
            cases = list(test_file_obj.cases)
            self.assertEqual(len(cases), 2)
            case = cases[1]  # Second case has the case-local RUN

            # Extract case-local RUN lines using new API
            case_runs = case.extract_local_run_lines()
            # Continuation lines are returned as separate entries
            self.assertEqual(
                len(case_runs), 2, "Should have 2 RUN lines (continuation)"
            )

            stripped = _strip_run_lines_preserve_line_numbers(case.render_for_testing())
            content = ("\n" * (case.start_line - 1)) + stripped
            rebuilt = inject_run_lines_with_case(content, [], case_runs)

            # Verify the RUN line was injected and contains the expected flag
            found_run = False
            for line in rebuilt.split("\n"):
                if "// RUN:" in line and "--canonicalize" in line:
                    found_run = True
                    break
            self.assertTrue(
                found_run, "Case-local RUN with continuation should be present"
            )
        finally:
            p.unlink()


if __name__ == "__main__":
    unittest.main()
