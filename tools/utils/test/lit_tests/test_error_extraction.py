# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for error extraction functions in lit_wrapper module."""

# Add project tools/utils to path for imports
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from lit_tools.core import lit_wrapper


class TestExtractTimeoutError(unittest.TestCase):
    """Tests for extract_timeout_error function."""

    def test_timeout_uppercase(self):
        """Test detecting TIMEOUT."""
        failure_text = "Command execution TIMEOUT after 60 seconds"
        error = lit_wrapper.extract_timeout_error(failure_text)
        self.assertIsNotNone(error)
        self.assertEqual(error, "Test exceeded timeout limit")

    def test_timeout_mixed_case(self):
        """Test detecting Timeout."""
        failure_text = "Test failed: Timeout reached"
        error = lit_wrapper.extract_timeout_error(failure_text)
        self.assertIsNotNone(error)
        self.assertEqual(error, "Test exceeded timeout limit")

    def test_reached_timeout(self):
        """Test detecting 'Reached timeout'."""
        failure_text = "Reached timeout of 120 seconds"
        error = lit_wrapper.extract_timeout_error(failure_text)
        self.assertIsNotNone(error)
        self.assertEqual(error, "Test exceeded timeout limit")

    def test_no_timeout(self):
        """Test no timeout in normal failure."""
        failure_text = "FileCheck error: expected string not found"
        error = lit_wrapper.extract_timeout_error(failure_text)
        self.assertIsNone(error)


class TestExtractCrashError(unittest.TestCase):
    """Tests for extract_crash_error function."""

    def test_segfault_signal_11(self):
        """Test detecting SIGSEGV (signal 11)."""
        failure_text = "Command terminated with signal 11"
        error = lit_wrapper.extract_crash_error(failure_text)
        self.assertIsNotNone(error)
        self.assertIn("SIGSEGV", error)
        self.assertIn("segmentation fault", error)

    def test_abort_signal_6(self):
        """Test detecting SIGABRT (signal 6)."""
        failure_text = "Process killed by signal 6"
        error = lit_wrapper.extract_crash_error(failure_text)
        self.assertIsNotNone(error)
        self.assertIn("SIGABRT", error)
        self.assertIn("abort", error)

    def test_illegal_instruction_signal_4(self):
        """Test detecting SIGILL (signal 4)."""
        failure_text = "Exited with signal 4"
        error = lit_wrapper.extract_crash_error(failure_text)
        self.assertIsNotNone(error)
        self.assertIn("SIGILL", error)
        self.assertIn("illegal instruction", error)

    def test_exit_code_139(self):
        """Test detecting segfault from exit code 139 (128 + 11)."""
        failure_text = "Command returned exit code 139"
        error = lit_wrapper.extract_crash_error(failure_text)
        self.assertIsNotNone(error)
        self.assertIn("SIGSEGV", error)
        self.assertIn("exit code 139", error)

    def test_exit_code_134(self):
        """Test detecting abort from exit code 134 (128 + 6)."""
        failure_text = "Command returned exit code 134"
        error = lit_wrapper.extract_crash_error(failure_text)
        self.assertIsNotNone(error)
        self.assertIn("SIGABRT", error)
        self.assertIn("exit code 134", error)

    def test_segfault_text(self):
        """Test detecting 'Segmentation fault' text."""
        failure_text = "Process terminated with: Segmentation fault (core dumped)"
        error = lit_wrapper.extract_crash_error(failure_text)
        self.assertIsNotNone(error)
        self.assertIn("segmentation fault", error.lower())

    def test_no_crash(self):
        """Test no crash in normal failure."""
        failure_text = "FileCheck error: expected string not found"
        error = lit_wrapper.extract_crash_error(failure_text)
        self.assertIsNone(error)

    def test_normal_exit_code(self):
        """Test normal exit code is not detected as crash."""
        failure_text = "Command returned exit code 1"
        error = lit_wrapper.extract_crash_error(failure_text)
        self.assertIsNone(error)


class TestExtractAssertionError(unittest.TestCase):
    """Tests for extract_assertion_error function."""

    def test_c_assertion_failure(self):
        """Test detecting C assertion failure."""
        failure_text = """
test.cpp:42: main: Assertion `x > 0' failed.
Aborted (core dumped)
"""
        error = lit_wrapper.extract_assertion_error(failure_text)
        self.assertIsNotNone(error)
        self.assertIn("Assertion failure", error)
        self.assertIn("Assertion `x > 0' failed", error)

    def test_check_failure(self):
        """Test detecting CHECK failure (glog style)."""
        failure_text = """
F0115 12:34:56.789012 12345 test.cc:42] Check failed: x > 0 (0 vs. 0)
Value was: 0
Expected: > 0
"""
        error = lit_wrapper.extract_assertion_error(failure_text)
        self.assertIsNotNone(error)
        self.assertIn("CHECK failure", error)
        self.assertIn("Check failed:", error)

    def test_no_assertion(self):
        """Test no assertion in normal failure."""
        failure_text = "FileCheck error: expected string not found"
        error = lit_wrapper.extract_assertion_error(failure_text)
        self.assertIsNone(error)


class TestExtractInvalidIRError(unittest.TestCase):
    """Tests for extract_invalid_ir_error function."""

    def test_verification_failed(self):
        """Test detecting MLIR verification failure."""
        failure_text = """
test.mlir:15:5: error: 'func.func' op verification failed
  %0 = arith.addi %arg0, %arg1 : tensor<10xf32>
       ^ note: expected integer type
"""
        error = lit_wrapper.extract_invalid_ir_error(failure_text)
        self.assertIsNotNone(error)
        self.assertIn("IR verification failed", error)
        self.assertIn("verification failed", error)

    def test_operation_error(self):
        """Test detecting operation validation error."""
        failure_text = """
test.mlir:20:10: error: 'tensor.empty' op result #0 must be shaped of any type values
  %1 = tensor.empty() : tensor<*xf32>
       ^
"""
        error = lit_wrapper.extract_invalid_ir_error(failure_text)
        self.assertIsNotNone(error)
        self.assertIn("IR validation error", error)
        self.assertIn("'tensor.empty' op", error)

    def test_no_ir_error(self):
        """Test no IR error in normal failure."""
        failure_text = "FileCheck error: expected string not found"
        error = lit_wrapper.extract_invalid_ir_error(failure_text)
        self.assertIsNone(error)

    def test_truncation_for_long_error(self):
        """Test that very long IR errors are truncated."""
        # Create a verification error with many lines.
        failure_text = "error: verification failed\n" + "\n".join(
            [f"Line {i} of error details" for i in range(20)]
        )
        error = lit_wrapper.extract_invalid_ir_error(failure_text)
        self.assertIsNotNone(error)
        # Should be truncated to 15 lines + ellipsis message.
        lines = error.splitlines()
        self.assertLessEqual(
            len(lines), 17
        )  # "IR verification failed:" + 15 + ellipsis
        self.assertIn("use -v for full output", error)


class TestExtractMissingFileError(unittest.TestCase):
    """Tests for extract_missing_file_error function."""

    def test_no_such_file_or_directory(self):
        """Test detecting 'No such file or directory'."""
        failure_text = "/usr/bin/iree-opt: error: input.mlir: No such file or directory"
        error = lit_wrapper.extract_missing_file_error(failure_text)
        self.assertIsNotNone(error)
        self.assertIn("File not found", error)
        self.assertIn("No such file or directory", error)

    def test_unable_to_open_file(self):
        """Test detecting 'unable to open file'."""
        failure_text = "Unable to open file: 'test_data.bin'"
        error = lit_wrapper.extract_missing_file_error(failure_text)
        self.assertIsNotNone(error)
        self.assertIn("Unable to open file", error)
        self.assertIn("test_data.bin", error)

    def test_could_not_open_file(self):
        """Test detecting 'Could not open file'."""
        failure_text = 'Error: Could not open: "/path/to/missing.mlir"'
        error = lit_wrapper.extract_missing_file_error(failure_text)
        self.assertIsNotNone(error)
        self.assertIn("Could not open file", error)
        self.assertIn("/path/to/missing.mlir", error)

    def test_no_file_error(self):
        """Test no file error in normal failure."""
        failure_text = "FileCheck error: expected string not found"
        error = lit_wrapper.extract_missing_file_error(failure_text)
        self.assertIsNone(error)


class TestParseRealLitOutput(unittest.TestCase):
    """Tests for parse_lit_failure with realistic lit output."""

    def test_filecheck_failure(self):
        """Test parsing FileCheck failure."""
        stdout = """
*********************** TEST 'test.mlir' FAILED ***********************
+ iree-opt --split-input-file test.mlir | FileCheck test.mlir
test.mlir:15:11: error: CHECK: expected string not found in input
// CHECK: %[[VAL:.+]] = arith.constant
          ^
<stdin>:10:1: note: scanning from here
func.func @test() {
^
*********************** TEST 'test.mlir' FAILED ***********************
"""
        summary, commands = lit_wrapper.parse_lit_failure(stdout, "")
        self.assertIsNotNone(summary)
        self.assertIn("Failed command:", summary)
        self.assertIn("iree-opt", summary)
        self.assertIn("CHECK: expected string not found", summary)
        self.assertEqual(len(commands), 1)
        self.assertIn("iree-opt", commands[0])

    def test_crash_failure(self):
        """Test parsing crash failure."""
        stdout = """
*********************** TEST 'test.mlir' FAILED ***********************
+ iree-opt --some-pass test.mlir
Command terminated with signal 11
*********************** TEST 'test.mlir' FAILED ***********************
"""
        summary, _ = lit_wrapper.parse_lit_failure(stdout, "")
        self.assertIsNotNone(summary)
        self.assertIn("Failed command:", summary)
        self.assertIn("SIGSEGV", summary)
        self.assertIn("segmentation fault", summary)

    def test_timeout_failure(self):
        """Test parsing timeout failure."""
        stdout = """
*********************** TEST 'test.mlir' FAILED ***********************
+ iree-compile test.mlir --iree-hal-target-backends=vulkan-spirv
TIMEOUT: test exceeded maximum time limit of 60 seconds
*********************** TEST 'test.mlir' FAILED ***********************
"""
        summary, _ = lit_wrapper.parse_lit_failure(stdout, "")
        self.assertIsNotNone(summary)
        self.assertIn("timeout limit", summary.lower())

    def test_assertion_failure(self):
        """Test parsing assertion failure."""
        stdout = """
*********************** TEST 'test.mlir' FAILED ***********************
+ iree-opt --some-pass test.mlir
iree-opt: SomePass.cpp:123: void runPass(): Assertion `result != nullptr' failed.
Aborted (core dumped)
*********************** TEST 'test.mlir' FAILED ***********************
"""
        summary, _ = lit_wrapper.parse_lit_failure(stdout, "")
        self.assertIsNotNone(summary)
        self.assertIn("Assertion", summary)
        self.assertIn("result != nullptr", summary)

    def test_ir_verification_failure(self):
        """Test parsing IR verification failure."""
        stdout = """
*********************** TEST 'test.mlir' FAILED ***********************
+ iree-opt --verify-each test.mlir
test.mlir:10:5: error: 'func.func' op verification failed
  func.func @test(%arg0: tensor<10xf32>) {
  ^
test.mlir:11:10: note: see current operation: "func.return"() : () -> ()
*********************** TEST 'test.mlir' FAILED ***********************
"""
        summary, _ = lit_wrapper.parse_lit_failure(stdout, "")
        self.assertIsNotNone(summary)
        self.assertIn("IR verification failed", summary)
        self.assertIn("verification failed", summary)

    def test_missing_file_failure(self):
        """Test parsing missing file failure."""
        stdout = """
*********************** TEST 'test.mlir' FAILED ***********************
+ iree-opt nonexistent.mlir
error: nonexistent.mlir: No such file or directory
*********************** TEST 'test.mlir' FAILED ***********************
"""
        summary, _ = lit_wrapper.parse_lit_failure(stdout, "")
        self.assertIsNotNone(summary)
        self.assertIn("File not found", summary)
        self.assertIn("No such file or directory", summary)


if __name__ == "__main__":
    unittest.main()
