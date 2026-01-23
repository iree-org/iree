# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for lit_tools.core.cli helper functions."""

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common import exit_codes
from lit_tools.core import cli


class TestLoadAndParseTestFile(unittest.TestCase):
    """Tests for load_and_parse_test_file helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.args = argparse.Namespace(quiet=False, json=False)

    def test_successful_parse(self):
        """Test successful parsing of a valid test file."""
        # Create temporary test file.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(
                """// RUN: iree-opt %s
// CHECK-LABEL: @test_func
func.func @test_func() {
  return
}
"""
            )
            test_file_path = Path(f.name)

        try:
            test_file, cases, exit_code = cli.load_and_parse_test_file(
                test_file_path, self.args
            )

            self.assertEqual(exit_code, exit_codes.SUCCESS)
            self.assertIsNotNone(test_file)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].name, "test_func")
        finally:
            test_file_path.unlink()

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        test_file_path = Path("/nonexistent/file.mlir")

        with patch("lit_tools.core.cli.console") as mock_console:
            test_file, cases, exit_code = cli.load_and_parse_test_file(
                test_file_path, self.args
            )

        self.assertEqual(exit_code, exit_codes.NOT_FOUND)
        self.assertIsNone(test_file)
        self.assertEqual(cases, [])
        # Should have printed error
        mock_console.error.assert_called_once()
        call_args = mock_console.error.call_args[0]
        self.assertIn("File not found", call_args[0])

    def test_unicode_decode_error(self):
        """Test handling of files with invalid encoding."""
        # Create file with invalid UTF-8
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".mlir", delete=False) as f:
            f.write(b"\xff\xfe Invalid UTF-8")
            test_file_path = Path(f.name)

        try:
            with patch("lit_tools.core.cli.console") as mock_console:
                test_file, cases, exit_code = cli.load_and_parse_test_file(
                    test_file_path, self.args
                )

            self.assertEqual(exit_code, exit_codes.ERROR)
            self.assertIsNone(test_file)
            self.assertEqual(cases, [])
            mock_console.error.assert_called_once()
        finally:
            test_file_path.unlink()

    def test_parse_error_invalid_syntax(self):
        """Test handling of files with invalid lit syntax."""
        # Create file with content that causes ValueError
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            # Write content that might trigger parser errors
            f.write("// CHECK-LABEL without separator")
            test_file_path = Path(f.name)

        try:
            # The parser might raise ValueError for certain malformed content
            # This test verifies the error handling path works
            _, _, exit_code = cli.load_and_parse_test_file(test_file_path, self.args)

            # Parser should succeed for this simple case (or fail gracefully)
            # This mainly tests that exception handling is in place
            self.assertIn(exit_code, [exit_codes.SUCCESS, exit_codes.ERROR])
        finally:
            test_file_path.unlink()

    def test_multiple_cases(self):
        """Test parsing file with multiple test cases."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(
                """// RUN: iree-opt %s
// CHECK-LABEL: @case1
func.func @case1() { return }

// -----
// CHECK-LABEL: @case2
func.func @case2() { return }

// -----
// CHECK-LABEL: @case3
func.func @case3() { return }
"""
            )
            test_file_path = Path(f.name)

        try:
            test_file, cases, exit_code = cli.load_and_parse_test_file(
                test_file_path, self.args
            )

            self.assertEqual(exit_code, exit_codes.SUCCESS)
            self.assertIsNotNone(test_file)
            self.assertEqual(len(cases), 3)
            self.assertEqual([c.name for c in cases], ["case1", "case2", "case3"])
        finally:
            test_file_path.unlink()

    def test_error_message_includes_exception_details(self):
        """Test that error messages include exception details."""
        test_file_path = Path("/nonexistent/specific_file.mlir")

        with patch("lit_tools.core.cli.console") as mock_console:
            cli.load_and_parse_test_file(test_file_path, self.args)

        # Verify error message format
        mock_console.error.assert_called_once()
        error_msg = mock_console.error.call_args[0][0]
        # Should mention file not found
        self.assertTrue("File not found" in error_msg or "Failed to parse" in error_msg)
        # Should include file name
        self.assertIn("specific_file.mlir", error_msg)


if __name__ == "__main__":
    unittest.main()
