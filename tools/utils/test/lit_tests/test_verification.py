# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for lit_tools.core.verification module."""

import argparse
import unittest
from unittest.mock import patch

from lit_tools.core import verification


class TestVerifyContentWithSkipCheck(unittest.TestCase):
    """Tests for verify_content_with_skip_check helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.args = argparse.Namespace(
            verify=True,
            quiet=False,
            json=False,
            verify_timeout=5,
        )

    def test_verification_disabled_globally(self):
        """Test that verification is skipped when --verify not set."""
        self.args.verify = False
        content = "func.func @test() { return }"

        skipped, valid, error = verification.verify_content_with_skip_check(
            content, case_number=1, case_name="test", args=self.args
        )

        self.assertFalse(skipped)  # Not skipped due to expected-error
        self.assertTrue(valid)  # Considered valid when disabled
        self.assertEqual(error, "")

    def test_skip_due_to_expected_error(self):
        """Test that content with expected-error skips verification."""
        content = """
        func.func @test() {
            // expected-error @+1 {{test error}}
            %bad = arith.invalid
            return
        }
        """

        with patch("common.console") as mock_console:
            skipped, valid, error = verification.verify_content_with_skip_check(
                content, case_number=2, case_name="test_func", args=self.args
            )

        self.assertTrue(skipped)
        self.assertTrue(valid)
        self.assertEqual(error, "")
        # Should have printed skip note
        mock_console.note.assert_called_once()
        call_args = mock_console.note.call_args[0]
        self.assertIn("Skipping verification for case 2", call_args[0])
        self.assertIn("expected-error", call_args[0])

    def test_skip_quiet_mode(self):
        """Test that skip note is suppressed in quiet mode."""
        self.args.quiet = True
        content = "// expected-error @+1 {{error}}\n%bad = invalid"

        with patch("common.console") as mock_console:
            skipped, valid, _ = verification.verify_content_with_skip_check(
                content, case_number=3, case_name=None, args=self.args
            )

        self.assertTrue(skipped)
        self.assertTrue(valid)
        # Should NOT have printed anything in quiet mode
        mock_console.note.assert_not_called()

    def test_verification_success(self):
        """Test successful verification of valid IR."""
        content = "func.func @test() { return }"

        # Mock verify_ir to return success
        with patch("lit_tools.core.verification.verify_ir") as mock_verify:
            mock_verify.return_value = (True, "")

            skipped, valid, error = verification.verify_content_with_skip_check(
                content, case_number=4, case_name="test", args=self.args
            )

        self.assertFalse(skipped)
        self.assertTrue(valid)
        self.assertEqual(error, "")

        # Verify that verify_ir was called with correct case_info
        mock_verify.assert_called_once()
        call_args = mock_verify.call_args[0]
        self.assertEqual(call_args[0], content)
        self.assertEqual(call_args[1], "Case 4 (@test)")

    def test_verification_failure(self):
        """Test verification failure with error message."""
        content = "func.func @test() { %invalid syntax }"
        error_msg = "Verification failed for Case 5: syntax error"

        # Mock verify_ir to return failure
        with patch("lit_tools.core.verification.verify_ir") as mock_verify:
            mock_verify.return_value = (False, error_msg)

            skipped, valid, error = verification.verify_content_with_skip_check(
                content, case_number=5, case_name=None, args=self.args
            )

        self.assertFalse(skipped)
        self.assertFalse(valid)
        self.assertEqual(error, error_msg)

        # Verify case_info without name
        call_args = mock_verify.call_args[0]
        self.assertEqual(call_args[1], "Case 5")

    def test_custom_timeout(self):
        """Test that custom timeout is passed through."""
        content = "func.func @test() { return }"

        with patch("lit_tools.core.verification.verify_ir") as mock_verify:
            mock_verify.return_value = (True, "")

            verification.verify_content_with_skip_check(
                content,
                case_number=6,
                case_name="test",
                args=self.args,
                timeout=10,
            )

        # Verify timeout parameter
        call_kwargs = mock_verify.call_args[1]
        self.assertEqual(call_kwargs["timeout"], 10)

    def test_default_timeout_from_args(self):
        """Test that default timeout comes from args.verify_timeout."""
        self.args.verify_timeout = 15
        content = "func.func @test() { return }"

        with patch("lit_tools.core.verification.verify_ir") as mock_verify:
            mock_verify.return_value = (True, "")

            verification.verify_content_with_skip_check(
                content,
                case_number=7,
                case_name="test",
                args=self.args,
            )

        # Verify default timeout from args
        call_kwargs = mock_verify.call_args[1]
        self.assertEqual(call_kwargs["timeout"], 15)

    def test_expected_warning_also_skips(self):
        """Test that expected-warning also triggers skip."""
        content = "// expected-warning @+1 {{deprecated}}\nfunc.func @old() { return }"

        with patch("common.console"):
            skipped, valid, _ = verification.verify_content_with_skip_check(
                content, case_number=8, case_name="old", args=self.args
            )

        self.assertTrue(skipped)
        self.assertTrue(valid)

    def test_expected_note_also_skips(self):
        """Test that expected-note also triggers skip."""
        content = "// expected-note @+1 {{info}}\nfunc.func @info() { return }"

        with patch("common.console"):
            skipped, valid, _ = verification.verify_content_with_skip_check(
                content, case_number=9, case_name="info", args=self.args
            )

        self.assertTrue(skipped)
        self.assertTrue(valid)


if __name__ == "__main__":
    unittest.main()
