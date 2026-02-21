# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for PytestErrorExtractor."""

import unittest

from common.extractors.pytest_error import PytestErrorExtractor
from common.issues import Severity
from common.log_buffer import LogBuffer


class TestPytestErrorExtractor(unittest.TestCase):
    """Test PytestErrorExtractor for Python test failures."""

    def setUp(self):
        """Set up extractor for tests."""
        self.extractor = PytestErrorExtractor()

    def test_qualified_exception_name(self):
        """Test pytest FAILED with qualified exception name (e.g., onnx_models.utils.IreeRunException)."""
        log = """
pytest session starts
FAILED iree-test-suites/onnx_models/tests/test.py::test_models[alexnet.onnx] - onnx_models.utils.IreeRunException: 'alexnet.vmfb' run failed
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.severity, Severity.HIGH)
        self.assertTrue(issue.actionable)
        self.assertEqual(issue.test_name, "test_models[alexnet.onnx]")
        self.assertEqual(issue.exception_type, "onnx_models.utils.IreeRunException")
        self.assertIn("alexnet.vmfb", issue.exception_message)

    def test_any_exception_type(self):
        """Test that we catch ANY exception type, not just standard Error/Exception suffixes."""
        log = """
pytest session starts
FAILED tests/test_custom.py::test_foo - CustomTestFailure: something went wrong
FAILED tests/test_timeout.py::test_bar - TestTimeout: test took too long
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 2)

        # Verify we caught CustomTestFailure (doesn't end in Error or Exception).
        self.assertEqual(issues[0].exception_type, "CustomTestFailure")
        self.assertIn("something went wrong", issues[0].exception_message)

        # Verify we caught TestTimeout (doesn't end in Error or Exception).
        self.assertEqual(issues[1].exception_type, "TestTimeout")
        self.assertIn("test took too long", issues[1].exception_message)


if __name__ == "__main__":
    unittest.main()
