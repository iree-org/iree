# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for Extractor base class.

Tests cover:
- ABC enforcement (can't instantiate without implementing methods)
- extract() contract
- Basic mock extractor implementation
"""

import unittest

from common.extractors.base import Extractor
from common.issues import Issue, Severity
from common.log_buffer import LogBuffer


class MockExtractor(Extractor):
    """Mock extractor for testing."""

    name = "mock"

    def __init__(self) -> None:
        """Initialize mock extractor."""
        self.extract_called = False

    def extract(self, log_buffer: LogBuffer) -> list[Issue]:
        """Mock extract implementation."""
        self.extract_called = True

        if "ERROR" in log_buffer.content:
            return [
                Issue(
                    severity=Severity.HIGH,
                    actionable=True,
                    message="Mock error found",
                    source_extractor=self.name,
                )
            ]
        return []


class TestExtractorABC(unittest.TestCase):
    """Test Extractor ABC enforcement."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that Extractor cannot be instantiated directly."""
        with self.assertRaises(TypeError) as context:
            Extractor()

        self.assertIn("abstract", str(context.exception).lower())

    def test_must_implement_extract(self):
        """Test that subclasses must implement extract()."""

        class IncompleteExtractor(Extractor):
            name = "incomplete"

        with self.assertRaises(TypeError) as context:
            IncompleteExtractor()

        self.assertIn("abstract", str(context.exception).lower())


class TestMockExtractor(unittest.TestCase):
    """Test MockExtractor implementation."""

    def test_name_attribute(self):
        """Test extractor has name attribute."""
        extractor = MockExtractor()
        self.assertEqual(extractor.name, "mock")

    def test_extract_with_error(self):
        """Test extract finds issues when log contains ERROR."""
        extractor = MockExtractor()
        log_buffer = LogBuffer(
            "Some log content\nERROR: Something failed\nMore content"
        )

        issues = extractor.extract(log_buffer)

        self.assertTrue(extractor.extract_called)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].message, "Mock error found")
        self.assertEqual(issues[0].severity, Severity.HIGH)
        self.assertTrue(issues[0].actionable)
        self.assertEqual(issues[0].source_extractor, "mock")

    def test_extract_without_error(self):
        """Test extract returns empty list when no issues found."""
        extractor = MockExtractor()
        log_buffer = LogBuffer("Clean log with no errors")

        issues = extractor.extract(log_buffer)

        self.assertTrue(extractor.extract_called)
        self.assertEqual(len(issues), 0)


if __name__ == "__main__":
    unittest.main()
