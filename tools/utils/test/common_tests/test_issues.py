# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for Issue type hierarchy.

Tests cover:
- Base Issue class instantiation
- All specific Issue subclasses
- Field validation and defaults
- Severity enum
"""

import unittest

from common.issues import (
    DeprecatedAPIIssue,
    Issue,
    LITTestIssue,
    MissingDependencyIssue,
    PythonTestIssue,
    ROCmInfrastructureIssue,
    SanitizerIssue,
    Severity,
)


class TestSeverity(unittest.TestCase):
    """Test Severity enum."""

    def test_severity_values(self):
        """Test all severity levels exist with correct ordering values."""
        # Severity uses numeric values for sorting (higher = more severe).
        self.assertEqual(Severity.CRITICAL.value, 4)
        self.assertEqual(Severity.HIGH.value, 3)
        self.assertEqual(Severity.MEDIUM.value, 2)
        self.assertEqual(Severity.LOW.value, 1)

    def test_severity_ordering(self):
        """Test severity levels can be sorted by value."""
        severities = [Severity.LOW, Severity.HIGH, Severity.CRITICAL, Severity.MEDIUM]
        sorted_severities = sorted(severities, key=lambda s: s.value, reverse=True)
        expected = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]
        self.assertEqual(sorted_severities, expected)


class TestBaseIssue(unittest.TestCase):
    """Test base Issue class."""

    def test_minimal_issue(self):
        """Test Issue with minimal required fields."""
        issue = Issue(
            severity=Severity.HIGH,
            actionable=True,
            message="Test issue",
        )

        self.assertEqual(issue.severity, Severity.HIGH)
        self.assertTrue(issue.actionable)
        self.assertEqual(issue.message, "Test issue")
        self.assertEqual(issue.context_lines, [])
        self.assertIsNone(issue.line_number)
        self.assertEqual(issue.source_extractor, "Unknown")

    def test_issue_with_all_fields(self):
        """Test Issue with all fields populated."""
        context = ["line 1", "line 2", ">>> error line", "line 4"]
        issue = Issue(
            severity=Severity.CRITICAL,
            actionable=False,
            message="Infrastructure flake",
            context_lines=context,
            line_number=42,
            source_extractor="TestExtractor",
        )

        self.assertEqual(issue.severity, Severity.CRITICAL)
        self.assertFalse(issue.actionable)
        self.assertEqual(issue.message, "Infrastructure flake")
        self.assertEqual(issue.context_lines, context)
        self.assertEqual(issue.line_number, 42)
        self.assertEqual(issue.source_extractor, "TestExtractor")


class TestSanitizerIssue(unittest.TestCase):
    """Test SanitizerIssue class."""

    def test_tsan_data_race(self):
        """Test TSAN data race issue."""
        issue = SanitizerIssue(
            severity=Severity.CRITICAL,
            actionable=True,
            message="Data race in Device::Submit",
            sanitizer_type="TSAN",
            error_type="data-race",
            primary_stack=[
                "#0 iree::hal::Device::Submit() device.cc:123",
                "#1 main() main.cc:42",
            ],
            thread_id="T1",
            address="0x7b0400001234",
        )

        self.assertEqual(issue.sanitizer_type, "TSAN")
        self.assertEqual(issue.error_type, "data-race")
        self.assertEqual(len(issue.primary_stack), 2)
        self.assertEqual(issue.thread_id, "T1")
        self.assertEqual(issue.address, "0x7b0400001234")
        self.assertEqual(len(issue.allocation_stack), 0)

    def test_asan_use_after_free(self):
        """Test ASAN use-after-free issue."""
        issue = SanitizerIssue(
            severity=Severity.CRITICAL,
            actionable=True,
            message="Heap use after free",
            sanitizer_type="ASAN",
            error_type="heap-use-after-free",
            primary_stack=["#0 iree::hal::Buffer::Map() buffer.cc:456"],
            allocation_stack=["Allocated at buffer.cc:123"],
            address="0x12345678",
        )

        self.assertEqual(issue.sanitizer_type, "ASAN")
        self.assertEqual(issue.error_type, "heap-use-after-free")
        self.assertEqual(len(issue.allocation_stack), 1)
        self.assertEqual(issue.allocation_stack[0], "Allocated at buffer.cc:123")


class TestLITTestIssue(unittest.TestCase):
    """Test LITTestIssue class."""

    def test_filecheck_mismatch(self):
        """Test FileCheck pattern mismatch."""
        issue = LITTestIssue(
            severity=Severity.HIGH,
            actionable=True,
            message="CHECK-SAME pattern mismatch",
            test_file="test.mlir",
            test_line=42,
            check_type="CHECK-SAME",
            check_pattern="expected pattern",
            expected="expected pattern",
            actual="actual output",
            run_command="iree-opt --pass-pipeline='...'",
        )

        self.assertEqual(issue.test_file, "test.mlir")
        self.assertEqual(issue.test_line, 42)
        self.assertEqual(issue.check_type, "CHECK-SAME")
        self.assertIsNotNone(issue.expected)
        self.assertIsNotNone(issue.actual)
        self.assertIsNotNone(issue.run_command)

    def test_minimal_lit_issue(self):
        """Test LIT issue with minimal fields."""
        issue = LITTestIssue(
            severity=Severity.HIGH,
            actionable=True,
            message="Test failed",
            test_file="test.mlir",
            test_line=10,
            check_type="CHECK",
        )

        self.assertEqual(issue.test_file, "test.mlir")
        self.assertEqual(issue.test_line, 10)
        self.assertIsNone(issue.check_pattern)
        self.assertIsNone(issue.expected)


class TestMissingDependencyIssue(unittest.TestCase):
    """Test MissingDependencyIssue class."""

    def test_bazel_missing_header(self):
        """Test Bazel missing header dependency."""
        issue = MissingDependencyIssue(
            severity=Severity.HIGH,
            actionable=True,
            message="Missing header mlir/IR/Builders.h",
            missing_header="mlir/IR/Builders.h",
            target="//compiler/src/iree:Foo",
            suggested_dep="MLIRIR",
            fix_suggestion="Add MLIRIR to BUILD.bazel deps",
        )

        self.assertEqual(issue.missing_header, "mlir/IR/Builders.h")
        self.assertEqual(issue.target, "//compiler/src/iree:Foo")
        self.assertEqual(issue.suggested_dep, "MLIRIR")
        self.assertIsNotNone(issue.fix_suggestion)


class TestDeprecatedAPIIssue(unittest.TestCase):
    """Test DeprecatedAPIIssue class."""

    def test_deprecated_api_usage(self):
        """Test deprecated API usage."""
        issue = DeprecatedAPIIssue(
            severity=Severity.MEDIUM,
            actionable=True,
            message="'oldFunction' is deprecated",
            deprecated_symbol="oldFunction",
            replacement="newFunction",
            file_path="src/foo.cpp",
            line=123,
            column=10,
        )

        self.assertEqual(issue.deprecated_symbol, "oldFunction")
        self.assertEqual(issue.replacement, "newFunction")
        self.assertEqual(issue.file_path, "src/foo.cpp")
        self.assertEqual(issue.line, 123)
        self.assertEqual(issue.column, 10)


class TestPythonTestIssue(unittest.TestCase):
    """Test PythonTestIssue class."""

    def test_pytest_assertion_error(self):
        """Test pytest assertion error."""
        issue = PythonTestIssue(
            severity=Severity.HIGH,
            actionable=True,
            message="Assertion failed in test_foo",
            test_name="test_foo",
            exception_type="AssertionError",
            exception_message="assert 1 == 2",
            stack_trace=[
                "test_foo.py:42: in test_foo",
                "    assert 1 == 2",
            ],
            assertion_details="Expected 1, got 2",
        )

        self.assertEqual(issue.test_name, "test_foo")
        self.assertEqual(issue.exception_type, "AssertionError")
        self.assertEqual(len(issue.stack_trace), 2)
        self.assertIsNotNone(issue.assertion_details)

    def test_python_exception(self):
        """Test general Python exception."""
        issue = PythonTestIssue(
            severity=Severity.HIGH,
            actionable=True,
            message="ValueError in test_bar",
            test_name="test_bar",
            exception_type="ValueError",
            exception_message="invalid literal for int()",
            stack_trace=["test_bar.py:10: in test_bar"],
        )

        self.assertEqual(issue.exception_type, "ValueError")
        self.assertIsNone(issue.assertion_details)


class TestROCmInfrastructureIssue(unittest.TestCase):
    """Test ROCmInfrastructureIssue class."""

    def test_cleanup_crash(self):
        """Test ROCm cleanup crash."""
        issue = ROCmInfrastructureIssue(
            severity=Severity.MEDIUM,
            actionable=False,
            message="ROCm cleanup crash after tests passed",
            error_pattern="cleanup_crash",
            test_passed=True,
        )

        self.assertEqual(issue.error_pattern, "cleanup_crash")
        self.assertTrue(issue.test_passed)
        self.assertFalse(issue.actionable)

    def test_device_lost(self):
        """Test ROCm device lost error."""
        issue = ROCmInfrastructureIssue(
            severity=Severity.HIGH,
            actionable=False,
            message="GPU device lost during test",
            error_pattern="device_lost",
            test_passed=False,
        )

        self.assertEqual(issue.error_pattern, "device_lost")
        self.assertFalse(issue.test_passed)


if __name__ == "__main__":
    unittest.main()
