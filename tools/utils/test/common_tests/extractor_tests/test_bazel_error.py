# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for BazelErrorExtractor."""

import unittest

from common.extractors.bazel_error import BazelErrorExtractor
from common.issues import Severity
from common.log_buffer import LogBuffer


class TestBazelErrorExtractor(unittest.TestCase):
    """Test BazelErrorExtractor for Bazel build failures."""

    def setUp(self):
        """Set up extractor for tests."""
        self.extractor = BazelErrorExtractor()

    def test_missing_target_error(self):
        """Test Bazel missing target error extraction."""
        log = """
INFO: Analyzed target //compiler/bindings/c:loader_test (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
ERROR: /github/home/.cache/bazel/_bazel_runner/install/d5f43c91aa9bf04c810a8cdf7/external/llvm-project/mlir/BUILD.bazel:1234:56: no such target '@@llvm-project//mlir:OpAsmInterfaceTdFiles': target 'OpAsmInterfaceTdFiles' not declared in package 'mlir'; did you mean 'OpBaseTdFiles'?
ERROR: /home/runner/work/iree/iree/compiler/bindings/c/BUILD.bazel:123:45 declared in package 'compiler/bindings/c'
Target //compiler/bindings/c:loader_test failed to build
Use --verbose_failures to see the command lines of failed build steps.
ERROR: Build did NOT complete successfully
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.severity, Severity.CRITICAL)
        self.assertTrue(issue.actionable)
        self.assertEqual(issue.error_type, "missing_target")
        self.assertIn("OpAsmInterfaceTdFiles", issue.message)
        self.assertEqual(issue.workspace, "llvm-project")
        self.assertEqual(issue.package, "mlir")
        self.assertEqual(issue.target, "OpAsmInterfaceTdFiles")
        self.assertIn("BUILD.bazel", issue.bazel_file)
        self.assertEqual(issue.bazel_line, 1234)

    def test_missing_target_without_line_number(self):
        """Test missing target error without line/column numbers."""
        log = """
Loading: 0 packages loaded
ERROR: /external/llvm-project/utils/bazel/llvm-project-overlay/mlir/BUILD.bazel: no such target '@@llvm-project//mlir:IR': target 'IR' not declared in package 'mlir'
ERROR: Build did NOT complete successfully
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.severity, Severity.CRITICAL)
        self.assertEqual(issue.error_type, "missing_target")
        self.assertEqual(issue.workspace, "llvm-project")
        self.assertEqual(issue.target, "IR")
        self.assertEqual(issue.bazel_line, 0)  # No line number in error.

    def test_build_failed_with_summary(self):
        """Test Bazel build failure with failed target summary."""
        log = """
INFO: Elapsed time: 1234.567s, Critical Path: 567.89s
INFO: 12345 processes: 10000 internal, 2345 linux-sandbox.
ERROR: Build did NOT complete successfully

//compiler/bindings/c:loader_test                               FAILED TO BUILD
//compiler/plugins/input/StableHLO/Conversion/Preprocessing/test:canonicalization.mlir.test FAILED TO BUILD

INFO: Build completed, 2 total actions
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.severity, Severity.CRITICAL)
        self.assertTrue(issue.actionable)
        self.assertEqual(issue.error_type, "build_failed")
        self.assertEqual(len(issue.failed_targets), 2)
        self.assertIn("//compiler/bindings/c:loader_test", issue.failed_targets)
        self.assertIn(
            "//compiler/plugins/input/StableHLO/Conversion/Preprocessing/test:canonicalization.mlir.test",
            issue.failed_targets,
        )
        self.assertIn("2 target(s)", issue.message)

    def test_multiple_missing_targets(self):
        """Test multiple missing target errors in one log."""
        log = """
ERROR: /path/to/BUILD.bazel:10:20: no such target '@@workspace1//pkg1:target1': target 'target1' not declared in package 'pkg1'
ERROR: /path/to/BUILD.bazel:30:40: no such target '@@workspace2//pkg2:target2': target 'target2' not declared in package 'pkg2'
ERROR: Build did NOT complete successfully
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 2)
        self.assertEqual(issues[0].error_type, "missing_target")
        self.assertEqual(issues[1].error_type, "missing_target")
        self.assertEqual(issues[0].target, "target1")
        self.assertEqual(issues[1].target, "target2")

    def test_no_errors_on_success(self):
        """Test that no errors are extracted on successful build."""
        log = """
INFO: Analyzed 1234 targets (0 packages loaded, 0 targets configured).
INFO: Found 1234 targets...
INFO: Elapsed time: 123.456s, Critical Path: 45.67s
INFO: 10000 processes: 9000 internal, 1000 linux-sandbox.
INFO: Build completed successfully, 10000 total actions
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 0)

    def test_deduplication_by_target(self):
        """Test that duplicate missing target errors are deduplicated."""
        log = """
ERROR: /path/to/BUILD.bazel:10:20: no such target '@@workspace//pkg:target': target 'target' not declared
ERROR: /path/to/BUILD.bazel:10:20: no such target '@@workspace//pkg:target': target 'target' not declared
ERROR: Build did NOT complete successfully
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        # Should only report once.
        self.assertEqual(len(issues), 1)

    def test_compiler_errors_not_extracted(self):
        """Test that compiler errors are not extracted by BazelErrorExtractor."""
        log = """
compiler/src/iree/compiler/API/test.cpp:123:45: error: use of undeclared identifier 'foo'
  foo();
  ^
ERROR: Build did NOT complete successfully
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        # Should not extract compiler error - that's BuildErrorExtractor's job.
        # Only build summary.
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].error_type, "build_failed")

    def test_single_at_workspace_reference(self):
        """Test missing target with single @ workspace reference (legacy format)."""
        log = """
ERROR: /path/to/BUILD.bazel:10:20: no such target '@llvm-project//mlir:IR': target 'IR' not declared in package 'mlir'
ERROR: Build did NOT complete successfully
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.workspace, "llvm-project")
        self.assertEqual(issue.package, "mlir")
        self.assertEqual(issue.target, "IR")

    def test_error_context_extraction(self):
        """Test that error context is extracted from following lines."""
        log = """
ERROR: /path/to/BUILD.bazel:10:20: no such target '@@workspace//pkg:target':
target 'TargetName' not declared in package 'pkg'
and referenced by '//other/package:other_target'
ERROR: Build did NOT complete successfully
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertIn("target 'TargetName' not declared", issue.message)
        self.assertIn("and referenced by", issue.message)

    def test_chronological_ordering(self):
        """Test that issues are returned in chronological order."""
        log = """
ERROR: /path/to/BUILD.bazel:10:20: no such target '@@ws1//pkg1:t1': target 't1' not declared
ERROR: /path/to/BUILD.bazel:30:40: no such target '@@ws2//pkg2:t2': target 't2' not declared
ERROR: /path/to/BUILD.bazel:50:60: no such target '@@ws3//pkg3:t3': target 't3' not declared
ERROR: Build did NOT complete successfully
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 3)
        # Should be in order of appearance (not reversed).
        self.assertEqual(issues[0].target, "t1")
        self.assertEqual(issues[1].target, "t2")
        self.assertEqual(issues[2].target, "t3")


if __name__ == "__main__":
    unittest.main()
