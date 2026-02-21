# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for PrecommitErrorExtractor.

All test data is extracted verbatim from actual CI failure logs in the corpus
at /home/ben/src/iree/.vscode/notes/tools/ci_failure_corpus/logs/.
"""

import unittest

from common.extractors.precommit import PrecommitErrorExtractor
from common.issues import Severity
from common.log_buffer import LogBuffer


class TestPrecommitErrorExtractor(unittest.TestCase):
    """Test PrecommitErrorExtractor for pre-commit hook failures."""

    def setUp(self):
        """Set up extractor for tests."""
        self.extractor = PrecommitErrorExtractor()

    def test_hook_modified_files(self):
        """Test pre-commit hook that modified files.

        Real data from: run_19267500160_job_55086837291.log
        """
        log = """pre-commit\tUNKNOWN STEP\t2025-11-11T13:42:12.4259043Z Run bazel_to_cmake.py on BUILD.bazel files...............................[41mFailed[m
pre-commit\tUNKNOWN STEP\t2025-11-11T13:42:12.4260318Z [2m- hook id: bazel_to_cmake_1[m
pre-commit\tUNKNOWN STEP\t2025-11-11T13:42:12.4263309Z [2m- files were modified by this hook[m
pre-commit\tUNKNOWN STEP\t2025-11-11T13:42:12.4263924Z
pre-commit\tUNKNOWN STEP\t2025-11-11T13:42:12.4265196Z Using repo root /home/runner/work/iree/iree
"""
        log_buffer = LogBuffer(log, auto_detect_format=True)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.severity, Severity.LOW)
        self.assertTrue(issue.actionable)
        self.assertEqual(issue.hook_id, "bazel_to_cmake_1")
        self.assertEqual(
            issue.hook_description, "Run bazel_to_cmake.py on BUILD.bazel files"
        )
        self.assertEqual(issue.error_type, "modified_files")
        self.assertTrue(issue.files_modified)

    def test_hook_check_failed_no_files_modified(self):
        """Test pre-commit hook that failed a check without modifying files.

        Real data from: run_19043165665_job_54385002311.log
        """
        log = """pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:17.4197193Z DO NOT SUBMIT............................................................[41mFailed[m
pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:17.4197958Z [2m- hook id: do-not-submit[m
pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:17.4198374Z [2m- exit code: 1[m
pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:17.4198575Z
pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:17.4198804Z Error: The string "DO NOT SUBMIT" was found!
"""
        log_buffer = LogBuffer(log, auto_detect_format=True)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.severity, Severity.MEDIUM)
        self.assertEqual(issue.hook_id, "do-not-submit")
        self.assertEqual(issue.error_type, "check_failed")
        self.assertFalse(issue.files_modified)

    def test_multiple_hooks_failed(self):
        """Test multiple pre-commit hooks failing.

        Real data from: run_19043165665_job_54385002311.log
        """
        log = """pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:06.5509294Z Fix End of Files.........................................................[41mFailed[m
pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:06.5510914Z [2m- hook id: end-of-file-fixer[m
pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:06.5511804Z [2m- exit code: 1[m
pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:06.5512735Z [2m- files were modified by this hook[m
pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:06.5513327Z
pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:06.5514300Z Fixing compiler/src/iree/compiler/Codegen/Common/test/convert_workgroup_forall_to_pcf.mlir
pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:06.5515592Z
pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:06.9833012Z Trim Trailing Whitespace.................................................[41mFailed[m
pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:06.9833593Z [2m- hook id: trailing-whitespace[m
pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:06.9833935Z [2m- exit code: 1[m
pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:06.9834225Z [2m- files were modified by this hook[m
"""
        log_buffer = LogBuffer(log, auto_detect_format=True)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 2)
        self.assertEqual(issues[0].hook_id, "end-of-file-fixer")
        self.assertEqual(issues[1].hook_id, "trailing-whitespace")
        self.assertTrue(all(i.files_modified for i in issues))
        self.assertTrue(all(i.severity == Severity.LOW for i in issues))

    def test_clang_format_with_ansi_codes(self):
        """Test pre-commit hook failure with ANSI color codes.

        Real data from: run_19011226316_job_54292428222.log
        """
        log = """pre-commit\tUNKNOWN STEP\t2025-11-02T10:54:26.3289419Z Run clang-format on C/C++/etc. files.....................................[41mFailed[m
pre-commit\tUNKNOWN STEP\t2025-11-02T10:54:26.3289932Z [2m- hook id: clang-format[m
pre-commit\tUNKNOWN STEP\t2025-11-02T10:54:26.3290217Z [2m- files were modified by this hook[m
"""
        log_buffer = LogBuffer(log, auto_detect_format=True)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.hook_id, "clang-format")
        self.assertTrue(issue.files_modified)
        self.assertEqual(issue.severity, Severity.LOW)

    def test_no_failures(self):
        """Test log with no pre-commit failures.

        Real data from: run_19011226316_job_54292428222.log
        """
        log = """pre-commit\tUNKNOWN STEP\t2025-11-02T10:54:17.1074234Z Check Yaml...............................................................[42mPassed[m
pre-commit\tUNKNOWN STEP\t2025-11-02T10:54:17.3922618Z Fix End of Files.........................................................[42mPassed[m
pre-commit\tUNKNOWN STEP\t2025-11-02T10:54:17.8250992Z Trim Trailing Whitespace.................................................[42mPassed[m
pre-commit\tUNKNOWN STEP\t2025-11-02T10:54:20.9548651Z Run Black to format Python files.........................................[42mPassed[m
"""
        log_buffer = LogBuffer(log, auto_detect_format=True)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 0)

    def test_hook_with_exit_code_and_files_modified(self):
        """Test pre-commit hook with both exit code and files modified.

        Real data from: run_19043165665_job_54385002311.log
        Even with exit code, files_modified takes precedence for severity.
        """
        log = """pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:06.9833012Z Trim Trailing Whitespace.................................................[41mFailed[m
pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:06.9833593Z [2m- hook id: trailing-whitespace[m
pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:06.9833935Z [2m- exit code: 1[m
pre-commit\tUNKNOWN STEP\t2025-11-03T17:17:06.9834225Z [2m- files were modified by this hook[m
"""
        log_buffer = LogBuffer(log, auto_detect_format=True)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.hook_id, "trailing-whitespace")
        # Even with exit code, files_modified takes precedence for severity.
        self.assertEqual(issue.severity, Severity.LOW)
        self.assertTrue(issue.files_modified)


if __name__ == "__main__":
    unittest.main()
