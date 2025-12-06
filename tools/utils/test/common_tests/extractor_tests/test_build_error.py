# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for BuildErrorExtractor."""

import unittest

from common.extractors.build_error import BuildErrorExtractor
from common.issues import Severity
from common.log_buffer import LogBuffer


class TestBuildErrorExtractor(unittest.TestCase):
    """Test BuildErrorExtractor for C/C++ compilation errors."""

    def setUp(self):
        """Set up extractor for tests."""
        self.extractor = BuildErrorExtractor()

    def test_simple_compile_error(self):
        """Test basic C++ compilation error."""
        log = """
[ 45%] Building CXX object runtime/src/iree/hal/CMakeFiles/iree_hal.dir/device.c.o
/home/user/iree/runtime/src/iree/hal/device.c:123:5: error: use of undeclared identifier 'foo'
  foo();
  ^
1 error generated.
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.severity, Severity.CRITICAL)
        self.assertTrue(issue.actionable)
        self.assertIn("undeclared identifier", issue.message)
        self.assertIn("device.c", issue.file_path or "")

    def test_multiple_errors_same_file(self):
        """Test multiple errors from same file."""
        log = """
/home/user/iree/compiler/src/test.cpp:10:3: error: expected ';' after expression
  x = 5
  ^
/home/user/iree/compiler/src/test.cpp:15:10: error: no matching function for call to 'bar'
  return bar();
         ^~~
2 errors generated.
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 2)
        self.assertEqual(issues[0].severity, Severity.CRITICAL)
        self.assertEqual(issues[1].severity, Severity.CRITICAL)
        self.assertIn("expected ';'", issues[0].message)
        self.assertIn("no matching function", issues[1].message)

    def test_linker_error(self):
        """Test linker undefined reference error."""
        log = """
[100%] Linking CXX executable iree-run-module
/usr/bin/ld: CMakeFiles/iree_run_module.dir/main.o: undefined reference to `iree_hal_device_create'
collect2: error: ld returned 1 exit status
ninja: build stopped: subcommand failed.
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.severity, Severity.CRITICAL)
        self.assertTrue(issue.actionable)
        self.assertIn(
            "Undefined reference", issue.message
        )  # Capital U to match actual output.

    def test_warning_not_extracted(self):
        """Test that warnings are not extracted (unless treated as errors)."""
        log = """
/home/user/iree/runtime/src/test.c:50:10: warning: unused variable 'x' [-Wunused-variable]
  int x = 5;
      ^
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        # BuildErrorExtractor only extracts errors, not warnings.
        self.assertEqual(len(issues), 0)

    def test_no_errors(self):
        """Test log with no build errors."""
        log = """
[ 45%] Building CXX object runtime/src/iree/hal/CMakeFiles/iree_hal.dir/device.c.o
[ 46%] Building CXX object runtime/src/iree/hal/CMakeFiles/iree_hal.dir/buffer.c.o
[100%] Built target iree_hal
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 0)


if __name__ == "__main__":
    unittest.main()
