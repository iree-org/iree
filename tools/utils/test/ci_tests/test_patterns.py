# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for CI pattern matching."""

import sys
import unittest
from pathlib import Path

# Add project tools/utils to path for imports.
sys.path.insert(0, str(Path(__file__).parents[2]))

from ci.core import patterns

# Module-level fixture directory (absolute path for CWD-independence).
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Synthetic log snippets for testing (small, inline).
FILECHECK_FAILURE_LOG = """
test.mlir:15:11: error: CHECK: expected string not found in input
// CHECK: %[[VAL:.+]] = arith.constant
          ^
<stdin>:10:1: note: scanning from here
func.func @test() {
^
"""

ROCM_CLEANUP_CRASH_LOG = """
==12345== ERROR: AddressSanitizer: heap-use-after-free
    #0 0x7f9876543210 in rocclr/device/device.cpp:2891
    #1 0x7f9876543211 in ~Memory() rocclr/device/memory.cpp:456
    #2 0x7f9876543212 in release() rocclr/platform/object.hpp:123

Aborted (core dumped)
"""

COMPILE_ERROR_LOG = """
/usr/bin/c++ -o runtime/src/iree/hal/buffer.o -c runtime/src/iree/hal/buffer.c
runtime/src/iree/hal/buffer.c:42:10: fatal error: 'iree/base/api.h' file not found
#include "iree/base/api.h"
         ^~~~~~~~~~~~~~~~~
1 error generated.
"""

TIMEOUT_LOG = """
*********************** TEST 'test.mlir' FAILED ***********************
+ iree-compile test.mlir --iree-hal-target-backends=vulkan-spirv
TIMEOUT: test exceeded maximum time limit of 60 seconds
Killing process group
*********************** TEST 'test.mlir' FAILED ***********************
"""


class TestPatternLoading(unittest.TestCase):
    """Tests for loading patterns from YAML."""

    def setUp(self):
        """Set up test fixtures."""
        self.patterns_file = _FIXTURES_DIR / "test_patterns.yaml"
        self.rules_file = _FIXTURES_DIR / "test_cooccurrence_rules.yaml"

    def test_load_patterns_from_yaml(self):
        """Test loading pattern definitions."""
        loader = patterns.PatternLoader(self.patterns_file, self.rules_file)
        loader.load()

        self.assertIn("filecheck_failed", loader.patterns)
        self.assertEqual(loader.patterns["filecheck_failed"].severity, "high")
        self.assertTrue(loader.patterns["filecheck_failed"].actionable)

    def test_load_rules_from_yaml(self):
        """Test loading co-occurrence rules."""
        loader = patterns.PatternLoader(self.patterns_file, self.rules_file)
        loader.load()

        self.assertGreater(len(loader.rules), 0)

        # Find rocm_cleanup_crash rule.
        rocm_rule = next(
            (r for r in loader.rules if r.name == "rocm_cleanup_crash"), None
        )
        self.assertIsNotNone(rocm_rule)
        self.assertEqual(rocm_rule.primary_pattern, "rocclr_memobj")
        self.assertIn("aborted", rocm_rule.secondary_patterns)
        self.assertFalse(rocm_rule.actionable)


class TestPatternMatching(unittest.TestCase):
    """Tests for pattern matching logic."""

    def setUp(self):
        """Load test patterns."""
        self.patterns_file = _FIXTURES_DIR / "test_patterns.yaml"
        self.rules_file = _FIXTURES_DIR / "test_cooccurrence_rules.yaml"
        self.loader = patterns.PatternLoader(self.patterns_file, self.rules_file)
        self.loader.load()
        self.matcher = patterns.PatternMatcher(self.loader)

    def test_filecheck_pattern_match(self):
        """Test FileCheck failure pattern detection."""
        matches = self.matcher.analyze_log(FILECHECK_FAILURE_LOG)

        self.assertIn("filecheck_failed", matches)
        self.assertEqual(len(matches["filecheck_failed"]), 1)

        match = matches["filecheck_failed"][0]
        self.assertIn("CHECK", match.match_text)
        self.assertEqual(match.line_number, 2)

    def test_filecheck_field_extraction(self):
        """Test extracting fields from FileCheck failure."""
        matches = self.matcher.analyze_log(FILECHECK_FAILURE_LOG)

        match = matches["filecheck_failed"][0]
        self.assertIn("file_path", match.extracted_fields)

        # Should extract test.mlir:15.
        file_paths = match.extracted_fields["file_path"]
        self.assertTrue(any("test.mlir" in fp for fp in file_paths))

    def test_rocm_crash_pattern_match(self):
        """Test ROCm cleanup crash pattern detection."""
        matches = self.matcher.analyze_log(ROCM_CLEANUP_CRASH_LOG)

        # Should match both rocclr_memobj and aborted.
        self.assertIn("rocclr_memobj", matches)
        self.assertIn("aborted", matches)

        rocm_match = matches["rocclr_memobj"][0]
        self.assertIn("rocclr/device/device.cpp", rocm_match.match_text)

    def test_compile_error_pattern_match(self):
        """Test compilation error pattern detection."""
        matches = self.matcher.analyze_log(COMPILE_ERROR_LOG)

        self.assertIn("compile_error", matches)

        match = matches["compile_error"][0]
        self.assertIn("fatal error", match.match_text)

    def test_compile_error_field_extraction(self):
        """Test extracting file path and error message from compile error."""
        matches = self.matcher.analyze_log(COMPILE_ERROR_LOG)

        match = matches["compile_error"][0]

        # Should extract file path.
        self.assertIn("file_path", match.extracted_fields)
        file_paths = match.extracted_fields["file_path"]
        self.assertTrue(any("buffer.c" in fp for fp in file_paths))

        # Should extract error message.
        self.assertIn("error_message", match.extracted_fields)
        errors = match.extracted_fields["error_message"]
        self.assertTrue(any("file not found" in e for e in errors))

    def test_timeout_pattern_match(self):
        """Test timeout pattern detection."""
        matches = self.matcher.analyze_log(TIMEOUT_LOG)

        self.assertIn("timeout", matches)

        match = matches["timeout"][0]
        self.assertIn("TIMEOUT", match.match_text)

    def test_context_lines_extraction(self):
        """Test context line extraction."""
        matches = self.matcher.analyze_log(FILECHECK_FAILURE_LOG)

        match = matches["filecheck_failed"][0]

        # Should have context before and after.
        self.assertGreater(len(match.context_before), 0)
        self.assertGreater(len(match.context_after), 0)

    def test_no_match_returns_empty(self):
        """Test that non-matching logs return empty results."""
        log = "This log has no error patterns"

        matches = self.matcher.analyze_log(log)

        self.assertEqual(len(matches), 0)


if __name__ == "__main__":
    unittest.main()
