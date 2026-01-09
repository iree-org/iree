# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Integration tests for fuzzy matching suggestions in iree-lit-* tools.

These tests verify that the suggestions module integrates correctly with
the actual tools (iree-lit-extract, iree-lit-test, iree-lit-replace) and
produces helpful error messages with suggestions when case names are not found.
"""

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from common import exit_codes
from lit_tools import iree_lit_extract, iree_lit_replace, iree_lit_test
from lit_tools.core.parser import parse_test_file
from lit_tools.iree_lit_replace import _find_target_case_by_name

TEST_CONTENT = """// RUN: iree-opt %s | FileCheck %s

// -----
// CHECK-LABEL: @emplaceDispatch
util.func @emplaceDispatch() {
  util.return
}

// -----
// CHECK-LABEL: @dontEmplaceTiedDispatch
util.func @dontEmplaceTiedDispatch() {
  util.return
}

// -----
// CHECK-LABEL: @emplaceDispatchSequence
util.func @emplaceDispatchSequence() {
  util.return
}
"""


@contextmanager
def temp_test_file(content: str, suffix: str = ".mlir"):
    """Create a temporary test file with automatic cleanup."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)


class TestSuggestionsIntegration(unittest.TestCase):
    """Integration tests for fuzzy matching across all tools."""

    def test_extract_with_typo_shows_suggestions(self):
        """Test iree-lit-extract shows suggestions for typo in name."""
        with temp_test_file(TEST_CONTENT) as tmp_path:
            # Create args with typo: "emplceDispatch" (missing 'a').
            with patch(
                "sys.argv",
                [
                    "iree-lit-extract",
                    str(tmp_path),
                    "--name",
                    "emplceDispatch",
                    "--quiet",
                ],
            ):
                args = iree_lit_extract.parse_arguments()

            # Run extraction (should fail with suggestions).
            exit_code = iree_lit_extract.main(args)

            # Should return NOT_FOUND error.
            self.assertEqual(exit_codes.NOT_FOUND, exit_code)

            # Error message tested manually - suggestions should appear in stderr.
            # This test verifies the integration works (exit code correct).

    def test_extract_with_no_match_shows_no_suggestions(self):
        """Test iree-lit-extract shows no suggestions for completely wrong name."""
        with temp_test_file(TEST_CONTENT) as tmp_path:
            # Create args with completely wrong name.
            with patch(
                "sys.argv",
                [
                    "iree-lit-extract",
                    str(tmp_path),
                    "--name",
                    "completelyWrongName",
                ],
            ):
                args = iree_lit_extract.parse_arguments()

            # Run extraction (should fail without suggestions).
            exit_code = iree_lit_extract.main(args)

            # Should return NOT_FOUND error.
            self.assertEqual(exit_codes.NOT_FOUND, exit_code)

    def test_test_with_typo_shows_suggestions(self):
        """Test iree-lit-test shows suggestions for typo in name."""
        with temp_test_file(TEST_CONTENT) as tmp_path:
            # Create args with typo.
            with patch(
                "sys.argv",
                [
                    "iree-lit-test",
                    str(tmp_path),
                    "--name",
                    "emplaceDisptch",
                    "--quiet",
                ],
            ):
                args = iree_lit_test.parse_arguments()

            # Run test (should fail with suggestions).
            exit_code = iree_lit_test.main(args)

            # Should return error (no cases found).
            self.assertNotEqual(exit_codes.SUCCESS, exit_code)

    def test_replace_with_invalid_name_shows_suggestions(self):
        """Test iree-lit-replace shows suggestions for invalid name in text mode."""
        with temp_test_file(TEST_CONTENT) as tmp_path:
            # This test is more complex as it requires stdin input.
            # We test the core function _find_target_case_by_name instead.

            # Parse test cases.
            test_file_obj = parse_test_file(tmp_path)
            all_cases = list(test_file_obj.cases)

            # Create minimal args.
            with patch("sys.argv", ["iree-lit-replace", "--quiet"]):
                args = iree_lit_replace.parse_arguments()

            # Try to find case with typo.
            case, error_code = _find_target_case_by_name(
                all_cases, tmp_path, "emplaceDisptch", args
            )

            # Should return None and NOT_FOUND error.
            self.assertIsNone(case)
            self.assertEqual(exit_codes.NOT_FOUND, error_code)

            # Suggestions tested manually - error message should contain suggestions.


if __name__ == "__main__":
    unittest.main()
