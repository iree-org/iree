# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for fuzzy matching suggestion utilities."""

import unittest
from dataclasses import dataclass

from lit_tools.core import suggestions


# Mock TestCase for testing (mimics the real TestCase structure).
@dataclass
class MockTestCase:
    """Mock test case for testing suggestions."""

    name: str | None
    number: int


class TestFindSimilarCaseNames(unittest.TestCase):
    """Tests for find_similar_case_names()."""

    def setUp(self):
        """Create test cases for use in tests."""
        self.test_cases = [
            MockTestCase(name="emplaceDispatch", number=1),
            MockTestCase(name="dontEmplaceTiedDispatch", number=2),
            MockTestCase(name="emplaceDispatchSequence", number=3),
            MockTestCase(name="foldConstants", number=4),
            MockTestCase(name="optimizeLoops", number=5),
            MockTestCase(name=None, number=6),  # Case without name.
        ]

    def test_exact_typo_single_character(self):
        """Test single-character typo finds correct suggestion."""
        suggestions_list = suggestions.find_similar_case_names(
            "emplceDispatch",  # Missing 'a'.
            self.test_cases,
        )
        # Should find emplaceDispatch as first (best) match.
        self.assertGreater(len(suggestions_list), 0)
        self.assertEqual("emplaceDispatch", suggestions_list[0])

    def test_similar_prefix_multiple_matches(self):
        """Test partial match returns multiple similar names."""
        suggestions_list = suggestions.find_similar_case_names(
            "emplaceDisp",  # Prefix of multiple names.
            self.test_cases,
        )
        # Should find emplaceDispatch and emplaceDispatchSequence.
        self.assertIn("emplaceDispatch", suggestions_list)
        self.assertIn("emplaceDispatchSequence", suggestions_list)
        self.assertLessEqual(len(suggestions_list), 3)  # Max 3 suggestions.

    def test_no_similar_names(self):
        """Test completely different name returns empty list."""
        suggestions_list = suggestions.find_similar_case_names(
            "completelyDifferentName", self.test_cases
        )
        self.assertEqual([], suggestions_list)

    def test_empty_case_list(self):
        """Test empty case list returns empty suggestions."""
        suggestions_list = suggestions.find_similar_case_names("anything", [])
        self.assertEqual([], suggestions_list)

    def test_all_cases_without_names(self):
        """Test cases with no names returns empty suggestions."""
        nameless_cases = [
            MockTestCase(name=None, number=1),
            MockTestCase(name=None, number=2),
        ]
        suggestions_list = suggestions.find_similar_case_names(
            "anything", nameless_cases
        )
        self.assertEqual([], suggestions_list)

    def test_max_suggestions_limit(self):
        """Test max_suggestions parameter limits results."""
        # Create many similar names.
        many_cases = [MockTestCase(name=f"test{i}", number=i) for i in range(10)]
        suggestions_list = suggestions.find_similar_case_names(
            "test", many_cases, max_suggestions=2
        )
        self.assertLessEqual(len(suggestions_list), 2)

    def test_cutoff_parameter(self):
        """Test cutoff parameter filters low-similarity matches."""
        # With high cutoff, only very similar matches returned.
        suggestions_list = suggestions.find_similar_case_names(
            "fold", self.test_cases, cutoff=0.9  # 90% similarity.
        )
        # "fold" is not very similar to "foldConstants" (only 44% match).
        # Should return empty with high cutoff.
        self.assertEqual([], suggestions_list)

        # With low cutoff, more permissive matching.
        suggestions_list = suggestions.find_similar_case_names(
            "fold", self.test_cases, cutoff=0.3  # 30% similarity.
        )
        # Now "foldConstants" should match.
        self.assertIn("foldConstants", suggestions_list)

    def test_case_sensitive_matching(self):
        """Test that matching is case-sensitive."""
        cases_with_mixed_case = [
            MockTestCase(name="TestCase", number=1),
            MockTestCase(name="testCase", number=2),
        ]
        # Exact case match should be prioritized.
        suggestions_list = suggestions.find_similar_case_names(
            "testcase", cases_with_mixed_case
        )
        # Both are similar, but lowercase is closer match.
        self.assertIn("testCase", suggestions_list)


class TestFormatCaseNameError(unittest.TestCase):
    """Tests for format_case_name_error()."""

    def setUp(self):
        """Create test cases for use in tests."""
        self.test_cases = [
            MockTestCase(name="emplaceDispatch", number=1),
            MockTestCase(name="dontEmplaceTiedDispatch", number=2),
            MockTestCase(name="emplaceDispatchSequence", number=3),
        ]

    def test_no_suggestions_no_file_path(self):
        """Test error with no suggestions and no file path."""
        error_msg = suggestions.format_case_name_error(
            "completelyWrong", self.test_cases
        )
        self.assertEqual("Case with name 'completelyWrong' not found", error_msg)

    def test_no_suggestions_with_file_path(self):
        """Test error with no suggestions but with file path."""
        error_msg = suggestions.format_case_name_error(
            "completelyWrong", self.test_cases, file_path="test.mlir"
        )
        self.assertEqual(
            "Case with name 'completelyWrong' not found in test.mlir", error_msg
        )

    def test_single_suggestion(self):
        """Test error message with suggestions."""
        error_msg = suggestions.format_case_name_error(
            "emplceDispatch",  # Typo: missing 'a'.
            self.test_cases,
        )
        # Should find emplaceDispatch as a suggestion.
        self.assertIn("emplaceDispatch", error_msg)
        self.assertIn("Case with name 'emplceDispatch' not found", error_msg)
        # Check for suggestion format (either single or multiple).
        self.assertTrue("Did you mean" in error_msg, "Should include 'Did you mean'")

    def test_multiple_suggestions(self):
        """Test error message with multiple suggestions."""
        error_msg = suggestions.format_case_name_error(
            "emplaceDisp",  # Partial match.
            self.test_cases,
        )
        self.assertIn("Did you mean one of:", error_msg)
        self.assertIn("emplaceDispatch", error_msg)
        self.assertIn("emplaceDispatchSequence", error_msg)

    def test_suggestion_with_file_path(self):
        """Test error with suggestion and file path."""
        error_msg = suggestions.format_case_name_error(
            "emplceDispatch",
            self.test_cases,
            file_path="emplace_allocations.mlir",
        )
        self.assertIn("emplace_allocations.mlir", error_msg)
        self.assertIn("emplaceDispatch", error_msg)
        self.assertIn("Did you mean", error_msg)

    def test_empty_case_list(self):
        """Test error with empty case list."""
        error_msg = suggestions.format_case_name_error("anything", [])
        self.assertEqual("Case with name 'anything' not found", error_msg)


class TestFormatCaseNumberError(unittest.TestCase):
    """Tests for format_case_number_error()."""

    def test_basic_error_no_file_path(self):
        """Test basic case number error without file path."""
        error_msg = suggestions.format_case_number_error(999, 10)
        self.assertEqual("Case 999 not found (file has 10 cases)", error_msg)

    def test_error_with_file_path(self):
        """Test case number error with file path."""
        error_msg = suggestions.format_case_number_error(0, 5, file_path="test.mlir")
        self.assertEqual("Case 0 not found in test.mlir (file has 5 cases)", error_msg)

    def test_singular_case(self):
        """Test message with single case (singular 'case' not 'cases')."""
        error_msg = suggestions.format_case_number_error(2, 1)
        self.assertEqual("Case 2 not found (file has 1 case)", error_msg)

    def test_plural_cases(self):
        """Test message with multiple cases (plural 'cases')."""
        error_msg = suggestions.format_case_number_error(100, 50)
        self.assertEqual("Case 100 not found (file has 50 cases)", error_msg)


if __name__ == "__main__":
    unittest.main()
