# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for lit_tools/core/cli.py."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lit_tools.core import cli


def _make_mock_case(number, name=None, check_count=0):
    """Create a mock TestCase for testing filters."""
    case = Mock()
    case.number = number
    case.name = name
    case.check_count = check_count
    return case


class TestParseCaseNumbers(unittest.TestCase):
    """Tests for parse_case_numbers() function."""

    def test_parse_single(self):
        """Test parsing single case number."""
        self.assertEqual(cli.parse_case_numbers("1"), [1])
        self.assertEqual(cli.parse_case_numbers("5"), [5])
        self.assertEqual(cli.parse_case_numbers("100"), [100])

    def test_parse_comma_separated(self):
        """Test parsing comma-separated case numbers."""
        self.assertEqual(cli.parse_case_numbers("1,3,5"), [1, 3, 5])
        self.assertEqual(cli.parse_case_numbers("10,20,30"), [10, 20, 30])
        self.assertEqual(cli.parse_case_numbers("1,2,3,4,5"), [1, 2, 3, 4, 5])

    def test_parse_range(self):
        """Test parsing range syntax."""
        self.assertEqual(cli.parse_case_numbers("1-3"), [1, 2, 3])
        self.assertEqual(cli.parse_case_numbers("5-8"), [5, 6, 7, 8])
        self.assertEqual(cli.parse_case_numbers("1-1"), [1])

    def test_parse_mixed_comma_and_range(self):
        """Test parsing mixed comma-separated and ranges."""
        self.assertEqual(cli.parse_case_numbers("1,3-5,7"), [1, 3, 4, 5, 7])
        self.assertEqual(cli.parse_case_numbers("1-3,5,7-9"), [1, 2, 3, 5, 7, 8, 9])
        self.assertEqual(cli.parse_case_numbers("10,1-3,20"), [1, 2, 3, 10, 20])

    def test_parse_list_input(self):
        """Test parsing list of strings (multiple --case flags)."""
        self.assertEqual(cli.parse_case_numbers(["1", "3", "5"]), [1, 3, 5])
        self.assertEqual(cli.parse_case_numbers(["1", "2", "3"]), [1, 2, 3])

    def test_parse_list_with_comma(self):
        """Test parsing list containing comma-separated values."""
        self.assertEqual(cli.parse_case_numbers(["1,2", "3", "5"]), [1, 2, 3, 5])
        self.assertEqual(cli.parse_case_numbers(["1-3", "5"]), [1, 2, 3, 5])

    def test_parse_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        self.assertEqual(cli.parse_case_numbers(" 1 "), [1])
        self.assertEqual(cli.parse_case_numbers("1, 3, 5"), [1, 3, 5])
        self.assertEqual(cli.parse_case_numbers(" 1 - 3 "), [1, 2, 3])
        self.assertEqual(cli.parse_case_numbers("1 , 3 - 5 , 7"), [1, 3, 4, 5, 7])

    def test_parse_deduplication(self):
        """Test that duplicate numbers are removed."""
        self.assertEqual(cli.parse_case_numbers("1,1,3"), [1, 3])
        self.assertEqual(cli.parse_case_numbers("1-3,2-4"), [1, 2, 3, 4])
        self.assertEqual(cli.parse_case_numbers(["1", "1", "3"]), [1, 3])

    def test_parse_sorted_output(self):
        """Test that output is always sorted."""
        self.assertEqual(cli.parse_case_numbers("5,1,3"), [1, 3, 5])
        self.assertEqual(cli.parse_case_numbers("10,5,1"), [1, 5, 10])
        self.assertEqual(cli.parse_case_numbers("3,1-2"), [1, 2, 3])

    def test_parse_empty_string_error(self):
        """Test that empty string raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            cli.parse_case_numbers("")
        self.assertIn("No case numbers specified", str(cm.exception))

    def test_parse_whitespace_only_error(self):
        """Test that whitespace-only string raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            cli.parse_case_numbers("   ")
        self.assertIn("No case numbers specified", str(cm.exception))

    def test_parse_empty_list_error(self):
        """Test that empty list raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            cli.parse_case_numbers([])
        self.assertIn("No case numbers specified", str(cm.exception))

    def test_parse_invalid_number_error(self):
        """Test that non-numeric input raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            cli.parse_case_numbers("abc")
        self.assertIn("Invalid case number: 'abc'", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            cli.parse_case_numbers("1,x,3")
        self.assertIn("Invalid case number: 'x'", str(cm.exception))

    def test_parse_zero_error(self):
        """Test that zero raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            cli.parse_case_numbers("0")
        self.assertIn("Case numbers must be positive, got 0", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            cli.parse_case_numbers("1,0,3")
        self.assertIn("Case numbers must be positive, got 0", str(cm.exception))

    def test_parse_negative_error(self):
        """Test that negative numbers raise ValueError."""
        with self.assertRaises(ValueError) as cm:
            cli.parse_case_numbers("-1")
        self.assertIn("Case numbers must be positive, got -1", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            cli.parse_case_numbers("1,-3,5")
        self.assertIn("Case numbers must be positive, got -3", str(cm.exception))

    def test_parse_invalid_range_format_error(self):
        """Test that invalid range format raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            cli.parse_case_numbers("1-")
        self.assertIn("Invalid case number: '1-'", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            cli.parse_case_numbers("-3")
        # This is parsed as a negative number, not a range.
        self.assertIn("Case numbers must be positive, got -3", str(cm.exception))

    def test_parse_invalid_range_order_error(self):
        """Test that backward ranges raise ValueError."""
        with self.assertRaises(ValueError) as cm:
            cli.parse_case_numbers("5-1")
        self.assertIn("start must be <= end", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            cli.parse_case_numbers("10-5")
        self.assertIn("start must be <= end", str(cm.exception))

    def test_parse_range_with_zero_error(self):
        """Test that ranges with zero raise ValueError."""
        with self.assertRaises(ValueError) as cm:
            cli.parse_case_numbers("0-3")
        self.assertIn("Case numbers must be positive", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            cli.parse_case_numbers("1-0")
        self.assertIn("Case numbers must be positive", str(cm.exception))

    def test_parse_range_with_negative_error(self):
        """Test that ranges with negative numbers raise ValueError."""
        with self.assertRaises(ValueError) as cm:
            cli.parse_case_numbers("-1-3")
        self.assertIn("Invalid case number: '-1-3'", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            cli.parse_case_numbers("1--3")
        # This is parsed as range 1 to -3, which fails validation.
        self.assertIn("Case numbers must be positive", str(cm.exception))

    def test_parse_non_integer_in_range_error(self):
        """Test that non-integer range bounds raise ValueError."""
        with self.assertRaises(ValueError) as cm:
            cli.parse_case_numbers("a-3")
        self.assertIn("both start and end must be integers", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            cli.parse_case_numbers("1-b")
        self.assertIn("both start and end must be integers", str(cm.exception))


class TestApplyFilters(unittest.TestCase):
    """Tests for apply_filters() function."""

    def setUp(self):
        """Create test cases for filtering."""
        self.cases = [
            _make_mock_case(1, "fold_constant", 2),
            _make_mock_case(2, "fold_operation", 3),
            _make_mock_case(3, "gpu_kernel", 1),
            _make_mock_case(4, "cpu_optimization", 4),
            _make_mock_case(5, None, 0),  # Unnamed case.
        ]

    def test_no_filters_returns_all_cases(self):
        """Test that no filters returns original list."""
        result = cli.apply_filters(self.cases, None, None)
        self.assertEqual(result, self.cases)

    def test_filter_include_single_match(self):
        """Test positive filter with single match."""
        result = cli.apply_filters(self.cases, "gpu", None)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "gpu_kernel")

    def test_filter_include_multiple_matches(self):
        """Test positive filter with multiple matches."""
        result = cli.apply_filters(self.cases, "fold", None)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "fold_constant")
        self.assertEqual(result[1].name, "fold_operation")

    def test_filter_include_regex_pattern(self):
        """Test positive filter with regex pattern."""
        result = cli.apply_filters(self.cases, "fold.*ion", None)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "fold_operation")

    def test_filter_include_or_pattern(self):
        """Test positive filter with OR pattern."""
        result = cli.apply_filters(self.cases, "gpu|cpu", None)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "gpu_kernel")
        self.assertEqual(result[1].name, "cpu_optimization")

    def test_filter_exclude_single_match(self):
        """Test negative filter excluding single case."""
        result = cli.apply_filters(self.cases, None, "gpu")
        self.assertEqual(len(result), 4)  # Excludes gpu_kernel, keeps unnamed.
        self.assertNotIn("gpu_kernel", [c.name for c in result])

    def test_filter_exclude_multiple_matches(self):
        """Test negative filter excluding multiple cases."""
        result = cli.apply_filters(self.cases, None, "fold")
        self.assertEqual(len(result), 3)  # Excludes 2 fold cases, keeps unnamed.
        self.assertEqual(result[0].name, "gpu_kernel")
        self.assertEqual(result[1].name, "cpu_optimization")
        self.assertIsNone(result[2].name)  # Unnamed case kept.

    def test_filter_exclude_or_pattern(self):
        """Test negative filter with OR pattern."""
        result = cli.apply_filters(self.cases, None, "gpu|cpu")
        self.assertEqual(len(result), 3)  # Keeps fold cases and unnamed.
        self.assertEqual(result[0].name, "fold_constant")
        self.assertEqual(result[1].name, "fold_operation")
        self.assertIsNone(result[2].name)  # Unnamed case kept.

    def test_combined_filters(self):
        """Test combining positive and negative filters."""
        # Include anything with 'fold' or 'cpu', exclude 'constant'.
        result = cli.apply_filters(self.cases, "fold|cpu", "constant")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "fold_operation")
        self.assertEqual(result[1].name, "cpu_optimization")

    def test_combined_filters_narrow_down(self):
        """Test using both filters to narrow down results."""
        # Include anything with 'o', exclude 'gpu'.
        result = cli.apply_filters(self.cases, "o", "gpu")
        self.assertEqual(len(result), 3)
        names = [c.name for c in result]
        self.assertIn("fold_constant", names)
        self.assertIn("fold_operation", names)
        self.assertIn("cpu_optimization", names)

    def test_filter_excludes_unnamed_cases(self):
        """Test that unnamed cases are excluded when filters are active."""
        # Apply filter that would match everything (if cases had names).
        result = cli.apply_filters(self.cases, ".*", None)
        self.assertEqual(len(result), 4)
        self.assertNotIn(None, [c.name for c in result])

    def test_filter_include_no_match_returns_none(self):
        """Test that no matches for positive filter returns None."""
        result = cli.apply_filters(self.cases, "nonexistent", None)
        self.assertIsNone(result)

    def test_filter_exclude_all_named_keeps_unnamed(self):
        """Test that excluding all named cases keeps unnamed cases."""
        result = cli.apply_filters(self.cases, None, ".*")
        self.assertEqual(len(result), 1)  # Only unnamed case remains.
        self.assertIsNone(result[0].name)

    def test_combined_filter_no_match_returns_none(self):
        """Test that combined filters with no results returns None."""
        # Include 'fold', exclude 'fold' -> no results.
        result = cli.apply_filters(self.cases, "fold", "fold")
        self.assertIsNone(result)

    def test_filter_case_sensitive(self):
        """Test that filtering is case-sensitive by default."""
        result = cli.apply_filters(self.cases, "GPU", None)
        self.assertIsNone(result)  # No match (gpu vs GPU).

    def test_filter_case_insensitive_pattern(self):
        """Test case-insensitive filtering with regex flag."""
        result = cli.apply_filters(self.cases, "(?i)GPU", None)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "gpu_kernel")

    def test_filter_preserves_order(self):
        """Test that filtering preserves original case order."""
        result = cli.apply_filters(self.cases, "fold|gpu", None)
        self.assertEqual(result[0].number, 1)
        self.assertEqual(result[1].number, 2)
        self.assertEqual(result[2].number, 3)

    def test_empty_case_list_returns_empty(self):
        """Test that filtering empty list returns empty list."""
        result = cli.apply_filters([], "fold", None)
        self.assertIsNone(result)  # No matches in empty list.

    def test_all_unnamed_cases_with_filter(self):
        """Test filtering list with all unnamed cases."""
        unnamed = [
            _make_mock_case(1, None, 0),
            _make_mock_case(2, None, 0),
        ]
        result = cli.apply_filters(unnamed, ".*", None)
        self.assertIsNone(result)  # All cases excluded (no names).


if __name__ == "__main__":
    unittest.main()
