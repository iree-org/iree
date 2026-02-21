# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for LogBuffer class.

Tests cover:
- Line access (get_line, get_lines_around)
- Offset to line number conversion
- Forward and backward pattern searching
- Edge cases (empty logs, single line, boundary conditions)
"""

import unittest

from common.log_buffer import LogBuffer


class TestLogBuffer(unittest.TestCase):
    """Test LogBuffer class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Simple multi-line log.
        self.simple_log = """Line 0: First line
Line 1: Second line
Line 2: Third line
Line 3: Fourth line
Line 4: Fifth line"""

        # Empty log.
        self.empty_log = ""

        # Single line log.
        self.single_line_log = "Only line"

        # Log with pattern matching test cases.
        self.pattern_log = """Build started
ERROR: Compilation failed
file.cpp:42:10: error: undefined symbol 'foo'
int x = foo();
        ^
ERROR: Build failed
Tests passed"""

        # Real-world CI log snippet (TSAN style).
        self.tsan_log = """==================
WARNING: ThreadSanitizer: data race (pid=12345)
  Write of size 4 at 0x7b0400001234 by thread T1:
    #0 iree::hal::Device::Submit() device.cc:123
    #1 iree::hal::CommandBuffer::End() command_buffer.cc:456

  Previous write of size 4 at 0x7b0400001234 by main thread:
    #0 iree::hal::Device::Create() device.cc:789
    #1 main() main.cc:42

SUMMARY: ThreadSanitizer: data race device.cc:123 in iree::hal::Device::Submit()
=================="""

    def test_get_line_valid(self):
        """Test get_line with valid line numbers."""
        buffer = LogBuffer(self.simple_log)

        self.assertEqual(buffer.get_line(0), "Line 0: First line")
        self.assertEqual(buffer.get_line(2), "Line 2: Third line")
        self.assertEqual(buffer.get_line(4), "Line 4: Fifth line")

    def test_get_line_invalid(self):
        """Test get_line with invalid line numbers."""
        buffer = LogBuffer(self.simple_log)

        self.assertIsNone(buffer.get_line(-1))
        self.assertIsNone(buffer.get_line(5))
        self.assertIsNone(buffer.get_line(100))

    def test_get_line_empty_log(self):
        """Test get_line on empty log."""
        buffer = LogBuffer(self.empty_log)

        self.assertIsNone(buffer.get_line(0))
        self.assertIsNone(buffer.get_line(1))

    def test_get_line_single_line(self):
        """Test get_line on single line log."""
        buffer = LogBuffer(self.single_line_log)

        self.assertEqual(buffer.get_line(0), "Only line")
        self.assertIsNone(buffer.get_line(1))

    def test_get_lines_around_center(self):
        """Test get_lines_around with center context."""
        buffer = LogBuffer(self.simple_log)

        # Line 2 with context=1.
        lines = buffer.get_lines_around(2, context=1)
        self.assertEqual(
            lines,
            ["Line 1: Second line", "Line 2: Third line", "Line 3: Fourth line"],
        )

    def test_get_lines_around_start_boundary(self):
        """Test get_lines_around at start of log."""
        buffer = LogBuffer(self.simple_log)

        # Line 0 with context=2 (should not include negative lines).
        lines = buffer.get_lines_around(0, context=2)
        self.assertEqual(
            lines,
            [
                "Line 0: First line",
                "Line 1: Second line",
                "Line 2: Third line",
            ],
        )

    def test_get_lines_around_end_boundary(self):
        """Test get_lines_around at end of log."""
        buffer = LogBuffer(self.simple_log)

        # Line 4 with context=2 (should not include lines beyond end).
        lines = buffer.get_lines_around(4, context=2)
        self.assertEqual(
            lines,
            [
                "Line 2: Third line",
                "Line 3: Fourth line",
                "Line 4: Fifth line",
            ],
        )

    def test_get_lines_around_large_context(self):
        """Test get_lines_around with context larger than log."""
        buffer = LogBuffer(self.simple_log)

        # Context=100 should return entire log.
        lines = buffer.get_lines_around(2, context=100)
        self.assertEqual(len(lines), 5)  # All 5 lines.
        self.assertEqual(lines[0], "Line 0: First line")
        self.assertEqual(lines[4], "Line 4: Fifth line")

    def test_line_count(self):
        """Test line_count property."""
        self.assertEqual(LogBuffer(self.simple_log).line_count, 5)
        self.assertEqual(
            LogBuffer(self.empty_log).line_count, 0
        )  # Empty string returns no lines.
        self.assertEqual(LogBuffer(self.single_line_log).line_count, 1)
        self.assertEqual(LogBuffer(self.tsan_log).line_count, 12)

    def test_get_line_number_from_offset_start(self):
        """Test offset to line conversion at start."""
        buffer = LogBuffer(self.simple_log)

        # Offset 0 is start of line 0.
        self.assertEqual(buffer.get_line_number_from_offset(0), 0)

        # Offset 5 is within line 0.
        self.assertEqual(buffer.get_line_number_from_offset(5), 0)

    def test_get_line_number_from_offset_middle(self):
        """Test offset to line conversion in middle."""
        buffer = LogBuffer(self.simple_log)

        # Calculate offset to line 2.
        # Line 0: "Line 0: First line\n" = 19 bytes
        # Line 1: "Line 1: Second line\n" = 20 bytes
        # Line 2 starts at offset 39.
        offset_line_2_start = len("Line 0: First line\n") + len("Line 1: Second line\n")
        self.assertEqual(buffer.get_line_number_from_offset(offset_line_2_start), 2)

        # Middle of line 2.
        self.assertEqual(buffer.get_line_number_from_offset(offset_line_2_start + 5), 2)

    def test_get_line_number_from_offset_end(self):
        """Test offset to line conversion at end."""
        buffer = LogBuffer(self.simple_log)

        # Offset beyond end should return last line.
        self.assertEqual(buffer.get_line_number_from_offset(1000), 4)

    def test_find_previous_match_found(self):
        """Test backward search finding a match."""
        buffer = LogBuffer(self.pattern_log)

        # Find line 1 ("ERROR: Compilation failed") when searching from line 4.
        # Calculate offset for line 4 (middle of log).
        lines_before = self.pattern_log.split("\n")[:4]
        offset = sum(len(line) + 1 for line in lines_before)

        match = buffer.find_previous_match(offset, r"ERROR:")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(), "ERROR:")

    def test_find_previous_match_not_found(self):
        """Test backward search not finding a match."""
        buffer = LogBuffer(self.pattern_log)

        # Search from line 1 for pattern that only exists later.
        lines_before = self.pattern_log.split("\n")[:1]
        offset = sum(len(line) + 1 for line in lines_before)

        match = buffer.find_previous_match(offset, r"Tests passed")
        self.assertIsNone(match)

    def test_find_previous_match_max_lines_limit(self):
        """Test backward search respects max_lines limit."""
        buffer = LogBuffer(self.pattern_log)

        # Search from last line for first "ERROR:" with max_lines=3.
        # Pattern log has ERROR on lines 1 and 5.
        # From line 6, max_lines=3 searches lines 5, 4, 3.
        # Should find ERROR on line 5, NOT line 1.
        offset = len(self.pattern_log)

        match = buffer.find_previous_match(offset, r"ERROR:", max_lines=3)
        self.assertIsNotNone(match)  # Should find line 5.
        self.assertIn("Build failed", buffer.get_line(5))  # Verify it's line 5 ERROR.

        # Now search with max_lines=1 - should NOT find ERROR.
        # From line 6, max_lines=1 only searches line 5.
        # But line 5 has ERROR, so this WILL find it.
        # Let's search from line 3 with max_lines=1 instead.
        lines_before = self.pattern_log.split("\n")[:3]
        offset_line_3 = sum(len(line) + 1 for line in lines_before)

        match = buffer.find_previous_match(offset_line_3, r"ERROR:", max_lines=1)
        self.assertIsNone(match)  # Should NOT find line 1 (too far).

    def test_find_next_match_found(self):
        """Test forward search finding a match."""
        buffer = LogBuffer(self.pattern_log)

        # Find "error:" when searching from line 0.
        match = buffer.find_next_match(0, r"error:")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(), "error:")

    def test_find_next_match_not_found(self):
        """Test forward search not finding a match."""
        buffer = LogBuffer(self.pattern_log)

        # Search from last line for pattern that doesn't exist.
        offset = len(self.pattern_log)

        match = buffer.find_next_match(offset, r"NONEXISTENT")
        self.assertIsNone(match)

    def test_find_next_match_max_lines_limit(self):
        """Test forward search respects max_lines limit."""
        buffer = LogBuffer(self.pattern_log)

        # Search from line 0 for "Tests passed" with max_lines=2.
        # Should NOT find it (it's on line 6).
        match = buffer.find_next_match(0, r"Tests passed", max_lines=2)
        self.assertIsNone(match)

    def test_find_all_matches(self):
        """Test finding all matches in log."""
        buffer = LogBuffer(self.pattern_log)

        # Find all "ERROR:" occurrences.
        matches = buffer.find_all_matches(r"ERROR:")
        self.assertEqual(len(matches), 2)

        # Verify line numbers.
        line_nums = [line_num for line_num, _ in matches]
        self.assertEqual(line_nums, [1, 5])

    def test_find_all_matches_no_matches(self):
        """Test find_all_matches with no matches."""
        buffer = LogBuffer(self.simple_log)

        matches = buffer.find_all_matches(r"NONEXISTENT")
        self.assertEqual(len(matches), 0)

    def test_find_all_matches_with_groups(self):
        """Test find_all_matches with regex groups."""
        buffer = LogBuffer(self.pattern_log)

        # Find file:line:col patterns.
        matches = buffer.find_all_matches(r"(\w+)\.cpp:(\d+):(\d+):")
        self.assertEqual(len(matches), 1)

        line_num, match = matches[0]
        self.assertEqual(line_num, 2)
        self.assertEqual(match.group(1), "file")
        self.assertEqual(match.group(2), "42")
        self.assertEqual(match.group(3), "10")

    def test_tsan_log_context_extraction(self):
        """Test realistic TSAN log context extraction."""
        buffer = LogBuffer(self.tsan_log)

        # Find "WARNING: ThreadSanitizer" line.
        matches = buffer.find_all_matches(r"WARNING: ThreadSanitizer")
        self.assertEqual(len(matches), 1)

        line_num, _ = matches[0]
        context = buffer.get_lines_around(line_num, context=2)

        # Verify context includes separator and error details.
        self.assertTrue(any("==================" in line for line in context))
        self.assertTrue(any("data race" in line for line in context))

    def test_tsan_log_stack_trace_search(self):
        """Test searching for stack frames in TSAN log."""
        buffer = LogBuffer(self.tsan_log)

        # Find all stack frames (lines starting with #N).
        matches = buffer.find_all_matches(r"^\s+#\d+")
        self.assertEqual(len(matches), 4)  # 2 frames in each stack trace.

        # Verify first frame contains function name.
        line_num, _ = matches[0]
        line = buffer.get_line(line_num)
        self.assertIn("iree::hal::Device::Submit()", line)

    def test_empty_log_operations(self):
        """Test all operations on empty log."""
        buffer = LogBuffer(self.empty_log)

        self.assertEqual(buffer.line_count, 0)
        self.assertIsNone(buffer.get_line(0))  # Empty log has no lines.
        self.assertEqual(buffer.get_lines_around(0, context=5), [])
        # Empty log returns -1 for offset conversion (no lines exist).
        self.assertEqual(buffer.get_line_number_from_offset(0), -1)
        self.assertIsNone(buffer.find_previous_match(0, r"test"))
        self.assertIsNone(buffer.find_next_match(0, r"test"))
        self.assertEqual(len(buffer.find_all_matches(r"test")), 0)


if __name__ == "__main__":
    unittest.main()
