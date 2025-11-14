# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for LogBuffer prefix stripping functionality.

Tests cover:
- GitHub Actions timestamp prefix detection and stripping
- ctest/cmake prefix stripping
- Auto-format detection
- Backward compatibility (no stripping by default)
- Original content preservation
- Edge cases (BOM, mixed prefixes, empty logs)
"""

import unittest

from common.log_buffer import LogBuffer


class TestGitHubActionsStripping(unittest.TestCase):
    """Test GitHub Actions log format handling."""

    def test_github_actions_auto_detection(self):
        """Test auto-detection of GitHub Actions format."""
        log_content = (
            "linux_x64_gcc\tUNKNOWN STEP\t2025-09-16T09:23:05.2141216Z Starting build\n"
            "linux_x64_gcc\tUNKNOWN STEP\t2025-09-16T09:23:06.1234567Z Running tests\n"
        )

        buffer = LogBuffer(log_content, auto_detect_format=True)

        # Verify prefix was stripped.
        self.assertEqual(buffer.get_line(0), "Starting build")
        self.assertEqual(buffer.get_line(1), "Running tests")

    def test_github_actions_explicit_stripping(self):
        """Test explicit GitHub Actions format stripping."""
        log_content = (
            "Test Job / task\tUNKNOWN STEP\t2025-11-13T16:46:34.2164128Z Content here\n"
        )

        buffer = LogBuffer(log_content, strip_formats=["github_actions"])

        self.assertEqual(buffer.get_line(0), "Content here")

    def test_github_actions_complex_job_name(self):
        """Test GitHub Actions with complex job names (slashes, spaces)."""
        log_content = (
            "linux_x64_bazel / linux_x64_bazel\tUNKNOWN STEP\t"
            "2025-11-13T16:46:34.9136874Z Building targets\n"
        )

        buffer = LogBuffer(log_content, auto_detect_format=True)

        self.assertEqual(buffer.get_line(0), "Building targets")

    def test_github_actions_with_bom(self):
        """Test GitHub Actions format with BOM character on first line."""
        # BOM character (\ufeff) can appear before timestamp.
        log_content = (
            "job\tstep\t\ufeff2025-09-16T09:23:05.2141216Z First line with BOM\n"
            "job\tstep\t2025-09-16T09:23:06.0000000Z Second line normal\n"
        )

        buffer = LogBuffer(log_content, auto_detect_format=True)

        # Both lines should be stripped (BOM is part of prefix).
        self.assertEqual(buffer.get_line(0), "First line with BOM")
        self.assertEqual(buffer.get_line(1), "Second line normal")

    def test_github_actions_bom_on_all_lines(self):
        """Test BOM character on every line (real corpus pattern).

        Analysis of 18 corpus logs shows BOM appears on EVERY line, not just
        the first line. This is the actual format from GitHub Actions.
        """
        log_content = (
            "job\tstep\t\ufeff2025-09-16T09:23:05.0000000Z Line 1\n"
            "job\tstep\t\ufeff2025-09-16T09:23:06.0000000Z Line 2\n"
            "job\tstep\t\ufeff2025-09-16T09:23:07.0000000Z Line 3\n"
        )

        buffer = LogBuffer(log_content, auto_detect_format=True)

        # All lines should strip BOM correctly.
        self.assertEqual(buffer.get_line(0), "Line 1")
        self.assertEqual(buffer.get_line(1), "Line 2")
        self.assertEqual(buffer.get_line(2), "Line 3")

    def test_github_actions_mixed_lines(self):
        """Test log with some lines having prefix, some without."""
        log_content = (
            "job\tstep\t2025-09-16T09:23:05.0000000Z Prefixed line\n"
            "Raw line without prefix\n"
            "job\tstep\t2025-09-16T09:23:07.0000000Z Another prefixed line\n"
        )

        buffer = LogBuffer(log_content, auto_detect_format=True)

        # Prefixed lines stripped, unprefixed preserved.
        self.assertEqual(buffer.get_line(0), "Prefixed line")
        self.assertEqual(buffer.get_line(1), "Raw line without prefix")
        self.assertEqual(buffer.get_line(2), "Another prefixed line")


class TestCtestCmakeStripping(unittest.TestCase):
    """Test ctest and cmake prefix handling."""

    def test_ctest_prefix_stripping(self):
        """Test stripping of [ctest] prefix."""
        log_content = (
            "[ctest] Running test suite\n" "[ctest] Test passed\n" "Raw output line\n"
        )

        buffer = LogBuffer(log_content, strip_formats=["ctest"])

        self.assertEqual(buffer.get_line(0), "Running test suite")
        self.assertEqual(buffer.get_line(1), "Test passed")
        self.assertEqual(buffer.get_line(2), "Raw output line")

    def test_cmake_prefix_stripping(self):
        """Test stripping of [cmake] and [build] prefixes."""
        log_content = (
            "[cmake] Configuring project\n"
            "[build] Building target foo\n"
            "[cmake] Build complete\n"
        )

        buffer = LogBuffer(log_content, strip_formats=["cmake"])

        self.assertEqual(buffer.get_line(0), "Configuring project")
        self.assertEqual(buffer.get_line(1), "Building target foo")
        self.assertEqual(buffer.get_line(2), "Build complete")

    def test_multiple_format_stripping(self):
        """Test stripping multiple formats in same log."""
        log_content = "[ctest] Test output\n" "[cmake] Build message\n" "Raw line\n"

        buffer = LogBuffer(log_content, strip_formats=["ctest", "cmake"])

        self.assertEqual(buffer.get_line(0), "Test output")
        self.assertEqual(buffer.get_line(1), "Build message")
        self.assertEqual(buffer.get_line(2), "Raw line")


class TestBackwardCompatibility(unittest.TestCase):
    """Test that default behavior is unchanged."""

    def test_no_stripping_by_default(self):
        """Test that logs are not stripped by default."""
        log_content = (
            "job\tstep\t2025-09-16T09:23:05.0000000Z Prefixed line\n"
            "[ctest] Test line\n"
        )

        # Default: no stripping.
        buffer = LogBuffer(log_content)

        # Original content preserved.
        self.assertIn("2025-09-16T09:23:05.0000000Z", buffer.get_line(0))
        self.assertIn("[ctest]", buffer.get_line(1))

    def test_empty_strip_formats_list(self):
        """Test that empty strip_formats list does nothing."""
        log_content = "job\tstep\t2025-09-16T09:23:05.0000000Z Content\n"

        buffer = LogBuffer(log_content, strip_formats=[])

        # No stripping with empty list.
        self.assertIn("2025-09-16T09:23:05.0000000Z", buffer.get_line(0))


class TestOriginalContentPreservation(unittest.TestCase):
    """Test that original content is preserved."""

    def test_get_original_content(self):
        """Test accessing original unstripped content."""
        original = (
            "job\tstep\t2025-09-16T09:23:05.0000000Z Line 1\n"
            "job\tstep\t2025-09-16T09:23:06.0000000Z Line 2\n"
        )

        buffer = LogBuffer(original, auto_detect_format=True)

        # Stripped content different from original.
        self.assertEqual(buffer.get_line(0), "Line 1")
        self.assertNotIn("2025-09-16", buffer.content)

        # Original content preserved.
        self.assertEqual(buffer.get_original_content(), original)
        self.assertIn("2025-09-16T09:23:05.0000000Z", buffer.get_original_content())


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_log(self):
        """Test stripping on empty log."""
        buffer = LogBuffer("", auto_detect_format=True)

        self.assertEqual(buffer.line_count, 0)
        self.assertEqual(buffer.content, "")
        self.assertEqual(buffer.get_original_content(), "")

    def test_log_with_only_newlines(self):
        """Test log with only newlines."""
        buffer = LogBuffer("\n\n\n", auto_detect_format=True)

        self.assertEqual(buffer.line_count, 3)
        self.assertEqual(buffer.get_line(0), "")
        self.assertEqual(buffer.get_line(1), "")

    def test_no_format_detected(self):
        """Test log where auto-detection finds no known format."""
        log_content = "Plain log line 1\nPlain log line 2"

        buffer = LogBuffer(log_content, auto_detect_format=True)

        # No format detected, content unchanged.
        self.assertEqual(buffer.get_line(0), "Plain log line 1")
        self.assertEqual(buffer.content, "Plain log line 1\nPlain log line 2")

    def test_line_offsets_after_stripping(self):
        """Test that line offsets are computed correctly after stripping."""
        log_content = (
            "job\tstep\t2025-09-16T09:23:05.0000000Z Line 1\n"
            "job\tstep\t2025-09-16T09:23:06.0000000Z Line 2\n"
        )

        buffer = LogBuffer(log_content, auto_detect_format=True)

        # Line offsets should correspond to stripped content.
        offset_line_0 = 0
        offset_line_1 = len("Line 1\n")

        self.assertEqual(buffer.get_line_number_from_offset(offset_line_0), 0)
        self.assertEqual(buffer.get_line_number_from_offset(offset_line_1), 1)

    def test_pattern_matching_on_stripped_content(self):
        """Test that pattern matching works on stripped content."""
        log_content = (
            "job\tstep\t2025-09-16T09:23:05.0000000Z ERROR: Something failed\n"
            "job\tstep\t2025-09-16T09:23:06.0000000Z Normal output\n"
        )

        buffer = LogBuffer(log_content, auto_detect_format=True)

        # Search for ERROR in stripped content.
        matches = buffer.find_all_matches(r"^ERROR:")

        self.assertEqual(len(matches), 1)
        line_num, _ = matches[0]
        self.assertEqual(line_num, 0)
        self.assertEqual(buffer.get_line(line_num), "ERROR: Something failed")

    def test_context_extraction_with_stripping(self):
        """Test get_lines_around works correctly with stripped content."""
        log_content = (
            "job\tstep\t2025-09-16T09:23:01.0000000Z Line 0\n"
            "job\tstep\t2025-09-16T09:23:02.0000000Z Line 1\n"
            "job\tstep\t2025-09-16T09:23:03.0000000Z ERROR HERE\n"
            "job\tstep\t2025-09-16T09:23:04.0000000Z Line 3\n"
            "job\tstep\t2025-09-16T09:23:05.0000000Z Line 4\n"
        )

        buffer = LogBuffer(log_content, auto_detect_format=True)

        # Get context around error line.
        context = buffer.get_lines_around(2, context=1)

        self.assertEqual(len(context), 3)
        self.assertEqual(context[0], "Line 1")
        self.assertEqual(context[1], "ERROR HERE")
        self.assertEqual(context[2], "Line 3")


if __name__ == "__main__":
    unittest.main()
