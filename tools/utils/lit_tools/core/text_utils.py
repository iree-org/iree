# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cross-platform text normalization and comparison utilities for lit tools."""

import os
import re


def normalize_line_endings(text: str) -> str:
    r"""Normalize line endings to platform-specific format.

    Args:
        text: Text with any line ending style (\n, \r\n, \r)

    Returns:
        Text with platform-specific line endings (os.linesep)
    """
    # Normalize to \n first, then convert to platform line ending.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if os.linesep != "\n":
        text = text.replace("\n", os.linesep)
    return text


def split_lines(text: str) -> list[str]:
    r"""Split text into lines in a cross-platform way.

    Handles \n, \r\n, and \r line endings.

    Args:
        text: Text to split

    Returns:
        List of lines (without line ending characters)
    """
    return text.splitlines()


def join_lines(lines: list[str]) -> str:
    """Join lines using platform-specific line separator.

    Args:
        lines: List of lines to join

    Returns:
        Text with platform-specific line endings
    """
    return os.linesep.join(lines)


def strip_run_lines(content: str) -> str:
    """Strip RUN lines from lit test content.

    Removes all lines starting with '// RUN:' (with optional leading whitespace).

    Args:
        content: Lit test content

    Returns:
        Content with RUN lines removed
    """
    lines = split_lines(content)
    non_run_lines = [line for line in lines if not line.strip().startswith("// RUN:")]
    return join_lines(non_run_lines)


def normalize_whitespace_for_comparison(content: str) -> str:
    r"""Normalize whitespace for content comparison.

    - Strips leading/trailing whitespace from each line
    - Removes empty lines from start and end
    - Normalizes line endings to \n
    - Preserves internal blank lines and indentation structure

    Args:
        content: Text content to normalize

    Returns:
        Normalized content for comparison
    """
    lines = split_lines(content)

    # Strip leading empty lines.
    while lines and not lines[0].strip():
        lines.pop(0)

    # Strip trailing empty lines.
    while lines and not lines[-1].strip():
        lines.pop()

    # Use \n for comparison (platform-independent).
    return "\n".join(lines)


def compare_ir_content(
    content1: str, content2: str, ignore_run_lines: bool = True
) -> bool:
    """Compare two IR content strings for semantic equality.

    Comparison is done after:
    - Optionally stripping RUN lines
    - Normalizing whitespace
    - Normalizing line endings

    Args:
        content1: First content to compare
        content2: Second content to compare
        ignore_run_lines: If True, strip RUN lines before comparing

    Returns:
        True if contents are semantically equal
    """
    # Strip RUN lines if requested.
    if ignore_run_lines:
        content1 = strip_run_lines(content1)
        content2 = strip_run_lines(content2)

    # Normalize whitespace for comparison.
    norm1 = normalize_whitespace_for_comparison(content1)
    norm2 = normalize_whitespace_for_comparison(content2)

    return norm1 == norm2


def extract_run_lines_from_content(content: str) -> list[str]:
    """Extract RUN lines from content.

    Args:
        content: Content containing RUN lines

    Returns:
        List of RUN line text (without '// RUN:' prefix)
    """
    lines = split_lines(content)
    run_lines = []

    for line in lines:
        match = re.match(r"^\s*//\s*RUN:\s?(.*)$", line)
        if match:
            run_lines.append(match.group(1))

    return run_lines


def _strip_run_lines_preserve_line_numbers(content: str) -> str:
    """Replace RUN lines with blank lines to preserve line numbers.

    Handles multi-line RUN commands with backslash continuation.
    RUN lines can appear anywhere in content, not just at the top.

    Args:
        content: Test case content potentially containing RUN lines

    Returns:
        Content with RUN lines replaced by blank lines
    """
    lines = content.splitlines()
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        # Check if this is a RUN line.
        if re.match(r"^\s*//\s*RUN:", line):
            # Replace RUN line with blank line.
            result.append("")

            # Handle continuation lines (backslash at end).
            while i < len(lines) - 1 and lines[i].rstrip().endswith("\\"):
                i += 1
                result.append("")  # Replace continuation with blank.
        else:
            result.append(line)

        i += 1

    return "\n".join(result)


def inject_run_lines(content: str, run_lines: list[str]) -> str:
    """Inject RUN lines by replacing leading blank lines.

    The content from parse_test_file() has blank lines where RUN lines were
    stripped. When lit_wrapper prepends more blanks for line preservation,
    we have a pile of blanks at the top. Just fill the first N of them with
    RUN lines.

    Args:
        content: Test case content with leading blanks (from line preservation)
        run_lines: List of RUN commands (from extract_run_lines)

    Returns:
        Content with RUN lines injected at top
    """
    if not run_lines:
        return content

    lines = content.splitlines()

    # Replace first N lines with RUN lines.
    for i, cmd in enumerate(run_lines):
        if i < len(lines):
            lines[i] = f"// RUN: {cmd}"

    return "\n".join(lines)
