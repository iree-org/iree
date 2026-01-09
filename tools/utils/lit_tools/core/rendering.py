# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rendering utilities for lit test files.

This module provides functions for rebuilding complete test files
from parsed TestFile objects, with support for both line-preserving
and normalized output modes.
"""

from .parser import TAG_RUN_HEADER, TestFile


def build_file_content(test_file: TestFile, normalize: bool = False) -> str:
    """Build complete file content from parsed TestFile.

    This is the main function for rebuilding lit test files after
    transformations. It handles:
    - Header RUN lines (always preserved at top)
    - Case separation with // ----- delimiters
    - Case rendering (line-preserving or normalized)

    Args:
        test_file: Parsed test file with structure
        normalize: If True, use normalized rendering (drop RUNs, trim blanks)
                   If False, preserve original structure

    Returns:
        Complete file content as string

    Example (normalize=False):
        // RUN: iree-opt %s | FileCheck %s
        // CHECK-LABEL: @foo
        func @foo() { }

        // -----

        // CHECK-LABEL: @bar
        func @bar() { }

    Example (normalize=True):
        // RUN: iree-opt %s | FileCheck %s
        // CHECK-LABEL: @foo
        func @foo() { }

        // -----

        // CHECK-LABEL: @bar
        func @bar() { }
    """
    result_parts = []

    # Render header RUN lines first (always at top of file).
    # For multi-case files, extract from header_span.
    # For single-case files, scan entire doc for RUN_HEADER tags.
    header_run_lines = []
    if test_file.header_span:
        header_lines = test_file.doc.slice(test_file.header_span)
        for line in header_lines:
            if TAG_RUN_HEADER in line.tags:
                header_run_lines.append(line.get_full_line())
    else:
        # Single-case file: scan all lines for RUN_HEADER.
        for line in test_file.doc.lines:
            if TAG_RUN_HEADER in line.tags:
                header_run_lines.append(line.get_full_line())

    if header_run_lines:
        result_parts.append("".join(header_run_lines))

    # Render each case.
    for idx, case in enumerate(test_file.cases):
        # Add delimiter before each case (except first).
        if idx > 0:
            if normalize:
                # Normalized mode: consistent spacing (2 newlines before/after).
                result_parts.append("\n\n// -----\n\n")
            else:
                # Preserving mode: just the delimiter, no added spacing.
                # Original spacing is preserved in case content (leading/trailing lines).
                result_parts.append("// -----\n")

        # Render case content.
        content = case.render_normalized() if normalize else case.render_for_testing()

        if content:  # Skip empty cases.
            result_parts.append(content)

    # Join all parts.
    full_content = "".join(result_parts)

    # Ensure file ends with newline.
    if not full_content.endswith("\n"):
        full_content += "\n"

    return full_content


def inject_run_lines_with_case(
    content: str, header_runs: list[str], case_runs: list[tuple[int, str]]
) -> str:
    """Injects RUN lines for both header and in-body case RUN commands.

    - Replaces the first N lines with header RUN commands.
    - Replaces specific relative lines (within the case body) with RUN commands.

    Args:
        content: Test case content (typically with blank lines where RUNs should go)
        header_runs: List of header RUN command strings (without // RUN: prefix)
        case_runs: List of (absolute_line_index, command) tuples for case-local RUNs

    Returns:
        Content with RUN lines injected

    Raises:
        ValueError: If case_runs indices are out of range for the content.
                    This can happen if replacement content has different structure
                    than the original (different number of lines or blank padding).

    Note:
        iree-lit-replace uses top-injection for case-local RUNs (not this function),
        so this error primarily affects iree-lit-test's lit_wrapper.
    """
    lines = content.splitlines()
    # Header runs at the very top.
    for i, cmd in enumerate(header_runs):
        if i < len(lines):
            lines[i] = f"// RUN: {cmd}"
        else:
            lines.append(f"// RUN: {cmd}")

    # Case-local runs within the case region: absolute positions in synthesized content.
    # Strict mode: error if indices go out of range due to content structure changes.
    for abs_index, cmd in case_runs:
        idx = abs_index
        if not (0 <= idx < len(lines)):
            raise ValueError(
                f"Cannot inject RUN line at index {idx}: content only has {len(lines)} lines.\n"
                f"This typically happens when replacement content has different structure than original.\n"
                f"Solutions:\n"
                f"  1. Use --replace-run-lines to replace RUN lines explicitly\n"
                f"  2. Adjust replacement content to match original line structure\n"
                f"  3. Remove RUN lines from replacement (they'll be re-added from original)"
            )
        lines[idx] = f"// RUN: {cmd}"
    return "\n".join(lines)
