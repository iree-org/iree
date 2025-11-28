# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Parsing utilities for MLIR lit test files.

Lit test files use `// -----` delimiters to separate multiple test cases.
This module provides utilities to parse these files, extract metadata, and
split them into individual test cases.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestCase:
    """Represents a single test case in a lit test file.

    Attributes:
        number: 1-indexed test case number
        name: Function name from CHECK-LABEL (if present)
        content: Full content of the test case
        start_line: 1-indexed line number where test case starts
        end_line: 1-indexed line number where test case ends
        line_count: Total number of lines in test case
        check_count: Number of CHECK lines in test case
    """

    number: int
    name: str | None
    content: str
    start_line: int
    end_line: int
    line_count: int
    check_count: int


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
                result.append("")  # replace continuation with blank
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


def parse_test_file(file_path: Path) -> list[TestCase]:
    """Parses lit test file and returns list of test cases.

    Args:
        file_path: Path to lit test file

    Returns:
        List of TestCase objects

    Example:
        >>> cases = parse_test_file(Path('test.mlir'))
        >>> len(cases)
        3
        >>> cases[0].name
        'single_transient'
        >>> cases[0].line_count
        36
    """
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Split by delimiter (// -----)
    # Use regex to find delimiter lines (at least 5 dashes)
    delimiter_pattern = r"^//\s*-{5,}.*$"
    lines = content.splitlines()

    # Find delimiter positions
    delimiter_indices = []
    for i, line in enumerate(lines):
        if re.match(delimiter_pattern, line):
            delimiter_indices.append(i)

    # If no delimiters, entire file is one test case.
    if not delimiter_indices:
        case = _parse_single_case(content, case_number=1, start_line=1)
        case.content = _strip_run_lines_preserve_line_numbers(case.content)
        return [case]

    # Split into test cases
    test_cases = []
    case_number = 1

    # First case: from start to first delimiter
    start_idx = 0
    end_idx = delimiter_indices[0]
    case_content = "\n".join(lines[start_idx:end_idx])
    test_cases.append(
        _parse_single_case(case_content, case_number=case_number, start_line=1)
    )
    case_number += 1

    # Middle cases: between delimiters
    for i in range(len(delimiter_indices) - 1):
        start_idx = delimiter_indices[i] + 1  # Skip delimiter line
        end_idx = delimiter_indices[i + 1]
        start_line = start_idx + 1  # 1-indexed
        case_content = "\n".join(lines[start_idx:end_idx])
        test_cases.append(
            _parse_single_case(
                case_content, case_number=case_number, start_line=start_line
            )
        )
        case_number += 1

    # Last case: from last delimiter to end
    start_idx = delimiter_indices[-1] + 1
    start_line = start_idx + 1
    case_content = "\n".join(lines[start_idx:])
    if case_content.strip():  # Only add if not empty
        test_cases.append(
            _parse_single_case(
                case_content, case_number=case_number, start_line=start_line
            )
        )

    # Strip RUN lines from all cases (preserving line numbers).
    for case in test_cases:
        case.content = _strip_run_lines_preserve_line_numbers(case.content)

    return test_cases


def _parse_single_case(content: str, case_number: int, start_line: int) -> TestCase:
    """Parses metadata for a single test case.

    Args:
        content: Test case content
        case_number: 1-indexed test case number
        start_line: 1-indexed starting line number

    Returns:
        TestCase object with extracted metadata
    """
    # Extract function name from CHECK-LABEL
    # Patterns to match:
    #   // CHECK-LABEL: @function_name
    #   // CHECK-LABEL: func @function_name
    #   // CHECK-LABEL: util.func @function_name
    name = None
    check_label_pattern = r"//\s*CHECK-LABEL:.*@([A-Za-z0-9_.$-]+)"
    match = re.search(check_label_pattern, content)
    if match:
        name = match.group(1)

    # Count lines
    lines = content.splitlines()
    line_count = len(lines)
    end_line = start_line + line_count - 1

    # Count CHECK lines (any line with // CHECK)
    check_pattern = r"^\s*//\s*CHECK"
    check_count = sum(1 for line in lines if re.match(check_pattern, line))

    return TestCase(
        number=case_number,
        name=name,
        content=content,
        start_line=start_line,
        end_line=end_line,
        line_count=line_count,
        check_count=check_count,
    )


def extract_case_by_number(file_path: Path, case_number: int) -> TestCase:
    """Extracts a specific test case by number.

    Args:
        file_path: Path to lit test file
        case_number: 1-indexed test case number

    Returns:
        TestCase object

    Raises:
        ValueError: If case_number is out of range

    Example:
        >>> case = extract_case_by_number(Path('test.mlir'), 2)
        >>> case.name
        'multiple_transients'
    """
    cases = parse_test_file(file_path)
    if case_number < 1 or case_number > len(cases):
        raise ValueError(f"Case number {case_number} out of range (1-{len(cases)})")
    return cases[case_number - 1]


def extract_case_by_name(file_path: Path, name: str) -> TestCase:
    """Extracts a specific test case by function name.

    Args:
        file_path: Path to lit test file
        name: Function name (without @ prefix)

    Returns:
        TestCase object

    Raises:
        ValueError: If no test case with given name found

    Example:
        >>> case = extract_case_by_name(Path('test.mlir'), 'single_transient')
        >>> case.number
        1
    """
    cases = parse_test_file(file_path)
    for case in cases:
        if case.name == name:
            return case

    # Try with @ prefix if user included it
    if name.startswith("@"):
        name_without_at = name[1:]
        for case in cases:
            if case.name == name_without_at:
                return case

    raise ValueError(f"No test case found with name '{name}'")


def extract_case_by_line_number(file_path: Path, line_number: int) -> TestCase:
    """Extracts the test case containing a specific line number.

    Useful for extracting test case from error messages that reference line numbers.

    Args:
        file_path: Path to lit test file
        line_number: Line number (1-indexed) within the test file

    Returns:
        TestCase object containing the specified line

    Raises:
        ValueError: If line number is out of range or invalid

    Example:
        >>> # Error at line 142 - which test case is that?
        >>> case = extract_case_by_line_number(Path('test.mlir'), 142)
        >>> case.name
        'nested_execute'
        >>> case.number
        3
    """
    cases = parse_test_file(file_path)

    if line_number < 1:
        raise ValueError(f"Line number must be >= 1, got {line_number}")

    for case in cases:
        if case.start_line <= line_number <= case.end_line:
            return case

    # Line number out of range
    if cases:
        last_line = cases[-1].end_line
        raise ValueError(
            f"Line number {line_number} out of range (file has {last_line} lines)"
        )
    raise ValueError(f"Line number {line_number} out of range (empty file)")


def extract_run_lines(file_path: Path, raw: bool = False) -> list[str]:
    """Extracts RUN lines from lit test file.

    RUN lines appear at the top of the file before test cases. This function
    extracts all RUN lines, handling multi-line continuations with backslash.

    Args:
        file_path: Path to lit test file
        raw: If True, return original lines with "// RUN:" prefix and
            backslash continuations preserved. If False (default), join
            continuations into single command strings.

    Returns:
        List of RUN commands. Format depends on raw parameter:
        - raw=False: Commands with continuations joined
        - raw=True: Original lines including "// RUN:" prefix

    Example:
        >>> run_lines = extract_run_lines(Path('test.mlir'))
        >>> run_lines[0]
        'iree-opt --split-input-file --pass-pipeline=... %s | FileCheck %s'

        >>> raw_lines = extract_run_lines(Path('test.mlir'), raw=True)
        >>> raw_lines[0]
        '// RUN: iree-opt --split-input-file \\'
    """
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    if raw:
        # Raw mode: preserve original lines exactly
        raw_run_lines = []
        for line in lines:
            line_stripped = line.rstrip()
            # Check if this is a RUN line
            if re.match(r"^\s*//\s*RUN:", line_stripped):
                raw_run_lines.append(line_stripped)
            # Stop at first non-comment, non-empty line
            elif line_stripped and not re.match(r"^\s*//", line_stripped):
                break
        return raw_run_lines

    # Normal mode: join continuations into single commands
    run_lines = []
    current_run = None
    continuing = False

    for line in lines:
        line = line.rstrip()

        # Check if this is a RUN line
        # Accept both "// RUN:" and "//RUN:" and tolerate leading spaces.
        m = re.match(r"^\s*//\s*RUN:\s?(.*)$", line)
        if m:
            content = m.group(1).strip()

            if continuing and current_run is not None:
                # This is a continuation of previous RUN line
                # Remove trailing backslash from previous part
                current_run = current_run.rstrip("\\").strip()
                # Append this continuation
                current_run += " " + content.lstrip("\\").strip()
            else:
                # Start new RUN command
                current_run = content

            # Check if this line continues
            continuing = content.endswith("\\")

            if not continuing:
                # Command is complete, save it
                run_lines.append(current_run.rstrip("\\").strip())
                current_run = None

        # Stop at first non-comment, non-empty line
        elif line and not re.match(r"^\s*//", line):
            break

    # Save any pending multi-line command
    if current_run is not None:
        run_lines.append(current_run.rstrip("\\").strip())

    return run_lines


def extract_case_run_lines(file_path: Path, case: TestCase) -> list[tuple[int, str]]:
    """Extracts RUN lines that appear inside a specific test case body.

    Returns a list of (absolute_line_index, command) entries, where
    absolute_line_index is the zero-based index within the synthesized content
    that starts with ``(case.start_line - 1)`` blank lines followed by the
    case content. This matches how lit_wrapper constructs the temp shard.
    """
    with open(file_path, encoding="utf-8") as f:
        all_lines = f.read().splitlines()

    # Convert case line range (1-based inclusive) to 0-based indices
    start = case.start_line - 1
    end = case.end_line - 1
    out = []
    i = start
    while i <= end:
        line = all_lines[i]
        m = re.match(r"^\s*//\s*RUN:\s?(.*)$", line.rstrip())
        if m:
            content = m.group(1).strip()
            # Gather continued lines and remember the first line index
            first_i = i
            cmd = content
            while i < end and all_lines[i].rstrip().endswith("\\"):
                i += 1
                cont = re.match(r"^\s*//\s*RUN:\s?(.*)$", all_lines[i].rstrip())
                if cont:
                    cmd = (
                        cmd.rstrip("\\").strip()
                        + " "
                        + cont.group(1).lstrip("\\").strip()
                    )
                else:
                    break
            # Absolute index is blank_prefix + relative index within case
            abs_index = (case.start_line - 1) + (first_i - start)
            out.append((abs_index, cmd.rstrip("\\").strip()))
        i += 1
    return out


def inject_run_lines_with_case(
    content: str, header_runs: list[str], case_runs: list[tuple[int, str]]
) -> str:
    """Injects RUN lines for both header and in-body case RUN commands.

    - Replaces the first N lines with header RUN commands.
    - Replaces specific relative lines (within the case body) with RUN commands.

    Raises:
        ValueError: If case_runs indices are out of range for the content.
                    This can happen if replacement content has different structure
                    than the original (different number of lines or blank padding).

    Note:
        iree-lit-replace uses top-injection for case-local RUNs (not this function),
        so this error primarily affects iree-lit-test's lit_wrapper.
    """
    lines = content.splitlines()
    # Header runs at the very top
    for i, cmd in enumerate(header_runs):
        if i < len(lines):
            lines[i] = f"// RUN: {cmd}"
        else:
            lines.append(f"// RUN: {cmd}")

    # Case-local runs within the case region: absolute positions in synthesized content
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
