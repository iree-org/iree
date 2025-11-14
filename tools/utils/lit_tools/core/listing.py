# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared listing helpers for lit tools.

Provides common JSON payload construction and text formatting so tools like
`iree-lit-list` and `iree-lit-extract --list` emit identical output.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003

from common import formatting

from lit_tools.core.parser import TestCase, parse_test_file


def get_cases(file_path: Path) -> list[TestCase]:
    """Parse test file and return test cases.

    Args:
        file_path: Path to test file

    Returns:
        List of parsed test cases
    """
    return list(parse_test_file(file_path).cases)


def build_json_payload(file_path: Path, cases: list[TestCase]) -> dict:
    """Build JSON payload for test case listing.

    Args:
        file_path: Path to test file
        cases: List of test cases

    Returns:
        Dictionary suitable for JSON serialization
    """
    return {
        "file": str(file_path),
        "count": len(cases),
        "cases": [
            {
                "number": c.number,
                "name": c.name,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "line_count": c.line_count,
                "check_count": c.check_count,
            }
            for c in cases
        ],
    }


def format_text_listing(
    file_path: Path,
    cases: list[TestCase],
    *,
    pretty: bool = False,
    header: bool = True,
) -> str:
    """Format test cases as human-readable text listing.

    Args:
        file_path: Path to test file
        cases: List of test cases
        pretty: Enable colorized output
        header: Include header line with file name and count

    Returns:
        Formatted text listing
    """
    header_line = (
        f"{file_path.name}: {len(cases)} test case{'s' if len(cases) != 1 else ''}"
    )
    if pretty:
        header_line = formatting.color("1;33", header_line, True)  # bold yellow
    lines = [header_line] if header else []
    for case in cases:
        name = f"@{case.name}" if case.name else "(unnamed)"
        line_range = f"lines {case.start_line}-{case.end_line}"
        line_count_str = f"{case.line_count} line{'s' if case.line_count != 1 else ''}"
        check_count_str = (
            f"{case.check_count} CHECK line{'s' if case.check_count != 1 else ''}"
        )
        number = f"{case.number:>2}"
        if pretty:
            number = formatting.color("1;36", number, True)
            name_fmt = formatting.color("1", f"{name:30s}", True)
            meta = formatting.color(
                "2;37", f"({line_range}, {line_count_str:4s}, {check_count_str})", True
            )
            lines.append(f"  {number}: {name_fmt} {meta}")
        else:
            lines.append(
                f"  {case.number}: {name:30s} ({line_range}, {line_count_str:4s}, {check_count_str})"
            )
    return "\n".join(lines)


def format_names(cases: list[TestCase]) -> str:
    """Format test case names as space-separated string.

    Args:
        cases: List of test cases

    Returns:
        Space-separated names (e.g., "@foo @bar case3")
    """
    names = [f"@{c.name}" if c.name else f"case{c.number}" for c in cases]
    return " ".join(names)


def count_cases(cases: list[TestCase]) -> int:
    """Count test cases.

    Args:
        cases: List of test cases

    Returns:
        Number of cases
    """
    return len(cases)
