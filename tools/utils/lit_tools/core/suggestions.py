# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fuzzy matching utilities for user-friendly error messages.

This module provides utilities for suggesting similar names when a user-provided
identifier (like a test case name) cannot be found. Uses Python's difflib for
fuzzy string matching with configurable similarity thresholds.
"""

import difflib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lit_tools.core.parser import TestCase


def find_similar_case_names(
    target_name: str,
    all_cases: list["TestCase"],
    max_suggestions: int = 3,
    cutoff: float = 0.6,
) -> list[str]:
    """Find case names similar to target using fuzzy matching.

    Uses difflib.get_close_matches() to find case names that are similar to
    the target name. The similarity is based on the SequenceMatcher algorithm
    which compares sequences of characters.

    Args:
        target_name: The name that was not found.
        all_cases: List of all test cases to search.
        max_suggestions: Maximum number of suggestions to return (default: 3).
        cutoff: Minimum similarity ratio (0.0-1.0) to consider a match.
            Default 0.6 means 60% similarity. Lower values are more permissive.

    Returns:
        List of similar case names, sorted by similarity (most similar first).
        Empty list if no similar names found or if all_cases has no names.

    Example:
        >>> cases = [
        ...     TestCase(name="emplaceDispatch", ...),
        ...     TestCase(name="dontEmplaceTiedDispatch", ...),
        ...     TestCase(name="emplaceDispatchSequence", ...),
        ... ]
        >>> find_similar_case_names("emplceDispatch", cases)
        ['emplaceDispatch', 'emplaceDispatchSequence']
    """
    # Extract non-empty names from test cases.
    available_names = [c.name for c in all_cases if c.name]
    if not available_names:
        return []

    # difflib.get_close_matches uses SequenceMatcher with configurable cutoff.
    # It returns matches sorted by similarity (highest first).
    return difflib.get_close_matches(
        target_name, available_names, n=max_suggestions, cutoff=cutoff
    )


def format_case_name_error(
    target_name: str, all_cases: list["TestCase"], file_path: str | None = None
) -> str:
    """Format error message with suggestions for case name not found.

    Generates a user-friendly error message that includes suggestions for
    similar case names if any are found. The format adapts based on the
    number of suggestions found.

    Args:
        target_name: The name that was not found.
        all_cases: List of all test cases.
        file_path: Optional path to include in message.

    Returns:
        Formatted error message with suggestions.

    Examples:
        No suggestions:
            "Case with name 'xyz' not found"

        Single suggestion:
            "Case with name 'emplceDispatch' not found. Did you mean 'emplaceDispatch'?"

        Multiple suggestions:
            "Case with name 'emplace' not found. Did you mean one of: 'emplaceDispatch', 'emplaceDispatchSequence'?"

        With file path:
            "Case with name 'xyz' not found in test.mlir"
    """
    # Build base error message.
    base_msg = f"Case with name '{target_name}' not found"
    if file_path:
        base_msg += f" in {file_path}"

    # Try to find similar names.
    suggestions = find_similar_case_names(target_name, all_cases)
    if not suggestions:
        return base_msg

    # Format suggestions based on count.
    if len(suggestions) == 1:
        return f"{base_msg}. Did you mean '{suggestions[0]}'?"

    # Multiple suggestions - format as comma-separated list.
    suggestion_list = "', '".join(suggestions)
    return f"{base_msg}. Did you mean one of: '{suggestion_list}'?"


def format_case_number_error(
    requested_number: int, total_cases: int, file_path: str | None = None
) -> str:
    """Format error message for invalid case number.

    Generates a user-friendly error message when a case number is out of range,
    providing context about the valid range.

    Args:
        requested_number: The case number that was requested.
        total_cases: Total number of cases in the file.
        file_path: Optional path to include in message.

    Returns:
        Formatted error message with range information.

    Examples:
        "Case 999 not found (file has 10 cases)"
        "Case 0 not found in test.mlir (file has 5 cases)"
    """
    base_msg = f"Case {requested_number} not found"
    if file_path:
        base_msg += f" in {file_path}"
    base_msg += f" (file has {total_cases} case{'s' if total_cases != 1 else ''})"
    return base_msg
