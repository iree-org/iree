# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lit-specific CLI helpers for lit tools.

Provides test case selection and filtering specific to lit test files.
Generic CLI helpers are in common.cli.
"""

from __future__ import annotations

import argparse  # noqa: TCH003
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from lit_tools.core.parser import TestCase, TestFile

from common import cli as common_cli
from common import console, exit_codes

from lit_tools.core import suggestions
from lit_tools.core.parser import parse_test_file

# Re-export generic CLI helpers from common module.
add_common_output_flags = common_cli.add_common_output_flags
require_exactly_one = common_cli.require_exactly_one
require_at_most_one = common_cli.require_at_most_one
parse_case_numbers = common_cli.parse_case_numbers


def load_and_parse_test_file(
    file_path: Path, args: argparse.Namespace
) -> tuple[TestFile | None, list[TestCase], int]:
    """Load and parse a lit test file with standardized error handling.

    This helper consolidates the common pattern of parsing test files with
    consistent error handling across all lit tools.

    Args:
        file_path: Path to the test file to parse
        args: Parsed arguments for error reporting (uses --quiet, --json)

    Returns:
        Tuple of (test_file, cases, exit_code)
        - test_file: Parsed TestFile object, or None if parsing failed
        - cases: List of test cases (empty if parsing failed)
        - exit_code: exit_codes.SUCCESS if successful, otherwise error code

    Example:
        >>> test_file, cases, exit_code = load_and_parse_test_file(file_path, args)
        >>> if exit_code != exit_codes.SUCCESS:
        ...     return exit_code
        >>> # Proceed with cases...

    Note:
        This function prints error messages to stderr using console.error(),
        so callers should not print duplicate errors.
    """
    try:
        test_file = parse_test_file(file_path)
        all_cases = list(test_file.cases)
        return (test_file, all_cases, exit_codes.SUCCESS)
    except FileNotFoundError:
        console.error(f"File not found: {file_path}", args=args)
        return (None, [], exit_codes.NOT_FOUND)
    except (OSError, UnicodeDecodeError, ValueError) as e:
        console.error(f"Failed to parse test file: {e}", args=args)
        return (None, [], exit_codes.ERROR)


def suggest_common_mistakes(argv: list[str]) -> str | None:
    """Check for common CLI mistakes and suggest corrections.

    Used by lit tools to detect when users pass case numbers or names as
    positional arguments instead of using the proper flags (--case, --name).
    This is integrated with run_with_argparse_suggestions() to show helpful
    error messages.

    Args:
        argv: Command line arguments (sys.argv)

    Returns:
        Suggestion string if a common mistake is detected, None otherwise

    Examples:
        >>> suggest_common_mistakes(["iree-lit-extract", "test.mlir", "2"])
        'Did you mean: iree-lit-extract test.mlir --case 2'

        >>> suggest_common_mistakes(["iree-lit-test", "test.mlir", "my_func"])
        'Did you mean: iree-lit-test test.mlir --name my_func'
    """
    if len(argv) < 2:
        return None

    # Check if last argument looks like a case number without --case flag.
    last_arg = argv[-1]
    if last_arg.isdigit():
        tool_name = Path(argv[0]).name
        file_arg = argv[-2] if len(argv) >= 3 else "FILE"
        return f"Did you mean: {tool_name} {file_arg} --case {last_arg}"

    # Check if there's an unquoted string that looks like a name without --name flag.
    if (
        len(argv) >= 3
        and not argv[-1].startswith("-")
        and not Path(argv[-1]).exists()
        and not argv[-1].endswith(".mlir")
    ):
        tool_name = Path(argv[0]).name
        file_arg = argv[-2] if len(argv) >= 3 else "FILE"
        return f"Did you mean: {tool_name} {file_arg} --name {last_arg}"

    return None


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add standardized test case selection arguments.

    Used by tools that operate on individual test cases (iree-lit-test,
    iree-lit-extract, iree-lit-lint). Provides consistent selection interface.

    Args:
        parser: ArgumentParser to add selection arguments to
    """
    selection = parser.add_argument_group("Test case selection")
    selection.add_argument(
        "--case",
        "-c",
        action="append",
        metavar="N[,N...]",
        help="Select case number(s): --case 1, --case 1,3,5, --case 1-3",
    )
    selection.add_argument(
        "--name",
        "-n",
        metavar="NAME",
        help="Select case with specific function name",
    )
    selection.add_argument(
        "--containing",
        type=int,
        metavar="LINE",
        help="Select case containing line number",
    )
    selection.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available test cases and exit",
    )


def add_filter_arguments(parser: argparse.ArgumentParser) -> None:
    """Add standardized --filter and --filter-out arguments.

    Used by tools that support regex filtering of test cases by name.
    Filters apply only when explicit selectors are not used.
    """
    parser.add_argument(
        "--filter",
        type=str,
        metavar="REGEX",
        help="Only include cases with name matching REGEX (when not using explicit selectors)",
    )
    parser.add_argument(
        "--filter-out",
        type=str,
        metavar="REGEX",
        help="Exclude cases with name matching REGEX (when not using explicit selectors)",
    )


def select_cases(
    cases: list[TestCase], args: argparse.Namespace
) -> list[TestCase] | None:
    """Select test cases based on command-line arguments.

    Handles --case, --name, --containing selection. If none specified,
    returns all cases (for use with --filter).

    Args:
        cases: All test cases from file
        args: Parsed arguments with selection flags

    Returns:
        Selected cases, or None if selection failed (error already printed).
        None signals error; caller should return exit_codes.NOT_FOUND.

    Note:
        This function provides the core selection logic shared across all lit tools.
        It handles explicit selectors (--case, --name, --containing) but not regex
        filters (--filter, --filter-out), which should be applied afterward using
        apply_filters().

    Examples:
        >>> cases = parse_test_file("test.mlir")
        >>> # Select by case number
        >>> args = Namespace(case=["2"], name=None, containing=None)
        >>> selected = select_cases(cases, args)
        >>>
        >>> # Select by name
        >>> args = Namespace(case=None, name="fold_op", containing=None)
        >>> selected = select_cases(cases, args)
        >>>
        >>> # No selector - return all for filtering
        >>> args = Namespace(case=None, name=None, containing=None)
        >>> selected = select_cases(cases, args)  # Returns all cases
    """
    # --containing: select by line number.
    if args.containing is not None:
        for case in cases:
            if case.start_line <= args.containing <= case.end_line:
                return [case]
        console.error(f"No case contains line {args.containing}", args=args)
        return None

    # --case: select by case number(s).
    if args.case:
        try:
            case_numbers = parse_case_numbers(args.case)
        except ValueError as e:
            console.error(str(e), args=args)
            return None

        selected = []
        for num in case_numbers:
            found = False
            for case in cases:
                if case.number == num:
                    selected.append(case)
                    found = True
                    break
            if not found:
                console.error(
                    f"Case {num} not found (file has {len(cases)} cases)", args=args
                )
                return None
        return selected

    # --name: select by function name.
    if args.name:
        for case in cases:
            if case.name == args.name:
                return [case]
        error_msg = suggestions.format_case_name_error(args.name, cases)
        console.error(error_msg, args=args)
        return None

    # No explicit selector - return all (for --filter usage).
    return cases


def apply_filters(
    cases: list[TestCase],
    filter_pattern: str | None,
    filter_out_pattern: str | None,
    args: argparse.Namespace | None = None,
) -> list[TestCase] | None:
    """Apply regex filters to test cases by name.

    Filters by case.name field (from CHECK-LABEL comments). Cases without names
    are excluded when filters are active.

    Args:
        cases: Test cases to filter
        filter_pattern: Regex pattern to include (or None to include all)
        filter_out_pattern: Regex pattern to exclude (or None to exclude none)
        args: Parsed arguments for error reporting (optional)

    Returns:
        Filtered list of cases, or None if no cases match (error already printed).
        None signals error condition; caller should return exit_codes.NOT_FOUND.

    Note:
        Both filters can be applied together. Positive filter applied first, then
        negative filter. This allows patterns like: --filter "op" --filter-out "gpu"

    Examples:
        >>> cases = [Case(name="fold_op"), Case(name="gpu_fold"), Case(name="add")]
        >>> apply_filters(cases, "fold", None)
        [Case(name="fold_op"), Case(name="gpu_fold")]
        >>> apply_filters(cases, "fold", "gpu")
        [Case(name="fold_op")]
    """
    filtered = cases

    # Apply positive filter (include only matching).
    if filter_pattern:
        pattern = re.compile(filter_pattern)
        filtered = [c for c in filtered if (c.name and pattern.search(c.name))]
        if not filtered:
            console.error("No cases matched --filter", args=args)
            return None

    # Apply negative filter (exclude matching).
    if filter_out_pattern:
        pattern = re.compile(filter_out_pattern)
        filtered = [c for c in filtered if not (c.name and pattern.search(c.name))]
        if not filtered:
            console.error("All cases excluded by --filter-out", args=args)
            return None

    return filtered


def run_with_argparse_suggestions(
    parse_fn: Callable[[], argparse.Namespace],
    main_fn: Callable[[argparse.Namespace], int],
    suggest_fn: Callable[[list[str]], str | None],
) -> int:
    """Run tool with argparse error suggestion support.

    This helper encapsulates the common pattern for tool entry points that want
    to provide helpful suggestions when users make common CLI mistakes (like
    passing positional arguments instead of using flags).

    Args:
        parse_fn: Function that parses arguments (typically parse_arguments())
        main_fn: Main function to run with parsed args (typically main())
        suggest_fn: Function that checks sys.argv for common mistakes and
                    returns a suggestion string, or None if no suggestion

    Returns:
        Exit code from main_fn

    Example:
        if __name__ == "__main__":
            sys.exit(cli.run_with_argparse_suggestions(
                parse_arguments,
                main,
                _suggest_common_mistakes
            ))
    """
    try:
        args = parse_fn()
    except SystemExit as e:
        # Check if user made a common mistake and show helpful suggestion.
        if e.code != 0:  # Non-zero exit means error occurred.
            suggestion = suggest_fn(sys.argv)
            if suggestion:
                console.error(f"\n{suggestion}", args=None)
        raise
    return main_fn(args)
