# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common CLI helpers for lit tools.

Keeps flag naming and selection behavior uniform across all tools.
"""

from __future__ import annotations

import argparse  # noqa: TCH003
import re
import sys
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from collections.abc import Callable

from lit_tools.core import console, test_file


def add_common_output_flags(parser: argparse.ArgumentParser) -> None:
    """Add common output flags to parser.

    Args:
        parser: ArgumentParser to add flags to
    """
    parser.add_argument("--json", action="store_true", help="JSON output for scripting")
    parser.add_argument(
        "--pretty", action="store_true", help="Human-friendly formatting"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress non-essential text"
    )


def require_exactly_one(
    args: argparse.Namespace, names: Iterable[str], message: str | None = None
) -> bool:
    """Ensure exactly one of the named flags/attrs is truthy.

    Args:
        args: Parsed arguments namespace
        names: Flag names to check
        message: Optional error message override

    Returns:
        True if exactly one is set, False otherwise
    """
    count = sum(1 for n in names if getattr(args, n, None))
    if count != 1:
        console.error(
            message or f"specify exactly one of: {', '.join(names)}", args=args
        )
        return False
    return True


def require_at_most_one(
    args: argparse.Namespace, names: Iterable[str], message: str | None = None
) -> bool:
    """Ensures at most one of the named flags/attrs is truthy.

    Useful for tools like list where multiple summary modes exist (--json, --count,
    --names) but only one should be chosen at a time.

    Args:
        args: Parsed arguments namespace
        names: Flag names to check
        message: Optional error message override

    Returns:
        True if at most one is set, False otherwise
    """
    count = sum(1 for n in names if getattr(args, n, None))
    if count > 1:
        console.error(
            message or f"choose at most one of: {', '.join(names)}", args=args
        )
        return False
    return True


def parse_case_numbers(spec: str | list[str]) -> list[int]:
    """Parse case number specification into list of integers.

    Supports:
    - Single number: "5" -> [5]
    - Comma-separated: "1,3,5" -> [1, 3, 5]
    - Ranges: "1-3" -> [1, 2, 3]
    - Mixed: "1,3-5,7" -> [1, 3, 4, 5, 7]
    - Multiple flags: ["1", "3", "5"] -> [1, 3, 5]

    Args:
        spec: String like "1,3-5" or list like ["1", "3", "5"]

    Returns:
        Sorted list of unique case numbers

    Raises:
        ValueError: Invalid format, negative/zero numbers, or invalid range

    Examples:
        >>> parse_case_numbers("1")
        [1]
        >>> parse_case_numbers("1,3,5")
        [1, 3, 5]
        >>> parse_case_numbers("1-3")
        [1, 2, 3]
        >>> parse_case_numbers("1,3-5,7")
        [1, 3, 4, 5, 7]
        >>> parse_case_numbers(["1", "3", "5"])
        [1, 3, 5]
    """
    # Convert list to comma-separated string.
    if isinstance(spec, list):
        spec = ",".join(spec)

    if not spec or not spec.strip():
        raise ValueError("No case numbers specified")

    numbers = set()
    parts = spec.split(",")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check if it's a range (e.g., "1-3").
        # Must have a hyphen not at the start or end.
        if "-" in part and not part.startswith("-") and not part.endswith("-"):
            hyphen_index = part.index("-")
            # Make sure there's content on both sides of the hyphen.
            if hyphen_index > 0 and hyphen_index < len(part) - 1:
                range_parts = part.split("-", 1)

                try:
                    start = int(range_parts[0].strip())
                    end = int(range_parts[1].strip())
                except ValueError:
                    raise ValueError(
                        f"Invalid range '{part}': both start and end must be integers"
                    ) from None

                if start <= 0 or end <= 0:
                    raise ValueError(
                        f"Case numbers must be positive, got range {start}-{end}"
                    )

                if start > end:
                    raise ValueError(
                        f"Invalid range {start}-{end}: start must be <= end"
                    )

                numbers.update(range(start, end + 1))
                continue

        # Single number (or invalid input).
        try:
            num = int(part)
        except ValueError:
            raise ValueError(f"Invalid case number: '{part}'") from None

        if num <= 0:
            raise ValueError(f"Case numbers must be positive, got {num}")

        numbers.add(num)

    if not numbers:
        raise ValueError("No valid case numbers found in specification")

    return sorted(numbers)


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


def apply_filters(
    cases: list[test_file.TestCase],
    filter_pattern: str | None,
    filter_out_pattern: str | None,
    args: argparse.Namespace | None = None,
) -> list[test_file.TestCase] | None:
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
