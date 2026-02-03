# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common CLI helpers for all tools.

Provides generic argument parsing and validation helpers.
Lit-specific helpers are in lit_tools.core.cli.
"""

from __future__ import annotations

import argparse  # noqa: TCH003
import sys
from typing import TYPE_CHECKING

from common import console

if TYPE_CHECKING:
    from collections.abc import Iterable


def add_common_output_flags(parser: argparse.ArgumentParser) -> None:
    """Add common output flags to parser.

    Auto-detects terminal color support by default. Pretty mode is enabled when:
    - stderr is a TTY (interactive terminal)
    - NO_COLOR environment variable is not set
    - --no-pretty is not specified

    Args:
        parser: ArgumentParser to add flags to
    """
    parser.add_argument("--json", action="store_true", help="JSON output for scripting")
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=sys.stderr.isatty(),
        help="Human-friendly formatting (default: auto-detect terminal)",
    )
    parser.add_argument(
        "--no-pretty",
        action="store_false",
        dest="pretty",
        help="Disable colored output",
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
