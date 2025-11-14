# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Console logging helpers for lit tools.

These helpers centralize printing, target stream selection, and styling. They
pull `pretty`/`quiet` flags from an `args` object when present to keep callers
simple and consistent.

All output goes to stderr except for primary data (like extracted test cases
or JSON output), which goes to stdout. This follows Unix conventions where
stderr is for diagnostics and stdout is for pipeline-able data.

Quiet mode behavior:
- error() - ALWAYS prints (errors are never suppressed)
- warn() - Suppressed when quiet=True
- note() - Suppressed when quiet=True
- success() - Suppressed when quiet=True
- print_json() - Outputs compact JSON when quiet=True

Pretty mode behavior:
- When pretty=True and output is a TTY, adds ANSI color codes
- When pretty=False or not a TTY, outputs plain text

Example:
    >>> import argparse
    >>> from common import console
    >>>
    >>> # Normal mode - all messages print with color if TTY
    >>> args = argparse.Namespace(pretty=True, quiet=False)
    >>> console.note("Processing test cases", args=args)
    note: Processing test cases  # (with color if TTY)
    >>>
    >>> console.warn("Large file detected", args=args)
    warning: Large file detected  # (with color if TTY)
    >>>
    >>> # Quiet mode - only errors print
    >>> args.quiet = True
    >>> console.note("This will not print", args=args)
    # (suppressed)
    >>>
    >>> console.warn("This warning is also suppressed", args=args)
    # (suppressed)
    >>>
    >>> console.error("File not found", args=args)
    error: File not found  # (errors ALWAYS print, even when quiet)
    >>>
    >>> # JSON output - compact when quiet
    >>> console.print_json({"status": "ok", "count": 5}, args=args)
    {"count":5,"status":"ok"}  # (compact, no whitespace)
    >>>
    >>> args.quiet = False
    >>> console.print_json({"status": "ok", "count": 5}, args=args)
    {
      "count": 5,
      "status": "ok"
    }  # (pretty-printed with indentation)

Integration with cli.add_common_output_flags():
    All tools should use cli.add_common_output_flags() to get --pretty, --quiet,
    and --json flags, then pass the args object to console functions. This ensures
    consistent behavior across all tools without duplicating flag logic.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from common import formatting


def _is_pretty(args: Any | None, pretty: bool | None) -> bool:
    """Determine if color output should be used.

    Checks pretty flag, TTY status, and NO_COLOR environment variable.
    All I/O-related decisions (TTY checks) are centralized here.

    Args:
        args: Parsed arguments with pretty flag (optional)
        pretty: Override pretty mode (optional)

    Returns:
        Whether to use color codes in output
    """
    # Extract the pretty flag value.
    if pretty is not None:
        pretty_flag = bool(pretty)
    else:
        pretty_flag = bool(getattr(args, "pretty", False))

    # Only use color if: pretty enabled AND stderr is TTY AND NO_COLOR not set.
    # We check stderr because that's where console output (note/warn/error) goes.
    return bool(
        pretty_flag and sys.stderr.isatty() and os.environ.get("NO_COLOR") is None
    )


def error(msg: str, *, args: Any | None = None, pretty: bool | None = None) -> None:
    """Print error message to stderr (never suppressed, even in quiet mode).

    Args:
        msg: Error message to display
        args: Parsed arguments with pretty/quiet flags (optional)
        pretty: Override pretty mode (optional)
    """
    sys.stderr.write(formatting.error(msg, pretty=_is_pretty(args, pretty)) + "\n")


def warn(msg: str, *, args: Any | None = None, pretty: bool | None = None) -> None:
    """Print warning message to stderr (suppressed in quiet mode).

    Args:
        msg: Warning message to display
        args: Parsed arguments with pretty/quiet flags (optional)
        pretty: Override pretty mode (optional)
    """
    # Suppress non-essential noise when quiet is requested.
    if getattr(args, "quiet", False):
        return
    sys.stderr.write(formatting.warn(msg, pretty=_is_pretty(args, pretty)) + "\n")


def note(msg: str, *, args: Any | None = None, pretty: bool | None = None) -> None:
    """Print informational note to stderr (suppressed in quiet mode).

    Args:
        msg: Note message to display
        args: Parsed arguments with pretty/quiet flags (optional)
        pretty: Override pretty mode (optional)
    """
    if getattr(args, "quiet", False):
        return
    sys.stderr.write(formatting.note(msg, pretty=_is_pretty(args, pretty)) + "\n")


def banner(case: Any, *, args: Any | None = None, pretty: bool | None = None) -> str:
    """Format test case banner string.

    Args:
        case: Test case to format banner for
        args: Parsed arguments with pretty/quiet flags (optional)
        pretty: Override pretty mode (optional)

    Returns:
        Formatted banner string
    """
    return formatting.case_banner(case, pretty=_is_pretty(args, pretty))


def maybe_highlight_run(
    line: str,
    *,
    args: Any | None = None,
    pretty: bool | None = None,
    to_file: bool = False,
) -> str:
    """Optionally highlight RUN lines in test output.

    Args:
        line: Line to potentially highlight
        args: Parsed arguments with pretty/quiet flags (optional)
        pretty: Override pretty mode (optional)
        to_file: If True, skip highlighting for file output

    Returns:
        Highlighted or plain line
    """
    if to_file:
        return line
    return formatting.highlight_run(line, pretty=_is_pretty(args, pretty))


def print_json(
    payload: Any, *, args: Any | None = None, quiet: bool | None = None
) -> None:
    """Print JSON to stdout (compact in quiet mode, pretty otherwise).

    Args:
        payload: Data to serialize as JSON
        args: Parsed arguments with quiet flag (optional)
        quiet: Override quiet mode (optional)
    """
    q = bool(getattr(args, "quiet", False)) if quiet is None else bool(quiet)
    if q:
        sys.stdout.write(
            json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
        )
    else:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def success(msg: str, *, args: Any | None = None, pretty: bool | None = None) -> None:
    """Print success message to stderr (suppressed in quiet mode).

    Args:
        msg: Success message to display
        args: Parsed arguments with pretty/quiet flags (optional)
        pretty: Override pretty mode (optional)
    """
    if getattr(args, "quiet", False):
        return
    sys.stderr.write(formatting.success(msg, pretty=_is_pretty(args, pretty)) + "\n")


def out(text: str) -> None:
    """Writes a single line of primary output to stdout (adds newline)."""
    sys.stdout.write(text + "\n")


def write(text: str) -> None:
    """Writes raw text to stdout without appending a newline."""
    sys.stdout.write(text)
