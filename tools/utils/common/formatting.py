# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared formatting helpers for human-friendly output.

All lit tools should use these helpers to keep banners and human output
consistent. Tools remain concise by default; callers can opt-in to `pretty`
styling which may add color when outputting to a TTY.

Note: TTY detection and NO_COLOR checks are handled by console.py before
calling these formatting functions. This module is pure string formatting.
"""

from __future__ import annotations


def _use_color(pretty: bool) -> bool:
    """Return whether to apply color codes based on pretty flag.

    The pretty flag is computed by console.py which handles TTY detection
    and NO_COLOR environment variable checks. This function just returns
    the flag value since all I/O decisions are made in the console layer.

    Args:
        pretty: Whether color output is enabled (from console.py)

    Returns:
        The pretty flag value
    """
    return bool(pretty)


def _c(code: str, text: str, pretty: bool) -> str:
    if _use_color(pretty):
        return f"\033[{code}m{text}\033[0m"
    return text


def color(code: str, text: str, pretty: bool = False) -> str:
    """Applies an ANSI color/style ``code`` when ``pretty`` is enabled.

    Prefer this for one-off emphasis in tools (headers, numbers, etc.).
    Keep usage minimal to preserve token-efficiency and readability.

    Example:
        >>> header = color("1;33", "summary:", pretty=True)  # bold yellow
    """
    return _c(code, text, pretty)


def case_banner(case: object, pretty: bool = False) -> str:
    """Format test case as banner comment.

    Args:
        case: TestCase object with number, name, start_line, end_line attributes
        pretty: Enable colorized output

    Returns:
        Formatted banner string
    """
    name = f"@{case.name}" if case.name else "(unnamed)"
    head = f"// Test case {case.number}: {name}"
    rng = f"(lines {case.start_line}-{case.end_line})"
    if _use_color(pretty):
        head = _c("1;36", head, True)  # bold cyan
        rng = _c("2;37", rng, True)  # dim grey
    return f"{head} {rng}"


def use_color(pretty: bool) -> bool:
    """Expose color decision for callers that need to branch behavior.

    Args:
        pretty: Pretty output mode enabled

    Returns:
        Whether to apply color codes
    """
    return _use_color(pretty)


def warn(msg: str, pretty: bool = False) -> str:
    """Format warning message with optional color.

    Args:
        msg: Warning message text
        pretty: Enable colorized output

    Returns:
        Formatted warning message
    """
    prefix = _c("1;33", "warning:", pretty)
    return f"{prefix} {msg}"


def error(msg: str, pretty: bool = False) -> str:
    """Format error message with optional color.

    Args:
        msg: Error message text
        pretty: Enable colorized output

    Returns:
        Formatted error message
    """
    prefix = _c("1;31", "error:", pretty)
    return f"{prefix} {msg}"


def note(msg: str, pretty: bool = False) -> str:
    """Format note message with optional color.

    Args:
        msg: Note message text
        pretty: Enable colorized output

    Returns:
        Formatted note message
    """
    prefix = _c("1;34", "note:", pretty)
    return f"{prefix} {msg}"


def highlight_run(line: str, pretty: bool = False) -> str:
    """Highlights a single `// RUN: ...` comment line for display.

    Should only be used when writing to a terminal. Do not colorize when
    writing to files.
    """
    if not _use_color(pretty):
        return line
    if line.lstrip().startswith("// RUN:"):
        # Colorize RUN: label and dim the rest
        prefix, rest = line.split("RUN:", 1)
        return f"{prefix}{_c('1;35', 'RUN:', True)}{_c('2;37', rest, True)}"
    return line


def success(msg: str, pretty: bool = False) -> str:
    """Format success message with optional color.

    Args:
        msg: Success message text
        pretty: Enable colorized output

    Returns:
        Formatted success message
    """
    prefix = _c("1;32", "ok:", pretty)
    return f"{prefix} {msg}"


def colorize_diff(diff_text: str, pretty: bool = False) -> str:
    """Add ANSI color codes to unified diff output.

    Only adds color when pretty=True and outputting to TTY.

    Args:
        diff_text: Unified diff text
        pretty: Whether to apply color codes

    Returns:
        Colorized diff (or original if color disabled)
    """
    if not _use_color(pretty):
        return diff_text

    # ANSI color codes
    RED = "31"  # Deletions
    GREEN = "32"  # Additions
    CYAN = "36"  # File headers (---)
    BOLD = "1"

    lines = []
    for line in diff_text.splitlines():
        if line.startswith("---") or line.startswith("+++"):
            # File headers: bold cyan
            lines.append(_c(f"{BOLD};{CYAN}", line, True))
        elif line.startswith("-") and not line.startswith("---"):
            # Deletions: red
            lines.append(_c(RED, line, True))
        elif line.startswith("+") and not line.startswith("+++"):
            # Additions: green
            lines.append(_c(GREEN, line, True))
        elif line.startswith("@@"):
            # Hunk headers: bold cyan
            lines.append(_c(f"{BOLD};{CYAN}", line, True))
        else:
            # Context lines: no color
            lines.append(line)

    return "\n".join(lines)
