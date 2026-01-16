# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared IR verification utilities for iree-lit-* tools.

This module provides functions for verifying MLIR IR using iree-opt and
detecting when IR is intentionally invalid (contains expected-error directives).
"""

import argparse
import subprocess
from typing import TYPE_CHECKING

from common import build_detection

if TYPE_CHECKING:
    from lit_tools.core.parser import TestCase


def verify_ir(
    content: str,
    case_info: str = "",
    args: argparse.Namespace | None = None,
    timeout: int = 5,
) -> tuple[bool, str]:
    """Verifies IR content using iree-opt (MLIR verifier).

    Runs iree-opt with no flags, which invokes the MLIR verifier to check
    structural validity of the IR.

    Args:
        content: MLIR IR content to verify
        case_info: Description of case being verified (for error messages)
        args: Parsed arguments (for console output control)
        timeout: Verification timeout in seconds

    Returns:
        Tuple of (success: bool, error_message: str)
        - If success=True, error_message is empty
        - If success=False, error_message contains verification failure details
    """
    try:
        # Find iree-opt
        iree_opt = build_detection.find_tool("iree-opt")
    except FileNotFoundError as e:
        return (False, str(e))

    try:
        # Run iree-opt on stdin (no flags = just verify IR structure)
        result = subprocess.run(
            [str(iree_opt), "-"],
            input=content,
            capture_output=True,
            timeout=timeout,
            text=True,
        )

        if result.returncode != 0:
            # Truncate stderr for human output, keep full for JSON.
            stderr = result.stderr
            if args and not args.json and len(stderr) > 1000:
                stderr = (
                    stderr[:1000]
                    + "\n... (output truncated, use --json for full diagnostics)"
                )

            error_msg = "Verification failed"
            if case_info:
                error_msg += f" for {case_info}"
            error_msg += f":\n{stderr}\nTip: Run iree-opt on extracted case to debug"
            return (False, error_msg)

        return (True, "")

    except subprocess.TimeoutExpired:
        error_msg = f"Verification timeout ({timeout}s)"
        if case_info:
            error_msg += f" for {case_info}"
        error_msg += ".\nCheck for infinite loops or increase timeout with --verify-timeout SECONDS"
        return (False, error_msg)


def verify_ir_file(
    file_path: str, case_info: str, args: argparse.Namespace, timeout: int = 5
) -> tuple[bool, str]:
    """Verifies IR from a file using iree-opt (MLIR verifier).

    Args:
        file_path: Path to MLIR file to verify
        case_info: Description of case being verified (for error messages)
        args: Parsed arguments
        timeout: Verification timeout in seconds

    Returns:
        Tuple of (success: bool, error_message: str)
        - If success=True, error_message is empty
        - If success=False, error_message contains verification failure details
    """
    try:
        # Find iree-opt
        iree_opt = build_detection.find_tool("iree-opt")
    except FileNotFoundError as e:
        return (False, str(e))

    try:
        # Run iree-opt on file (no flags = just verify IR structure)
        result = subprocess.run(
            [str(iree_opt), file_path],
            capture_output=True,
            timeout=timeout,
            text=True,
        )

        if result.returncode != 0:
            # Truncate stderr for human output, keep full for JSON
            stderr = result.stderr
            if not args.json and len(stderr) > 1000:
                stderr = (
                    stderr[:1000]
                    + "\n... (output truncated, use --json for full diagnostics)"
                )

            error_msg = (
                f"Verification failed for {case_info}:\n"
                f"{stderr}\n"
                f"Tip: Run iree-opt on extracted case to debug"
            )
            return (False, error_msg)

        return (True, "")

    except subprocess.TimeoutExpired:
        error_msg = (
            f"Verification timeout ({timeout}s) for {case_info}.\n"
            f"Check for infinite loops or increase timeout with --verify-timeout SECONDS"
        )
        return (False, error_msg)


def should_skip_verification_for_case(case: "TestCase") -> bool:
    """Check if test case expects diagnostic errors and should skip verification.

    Args:
        case: Test case to check

    Returns:
        True if case contains expected-error directives, False otherwise
    """
    # MLIR diagnostic directives indicate IR is intentionally invalid.
    diagnostic_patterns = [
        "// expected-error",
        "// expected-warning",
        "// expected-note",
    ]
    return any(pattern in case.content for pattern in diagnostic_patterns)


def should_skip_verification_for_content(content: str) -> bool:
    """Check if IR content expects diagnostic errors and should skip verification.

    Args:
        content: IR content to check

    Returns:
        True if content contains expected-error directives, False otherwise
    """
    # MLIR diagnostic directives indicate IR is intentionally invalid.
    diagnostic_patterns = [
        "// expected-error",
        "// expected-warning",
        "// expected-note",
    ]
    return any(pattern in content for pattern in diagnostic_patterns)


def verify_content_with_skip_check(
    content: str,
    case_number: int,
    case_name: str | None,
    args: argparse.Namespace,
    timeout: int | None = None,
) -> tuple[bool, bool, str]:
    """Verify IR content with automatic skip check for expected-error directives.

    This helper consolidates the common pattern of:
    1. Check if content should skip verification (has expected-error)
    2. Print skip note if not quiet
    3. Run verification if not skipped
    4. Return structured result

    Args:
        content: IR content to verify
        case_number: Case number for error messages
        case_name: Case name (from CHECK-LABEL) or None
        args: Parsed arguments (for --quiet, --json, --verify flags)
        timeout: Optional verification timeout override

    Returns:
        Tuple of (was_skipped, is_valid, error_message)
        - was_skipped: True if verification was skipped due to expected-error
        - is_valid: True if verification passed (always True if skipped)
        - error_message: Error details if verification failed, empty otherwise

    Example:
        >>> skipped, valid, error = verify_content_with_skip_check(
        ...     content, case.number, case.name, args
        ... )
        >>> if not skipped and not valid:
        ...     console.error(error, args=args)
        ...     return exit_codes.ERROR
    """
    from common import console  # noqa: PLC0415

    # Check if verification is disabled globally.
    if not args.verify:
        return (False, True, "")

    # Check if content has expected-error directives (skip verification).
    if should_skip_verification_for_content(content):
        if not args.quiet:
            console.note(
                f"Skipping verification for case {case_number}: "
                "contains expected-error directives",
                args=args,
            )
        return (True, True, "")

    # Build case info string for error messages.
    case_info = f"Case {case_number}"
    if case_name:
        case_info += f" (@{case_name})"

    # Run verification.
    if timeout is None:
        timeout = getattr(args, "verify_timeout", 5)

    valid, error_msg = verify_ir(content, case_info, args, timeout=timeout)
    return (False, valid, error_msg)
