# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run individual test cases from MLIR lit test files.

Lit test files use // ----- to separate multiple test cases. This tool runs
individual cases in isolation using LLVM lit, making debugging and iteration
much faster than running entire files.

Quick Start:
  # Run all test cases
  iree-lit-test test.mlir

  # Run specific case by number
  iree-lit-test test.mlir --case 2

  # Run specific case by name
  iree-lit-test test.mlir --name function_name

  # List available cases
  iree-lit-test test.mlir --list

  # Debug with extra flags
  iree-lit-test test.mlir --case 3 --extra-flags="--mlir-print-ir-after-all"

Key Features:

  Case Selection:
    Select by number (--case 1,3,5), range (--case 1-3), or name (--name foo).
    Use --containing LINE to select case containing a specific line number.
    Combine multiple --case flags: --case 1 --case 3 --case 5

  Filtering:
    --filter REGEX: Include only cases matching pattern (e.g., --filter "fold")
    --filter-out REGEX: Exclude cases matching pattern
    Combine both to narrow down: --filter "op" --filter-out "gpu"

  Timeout Protection:
    Tests timeout after 60s by default to catch infinite loops.
    Configure with --timeout SECONDS or disable with --timeout 0.

  Debug Flag Injection:
    Add flags to iree-opt/iree-compile without editing test files.
    Example: --extra-flags="--mlir-print-ir-after-all --debug"
    Flags are inserted after tool name in RUN: lines.

  Parallel Execution:
    Run multiple cases concurrently with --workers N.
    Use --keep-going to continue after failures.

  JSON Output:
    Machine-readable results with --json.
    Write to file: --json-output FILE
    Include full output: --full-json
    Suppress human-readable output: --quiet

Common Workflows:

  Debug single failing case:
    $ iree-lit-test test.mlir --case 5 --verbose

  Run subset by name pattern:
    $ iree-lit-test test.mlir --filter "fold" --keep-going

  Parallel execution for large files:
    $ iree-lit-test test.mlir --workers 4 --keep-going

  JSON output for automation:
    $ iree-lit-test test.mlir --json --quiet > results.json

  Preview what would run:
    $ iree-lit-test test.mlir --filter "fold" --dry-run

  Inject debug flags:
    $ iree-lit-test test.mlir --case 3 --extra-flags="--debug" --verbose

Exit Codes:
  0 - All selected test cases passed
  1 - One or more test cases failed or execution error
  2 - Not found (file doesn't exist, no cases match selection)

Environment Variables:
  IREE_BUILD_DIR: Override build directory auto-detection

Build Detection:
  Automatically finds IREE build directory in this order:
  1. IREE_BUILD_DIR environment variable
  2. ./build/ (in-tree build)
  3. ../<worktree>-build/ (for git worktree pattern)
  4. ../iree-build/ (main repo build)

For comprehensive troubleshooting guide and advanced usage, see:
  tools/utils/lit_tools/README.md

See Also:
  iree-lit-list - List test cases in a file
  iree-lit-extract - Extract test case to separate file
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lit_tools.core.listing import TestCase
    from lit_tools.core.lit_wrapper import LitResult

from common import build_detection, fs

from lit_tools.core import (
    cli,
    console,
    exit_codes,
    listing,
    lit_wrapper,
    suggestions,
)
from lit_tools.core import test_file as test_file_module


def _suggest_common_mistakes(argv: list[str]) -> str | None:
    """Check for common CLI mistakes and suggest corrections.

    Args:
        argv: Command line arguments (sys.argv)

    Returns:
        Suggestion string if a common mistake is detected, None otherwise
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


def parse_arguments() -> argparse.Namespace:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Run individual test cases from MLIR lit test files",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("test_file", type=Path, help="Test file to run (.mlir)")

    # Test case selection (shared with iree-lit-extract).
    selection = parser.add_argument_group("Test case selection")
    selection.add_argument(
        "--case",
        "-c",
        action="append",
        metavar="N[,N...]",
        help="Run specific case number(s): --case 1, --case 1,3,5, --case 1-3",
    )
    selection.add_argument(
        "--name", "-n", metavar="NAME", help="Run case with specific function name"
    )
    selection.add_argument(
        "--containing", metavar="LINE", type=int, help="Run case containing line number"
    )
    selection.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all test cases and exit (ignores filters)",
    )
    # Note: If none provided, run ALL cases.

    # Execution options.
    exec_group = parser.add_argument_group("Execution options")
    exec_group.add_argument(
        "--timeout",
        type=int,
        default=60,
        metavar="SEC",
        help="Test timeout in seconds (default: 60, 0=disable)",
    )
    exec_group.add_argument(
        "--extra-flags",
        type=str,
        metavar="FLAGS",
        help="Inject flags into iree-* tools (e.g., '--debug')",
    )
    exec_group.add_argument(
        "--keep-going", "-k", action="store_true", help="Continue after first failure"
    )
    exec_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show full lit output (including passing tests)",
    )
    exec_group.add_argument(
        "--keep-temps", action="store_true", help="Don't delete temporary test files"
    )
    exec_group.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of cases to run in parallel (default: 1)",
    )
    # Standardized filters (shared across tools)
    cli.add_filter_arguments(parser)
    exec_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing tests",
    )

    # Build detection.
    parser.add_argument(
        "--build-dir", type=Path, help="IREE build directory (auto-detected if omitted)"
    )

    # Common output flags.
    cli.add_common_output_flags(parser)

    # JSON controls
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write JSON output to this file (instead of stdout)",
    )
    parser.add_argument(
        "--full-json",
        action="store_true",
        help="Include full per-case output in JSON (may be large)",
    )

    return parser.parse_args()


def _print_result(result: LitResult, args: argparse.Namespace) -> None:
    """Print test result with appropriate detail level.

    Args:
        result: LitResult from running a test case
        args: Parsed command-line arguments
    """
    if result.passed:
        if not args.quiet:
            console.note(f"  ✓ PASS ({result.duration:.2f}s)", args=args)
            if args.verbose and result.stdout and not args.json:
                console.out("")
                console.out("Full lit output:")
                console.out(result.stdout)
    else:
        console.error(f"  ✗ FAIL ({result.duration:.2f}s)", args=args)
        if not args.json:
            if result.failure_summary:
                console.note(result.failure_summary, args=args)
            if args.verbose and result.stdout:
                console.out("")
                console.out("Full lit output:")
                console.out(result.stdout)


def _handle_result(
    result: LitResult, results: list[LitResult], args: argparse.Namespace
) -> bool:
    """Process test result and determine if execution should continue.

    Args:
        result: LitResult from test execution
        results: List to append result to
        args: Parsed arguments

    Returns:
        True if should continue execution, False if should stop early
    """
    results.append(result)
    _print_result(result, args)
    # Continue if test passed OR if --keep-going is enabled.
    return result.passed or args.keep_going


def filter_cases(cases: list[TestCase], args: argparse.Namespace) -> list[TestCase]:
    """Filter test cases based on selection arguments.

    Args:
        cases: List of test cases to filter
        args: Parsed command-line arguments

    Returns:
        List of selected TestCase objects.
        Empty list if no matches OR if error occurred.

    Note:
        Errors are signaled by calling console.error() then returning [].
        Caller should check for empty result and return exit_codes.NOT_FOUND.
        This implicit error signaling pattern keeps error messages near the
        point of failure while allowing simple boolean checks in caller.
    """
    if args.case:
        # Find by case number(s).
        try:
            case_numbers = cli.parse_case_numbers(args.case)
        except ValueError as e:
            console.error(str(e), args=args)
            return []

        # Find all matching cases.
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
                return []

        return selected

    if args.name:
        # Find by function name.
        for case in cases:
            if case.name == args.name:
                return [case]
        error_msg = suggestions.format_case_name_error(args.name, cases)
        console.error(error_msg, args=args)
        return []

    if args.containing:
        # Find by line number.
        for case in cases:
            if case.start_line <= args.containing <= case.end_line:
                return [case]
        console.error(f"No case contains line {args.containing}", args=args)
        return []

    # No filter - return all cases.
    return cases


def _handle_list_mode(
    test_file: Path, cases: list[TestCase], args: argparse.Namespace
) -> int:
    """Handles --list mode to display test cases.

    Args:
        test_file: Path to test file
        cases: List of parsed test cases
        args: Parsed command-line arguments

    Returns:
        exit_codes.SUCCESS
    """
    if args.json:
        payload = listing.build_json_payload(test_file, cases)
        console.print_json(payload, args=args)
    else:
        console.out(
            listing.format_text_listing(
                test_file,
                cases,
                pretty=args.pretty,
                header=not args.quiet,
            )
        )
    return exit_codes.SUCCESS


def _handle_dry_run_mode(
    test_file: Path, selected_cases: list[TestCase], args: argparse.Namespace
) -> int:
    """Handles --dry-run mode to show what would be executed.

    Args:
        test_file: Path to test file
        selected_cases: List of cases that would be run
        args: Parsed command-line arguments

    Returns:
        exit_codes.SUCCESS
    """
    if args.json:
        payload = {
            "file": str(test_file),
            "total_cases": len(selected_cases),
            "selected_cases": [
                {
                    "number": c.number,
                    "name": c.name,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                }
                for c in selected_cases
            ],
        }
        console.print_json(payload, args=args)
    else:
        if not args.quiet:
            console.note(
                f"Dry-run: would execute {len(selected_cases)} case(s)", args=args
            )
        for i, case in enumerate(selected_cases):
            case_label = f"  [{i + 1}] Case {case.number}"
            if case.name:
                case_label += f": @{case.name}"
            case_label += f" (lines {case.start_line}-{case.end_line})"
            console.note(case_label, args=args)
    return exit_codes.SUCCESS


def _run_cases_parallel(
    selected_cases: list[TestCase],
    test_file: Path,
    build_dir: Path,
    args: argparse.Namespace,
) -> list[LitResult]:
    """Runs test cases in parallel using ThreadPoolExecutor.

    Args:
        selected_cases: List of test cases to run
        test_file: Path to test file
        build_dir: Build directory path
        args: Parsed command-line arguments

    Returns:
        List of LitResult objects
    """
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_to_case = {
            ex.submit(
                lit_wrapper.run_lit_on_case,
                case,
                test_file,
                build_dir,
                args.timeout,
                args.extra_flags,
                args.verbose,
                args.keep_temps,
            ): case
            for case in selected_cases
        }
        for i, fut in enumerate(as_completed(fut_to_case), start=1):
            case = fut_to_case[fut]
            if not args.quiet and len(selected_cases) > 1:
                label = (
                    f"case {case.number}{' (' + case.name + ')' if case.name else ''}"
                )
                console.note(
                    f"[{i}/{len(selected_cases)}] Running {label}...", args=args
                )
            result = fut.result()
            if not _handle_result(result, results, args):
                break
    return results


def _run_cases_sequential(
    selected_cases: list[TestCase],
    test_file: Path,
    build_dir: Path,
    args: argparse.Namespace,
) -> list[LitResult]:
    """Runs test cases sequentially.

    Args:
        selected_cases: List of test cases to run
        test_file: Path to test file
        build_dir: Build directory path
        args: Parsed command-line arguments

    Returns:
        List of LitResult objects
    """
    results = []
    for i, case in enumerate(selected_cases):
        if not args.quiet and len(selected_cases) > 1:
            case_label = f"case {case.number}"
            if case.name:
                case_label += f" ({case.name})"
            console.note(
                f"[{i + 1}/{len(selected_cases)}] Running {case_label}...", args=args
            )

        result = lit_wrapper.run_lit_on_case(
            case=case,
            test_file=test_file,
            build_dir=build_dir,
            timeout=args.timeout,
            extra_flags=args.extra_flags,
            verbose=args.verbose,
            keep_temps=args.keep_temps,
        )

        if not _handle_result(result, results, args):
            break
    return results


def _build_json_output(
    test_file: Path, results: list[LitResult], args: argparse.Namespace
) -> dict:
    """Builds JSON output from test results.

    Args:
        test_file: Path to test file
        results: List of LitResult objects
        args: Parsed command-line arguments

    Returns:
        Dictionary containing JSON output structure
    """
    passed = sum(r.passed for r in results)
    total = len(results)
    total_time = sum(r.duration for r in results)

    return {
        "file": str(test_file),
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "total_time": round(total_time, 2),
        "results": [
            {
                "case_number": r.case_number,
                "case_name": r.case_name,
                "passed": r.passed,
                "duration": round(r.duration, 2),
                "failure_summary": r.failure_summary if not r.passed else None,
                **({"run_commands": r.run_commands} if r.run_commands else {}),
                **({"output": r.stdout} if args.full_json else {}),
            }
            for r in results
        ],
    }


def _output_json_results(json_output: dict, args: argparse.Namespace) -> None:
    """Outputs JSON results to file or stdout.

    Args:
        json_output: JSON output dictionary
        args: Parsed command-line arguments
    """
    if args.json_output:
        fs.safe_write_text(args.json_output, json.dumps(json_output, indent=2))
        if not args.quiet:
            console.note(f"Wrote JSON to {args.json_output}", args=args)
    else:
        console.print_json(json_output, args=args)


def _compute_final_exit_code(results: list[LitResult], args: argparse.Namespace) -> int:
    """Computes final exit code based on test results.

    Args:
        results: List of LitResult objects
        args: Parsed command-line arguments

    Returns:
        exit_codes.SUCCESS if all passed, exit_codes.ERROR otherwise
    """
    passed = sum(r.passed for r in results)
    total = len(results)
    total_time = sum(r.duration for r in results)

    if args.json:
        return exit_codes.SUCCESS if passed == total else exit_codes.ERROR

    # Human-readable output.
    if passed == total:
        if not args.quiet:
            console.success(
                f"All {total} test case(s) passed ({total_time:.2f}s total)",
                args=args,
            )
        return exit_codes.SUCCESS
    console.error(f"{total - passed}/{total} test case(s) failed", args=args)
    return exit_codes.ERROR


def _parse_and_validate_test_file(
    test_file: Path, args: argparse.Namespace
) -> tuple[list[TestCase] | None, int]:
    """Parses test file and validates it contains test cases.

    Args:
        test_file: Path to test file
        args: Parsed command-line arguments

    Returns:
        Tuple of (cases, exit_code). Exit code is ERROR/NOT_FOUND on failure.
    """
    if not test_file.exists():
        console.error(f"Test file not found: {test_file}", args=args)
        return None, exit_codes.NOT_FOUND

    try:
        cases = test_file_module.parse_test_file(test_file)
    except (OSError, UnicodeDecodeError, ValueError) as e:
        console.error(f"Failed to parse test file: {e}", args=args)
        return None, exit_codes.ERROR

    if not cases:
        console.error(f"No test cases found in {test_file}", args=args)
        return None, exit_codes.ERROR

    return cases, exit_codes.SUCCESS


def _select_and_filter_cases(
    cases: list[TestCase], args: argparse.Namespace
) -> tuple[list[TestCase] | None, int]:
    """Selects and filters test cases based on arguments.

    Args:
        cases: List of test cases to filter
        args: Parsed command-line arguments

    Returns:
        Tuple of (selected_cases, exit_code). Exit code is NOT_FOUND on failure.
    """
    selected_cases = filter_cases(cases, args)
    if not selected_cases:
        return None, exit_codes.NOT_FOUND

    # Apply regex filters (only when not using explicit selectors).
    if (args.filter or args.filter_out) and not (
        args.case or args.name or args.containing
    ):
        selected_cases = cli.apply_filters(
            selected_cases, args.filter, args.filter_out, args=args
        )
        if selected_cases is None:
            return None, exit_codes.NOT_FOUND

    return selected_cases, exit_codes.SUCCESS


def _run_tests_and_output_results(
    selected_cases: list[TestCase],
    test_file: Path,
    build_dir: Path,
    args: argparse.Namespace,
) -> int:
    """Runs test cases and outputs results.

    Args:
        selected_cases: List of test cases to run
        test_file: Path to test file
        build_dir: Build directory path
        args: Parsed command-line arguments

    Returns:
        Exit code (SUCCESS or ERROR)
    """
    if not args.quiet:
        case_desc = f"{len(selected_cases)} case(s)"
        if args.extra_flags:
            case_desc += f" with extra flags: {args.extra_flags}"
        console.note(f"Running {case_desc}", args=args)

    # Run test cases (parallel or sequential).
    if args.workers > 1 and len(selected_cases) > 1:
        results = _run_cases_parallel(selected_cases, test_file, build_dir, args)
    else:
        results = _run_cases_sequential(selected_cases, test_file, build_dir, args)

    # Output JSON results if requested.
    if args.json:
        json_output = _build_json_output(test_file, results, args)
        _output_json_results(json_output, args)

    # Compute and return final exit code.
    return _compute_final_exit_code(results, args)


def main(args: argparse.Namespace) -> int:
    """Main entry point.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (SUCCESS, ERROR, or NOT_FOUND)
    """
    # Parse and validate test file.
    cases, result = _parse_and_validate_test_file(args.test_file, args)
    if result != exit_codes.SUCCESS:
        return result

    # Handle --list mode early exit.
    if args.list:
        return _handle_list_mode(args.test_file, cases, args)

    # Select and filter cases to run.
    selected_cases, result = _select_and_filter_cases(cases, args)
    if result != exit_codes.SUCCESS:
        return result

    # Handle --dry-run mode early exit.
    if args.dry_run:
        return _handle_dry_run_mode(args.test_file, selected_cases, args)

    # Detect build directory (only needed when running tests).
    try:
        build_dir = args.build_dir or build_detection.detect_build_dir()
    except FileNotFoundError as e:
        console.error(str(e), args=args)
        return exit_codes.NOT_FOUND

    # Run tests and output results.
    return _run_tests_and_output_results(
        selected_cases, args.test_file, build_dir, args
    )


if __name__ == "__main__":
    sys.exit(
        cli.run_with_argparse_suggestions(
            parse_arguments, main, _suggest_common_mistakes
        )
    )
