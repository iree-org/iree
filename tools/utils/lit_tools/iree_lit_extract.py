# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Extract individual test cases from MLIR lit test files.

Lit test files often contain multiple test cases separated by `// -----` delimiters.
This tool extracts individual test cases for isolated testing and debugging.

Usage:
  # Extract by case number
  iree-lit-extract test.mlir --case 2

  # Extract multiple cases (separated by // -----)
  iree-lit-extract test.mlir --case 1,3,5
  iree-lit-extract test.mlir --case 1-3
  iree-lit-extract test.mlir --case 1 --case 3 --case 5

  # Extract by function name
  iree-lit-extract test.mlir --name function_name

  # Extract test containing a specific line (useful from error messages)
  iree-lit-extract test.mlir --containing 142

  # Filter by name pattern (extract all cases with "fold" in name)
  iree-lit-extract test.mlir --filter "fold"

  # Exclude by name pattern (extract all except GPU cases)
  iree-lit-extract test.mlir --filter-out "gpu"

  # List available test cases first
  iree-lit-extract test.mlir --list

  # JSON listing (scripting)
  iree-lit-extract test.mlir --list --json > cases.json

  # Save to file instead of stdout
  iree-lit-extract test.mlir --case 2 -o /tmp/case2.mlir

  # Extract without RUN lines (just IR)
  iree-lit-extract test.mlir --case 2 --exclude-run-lines > /tmp/case2.mlir

  # Verify extracted IR is structurally valid
  iree-lit-extract test.mlir --case 2 --verify

  # Extract with JSON output (array of case objects)
  iree-lit-extract test.mlir --case 2 --json > cases.json

Examples:
  # Extract second test case to stdout
  $ iree-lit-extract test.mlir --case 2
  // CHECK-LABEL: @second_case
  util.func @second_case(%arg0: tensor<8xf32>) -> tensor<8xf32> {
    ...
  }

  # Extract test case by name
  $ iree-lit-extract test.mlir --name multiple_transients
  // CHECK-LABEL: @multiple_transients
  util.func @multiple_transients() {
    ...
  }

  # Find which test contains error at line 142
  $ iree-lit-extract test.mlir --containing 142
  // Test case 3: @nested_execute (lines 91-142)
  // CHECK-LABEL: @nested_execute
  util.func @nested_execute() {
    ...  // <- line 142 is here
  }

  # Extract to stdout for standalone execution (RUN lines included by default)
  $ iree-lit-extract test.mlir --case 2 > /tmp/case2.mlir
  $ iree-opt /tmp/case2.mlir  # Can run directly

  # Verify extraction
  $ iree-lit-extract test.mlir --case 2 --verify
  Test case 2: @second_case (lines 14-24) extracted successfully
  Verification: IR is structurally valid

  # Extract all cases with "fold" in name (produces split-input file)
  $ iree-lit-extract test.mlir --filter "fold" -o /tmp/fold_cases.mlir
  # Creates file with multiple cases separated by // -----

  # Extract all cases except GPU-related ones

  # Write JSON array to a file
  $ iree-lit-extract test.mlir --case 2 --json -o /tmp/case2.json
  $ iree-lit-extract test.mlir --filter-out "gpu" -o /tmp/no_gpu.mlir

Exit codes:
  0 - Success
  1 - Error (parse failure, verification failure)
  2 - Not found (file doesn't exist, case not found, invalid case number)

See Also:
  iree-lit-list - List all test cases in a file
  iree-lit-test - Run test cases in isolation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lit_tools.core.parser import TestCase, TestFile

# Import from other categories (added to sys.path as top-level packages)
from common import console, exit_codes, formatting, fs

# Import from own category (as absolute path within sys.path)
from lit_tools.core import (
    cli,
    listing,
    verification,
)


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract individual test cases from MLIR lit test files",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", help="Path to lit test file")

    # Selection options (shared with iree-lit-test and iree-lit-lint).
    cli.add_selection_arguments(parser)

    # Output options
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--exclude-run-lines",
        action="store_true",
        help="Exclude RUN lines from output (default: include for complete, runnable tests)",
    )

    # Verification options
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify IR is structurally valid with iree-opt (recommended after rebuilding tools)",
    )

    # Filtering options (consistent with iree-lit-test).
    cli.add_filter_arguments(parser)

    # Common output flags (--json/--pretty/--quiet)
    cli.add_common_output_flags(parser)

    return parser.parse_args()


def format_case_info(case: TestCase) -> str:
    """Formats test case information for comment header.

    Args:
        case: TestCase object

    Returns:
        Formatted comment string
    """
    return formatting.case_banner(case, pretty=False)


def list_cases(file_path: Path, *, pretty: bool = False, quiet: bool = False) -> int:
    """Lists all test cases (delegated to iree-lit-list behavior).

    Args:
        file_path: Path to lit test file
        pretty: Enable colorized output
        quiet: Suppress header

    Returns:
        Exit code
    """
    try:
        cases = listing.get_cases(file_path)
        console.out(
            listing.format_text_listing(
                file_path, cases, pretty=pretty, header=not quiet
            )
        )
        return exit_codes.SUCCESS
    except (OSError, UnicodeDecodeError, ValueError) as e:
        console.error(f"Listing failed: {e}", args=None)
        return exit_codes.ERROR


def _validate_selection_args(args: argparse.Namespace) -> int:
    """Validates that exactly one selection method is specified.

    Args:
        args: Parsed command-line arguments

    Returns:
        exit_codes.SUCCESS if valid, exit_codes.NOT_FOUND if invalid
    """
    has_case = bool(args.case)
    has_name = bool(args.name)
    selection_count = (
        int(has_case)
        + int(has_name)
        + int(args.containing is not None)
        + int(args.list)
    )
    using_filters = args.filter or args.filter_out
    if selection_count != 1 and not (selection_count == 0 and using_filters):
        console.error(
            "specify exactly one of: --case, --name, --containing, --list, or --filter",
            args=args,
        )
        return exit_codes.NOT_FOUND
    return exit_codes.SUCCESS


def _handle_list_mode(file_path: Path, args: argparse.Namespace) -> int:
    """Handles --list mode for displaying available test cases.

    Args:
        file_path: Path to test file
        args: Parsed command-line arguments

    Returns:
        exit code (SUCCESS or ERROR)
    """
    if args.json:
        try:
            cases = listing.get_cases(file_path)
            payload = listing.build_json_payload(file_path, cases)
            console.print_json(payload, args=args)
            return exit_codes.SUCCESS
        except (OSError, UnicodeDecodeError, ValueError) as e:
            console.error(f"Listing failed: {e}", args=args)
            return exit_codes.ERROR
    return list_cases(file_path, pretty=args.pretty, quiet=args.quiet)


def _build_text_output(
    selected_cases: list[TestCase], test_file: TestFile, args: argparse.Namespace
) -> str:
    """Builds text output content from selected cases.

    Args:
        selected_cases: List of TestCase objects to output
        test_file: Parsed test file object
        args: Parsed command-line arguments

    Returns:
        String containing formatted output content
    """
    output_parts = []

    # Include RUN lines by default for complete, runnable test cases.
    # - Default: include RUN lines (makes extract | test pipeline work naturally)
    # - --exclude-run-lines: opt-out for workflows that need clean IR
    # - JSON: never include RUN lines (metadata separate from content)
    include_runs = not args.exclude_run_lines and not args.json
    num_header_run_lines = 0
    if include_runs:
        run_lines_section = _extract_run_lines_section(test_file, selected_cases, args)
        if run_lines_section:
            output_parts.append(run_lines_section)
            # Count number of header RUN lines (raw format, to get physical line count).
            num_header_run_lines = len(test_file.extract_run_lines(raw=True))

    # Add case content.
    # If we included RUN lines, strip corresponding leading newlines from first case
    # (case.content has leading newlines for each header RUN line that was stripped).
    case_outputs = []
    for i, case in enumerate(selected_cases):
        content = case.content
        # Strip leading newlines from first case when RUN lines are included.
        if i == 0 and include_runs:
            # Strip newlines for header RUN lines.
            for _ in range(num_header_run_lines):
                if content.startswith("\n"):
                    content = content[1:]
            # Also strip newlines for case-local RUN lines (they were also stripped from content).
            case_runs = case.extract_local_run_lines()
            for _ in range(len(case_runs)):
                if content.startswith("\n"):
                    content = content[1:]
        case_outputs.append(content)

    if len(case_outputs) > 1:
        output_parts.append("\n\n// -----\n\n".join(case_outputs))
    else:
        output_parts.append(case_outputs[0])

    return "\n".join(output_parts)


def _extract_run_lines_section(
    test_file: TestFile, selected_cases: list[TestCase], args: argparse.Namespace
) -> str | None:
    """Extracts and formats RUN lines section including header and case-local RUN lines.

    Args:
        test_file: Parsed test file object
        selected_cases: List of selected test cases
        args: Parsed command-line arguments

    Returns:
        Formatted RUN lines string or None if extraction failed
    """
    try:
        # Get header RUN lines.
        header_runs = test_file.extract_run_lines()
        run_lines = list(header_runs)  # Copy to avoid modifying original.

        # If extracting specific cases, also include case-local RUN lines.
        # Combine all case-local RUN lines from selected cases.
        # Note: We preserve case-local RUNs even if they duplicate header RUNs,
        # as they may be intentionally specified for that case.
        for case in selected_cases:
            case_runs = case.extract_local_run_lines()
            for _, cmd in case_runs:
                # Always include case-local RUNs (even if they duplicate header).
                run_lines.append(cmd)

        if not run_lines:
            return None

        run_lines_section = []
        for run_line in run_lines:
            line = f"// RUN: {run_line}"
            if args.pretty and not args.output:
                line = console.maybe_highlight_run(line, args=args)
            run_lines_section.append(line)
        # Note: Don't add blank line here - case content preserves the original
        # blank line after stripping leading newlines for header RUN lines.
        return "\n".join(run_lines_section)
    except (OSError, UnicodeDecodeError, ValueError) as e:
        console.warn(f"Could not extract RUN lines: {e}", args=args)
        return None


def _verify_selected_cases(
    selected_cases: list[TestCase], args: argparse.Namespace
) -> int:
    """Verifies IR for all selected cases if requested.

    Args:
        selected_cases: List of test cases to verify
        args: Parsed command-line arguments

    Returns:
        exit_codes.SUCCESS if all valid, exit_codes.ERROR if any invalid
    """
    if not args.verify:
        return exit_codes.SUCCESS

    all_valid = True
    for case in selected_cases:
        _, valid, error_msg = verification.verify_content_with_skip_check(
            case.content, case.number, case.name, args
        )
        if not valid:
            console.error(error_msg, args=args)
            all_valid = False
    return exit_codes.SUCCESS if all_valid else exit_codes.ERROR


def _build_json_payload(selected_cases: list[TestCase]) -> list[dict]:
    """Builds JSON payload from selected cases.

    Args:
        selected_cases: List of test cases to include in payload

    Returns:
        List of case dictionaries
    """
    return [
        {
            "number": c.number,
            "name": c.name,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "line_count": c.line_count,
            "check_count": c.check_count,
            "content": c.content,
        }
        for c in selected_cases
    ]


def _write_output_file(
    selected_cases: list[TestCase],
    output_content: str,
    output_path: Path,
    args: argparse.Namespace,
) -> int:
    """Writes output to file in JSON or text mode.

    Args:
        selected_cases: List of test cases to write
        output_content: Text content to write (if not JSON mode)
        output_path: Path to output file
        args: Parsed command-line arguments

    Returns:
        exit_codes.SUCCESS if write succeeded, exit_codes.ERROR if failed
    """
    try:
        if args.json:
            payload = _build_json_payload(selected_cases)
            text = (
                json.dumps(payload, separators=(",", ":"), sort_keys=True)
                if args.quiet
                else json.dumps(payload, indent=2, sort_keys=True)
            )
            fs.safe_write_text(output_path, text)

            if not args.quiet:
                console.note(
                    f"Wrote JSON for {len(selected_cases)} case(s) to {output_path}",
                    args=args,
                )
        else:
            fs.safe_write_text(output_path, output_content)
            if not args.quiet:
                _print_file_write_summary(selected_cases, output_path, args)
        return exit_codes.SUCCESS
    except OSError as e:
        console.error(f"Write failed: {e}", args=args)
        return exit_codes.ERROR


def _print_file_write_summary(
    selected_cases: list[TestCase], output_path: Path, args: argparse.Namespace
) -> None:
    """Prints summary message after writing cases to file.

    Args:
        selected_cases: List of test cases written
        output_path: Path where cases were written
        args: Parsed command-line arguments
    """
    if len(selected_cases) == 1:
        case = selected_cases[0]
        name = f"@{case.name}" if case.name else "(unnamed)"
        banner = (
            f"Test case {case.number}: {name} (lines {case.start_line}-{case.end_line})"
        )
        console.note(f"{banner} extracted to {output_path}", args=args)
    else:
        case_nums = ", ".join(str(c.number) for c in selected_cases)
        console.note(
            f"{len(selected_cases)} test cases ({case_nums}) extracted to {output_path}",
            args=args,
        )
    if args.verify:
        console.success("All IR parses correctly", args=args)


def _output_to_stdout(
    selected_cases: list[TestCase], output_content: str, args: argparse.Namespace
) -> None:
    """Outputs selected cases to stdout in JSON or text mode.

    Args:
        selected_cases: List of test cases to output
        output_content: Text content to output (if not JSON mode)
        args: Parsed command-line arguments
    """
    if args.json:
        payload = _build_json_payload(selected_cases)
        console.print_json(payload, args=args)
    else:
        console.write(output_content)


def _apply_filters_if_needed(
    all_cases: list[TestCase],
    selected_cases: list[TestCase],
    args: argparse.Namespace,
) -> tuple[list[TestCase] | None, int]:
    """Applies regex filters if specified and not using explicit selectors.

    Args:
        all_cases: List of all parsed test cases
        selected_cases: Currently selected test cases
        args: Parsed command-line arguments

    Returns:
        Tuple of (filtered_cases, exit_code). Exit code is NOT_FOUND on filter error.
    """
    if (
        (args.filter or args.filter_out)
        and not args.case
        and not args.name
        and args.containing is None
    ):
        filtered = cli.apply_filters(all_cases, args.filter, args.filter_out, args=args)
        if filtered is None:
            return None, exit_codes.NOT_FOUND
        return filtered, exit_codes.SUCCESS
    return selected_cases, exit_codes.SUCCESS


def main(args: argparse.Namespace) -> int:
    """Main entry point.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (SUCCESS, ERROR, or NOT_FOUND)
    """
    file_path = Path(args.file)

    # Enforce selection exclusivity.
    result = _validate_selection_args(args)
    if result != exit_codes.SUCCESS:
        return result

    # Handle --list mode.
    if args.list:
        return _handle_list_mode(file_path, args)

    # Parse test file to get all cases.
    test_file, all_cases, exit_code = cli.load_and_parse_test_file(file_path, args)
    if exit_code != exit_codes.SUCCESS:
        return exit_code

    # Select cases based on criteria.
    selected_cases = cli.select_cases(all_cases, args)
    if selected_cases is None:
        return exit_codes.NOT_FOUND

    # Apply regex filters if specified.
    selected_cases, result = _apply_filters_if_needed(all_cases, selected_cases, args)
    if result != exit_codes.SUCCESS:
        return result

    # Build text output.
    output_content = _build_text_output(selected_cases, test_file, args)

    # Verify if requested.
    result = _verify_selected_cases(selected_cases, args)
    if result != exit_codes.SUCCESS:
        return result

    # Output results.
    if args.output:
        output_path = Path(args.output)
        return _write_output_file(selected_cases, output_content, output_path, args)
    _output_to_stdout(selected_cases, output_content, args)
    return exit_codes.SUCCESS


if __name__ == "__main__":
    sys.exit(
        cli.run_with_argparse_suggestions(
            parse_arguments, main, cli.suggest_common_mistakes
        )
    )
