# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lint MLIR lit test files for common authoring mistakes.

Catches test authoring errors before running the compiler, provides actionable
warnings for test smells, and generates training data for fine-tuning models.

Quick Start:
  # Lint entire file
  iree-lit-lint test.mlir

  # Lint specific cases
  iree-lit-lint test.mlir --case 2
  iree-lit-lint test.mlir --name fold_constants

  # Show only errors
  iree-lit-lint test.mlir --errors-only

  # JSON output
  iree-lit-lint test.mlir --json

Key Features:

  Selection:
    Same selection flags as iree-lit-test and iree-lit-extract:
    --case, --name, --containing, --filter, --filter-out

  Severity Levels:
    ERROR: Must fix (raw SSA, zero CHECKs, etc.)
    WARNING: Should review (wildcards, non-semantic names, etc.)
    INFO: Suggestions (consider CHECK-NEXT, semantic naming, etc.)

  Filtering:
    --errors-only: Show only errors (exit code 1 if errors found)
    --min-severity=warning: Show warnings and errors only

  Output Modes:
    Grouped (default): Group errors by type, show help once per error
    Individual (--individual-errors): Show each occurrence separately
    JSON (--json): Machine-readable for tooling integration
    Quiet (--quiet): Suppress non-essential output

  Grouped Output Format (default):
    [ERROR] error_code (N occurrences)
    Full help text explaining the issue and how to fix it

    file.mlir:45:12: error: error_code: short description
    file.mlir:67:8: error: error_code: short description

    Each line maintains IDE-parseable format for jump-to-location while
    deduplicating verbose help text

Common Workflows:

  Check entire file:
    $ iree-lit-lint test.mlir

  CI integration (block on errors):
    $ iree-lit-lint test.mlir --errors-only --quiet
    $ echo $?  # 1 if errors found, 0 otherwise

  Generate training data:
    $ iree-lit-lint test.mlir --json > training.json

  Lint specific test case:
    $ iree-lit-lint test.mlir --case 5

Exit Codes:
  0 - No errors found (warnings/info may exist)
  1 - One or more errors found
  2 - File not found or parse error

See Also:
  iree-lit-test - Run individual test cases
  iree-lit-extract - Extract test cases to separate files
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lit_tools.core.lint import LintIssue
    from lit_tools.core.parser import TestCase, TestFile

from common import console, exit_codes, fs

from lit_tools.core import cli, lint, listing
from lit_tools.core.parser import parse_test_file


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Build argument parser.

    Args:
        argv: Optional argument list (for testing). If None, uses sys.argv.
    """
    parser = argparse.ArgumentParser(
        description="Lint MLIR lit test files for common authoring mistakes",
        epilog=(
            __doc__ + "\n\n"
            "For detailed style guidelines and rationale, see:\n"
            "  iree-lit-lint --help-style-guide\n"
            "  tools/utils/lit_tools/STYLE_GUIDE.md"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "test_file",
        type=Path,
        nargs="?",  # Make optional for --help-style-guide.
        help="Test file to lint (.mlir)",
    )

    # Style guide help.
    parser.add_argument(
        "--help-style-guide",
        action="store_true",
        help="Print detailed style guide and exit",
    )

    # Test case selection (shared with iree-lit-test and iree-lit-extract).
    cli.add_selection_arguments(parser)

    # Filtering options.
    cli.add_filter_arguments(parser)

    # Lint-specific options.
    lint_opts = parser.add_argument_group("Linting options")
    lint_opts.add_argument(
        "--errors-only",
        action="store_true",
        help="Show only errors (exit 1 if any errors found)",
    )
    lint_opts.add_argument(
        "--min-severity",
        choices=["error", "warning", "info"],
        default="info",
        help="Minimum severity level to show (default: info shows all)",
    )
    lint_opts.add_argument(
        "--individual-errors",
        action="store_true",
        help="Show individual error occurrences (ungrouped) instead of grouped by error type",
    )

    # Common output flags.
    cli.add_common_output_flags(parser)

    # JSON output file.
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write JSON output to file instead of stdout",
    )
    parser.add_argument(
        "--full-json",
        action="store_true",
        help="Include full IR context around each issue in JSON output",
    )
    parser.add_argument(
        "--context-lines",
        type=int,
        default=5,
        metavar="N",
        help="Number of lines before/after to include in context (default: 5, used with --full-json)",
    )

    return parser.parse_args(argv)


def _filter_issues_by_severity(
    issues: list[LintIssue], min_severity: str, errors_only: bool
) -> list[LintIssue]:
    """Filter issues based on severity settings.

    Args:
        issues: All issues found
        min_severity: Minimum severity to include ("error", "warning", "info")
        errors_only: If True, show only errors

    Returns:
        Filtered list of issues
    """
    if errors_only:
        return [i for i in issues if i.severity == "error"]

    # Severity hierarchy: error > warning > info (lower numbers = more severe)
    severity_levels = {"error": 0, "warning": 1, "info": 2}
    min_level = severity_levels[min_severity]

    return [i for i in issues if severity_levels.get(i.severity, 999) <= min_level]


def _format_text_output(
    issues: list[LintIssue], test_file: Path, args: argparse.Namespace
) -> str:
    """Format issues as text output.

    Args:
        issues: List of lint issues
        test_file: Path to test file
        args: Parsed arguments

    Returns:
        Formatted text string
    """
    if not issues:
        return ""

    lines = []
    for issue in issues:
        # Format: file:line:col: severity: message
        location = f"{test_file}:{issue.line}"
        if issue.column is not None:
            location += f":{issue.column}"

        lines.append(f"{location}: {issue.severity}: {issue.message}")
        lines.append(f"  {issue.snippet}")
        lines.append(f"  help: {issue.help}")
        lines.append("")  # Blank line between issues

    # Summary line.
    error_count = sum(1 for i in issues if i.severity == "error")
    warning_count = sum(1 for i in issues if i.severity == "warning")
    info_count = sum(1 for i in issues if i.severity == "info")

    summary_parts = []
    if error_count:
        summary_parts.append(f"{error_count} error(s)")
    if warning_count:
        summary_parts.append(f"{warning_count} warning(s)")
    if info_count:
        summary_parts.append(f"{info_count} info message(s)")

    lines.append(", ".join(summary_parts))

    return "\n".join(lines)


def _format_grouped_output(
    issues: list[LintIssue], test_file: Path, args: argparse.Namespace
) -> str:
    """Format issues grouped by error code.

    Groups issues by code to reduce duplicate help text. Each occurrence
    maintains IDE-parseable file:line:severity format while showing the
    full help text only once per error type.

    Output format:
        [SEVERITY] code (N occurrences)
        Full help text here

        file:line:col: severity: code: message
        file:line:col: severity: code: message

        [SEVERITY] next_code (M occurrences)
        ...

    Args:
        issues: List of lint issues
        test_file: Path to test file
        args: Parsed arguments

    Returns:
        Formatted text string
    """
    if not issues:
        return ""

    # Group issues by code.
    grouped: dict[str, list[LintIssue]] = {}
    for issue in issues:
        if issue.code not in grouped:
            grouped[issue.code] = []
        grouped[issue.code].append(issue)

    # Severity hierarchy for sorting: error (0) > warning (1) > info (2)
    severity_order = {"error": 0, "warning": 1, "info": 2}

    # Sort groups by severity (most severe first) then alphabetically by code.
    def group_sort_key(item: tuple[str, list[LintIssue]]) -> tuple[int, str]:
        code, group_issues = item
        # Use the most severe issue in the group for sorting.
        min_severity = min(
            severity_order.get(issue.severity, 999) for issue in group_issues
        )
        return (min_severity, code)

    sorted_groups = sorted(grouped.items(), key=group_sort_key)

    lines = []
    for code, group_issues in sorted_groups:
        # Determine group severity (use most severe issue).
        group_severity = min(
            group_issues, key=lambda i: severity_order.get(i.severity, 999)
        ).severity

        # Header: [SEVERITY] code (N occurrences)
        count = len(group_issues)
        occurrence_text = "occurrence" if count == 1 else "occurrences"
        lines.append(f"[{group_severity.upper()}] {code} ({count} {occurrence_text})")

        # Full help text (from first issue, should be same for all).
        lines.append(group_issues[0].help)
        lines.append("")  # Blank line after help text

        # Individual occurrences in IDE-parseable format.
        for issue in group_issues:
            location = f"{test_file}:{issue.line}"
            if issue.column is not None:
                location += f":{issue.column}"

            lines.append(f"{location}: {issue.severity}: {issue.code}: {issue.message}")

        lines.append("")  # Blank line between groups

    # Summary line.
    error_count = sum(1 for i in issues if i.severity == "error")
    warning_count = sum(1 for i in issues if i.severity == "warning")
    info_count = sum(1 for i in issues if i.severity == "info")

    summary_parts = []
    if error_count:
        summary_parts.append(f"{error_count} error(s)")
    if warning_count:
        summary_parts.append(f"{warning_count} warning(s)")
    if info_count:
        summary_parts.append(f"{info_count} info message(s)")

    if summary_parts:
        lines.append(", ".join(summary_parts))

    return "\n".join(lines)


def _build_json_output(
    issues: list[LintIssue],
    test_file: Path,
    total_cases: int,
    linted_cases: int,
    full_json: bool = False,
) -> dict:
    """Build JSON output structure.

    Args:
        issues: All lint issues found
        test_file: Path to test file
        total_cases: Total number of test cases in file
        linted_cases: Number of cases that were linted
        full_json: Include full IR context around each issue

    Returns:
        Dictionary for JSON serialization
    """
    error_count = sum(1 for i in issues if i.severity == "error")
    warning_count = sum(1 for i in issues if i.severity == "warning")
    info_count = sum(1 for i in issues if i.severity == "info")

    return {
        "file": str(test_file),
        "total_cases": total_cases,
        "cases_linted": linted_cases,
        "total_issues": len(issues),
        "errors": error_count,
        "warnings": warning_count,
        "info": info_count,
        "issues": [
            {
                "severity": issue.severity,
                "code": issue.code,
                "message": issue.message,
                "line": issue.line,
                "column": issue.column,
                "snippet": issue.snippet,
                "help": issue.help,
                **({"suggestions": issue.suggestions} if issue.suggestions else {}),
                **(
                    {"context_lines": issue.context_lines}
                    if full_json and issue.context_lines
                    else {}
                ),
            }
            for issue in issues
        ],
    }


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


def _handle_style_guide_help() -> int:
    """Print style guide and exit.

    Returns:
        exit_codes.SUCCESS
    """
    style_guide_path = Path(__file__).parent / "STYLE_GUIDE.md"
    if style_guide_path.exists():
        console.out(style_guide_path.read_text())
    else:
        console.error(f"Style guide not found at: {style_guide_path}")
        console.error("Expected location: tools/utils/lit_tools/STYLE_GUIDE.md")
    return exit_codes.SUCCESS


def _load_and_select_cases(
    test_file: Path, args: argparse.Namespace
) -> tuple[TestFile | None, list[TestCase] | None, list[TestCase] | None]:
    """Load test file and select cases to lint.

    Args:
        test_file: Path to test file
        args: Parsed command-line arguments

    Returns:
        Tuple of (test_file_obj, all_cases, selected_cases) or (None, None, None) on error
    """
    if not test_file.exists():
        console.error(f"Test file not found: {test_file}", args=args)
        return (None, None, None)

    try:
        test_file_obj = parse_test_file(test_file)
        all_cases = list(test_file_obj.cases)
    except (OSError, UnicodeDecodeError, ValueError) as e:
        console.error(f"Failed to parse test file: {e}", args=args)
        return (None, None, None)

    if not all_cases:
        console.error(f"No test cases found in {test_file}", args=args)
        return (None, None, None)

    # Handle --list mode.
    if args.list:
        _handle_list_mode(test_file, all_cases, args)
        return (test_file_obj, all_cases, None)

    # Select cases to lint.
    selected_cases = cli.select_cases(all_cases, args)
    if selected_cases is None:
        return (test_file_obj, all_cases, None)

    # Apply regex filters if specified.
    if (args.filter or args.filter_out) and not (
        args.case or args.name or args.containing
    ):
        selected_cases = cli.apply_filters(
            selected_cases, args.filter, args.filter_out, args=args
        )

    return (test_file_obj, all_cases, selected_cases)


def _run_linting_and_output(
    test_file_obj: TestFile,
    all_cases: list[TestCase],
    selected_cases: list[TestCase],
    test_file_path: Path,
    args: argparse.Namespace,
) -> int:
    """Run linters and output results.

    Args:
        test_file_obj: Parsed test file object
        all_cases: All test cases in file
        selected_cases: Cases to lint
        test_file_path: Path to test file
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    if not args.quiet and not args.json:
        console.note(f"Linting {len(selected_cases)} test case(s)", args=args)

    all_issues = []

    # Check file-level issues (separators, split boundary checks, etc.).
    file_level_issues = lint.check_file_level_issues(test_file_obj)
    all_issues.extend(file_level_issues)

    for case in selected_cases:
        issues = lint.run_all_checkers(case, context_lines=args.context_lines)
        all_issues.extend(issues)

    # Filter by severity.
    filtered_issues = _filter_issues_by_severity(
        all_issues, args.min_severity, args.errors_only
    )

    # Output results.
    if args.json:
        json_output = _build_json_output(
            filtered_issues,
            test_file_path,
            len(all_cases),
            len(selected_cases),
            args.full_json,
        )
        if args.json_output:
            fs.safe_write_text(
                args.json_output, json.dumps(json_output, indent=2, sort_keys=True)
            )
            if not args.quiet:
                console.note(f"Wrote JSON to {args.json_output}", args=args)
        else:
            console.print_json(json_output, args=args)
    else:
        # Use grouped output by default, individual output with --individual-errors.
        if args.individual_errors:
            text_output = _format_text_output(filtered_issues, test_file_path, args)
        else:
            text_output = _format_grouped_output(filtered_issues, test_file_path, args)

        if text_output:
            console.out(text_output)
        elif not args.quiet:
            console.success("No issues found", args=args)

    # Exit code: ERROR if any errors found, SUCCESS otherwise.
    has_errors = any(i.severity == "error" for i in filtered_issues)
    return exit_codes.ERROR if has_errors else exit_codes.SUCCESS


def main(args: argparse.Namespace) -> int:
    """Main entry point.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (SUCCESS if no errors, ERROR if errors found, NOT_FOUND on failure)
    """
    # Handle --help-style-guide early exit.
    if args.help_style_guide:
        return _handle_style_guide_help()

    # Validate test_file is provided.
    if not args.test_file:
        console.error("test_file is required (unless using --help-style-guide)")
        return exit_codes.ERROR

    # Load and select test cases.
    test_file_obj, all_cases, selected_cases = _load_and_select_cases(
        args.test_file, args
    )
    if test_file_obj is None or all_cases is None:
        return exit_codes.NOT_FOUND

    # --list mode already handled in _load_and_select_cases.
    if selected_cases is None:
        return exit_codes.SUCCESS if args.list else exit_codes.NOT_FOUND

    # Run linting and output results.
    return _run_linting_and_output(
        test_file_obj, all_cases, selected_cases, args.test_file, args
    )


if __name__ == "__main__":
    sys.exit(main(parse_arguments()))
