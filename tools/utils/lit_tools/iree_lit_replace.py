# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

r"""Replace test cases in MLIR lit test files.

Designed for extract → edit → replace workflows, particularly with LLMs.
Supports single-case text replacement and batch JSON operations.

Usage:
  # Text mode (replace single case)
  iree-lit-replace test.mlir --case 2 < new_content.mlir
  echo "new content" | iree-lit-replace test.mlir --case 2
  iree-lit-replace test.mlir --name function_name -i replacement.mlir

  # JSON mode (batch replacements)
  iree-lit-replace < replacements.json
  cat edits.json | iree-lit-replace
  cat edits.json | iree-lit-replace target.mlir  # Override file field

  # Preview changes
  iree-lit-replace test.mlir --case 2 --dry-run < new_content.mlir

  # Verify IR before replacing
  iree-lit-replace test.mlir --case 2 --verify < new_content.mlir

Examples:
  # Simple text replacement
  $ cat new_case.mlir | iree-lit-replace test.mlir --case 2
  note: Case 2 replaced successfully (1 file modified)

  # Batch JSON replacement
  $ iree-lit-extract test.mlir --case 1,3,5 --json > cases.json
  # Edit cases.json...
  $ iree-lit-replace < cases.json
  note: Replaced 3 cases in test.mlir

  # Cross-file move
  $ iree-lit-extract old.mlir --case 5 --json | \\
      jq '.[0].file = "new.mlir"' | \\
      iree-lit-replace

  # Dry-run to preview changes
  $ cat edited.mlir | iree-lit-replace test.mlir --case 2 --dry-run
  --- test.mlir (original)
  +++ test.mlir (modified)
  @@ -14,3 +14,2 @@
  -old content
  +new content

Exit codes:
  0 - Success (including no-op replacements)
  1 - Error (verification failure, invalid JSON, etc.)
  2 - Not found (file doesn't exist, case not found)

Validation:
  The tool performs strict validation to prevent errors:

  - Name/number consistency: When JSON includes both 'name' and 'number', they
    must point to the same test case. This catches copy-paste errors and
    ensures correct case targeting.

  - Duplicate case names: If multiple cases share the same CHECK-LABEL name,
    you must use --case NUMBER (text mode) or "number" field (JSON mode) to
    disambiguate.

  - Duplicate replacement entries: JSON mode rejects multiple replacement
    entries targeting the same case (by number or name) to prevent
    unintentional overwrites.

  - CLI override warnings: When --test-file is used with JSON mode, warns if
    JSON entries have different "file" values being overridden.

  All validation errors include actionable "Fix:" suggestions for resolution.

RUN Line Handling:
  The tool distinguishes between header RUN lines (at file top) and case-local
  RUN lines (inside individual test cases):

  - Header RUN lines: Preserved automatically. Appear at top of file, apply to
    all test cases. Example: "// RUN: iree-opt --split-input-file %s"

  - Case-local RUN lines: Validated against original. By default, replacement
    content must match either header RUN lines OR the original case's RUN lines.
    Use --replace-run-lines to allow changing them.

  - Replacement content RUN lines: Stripped from replacement by default and
    re-injected from original. This prevents accidental RUN line changes.

  Validation rules:
  1. If replacement contains RUN lines, they must match either:
     - Header RUN lines from file top, OR
     - Case-local RUN lines from original case
  2. Use --replace-run-lines to bypass this check (allows RUN changes)
  3. In JSON mode, per-case "replace_run_lines": true overrides global flag

  This prevents accidentally changing test semantics while allowing intentional
  RUN line updates when needed.

JSON Schema:
  The tool uses a unified JSON schema for all modes (text --json, JSON batch):

  Input Format (JSON mode):
    [
      {
        "file": "test.mlir",      // optional if --test-file provided
        "number": 2,              // XOR with name
        "name": "func_name",      // XOR with number
        "content": "...",         // replacement content

        // Optional per-case flags (override global flags):
        "replace_run_lines": true,  // allow changing RUN lines
        "allow_empty": true,        // allow empty content
        "require_label": false      // require CHECK-LABEL
      }
    ]

  Output Format (all modes with --json):
    {
      "modified_files": 1,
      "modified_cases": 2,
      "unchanged_cases": 0,
      "dry_run": false,           // true for --dry-run mode
      "file_results": [
        {
          "file": "test.mlir",
          "total_cases": 3,       // total cases replaced in this file
          "modified": 2,          // how many were actually changed
          "unchanged": 1,         // how many were skipped (identical)
          "dry_run": false,
          "cases": [
            {
              "number": 1,
              "name": "function_name",
              "changed": true,
              "reason": "content differs"  // optional explanation
            }
          ],
          "diff": "..."           // unified diff (empty if no changes)
        }
      ],
      "errors": [
        {
          "file": "test.mlir",
          "replacement": 1,       // optional: which replacement entry
          "error": "..."          // error message
        }
      ],
      "warnings": [
        {
          "file": "test.mlir",
          "warning": "..."        // warning message
        }
      ]
    }

  The schema is consistent across text mode (with --json) and JSON batch mode,
  both for dry-run and commit operations.

See Also:
  iree-lit-extract - Extract test cases to JSON or text
  iree-lit-list - List available test cases
  iree-lit-test - Run test cases in isolation
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os

    from lit_tools.core.parser import TestCase, TestFile

# Import from other categories (added to sys.path as top-level packages)
from common import console, exit_codes, formatting, fs

# Import from own category (as absolute path within sys.path)
from lit_tools.core import (
    cli,
    suggestions,
    verification,
)
from lit_tools.core.parser import parse_test_file


class _TemporaryArgsOverride:
    """Context manager for temporarily overriding argparse.Namespace attributes.

    Usage:
        with _TemporaryArgsOverride(args, replace_run_lines=True, allow_empty=True):
            # args.replace_run_lines and args.allow_empty are temporarily set
            do_something(args)
        # Original values restored automatically

    This replaces the manual save/restore pattern used in validation code.
    """

    def __init__(self, args: argparse.Namespace, **overrides: object) -> None:
        """Initialize with args object and attribute overrides.

        Args:
            args: argparse.Namespace object to modify
            **overrides: Keyword arguments specifying attribute overrides
        """
        self.args = args
        self.overrides = overrides
        self.originals = {}

    def __enter__(self) -> _TemporaryArgsOverride:
        """Save original values and apply overrides."""
        for key, value in self.overrides.items():
            # Save original value (use getattr to handle missing attributes).
            self.originals[key] = getattr(self.args, key, None)
            # Apply override.
            setattr(self.args, key, value)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        """Restore original values."""
        for key, original_value in self.originals.items():
            setattr(self.args, key, original_value)
        # Return False to propagate exceptions.
        return False


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


def check_concurrent_modification(
    file_path: Path,
    original_stat: os.stat_result,
    original_hash: str,
    args: argparse.Namespace,
) -> None:
    """Check if file was modified after parsing (concurrency protection).

    Args:
        file_path: Path to file being checked
        original_stat: os.stat_result from when file was parsed
        original_hash: SHA256 hash of file content when parsed
        args: Parsed arguments (for fail_if_changed flag)

    Raises:
        RuntimeError: If file was modified and --fail-if-changed is set
    """
    if not args.fail_if_changed:
        return  # Skip check if flag not set

    # Fast check: compare mtime (modification time)
    current_stat = file_path.stat()
    if current_stat.st_mtime == original_stat.st_mtime:
        return  # File hasn't been modified (mtime unchanged)

    # Slow check: mtime changed, but verify with content hash
    # (mtime can change due to touch, metadata changes, etc.)
    current_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    if current_hash == original_hash:
        return  # False alarm: mtime changed but content identical

    # Content actually changed - abort to prevent data loss
    raise RuntimeError(
        f"Concurrent modification detected: {file_path} was modified after parsing.\n"
        f"Aborting to prevent data loss.\n"
        f"Tip: Re-run without --fail-if-changed to force replacement, "
        f"or ensure no other processes are modifying the file."
    )


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Replace test cases in MLIR lit test files",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional argument (optional in JSON mode)
    parser.add_argument(
        "test_file",
        nargs="?",
        help="Target test file (optional in JSON mode if file field present)",
    )

    # Selection options (text mode only - mutually exclusive)
    selection = parser.add_argument_group("test case selection (text mode)")
    selection.add_argument(
        "--case",
        "-c",
        type=int,
        metavar="N",
        help="Replace case by number",
    )
    selection.add_argument(
        "--name",
        "-n",
        metavar="NAME",
        help="Replace case by name (from CHECK-LABEL)",
    )
    selection.add_argument(
        "--append",
        action="store_true",
        help="Append as new case at end of file",
    )
    selection.add_argument(
        "--insert-after",
        type=int,
        metavar="N",
        help="Insert as new case after case N",
    )
    selection.add_argument(
        "--insert-before",
        type=int,
        metavar="N",
        help="Insert as new case before case N",
    )

    # Input options
    input_group = parser.add_argument_group("input")
    input_group.add_argument(
        "-i",
        "--input",
        metavar="FILE",
        help="Read replacement from file (default: stdin)",
    )
    input_group.add_argument(
        "--mode",
        choices=["json", "text"],
        help="Explicit input mode (default: auto-detect from content)",
    )

    # RUN line handling
    run_group = parser.add_argument_group("RUN line handling")
    run_group.add_argument(
        "--replace-run-lines",
        action="store_true",
        help="Use replacement's RUN lines instead of original (default: strip and preserve original)",
    )

    # Validation options
    validation_group = parser.add_argument_group("validation")
    validation_group.add_argument(
        "--verify",
        action="store_true",
        help="Verify IR structural validity with iree-opt (recommended after rebuilding tools)",
    )
    validation_group.add_argument(
        "--require-label",
        action="store_true",
        help="Error if replacement lacks CHECK-LABEL",
    )
    validation_group.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow empty replacement content (default: error)",
    )
    validation_group.add_argument(
        "--verify-timeout",
        type=int,
        default=10,
        metavar="SECONDS",
        help="Timeout for iree-opt verification (default: 10 seconds)",
    )

    # Preview/Safety options
    safety_group = parser.add_argument_group("preview and safety")
    safety_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing file (outputs diff)",
    )
    safety_group.add_argument(
        "--diff-context",
        type=int,
        default=3,
        metavar="N",
        help="Number of context lines in unified diff (default: 3)",
    )
    safety_group.add_argument(
        "--backup",
        metavar="SUFFIX",
        default=".bak",
        help="Backup file suffix (default: .bak)",
    )
    safety_group.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup creation (not recommended)",
    )
    safety_group.add_argument(
        "--fail-if-changed",
        action="store_true",
        help="Abort if target file was modified after parsing (concurrency protection)",
    )

    # Common output flags (--json/--pretty/--quiet)
    cli.add_common_output_flags(parser)

    # JSON output file
    parser.add_argument(
        "--json-output",
        metavar="FILE",
        help="Write JSON output to file (implies --json)",
    )

    args = parser.parse_args()

    # --json-output implies --json
    if args.json_output:
        args.json = True

    return args


def write_json_output(payload: dict, args: argparse.Namespace) -> None:
    """Write JSON output to file or stdout.

    Args:
        payload: JSON payload to write
        args: Parsed arguments (for json_output flag)
    """
    if args.json_output:
        # Write to file
        try:
            with open(args.json_output, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2 if args.pretty else None)
                f.write("\n")  # Trailing newline
        except OSError as e:
            # Fallback to stderr if file write fails
            console.error(
                f"Failed to write JSON output to {args.json_output}: {e}", args=args
            )
            # Still output to stdout as fallback
            console.print_json(payload, args=args)
    else:
        # Write to stdout (normal behavior)
        console.print_json(payload, args=args)


def detect_input_mode(raw_input: bytes, args: argparse.Namespace) -> str:
    """Auto-detect JSON vs text mode from input content.

    Args:
        raw_input: Raw bytes from stdin or file
        args: Parsed arguments

    Returns:
        "json" or "text"
    """
    if args.mode:
        return args.mode  # Explicit override

    # Strip UTF-8 BOM if present
    if raw_input.startswith(b"\xEF\xBB\xBF"):
        raw_input = raw_input[3:]

    # Find first non-whitespace character
    content = raw_input.lstrip()
    if not content:
        return "text"  # Empty input defaults to text

    first_char = chr(content[0])
    return "json" if first_char in "{[" else "text"


def extract_run_lines_from_string(content: str) -> tuple[list[str], str]:
    """Extract RUN lines from replacement content string.

    Args:
        content: Replacement content that may contain RUN lines

    Returns:
        Tuple of (run_commands, content_without_runs)
        - run_commands: List of normalized RUN commands (continuations joined)
        - content_without_runs: Content with RUN lines stripped
    """
    lines = content.splitlines()
    run_commands = []
    current_run = None
    continuing = False
    run_line_indices = set()

    for i, line in enumerate(lines):
        line_stripped = line.rstrip()

        # Check if this is a RUN line
        m = re.match(r"^\s*//\s*RUN:\s?(.*)$", line_stripped)
        if m:
            run_line_indices.add(i)
            cmd = m.group(1).strip()

            if continuing and current_run is not None:
                # Continuation of previous RUN line
                current_run = current_run.rstrip("\\").strip()
                current_run += " " + cmd.lstrip("\\").strip()
            else:
                # Start new RUN command
                current_run = cmd

            # Check if this line continues
            continuing = cmd.endswith("\\")

            if not continuing:
                # Command is complete, save it
                run_commands.append(current_run.rstrip("\\").strip())
                current_run = None

    # Save any pending multi-line command
    if current_run is not None:
        run_commands.append(current_run.rstrip("\\").strip())

    # Remove RUN lines from content
    content_lines = [line for i, line in enumerate(lines) if i not in run_line_indices]
    content_without_runs = "\n".join(content_lines)

    return run_commands, content_without_runs


def _normalize_run_cmd(cmd: str) -> str:
    """Normalize a RUN command string for comparison (collapse whitespace)."""
    return " ".join(cmd.split()).strip()


def _filter_header_runs_from_case_runs(
    case_runs_absolute: list[tuple[int, str]],
    case_content: str,
) -> list[tuple[int, str]]:
    """Filter out header RUN lines from case-local RUN lines.

    Header RUN lines appear before actual case content starts. We find the first
    non-blank line in the case content - RUN lines before that are headers.

    Args:
        case_runs_absolute: List of (absolute_index, command) from extract_case_run_lines().
        case_content: The case content string from TestCase.content.

    Returns:
        Filtered list containing only case-local RUN lines (not header RUN lines).
    """
    # Find first non-blank line in case content
    lines = case_content.split("\n")
    first_content_idx = 0
    for i, line in enumerate(lines):
        if line.strip():
            first_content_idx = i
            break

    # Keep only RUN lines at or after the first actual content
    return [(idx, cmd) for (idx, cmd) in case_runs_absolute if idx >= first_content_idx]


def build_file_content(
    header_runs: list[str],
    cases: list[TestCase],
    content_overrides: dict[int, str] | None = None,
) -> str:
    """Rebuild complete test file from header RUNs and case objects.

    Args:
        header_runs: List of header RUN lines. Can be either:
            - Raw format: Lines with "// RUN:" prefix (from extract_run_lines(raw=True))
            - Joined format: Commands without prefix (from extract_run_lines(raw=False))
        cases: List of TestCase objects in original order
        content_overrides: Optional dict mapping case numbers to replacement content

    Returns:
        Complete file content as string
    """
    result = ""

    # 1. Header RUN lines (if present)
    if header_runs:
        # Detect format: raw lines start with "// RUN:", joined lines don't
        is_raw = header_runs and header_runs[0].lstrip().startswith("//")

        for run_line in header_runs:
            if is_raw:
                # Raw format: already has "// RUN:" prefix
                result += f"{run_line}\n"
            else:
                # Joined format: add "// RUN:" prefix
                result += f"// RUN: {run_line}\n"

    # 2. Cases with separators
    for i, case in enumerate(cases):
        # Use override content if provided, otherwise use case.content
        if content_overrides and case.number in content_overrides:
            content = content_overrides[case.number]
        else:
            content = case.content

        # First case: strip newlines that correspond to stripped RUN lines
        if i == 0 and header_runs:
            # Case content has len(header_runs) leading newlines from RUN stripping
            # Strip them since we've already added the actual RUN lines
            for _ in range(len(header_runs)):
                if content.startswith("\n"):
                    content = content[1:]

        # Normalize trailing blank lines: strip all trailing newlines, then add exactly one.
        # This ensures consistent formatting and idempotency.
        content = content.rstrip("\n")
        if content:  # Only add newline if content is not empty
            content += "\n"

        # Add separator before subsequent cases
        if i > 0:
            result += "\n// -----\n"

        # Add case content
        result += content

    # 3. Ensure final newline
    if not result.endswith("\n"):
        result += "\n"

    return result


def generate_unified_diff(
    original: str, modified: str, file_path: str, args: argparse.Namespace
) -> str:
    """Generate unified diff between original and modified content.

    Args:
        original: Original file content
        modified: Modified file content after replacement
        file_path: Path for file labels in diff
        args: Parsed arguments (for diff_context)

    Returns:
        Unified diff as string
    """
    # splitlines(keepends=True) preserves line endings for difflib
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    # Get context lines (default 3, matches standard diff)
    context = args.diff_context

    diff_lines = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=f"{file_path} (original)",
        tofile=f"{file_path} (modified)",
        n=context,
        lineterm="",  # Don't add extra newlines, we'll handle it
    )

    return "\n".join(diff_lines)


def main(args: argparse.Namespace) -> int:
    """Main entry point."""
    # Read input (stdin or -i FILE)
    try:
        if args.input:
            with open(args.input, "rb") as f:
                raw_input = f.read()
        else:
            raw_input = sys.stdin.buffer.read()
    except OSError as e:
        console.error(f"Failed to read input: {e}", args=args)
        return exit_codes.ERROR

    # Strip UTF-8 BOM if present.
    if raw_input.startswith(b"\xEF\xBB\xBF"):
        raw_input = raw_input[3:]

    # Detect input mode
    input_mode = detect_input_mode(raw_input, args)

    # Decode input
    try:
        input_text = raw_input.decode("utf-8")
    except UnicodeDecodeError as e:
        console.error(
            f"Input contains invalid UTF-8 encoding at byte offset {e.start}.\n"
            f"Ensure replacement content uses UTF-8 encoding.",
            args=args,
        )
        return exit_codes.ERROR

    # Text mode vs JSON mode
    if input_mode == "text":
        return handle_text_mode(input_text, args)
    return handle_json_mode(input_text, args)


def _validate_text_mode_args(args: argparse.Namespace) -> int | None:
    """Validate text mode arguments.

    Returns:
        Exit code if validation fails, None if successful.
    """
    if not args.test_file:
        console.error(
            "Text mode requires test file argument.\n"
            "Usage: iree-lit-replace <file> --case N < replacement.mlir",
            args=args,
        )
        return exit_codes.ERROR

    if not args.case and not args.name:
        console.error(
            "Text mode requires --case N or --name NAME selector.\n"
            "Use JSON mode for batch operations.",
            args=args,
        )
        return exit_codes.ERROR

    if args.case and args.name:
        console.error("specify either --case or --name, not both", args=args)
        return exit_codes.ERROR

    return None


def _load_and_parse_test_file(
    args: argparse.Namespace,
) -> tuple[
    Path | None,
    TestFile | None,
    list | None,
    os.stat_result | None,
    str | None,
    int | None,
]:
    """Load and parse test file, recording state for concurrency detection.

    Returns:
        Tuple of (file_path, test_file_obj, all_cases, original_stat, original_hash, error_code).
        If error_code is not None, an error occurred and other values should be ignored.
    """
    file_path = Path(args.test_file)
    if not file_path.exists():
        console.error(f"File not found: {file_path}", args=args)
        return None, None, None, None, None, exit_codes.NOT_FOUND

    try:
        test_file_obj = parse_test_file(file_path)
        all_cases = list(test_file_obj.cases)
    except (OSError, UnicodeDecodeError, ValueError) as e:
        console.error(f"Failed to parse test file: {e}", args=args)
        return None, None, None, None, None, exit_codes.ERROR

    original_stat = file_path.stat()
    original_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()

    return file_path, test_file_obj, all_cases, original_stat, original_hash, None


def _find_target_case_by_number(
    all_cases: list, file_path: Path, case_number: int, args: argparse.Namespace
) -> tuple[TestCase | None, int | None]:
    """Find target case by number.

    Returns:
        Tuple of (target_case, error_code).
    """
    for case in all_cases:
        if case.number == case_number:
            return case, None

    available = ", ".join(str(c.number) for c in all_cases[:10])
    if len(all_cases) > 10:
        available += f", ... and {len(all_cases) - 10} more"
    console.error(
        f"Case {case_number} not found in {file_path}\n"
        f"Available cases: {available}",
        args=args,
    )
    return None, exit_codes.NOT_FOUND


def _find_target_case_by_name(
    all_cases: list, file_path: Path, case_name: str, args: argparse.Namespace
) -> tuple[TestCase | None, int | None]:
    """Find target case by name.

    Returns:
        Tuple of (target_case, error_code).
    """
    name_matches = [c for c in all_cases if c.name == case_name]

    if len(name_matches) > 1:
        case_numbers = [c.number for c in name_matches]
        console.error(
            f"Multiple cases named '{case_name}' found at numbers {case_numbers}.\n"
            f"Use --case NUMBER to specify which one.",
            args=args,
        )
        return None, exit_codes.ERROR

    if not name_matches:
        error_msg = suggestions.format_case_name_error(
            case_name, all_cases, file_path=str(file_path)
        )
        console.error(error_msg, args=args)
        return None, exit_codes.NOT_FOUND

    return name_matches[0], None


def _validate_case_run_lines(
    replacement_runs: list[str],
    original_case_runs_cmds: list[str],
    header_runs: list[str],
    args: argparse.Namespace,
) -> int | None:
    """Validate replacement RUN lines against header and case-local RUN lines.

    Replacement RUN lines are allowed if they match either:
    - Header RUN lines (will be stripped, header takes precedence)
    - Case-local RUN lines (will be validated for consistency)

    Returns:
        Exit code if validation fails, None if successful.
    """
    if not replacement_runs or args.replace_run_lines:
        return None

    norm_orig = [_normalize_run_cmd(c) for c in original_case_runs_cmds]
    norm_repl = [_normalize_run_cmd(c) for c in replacement_runs]
    norm_header = [_normalize_run_cmd(c) for c in header_runs]

    # Allow if replacement RUNs match header RUNs (user included header in replacement).
    if norm_repl == norm_header:
        return None

    # Allow if replacement RUNs match case-local RUNs.
    if norm_repl == norm_orig:
        return None

    # Allow if replacement RUNs match header + case-local RUNs.
    # This happens when extract includes both header and case-local RUN lines
    # (e.g., with_case_runs.mlir has both).
    norm_header_plus_case = norm_header + norm_orig
    if norm_repl == norm_header_plus_case:
        return None

    # Otherwise, error.
    console.error(
        "Replacement contains RUN lines that differ from both header and case-local RUN lines.\n"
        "Remove RUN lines from replacement or use --replace-run-lines to accept them.",
        args=args,
    )
    return exit_codes.ERROR


def _validate_replacement_content_checks(
    replacement_content_clean: str,
    target_case: TestCase,
    args: argparse.Namespace,
) -> int | None:
    """Validate replacement content (empty check, CHECK-LABEL, iree-opt).

    Returns:
        Exit code if validation fails, None if successful.
    """
    # Empty content check.
    if not replacement_content_clean.strip():
        if not args.allow_empty:
            console.error(
                f"Replacement content is empty.\n"
                f"This would delete case {target_case.number} content (delimiters remain).\n"
                f"Use --allow-empty to confirm this is intentional.",
                args=args,
            )
            return exit_codes.ERROR

        if not args.quiet:
            console.warn(
                f"Replacing case {target_case.number} with empty content (--allow-empty specified)",
                args=args,
            )

    # CHECK-LABEL validation.
    if args.require_label and not re.search(
        r"//\s*CHECK-LABEL:\s*@\S+", replacement_content_clean
    ):
        console.error(
            f"Replacement content for case {target_case.number} lacks CHECK-LABEL.\n"
            f"Use --require-label only when replacement should contain a CHECK-LABEL.\n"
            f"Expected pattern: // CHECK-LABEL: @function_name",
            args=args,
        )
        return exit_codes.ERROR

    # iree-opt verification.
    _, valid, error_msg = verification.verify_content_with_skip_check(
        replacement_content_clean,
        target_case.number,
        target_case.name,
        args,
        timeout=args.verify_timeout,
    )
    if not valid:
        console.error(error_msg, args=args)
        return exit_codes.ERROR

    return None


def _build_final_case_content(
    replacement_content: str,
    replacement_content_clean: str,
    replacement_runs: list[str],
    case_runs_absolute: list[tuple[int, str]],
    args: argparse.Namespace,
) -> tuple[str | None, int | None]:
    """Build final case content by injecting or replacing RUN lines.

    Args:
        replacement_content: Original replacement content with RUN lines.
        replacement_content_clean: Replacement content with RUN lines stripped.
        replacement_runs: RUN lines extracted from replacement content.
        case_runs_absolute: Case-local RUN lines with absolute indices.
        args: Parsed arguments.

    Returns:
        Tuple of (final_content, error_code).
    """
    # When --replace-run-lines is used, replacement RUNs become header RUNs
    # (handled by caller), so return clean content without RUN lines
    if args.replace_run_lines and replacement_runs:
        return replacement_content_clean, None

    # Inject case-local RUN lines into replacement content.
    # SAFE APPROACH: Since replacement content structure may differ from original
    # (different line counts, blank padding), we use top-injection instead of
    # position-based injection. This avoids fragile index-based reinjection errors.
    # Case-local RUN lines appear at top of case (after header RUNs in final file).
    if case_runs_absolute:
        run_lines = [f"// RUN: {cmd}\n" for (_, cmd) in case_runs_absolute]
        final_content = "".join(run_lines) + replacement_content_clean
    else:
        final_content = replacement_content_clean

    return final_content, None


def _handle_unchanged_case(
    target_case: TestCase, file_path: Path, args: argparse.Namespace
) -> int:
    """Handle case where content is unchanged.

    Returns:
        Exit code (always SUCCESS).
    """
    if not args.quiet:
        console.note(
            f"Case {target_case.number} unchanged (content identical)", args=args
        )

    if args.json:
        payload = {
            "modified_files": 0,
            "modified_cases": 0,
            "unchanged_cases": 1,
            "dry_run": args.dry_run,
            "file_results": [
                {
                    "file": str(file_path),
                    "total_cases": 1,
                    "modified": 0,
                    "unchanged": 1,
                    "dry_run": args.dry_run,
                    "cases": [
                        {
                            "number": target_case.number,
                            "name": target_case.name,
                            "changed": False,
                            "reason": "content identical",
                        }
                    ],
                    "diff": "",
                }
            ],
            "errors": [],
            "warnings": [],
        }
        write_json_output(payload, args)

    return exit_codes.SUCCESS


def _handle_dry_run_output(
    original_content: str,
    new_content: str,
    file_path: Path,
    target_case: TestCase,
    args: argparse.Namespace,
) -> int:
    """Handle dry-run mode output (diff or JSON).

    Returns:
        Exit code (always SUCCESS).
    """
    diff_text = generate_unified_diff(
        original_content, new_content, str(file_path), args
    )

    if args.json:
        payload = {
            "modified_files": 0 if not diff_text else 1,
            "modified_cases": 0 if not diff_text else 1,
            "unchanged_cases": 1 if not diff_text else 0,
            "dry_run": True,
            "file_results": [
                {
                    "file": str(file_path),
                    "total_cases": 1,
                    "modified": 1 if diff_text else 0,
                    "unchanged": 0 if diff_text else 1,
                    "dry_run": True,
                    "cases": [
                        {
                            "number": target_case.number,
                            "name": target_case.name,
                            "changed": bool(diff_text),
                            "reason": "would modify" if diff_text else "no changes",
                        }
                    ],
                    "diff": diff_text if diff_text else "",
                }
            ],
            "errors": [],
            "warnings": [],
        }
        write_json_output(payload, args)
    else:
        if not diff_text:
            console.note(
                f"Case {target_case.number} unchanged (no differences)", args=args
            )
        else:
            if args.pretty:
                diff_text = formatting.colorize_diff(diff_text, pretty=True)
            console.out(diff_text)
            if not args.quiet:
                console.note(
                    f"Dry-run: would modify {file_path} (case {target_case.number})",
                    args=args,
                )

    return exit_codes.SUCCESS


def _write_modified_file(
    file_path: Path,
    new_content: str,
    original_stat: os.stat_result,
    original_hash: str,
    target_case: TestCase,
    args: argparse.Namespace,
) -> int:
    """Write modified file content after concurrency check.

    Returns:
        Exit code (SUCCESS or ERROR).
    """
    try:
        check_concurrent_modification(file_path, original_stat, original_hash, args)
    except RuntimeError as e:
        console.error(str(e), args=args)
        return exit_codes.ERROR

    try:
        fs.safe_write_text(
            file_path, new_content, atomic=True, backup=not args.no_backup
        )

        if not args.quiet:
            console.success(
                f"Case {target_case.number} replaced successfully in {file_path}",
                args=args,
            )

        if args.json:
            payload = {
                "modified_files": 1,
                "modified_cases": 1,
                "unchanged_cases": 0,
                "dry_run": False,
                "file_results": [
                    {
                        "file": str(file_path),
                        "total_cases": 1,
                        "modified": 1,
                        "unchanged": 0,
                        "dry_run": False,
                        "cases": [
                            {
                                "number": target_case.number,
                                "name": target_case.name,
                                "changed": True,
                                "reason": "content differs",
                            }
                        ],
                        "diff": "",
                    }
                ],
                "errors": [],
                "warnings": [],
            }
            write_json_output(payload, args)

        return exit_codes.SUCCESS
    except OSError as e:
        console.error(f"Failed to write file: {e}", args=args)
        return exit_codes.ERROR


def _extract_and_filter_run_lines(
    target_case: TestCase,
    file_path: Path,
    header_runs_raw: list[str],
    header_runs_normalized: list[str],
) -> tuple[list[str], list[str], list[str], list[tuple[int, str]]]:
    """Extract and filter RUN lines, separating header and case-local RUNs.

    Args:
        target_case: Target case being modified (contains cached run lines)
        file_path: Path to the test file (needed for re-reading when filtering)
        header_runs_raw: Raw header RUN lines
        header_runs_normalized: Normalized header RUN commands

    Returns:
        Tuple of (filtered_header_runs_raw, filtered_header_runs_normalized, case_run_cmds, case_runs_filtered)
    """
    # Extract case-local RUN lines and filter out header RUN lines.
    # (Header RUN lines are handled separately by build_file_content)
    original_case_runs_absolute = target_case.extract_local_run_lines()
    case_runs_filtered = _filter_header_runs_from_case_runs(
        original_case_runs_absolute, target_case.content
    )
    original_case_runs_cmds = [cmd for (_, cmd) in case_runs_filtered]

    # Also filter header RUN lines to exclude case-local RUN lines
    # (extract_run_lines includes ALL RUN lines before first non-comment line)
    #
    # Use normalized equality for filtering instead of substring matching to avoid
    # false positives (e.g., "iree-opt" substring matching in "// RUN: iree-opt ...").
    case_run_cmds_set = set(original_case_runs_cmds)

    # Filter normalized header runs (used for validation).
    # Note: header_runs_raw and header_runs_normalized have different lengths
    # when multi-line RUN commands are used (raw has one entry per physical line,
    # normalized has one entry per logical command), so we can't zip them together.
    filtered_normalized = [
        cmd for cmd in header_runs_normalized if cmd not in case_run_cmds_set
    ]

    # If we filtered out any commands, we need to re-extract the raw lines.
    # Otherwise, keep the original raw lines (preserves multi-line formatting).
    if len(filtered_normalized) != len(header_runs_normalized):
        # Re-extract raw lines, but only for the filtered normalized commands.
        # Build a set of normalized commands to keep for fast lookup.
        keep_cmds = set(filtered_normalized)

        # Re-read the file and extract only the raw lines for commands we're keeping.
        with open(file_path, encoding="utf-8") as f:
            file_lines = f.readlines()

        filtered_raw = []
        current_raw_lines = []
        current_normalized = None
        continuing = False

        for line in file_lines:
            line_stripped = line.rstrip()
            m = re.match(r"^\s*//\s*RUN:\s?(.*)$", line_stripped)

            if m:
                cmd = m.group(1).strip()

                if continuing and current_normalized is not None:
                    # Continuation of multi-line RUN command.
                    current_raw_lines.append(line_stripped)
                    current_normalized = current_normalized.rstrip("\\").strip()
                    current_normalized += " " + cmd.lstrip("\\").strip()
                else:
                    # Start of new RUN command.
                    current_raw_lines = [line_stripped]
                    current_normalized = cmd

                continuing = cmd.endswith("\\")

                if not continuing:
                    # Command is complete - check if we should keep it.
                    final_cmd = current_normalized.rstrip("\\").strip()
                    if final_cmd in keep_cmds:
                        filtered_raw.extend(current_raw_lines)
                    current_raw_lines = []
                    current_normalized = None
            elif line_stripped and not re.match(r"^\s*//", line_stripped):
                # Stop at first non-comment, non-empty line.
                break

        header_runs_raw = filtered_raw
    # else: No filtering needed, keep original raw lines

    return (
        header_runs_raw,
        filtered_normalized,
        original_case_runs_cmds,
        case_runs_filtered,
    )


def handle_text_mode(replacement_content: str, args: argparse.Namespace) -> int:
    """Handle text mode replacement.

    Args:
        replacement_content: Replacement text for single case
        args: Parsed arguments

    Returns:
        Exit code
    """
    # 1. Validate arguments.
    error_code = _validate_text_mode_args(args)
    if error_code is not None:
        return error_code

    # 2. Load and parse test file.
    (
        file_path,
        test_file_obj,
        all_cases,
        original_stat,
        original_hash,
        error_code,
    ) = _load_and_parse_test_file(args)
    if error_code is not None:
        return error_code

    # 3. Find target case.
    if args.case:
        target_case, error_code = _find_target_case_by_number(
            all_cases, file_path, args.case, args
        )
    else:
        target_case, error_code = _find_target_case_by_name(
            all_cases, file_path, args.name, args
        )

    if error_code is not None:
        return error_code

    # 4. Extract and validate RUN lines.
    replacement_runs, replacement_content_clean = extract_run_lines_from_string(
        replacement_content
    )

    try:
        header_runs_raw = test_file_obj.extract_run_lines(raw=True)
        header_runs_normalized = test_file_obj.extract_run_lines(raw=False)
    except (OSError, UnicodeDecodeError, ValueError) as e:
        console.error(f"Failed to extract header RUN lines: {e}", args=args)
        return exit_codes.ERROR

    # Extract and filter RUN lines, separating header and case-local RUNs.
    (
        header_runs_raw,
        header_runs_normalized,
        original_case_runs_cmds,
        case_runs_filtered,
    ) = _extract_and_filter_run_lines(
        target_case, file_path, header_runs_raw, header_runs_normalized
    )

    error_code = _validate_case_run_lines(
        replacement_runs, original_case_runs_cmds, header_runs_normalized, args
    )
    if error_code is not None:
        return error_code

    # 5. Validate replacement content.
    error_code = _validate_replacement_content_checks(
        replacement_content_clean, target_case, args
    )
    if error_code is not None:
        return error_code

    # 6. Build final content.
    final_content, error_code = _build_final_case_content(
        replacement_content,
        replacement_content_clean,
        replacement_runs,
        case_runs_filtered,
        args,
    )
    if error_code is not None:
        return error_code

    # 7. Replace case content and rebuild file.
    # If this is the first case and there are header RUN lines, we need to add back
    # the leading newlines that extract stripped (extract strips len(header_runs_raw)
    # newlines when including RUN lines in output, but build_file_content expects
    # the original format and will strip them again).
    if target_case.number == 1 and header_runs_raw:
        # Add back the newlines that were stripped by extract.
        final_content = ("\n" * len(header_runs_raw)) + final_content

    # Create content overrides dict for build_file_content.
    content_overrides = {target_case.number: final_content}

    # If --replace-run-lines was used, replacement RUNs become new header RUNs
    if args.replace_run_lines and replacement_runs:
        # Format as raw RUN lines (with "// RUN:" prefix)
        new_header_runs_raw = [f"// RUN: {cmd}" for cmd in replacement_runs]
        new_content = build_file_content(
            new_header_runs_raw, all_cases, content_overrides
        )
    else:
        new_content = build_file_content(header_runs_raw, all_cases, content_overrides)

    # 8. Handle unchanged, dry-run, or write cases.
    original_content = file_path.read_text(encoding="utf-8")
    if new_content == original_content:
        return _handle_unchanged_case(target_case, file_path, args)

    if args.dry_run:
        return _handle_dry_run_output(
            original_content, new_content, file_path, target_case, args
        )

    return _write_modified_file(
        file_path, new_content, original_stat, original_hash, target_case, args
    )


# Constants for JSON mode.
MAX_OVERRIDE_CASES_SHOWN = 5


# ==============================================================================
# Case-finding helper functions
# ==============================================================================


def _find_case_by_number(all_cases: list, case_num: int) -> TestCase | None:
    """Find case by number.

    Args:
        all_cases: List of TestCase objects from test_file.parse_test_file().
        case_num: Case number to find.

    Returns:
        TestCase object if found, None otherwise.
    """
    for case in all_cases:
        if case.number == case_num:
            return case
    return None


def _find_cases_by_name(all_cases: list, case_name: str) -> list[TestCase]:
    """Find all cases matching the given name.

    Args:
        all_cases: List of TestCase objects from test_file.parse_test_file().
        case_name: Case name to find (CHECK-LABEL value).

    Returns:
        List of matching TestCase objects (may be empty or contain multiple entries).
    """
    return [c for c in all_cases if c.name == case_name]


# ==============================================================================
# JSON mode helper functions (refactored from handle_json_mode)
# ==============================================================================


def _parse_and_validate_json_schema(
    json_input: str, args: argparse.Namespace
) -> list | None:
    """Parse JSON input and validate schema.

    Args:
        json_input: JSON input string.
        args: Parsed arguments.

    Returns:
        List of replacement objects if valid, None on error.
        Errors are reported via console.error().
    """
    # 1. Parse JSON.
    try:
        replacements = json.loads(json_input)
    except json.JSONDecodeError as e:
        console.error(
            f"Invalid JSON input at line {e.lineno}, column {e.colno}:\n{e.msg}",
            args=args,
        )
        return None

    # 2. Validate schema.
    validation_errors = []

    if not isinstance(replacements, list):
        console.error(
            "JSON input must be an array of replacement objects.\n"
            'Expected: [{"file": "test.mlir", "number": 2, "content": "..."}]',
            args=args,
        )
        return None

    if not replacements:
        console.error("JSON input is empty (no replacements specified)", args=args)
        return None

    # Validate each replacement object.
    for i, repl in enumerate(replacements):
        if not isinstance(repl, dict):
            validation_errors.append(
                f"Replacement {i + 1}: must be an object, got {type(repl).__name__}"
            )
            continue

        # Required: content field.
        if "content" not in repl:
            validation_errors.append(
                f"Replacement {i + 1}: missing required 'content' field"
            )
            continue

        # Required: either number or name.
        # Note: iree-lit-extract includes both, so we allow both (prefer number).
        has_number = "number" in repl
        has_name = "name" in repl
        if not has_number and not has_name:
            validation_errors.append(
                f"Replacement {i + 1}: must specify either 'number' or 'name' field"
            )

        # Required (unless CLI override): file field.
        if not args.test_file and "file" not in repl:
            validation_errors.append(
                f"Replacement {i + 1}: missing 'file' field (required when target file not specified via CLI)"
            )

    if validation_errors:
        console.error(
            "JSON schema validation failed:\n"
            + "\n".join(f"  - {e}" for e in validation_errors),
            args=args,
        )
        return None

    return replacements


def _group_by_file_and_check_overrides(
    replacements: list, args: argparse.Namespace
) -> dict:
    """Group replacements by file and check for CLI file overrides.

    Args:
        replacements: List of replacement dictionaries from JSON.
        args: Parsed arguments.

    Returns:
        Dictionary mapping Path -> list of (repl_idx, repl) tuples.
        Warns via console.warn() if CLI overrides JSON file fields.
    """
    file_groups = {}  # file_path -> list of (repl_idx, repl)

    for i, repl in enumerate(replacements):
        # CLI arg overrides JSON file field.
        target_file = args.test_file if args.test_file else repl.get("file")
        file_path = Path(target_file)

        if file_path not in file_groups:
            file_groups[file_path] = []
        file_groups[file_path].append((i, repl))

    # Check for CLI file overrides and warn.
    if args.test_file:
        overridden_count = 0
        overridden_cases = []
        for _file_path, replacements_for_file in file_groups.items():
            for _repl_idx, repl in replacements_for_file:
                if "file" in repl and repl["file"] != args.test_file:
                    overridden_count += 1
                    case_id = repl.get("number") or repl.get("name")
                    overridden_cases.append(str(case_id))

        if overridden_count > 0 and not args.quiet:
            case_list = ", ".join(overridden_cases[:MAX_OVERRIDE_CASES_SHOWN])
            if len(overridden_cases) > MAX_OVERRIDE_CASES_SHOWN:
                case_list += (
                    f", ... ({len(overridden_cases) - MAX_OVERRIDE_CASES_SHOWN} more)"
                )
            console.warn(
                f"CLI argument '{args.test_file}' overriding JSON 'file' field for "
                f"{overridden_count} replacement(s): cases {case_list}",
                args=args,
            )

    return file_groups


def _resolve_and_check_duplicates(
    file_groups: dict, args: argparse.Namespace
) -> dict | None:
    """Resolve selectors to canonical form and check for duplicate replacements.

    This implements two-pass duplicate detection:
    - Pass 1: Resolve all selectors (name/number) to canonical (file, case_number)
    - Pass 2: Check for duplicates using canonical identifiers

    This fixes the bug where {"number": 2} and {"name": "second_case"} were
    treated as different selectors even when targeting the same case.

    Args:
        file_groups: Dictionary from _group_by_file_and_check_overrides().
        args: Parsed arguments.

    Returns:
        Dictionary mapping (file_path, case_number) -> (repl_idx, repl, all_cases).
        Returns None on error (errors reported via console.error()).
    """
    all_errors = []
    resolved_replacements = (
        {}
    )  # (file_path, case_num) -> (repl_idx, repl, test_file_obj, all_cases)

    # PASS 1: Resolve all selectors to canonical (file, case_number) form.
    for file_path, replacements_for_file in file_groups.items():
        # Check file exists.
        if not file_path.exists():
            all_errors.append(
                {
                    "file": str(file_path),
                    "error": f"File not found: {file_path}",
                    "replacements": [i + 1 for i, _ in replacements_for_file],
                }
            )
            continue

        # Parse test file.
        try:
            test_file_obj = parse_test_file(file_path)
            all_cases = list(test_file_obj.cases)
        except (OSError, UnicodeDecodeError, ValueError) as e:
            all_errors.append(
                {
                    "file": str(file_path),
                    "error": f"Failed to parse test file: {e}",
                    "replacements": [i + 1 for i, _ in replacements_for_file],
                }
            )
            continue

        # Resolve each replacement to canonical case number.
        for repl_idx, repl in replacements_for_file:
            case_num = None

            # Determine case number from selector.
            if "number" in repl:
                case_num = repl["number"]
                # Verify case exists.
                case = _find_case_by_number(all_cases, case_num)
                if not case:
                    all_errors.append(
                        {
                            "file": str(file_path),
                            "replacement": repl_idx + 1,
                            "error": f"Case {case_num} not found (file has {len(all_cases)} cases)",
                        }
                    )
                    continue
            elif "name" in repl:
                case_name = repl["name"]
                # Find by name - check for duplicates FIRST (fixes P1 issue #4).
                name_matches = _find_cases_by_name(all_cases, case_name)
                if len(name_matches) > 1:
                    case_numbers = [c.number for c in name_matches]
                    all_errors.append(
                        {
                            "file": str(file_path),
                            "replacement": repl_idx + 1,
                            "error": (
                                f"Multiple cases named '{case_name}' found at numbers {case_numbers}. "
                                f"Fix: Use 'number' field instead of 'name' to disambiguate."
                            ),
                        }
                    )
                    continue
                if not name_matches:
                    error_msg = suggestions.format_case_name_error(
                        case_name, all_cases, file_path=str(file_path)
                    )
                    all_errors.append(
                        {
                            "file": str(file_path),
                            "replacement": repl_idx + 1,
                            "error": error_msg,
                        }
                    )
                    continue
                case_num = name_matches[0].number
            else:
                # Should be caught by schema validation, but be defensive.
                all_errors.append(
                    {
                        "file": str(file_path),
                        "replacement": repl_idx + 1,
                        "error": "Must specify either 'number' or 'name' field",
                    }
                )
                continue

            # Store resolved replacement with canonical key.
            canonical_key = (file_path, case_num)
            if canonical_key in resolved_replacements:
                # PASS 2: Duplicate detected!
                prev_idx = resolved_replacements[canonical_key][0]
                all_errors.append(
                    {
                        "file": str(file_path),
                        "error": (
                            f"Duplicate replacement for case {case_num}: "
                            f"entries {prev_idx + 1} and {repl_idx + 1}. "
                            f"Fix: Remove duplicate entries, keep only one."
                        ),
                    }
                )
            else:
                resolved_replacements[canonical_key] = (
                    repl_idx,
                    repl,
                    test_file_obj,
                    all_cases,
                )

    # Report errors if any.
    if all_errors:
        console.error(
            "Duplicate replacement entries detected:\n"
            + "\n".join(f"  - {e['error']}" for e in all_errors),
            args=args,
        )
        return None

    return resolved_replacements


def _build_final_json_content(
    use_replacement_runs: bool,
    original_case_runs_cmds: list[str],
    replacement_content_clean: str,
) -> str:
    """Build final content by prepending case-local RUN lines if needed.

    Args:
        use_replacement_runs: Whether replacement content uses new RUN lines.
        original_case_runs_cmds: Original case-local RUN commands.
        replacement_content_clean: Replacement content with RUN lines stripped.

    Returns:
        Final content with case-local RUN lines prepended if appropriate.
    """
    # When --replace-run-lines is used, replacement RUNs become new header RUNs
    # (handled by caller), so don't prepend case-local RUNs.
    if use_replacement_runs:
        return replacement_content_clean

    # Prepend case-local RUN lines to replacement content.
    # This mirrors text mode logic in _build_final_case_content().
    if original_case_runs_cmds:
        run_lines = [f"// RUN: {cmd}\n" for cmd in original_case_runs_cmds]
        return "".join(run_lines) + replacement_content_clean

    return replacement_content_clean


def _validate_single_replacement_content(
    file_path: Path,
    repl_idx: int,
    repl: dict,
    target_case: TestCase,
    header_runs_normalized: list[str],
    args: argparse.Namespace,
    all_errors: list,
    all_warnings: list,
) -> tuple[bool, dict] | None:
    """Validate a single replacement's content.

    Returns:
        Tuple of (use_replacement_runs, replacement_data) if valid, None if error.
    """
    # Check for name/number mismatch if both present.
    if "number" in repl and "name" in repl:
        case_name = repl["name"]
        if target_case.name != case_name:
            all_errors.append(
                {
                    "file": str(file_path),
                    "replacement": repl_idx + 1,
                    "error": (
                        f"Name/number mismatch: 'number' {repl['number']} has name '{target_case.name}', "
                        f"but 'name' field specifies '{case_name}'. "
                        f"Fix: Remove one field or ensure they match."
                    ),
                }
            )
            return None

    # Extract and validate RUN lines.
    replacement_content = repl["content"]
    replacement_runs, replacement_content_clean = extract_run_lines_from_string(
        replacement_content
    )

    # Extract case-local RUN lines from the original case and filter out header RUNs.
    # This mirrors the text mode logic for consistent validation.
    original_case_runs_absolute = target_case.extract_local_run_lines()
    case_runs_filtered = _filter_header_runs_from_case_runs(
        original_case_runs_absolute, target_case.content
    )
    original_case_runs_cmds = [cmd for (_, cmd) in case_runs_filtered]

    # Temporarily override args with per-case values if present.
    # Use context manager to ensure automatic restoration.
    with _TemporaryArgsOverride(
        args,
        replace_run_lines=repl.get("replace_run_lines", args.replace_run_lines),
        allow_empty=repl.get("allow_empty", args.allow_empty),
        require_label=repl.get("require_label", args.require_label),
    ):
        # Validate replacement RUN lines against both header and case-local RUNs.
        # This validates: replacement runs must match either header or case-local,
        # unless --replace-run-lines is set.
        validation_error = _validate_case_run_lines(
            replacement_runs, original_case_runs_cmds, header_runs_normalized, args
        )

        if validation_error is not None:
            all_errors.append(
                {
                    "file": str(file_path),
                    "replacement": repl_idx + 1,
                    "case": target_case.number,
                    "name": target_case.name,
                    "error": "RUN line validation failed (see stderr for details)",
                }
            )
            return None

        # Determine if we're using replacement RUNs.
        use_replacement_runs = False
        if replacement_runs and args.replace_run_lines:
            use_replacement_runs = True
            # Only warn if replacement RUNs differ from both header and case-local.
            norm_repl = [_normalize_run_cmd(c) for c in replacement_runs]
            norm_header = [_normalize_run_cmd(c) for c in header_runs_normalized]
            norm_case = [_normalize_run_cmd(c) for c in original_case_runs_cmds]
            if norm_repl != norm_header and norm_repl != norm_case:
                all_warnings.append(
                    {
                        "file": str(file_path),
                        "replacement": repl_idx + 1,
                        "case": target_case.number,
                        "warning": f"Replacing RUN lines with {len(replacement_runs)} new RUN line(s)",
                    }
                )

        # Validate empty content.
        if not replacement_content_clean.strip():
            if not args.allow_empty:
                all_errors.append(
                    {
                        "file": str(file_path),
                        "replacement": repl_idx + 1,
                        "case": target_case.number,
                        "name": target_case.name,
                        "error": (
                            "Replacement content is empty. "
                            "This would delete case content (delimiters remain). "
                            "Fix: Use --allow-empty to confirm this is intentional."
                        ),
                    }
                )
                return None

            all_warnings.append(
                {
                    "file": str(file_path),
                    "replacement": repl_idx + 1,
                    "case": target_case.number,
                    "warning": f"Replacing case {target_case.number} with empty content (--allow-empty specified)",
                }
            )

        # Validate CHECK-LABEL requirement.
        if args.require_label and not re.search(
            r"//\s*CHECK-LABEL:\s*@\S+", replacement_content_clean
        ):
            all_errors.append(
                {
                    "file": str(file_path),
                    "replacement": repl_idx + 1,
                    "case": target_case.number,
                    "name": target_case.name,
                    "error": (
                        f"Replacement content for case {target_case.number} lacks CHECK-LABEL. "
                        "Expected pattern: // CHECK-LABEL: @function_name"
                    ),
                }
            )
            return None

        # Verify with iree-opt.
        _, valid, error_msg = verification.verify_content_with_skip_check(
            replacement_content_clean,
            target_case.number,
            target_case.name,
            args,
            timeout=args.verify_timeout,
        )
        if not valid:
            all_errors.append(
                {
                    "file": str(file_path),
                    "replacement": repl_idx + 1,
                    "case": target_case.number,
                    "name": target_case.name,
                    "error": error_msg,
                }
            )
            return None

        # Build final content by prepending case-local RUN lines if needed.
        final_content = _build_final_json_content(
            use_replacement_runs, original_case_runs_cmds, replacement_content_clean
        )

    # Return validated replacement data.
    replacement_data = {
        "case": target_case,
        "content": final_content,
        "use_replacement_runs": use_replacement_runs,
        "replacement_runs": replacement_runs if use_replacement_runs else None,
        "repl_idx": repl_idx,
    }
    return use_replacement_runs, replacement_data


def _check_batch_run_line_consistency(
    file_path: Path,
    run_line_replacements: list[tuple[int, list[str]]],
    all_errors: list,
) -> None:
    """Check that all RUN line replacements for a file agree (P0 issue #3)."""
    if len(run_line_replacements) <= 1:
        return

    unique_run_lines = set(tuple(runs) for _, runs in run_line_replacements)
    if len(unique_run_lines) > 1:
        indices = [idx + 1 for idx, _ in run_line_replacements]
        all_errors.append(
            {
                "file": str(file_path),
                "error": (
                    f"Batch mode conflict: Multiple replacements for {file_path} "
                    f"specify different RUN lines (entries {indices}). "
                    f"Fix: Ensure all replacements agree on header RUN lines, or use text mode."
                ),
            }
        )


def _validate_file_group_replacements(
    file_path: Path,
    test_file_obj: TestFile,
    replacements_list: list[tuple[int, int, dict]],
    all_cases: list,
    args: argparse.Namespace,
    all_errors: list,
    all_warnings: list,
) -> dict | None:
    """Validate all replacements for a single file.

    Returns:
        Validation results dict for this file, or None if errors occurred.
    """
    # Record file state for concurrency detection.
    original_stat = file_path.stat()
    original_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()

    # Get header RUN lines.
    try:
        header_runs_normalized = test_file_obj.extract_run_lines(raw=False)
        header_runs_raw = test_file_obj.extract_run_lines(raw=True)
    except (OSError, UnicodeDecodeError, ValueError) as e:
        all_errors.append(
            {
                "file": str(file_path),
                "error": f"Failed to extract header RUN lines: {e}",
            }
        )
        return None

    # Validate each replacement.
    run_line_replacements = []
    replacements_data = []

    for case_num, repl_idx, repl in replacements_list:
        # Find target case.
        target_case = _find_case_by_number(all_cases, case_num)
        if not target_case:
            all_errors.append(
                {
                    "file": str(file_path),
                    "replacement": repl_idx + 1,
                    "error": f"Case {case_num} not found (internal error)",
                }
            )
            continue

        # Validate replacement content.
        result = _validate_single_replacement_content(
            file_path,
            repl_idx,
            repl,
            target_case,
            header_runs_normalized,
            args,
            all_errors,
            all_warnings,
        )
        if result is None:
            continue

        use_replacement_runs, replacement_data = result
        replacements_data.append(replacement_data)

        if use_replacement_runs:
            run_line_replacements.append(
                (repl_idx, replacement_data["replacement_runs"])
            )

    # Check RUN line batch consistency (P0 issue #3).
    _check_batch_run_line_consistency(file_path, run_line_replacements, all_errors)

    # Return validation results for this file.
    if not replacements_data:
        return None

    return {
        "all_cases": all_cases,
        "header_runs_raw": header_runs_raw,
        "replacements_data": replacements_data,
        "original_stat": original_stat,
        "original_hash": original_hash,
    }


def _validate_all_file_replacements(
    resolved_replacements: dict, args: argparse.Namespace
) -> tuple[dict, list] | None:
    """Validate all resolved replacements.

    Performs content validation for each replacement including:
    - Name/number consistency checking (when both provided)
    - RUN line extraction and comparison
    - Empty content checks
    - CHECK-LABEL requirements
    - iree-opt validation

    Args:
        resolved_replacements: Dictionary from _resolve_and_check_duplicates().
        args: Parsed arguments.

    Returns:
        Dictionary mapping file_path -> validation results dict.
        Returns None on error (errors reported via console.error()).
    """
    all_errors = []
    all_warnings = []
    validation_results = {}

    # Group by file for efficiency.
    file_to_replacements = {}
    for (file_path, case_num), (
        repl_idx,
        repl,
        test_file_obj,
        all_cases,
    ) in resolved_replacements.items():
        if file_path not in file_to_replacements:
            file_to_replacements[file_path] = {
                "test_file_obj": test_file_obj,
                "all_cases": all_cases,
                "replacements": [],
            }
        file_to_replacements[file_path]["replacements"].append(
            (case_num, repl_idx, repl)
        )

    # Validate each file's replacements.
    for file_path, file_data in file_to_replacements.items():
        file_results = _validate_file_group_replacements(
            file_path,
            file_data["test_file_obj"],
            file_data["replacements"],
            file_data["all_cases"],
            args,
            all_errors,
            all_warnings,
        )
        if file_results:
            validation_results[file_path] = file_results

    # Check if any errors occurred.
    if all_errors:
        console.error(
            f"Validation failed with {len(all_errors)} error(s):\n"
            + "\n".join(
                f"  - File {e.get('file', 'unknown')}, replacement {e.get('replacement', '?')}: {e['error']}"
                for e in all_errors
            )
            + "\nFix: Review individual error messages above for specific solutions.",
            args=args,
        )
        return None

    return validation_results, all_warnings


def _execute_batch_file_writes(
    validation_results: dict, all_warnings: list, args: argparse.Namespace
) -> tuple[int, int, int, list, list]:
    """Execute batch file writes after all validation passed.

    Args:
        validation_results: Dictionary from _validate_all_file_replacements().
        all_warnings: List of warning dicts from validation phase.
        args: Parsed arguments.

    Returns:
        Tuple of (exit_code, modified_files, modified_cases, file_results, all_errors).
    """
    modified_files = 0
    modified_cases = 0
    file_results = []
    all_errors = []

    for file_path, results in validation_results.items():
        all_cases = results["all_cases"]
        header_runs = results["header_runs_raw"]
        replacements_data = results["replacements_data"]

        # Build content overrides dict for all replacements.
        content_overrides = {}
        for repl_data in replacements_data:
            target_case = repl_data["case"]
            content_overrides[target_case.number] = repl_data["content"]

            # If this replacement uses different RUN lines, update header.
            # Note: At this point, batch mode RUN line conflicts have been detected.
            if repl_data["use_replacement_runs"]:
                # Convert normalized RUN lines to raw format (with // RUN: prefix).
                # This fixes P1 issue #5.
                header_runs = [
                    f"// RUN: {cmd}" for cmd in repl_data["replacement_runs"]
                ]

        # Rebuild file content.
        new_content = build_file_content(header_runs, all_cases, content_overrides)

        # Check if content actually changed.
        original_content = file_path.read_text(encoding="utf-8")
        if new_content == original_content:
            # No changes needed for this file.
            file_results.append(
                {
                    "file": str(file_path),
                    "modified": False,
                    "case_count": len(replacements_data),
                }
            )
            continue

        # Dry-run mode: don't write, but compute diffs.
        if args.dry_run:
            # Generate diff for this file.
            diff_text = generate_unified_diff(
                original_content, new_content, str(file_path), args
            )

            file_results.append(
                {
                    "file": str(file_path),
                    "total_cases": len(replacements_data),
                    "modified": len(replacements_data),
                    "unchanged": 0,
                    "dry_run": True,
                    "cases": [
                        {
                            "number": r["case"].number,
                            "name": r["case"].name,
                            "changed": True,
                        }
                        for r in replacements_data
                    ],
                    "diff": diff_text,  # Unified diff as string
                }
            )
            modified_files += 1
            modified_cases += len(replacements_data)
            continue

        # Check for concurrent modification before writing.
        try:
            check_concurrent_modification(
                file_path,
                results["original_stat"],
                results["original_hash"],
                args,
            )
        except RuntimeError as e:
            all_errors.append(
                {
                    "file": str(file_path),
                    "error": str(e),
                }
            )
            continue

        # Write file atomically with backup.
        try:
            fs.safe_write_text(
                file_path, new_content, atomic=True, backup=not args.no_backup
            )
            modified_files += 1
            modified_cases += len(replacements_data)

            file_results.append(
                {
                    "file": str(file_path),
                    "total_cases": len(replacements_data),
                    "modified": len(replacements_data),
                    "unchanged": 0,
                    "dry_run": False,
                    "cases": [
                        {
                            "number": r["case"].number,
                            "name": r["case"].name,
                            "changed": True,
                        }
                        for r in replacements_data
                    ],
                    "diff": "",
                }
            )
        except OSError as e:
            all_errors.append(
                {
                    "file": str(file_path),
                    "error": f"Failed to write file: {e}",
                }
            )

    # Determine exit code.
    exit_code = exit_codes.ERROR if all_errors else exit_codes.SUCCESS

    return exit_code, modified_files, modified_cases, file_results, all_errors


def handle_json_mode(json_input: str, args: argparse.Namespace) -> int:
    """Handle JSON mode replacement (batch operations).

    Refactored to use focused helper functions for maintainability.
    Implements two-pass duplicate detection to fix selector resolution bug.

    Accepts JSON array matching iree-lit-extract output format:
    [
      {
        "file": "test.mlir",      // optional if CLI arg provided
        "number": 2,              // XOR with name
        "name": "func_name",      // XOR with number
        "content": "...",         // replacement content
        // Optional fields from extract (ignored):
        "start_line": 10, "end_line": 20, "line_count": 11, "check_count": 3
      }
    ]

    Args:
        json_input: JSON input string.
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    # 1. Parse and validate JSON schema (~50 lines extracted).
    replacements = _parse_and_validate_json_schema(json_input, args)
    if replacements is None:
        return exit_codes.ERROR

    # 2. Group by file and check CLI overrides (~40 lines extracted).
    file_groups = _group_by_file_and_check_overrides(replacements, args)

    # 3. Resolve selectors to canonical form and check duplicates (~130 lines extracted).
    # This fixes the duplicate detection bug (P0 issue #2).
    resolved_replacements = _resolve_and_check_duplicates(file_groups, args)
    if resolved_replacements is None:
        return exit_codes.ERROR

    # 4. Validate all replacements (~250 lines extracted).
    # This includes P0 issue #3 fix (RUN line consistency in batch mode).
    validation_result = _validate_all_file_replacements(resolved_replacements, args)
    if validation_result is None:
        return exit_codes.ERROR
    validation_results, all_warnings = validation_result

    # 5. Execute batch writes (~100 lines extracted).
    # Returns exit_code based on whether errors occurred.
    (
        result_exit_code,
        modified_files,
        modified_cases,
        file_results,
        all_errors,
    ) = _execute_batch_file_writes(validation_results, all_warnings, args)

    # 6. Report final results and handle JSON output.
    # Only output messages if there were errors OR if not quiet mode.
    if all_errors:
        console.error(
            f"Write phase failed with {len(all_errors)} error(s)",
            args=args,
        )
    elif not args.quiet:
        if modified_files == 0:
            console.note("No files modified (all content identical)", args=args)
        else:
            console.success(
                f"Replaced {modified_cases} case(s) in {modified_files} file(s)",
                args=args,
            )

    # JSON output for both success and error cases.
    if args.json:
        payload = {
            "modified_files": modified_files,
            "modified_cases": modified_cases,
            "unchanged_cases": 0,  # JSON batch mode doesn't track unchanged cases currently
            "dry_run": args.dry_run,
            "file_results": file_results,
            "errors": all_errors,
            "warnings": all_warnings,
        }
        write_json_output(payload, args)

    return result_exit_code


if __name__ == "__main__":
    sys.exit(
        cli.run_with_argparse_suggestions(
            parse_arguments, main, _suggest_common_mistakes
        )
    )
