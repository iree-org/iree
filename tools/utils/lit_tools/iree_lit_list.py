# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""List test cases in MLIR lit test files.

Parses lit test files and displays metadata about test cases: case number,
function name (from CHECK-LABEL), line count, and CHECK pattern count.

Lit test files use `// -----` delimiters to separate test cases. This tool
identifies these boundaries and extracts metadata for each case.

Usage:
  # List all test cases with metadata
  iree-lit-list test.mlir

  # Just count test cases
  iree-lit-list test.mlir --count

  # Just list test case names
  iree-lit-list test.mlir --names

  # JSON for scripting
  iree-lit-list test.mlir --json > cases.json

Examples:
  $ iree-lit-list compiler/.../test/emplace_transients.mlir
  emplace_transients.mlir: 5 test cases
    1: @single_transient       (lines 10-45,  36 lines, 2 CHECK lines)
    2: @multiple_transients    (lines 47-89,  43 lines, 8 CHECK lines)
    3: @nested_execute         (lines 91-142, 52 lines, 12 CHECK lines)
    4: @external_transient     (lines 144-178, 35 lines, 4 CHECK lines)
    5: @scf_for                (lines 180-225, 46 lines, 10 CHECK lines)

  $ iree-lit-list test.mlir --count
  5

  $ iree-lit-list test.mlir --names
  @single_transient @multiple_transients @nested_execute @external_transient @scf_for

Exit codes:
  0 - Success
  1 - Error (file read failure, parse error)
  2 - File not found

See Also:
  iree-lit-extract - Extract individual test cases
  iree-lit-test - Run test cases in isolation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import from own category (as absolute path within sys.path)
from common import console, exit_codes

from lit_tools.core import cli, listing


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="List test cases in MLIR lit test files",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", help="Path to lit test file")
    parser.add_argument(
        "--count",
        action="store_true",
        help="Just print count of test cases",
    )
    parser.add_argument(
        "--names",
        action="store_true",
        help="Just print test case names (space-separated)",
    )
    # Common output flags (json/pretty/quiet)
    cli.add_common_output_flags(parser)
    # Note: --quiet on this tool suppresses the header line in text output.
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    """Main entry point.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (SUCCESS, ERROR, or NOT_FOUND)
    """
    file_path = Path(args.file)

    # Validate input
    if not file_path.exists():
        console.error(f"File not found: {file_path}", args=args)
        return exit_codes.NOT_FOUND

    # Parse test file
    try:
        cases = listing.get_cases(file_path)
    except (OSError, UnicodeDecodeError, ValueError) as e:
        console.error(f"Parsing failed: {e}", args=args)
        return exit_codes.ERROR

    # Enforce exclusivity uniformly via cli helpers
    if not cli.require_at_most_one(
        args,
        ["count", "names", "json"],
        "--json cannot be combined with --count or --names",
    ):
        return exit_codes.NOT_FOUND

    # Output based on mode
    if args.json:
        payload = listing.build_json_payload(file_path, cases)
        console.print_json(payload, args=args)
    elif args.count:
        # Just print count
        console.out(str(listing.count_cases(cases)))

    elif args.names:
        # Just print names
        console.out(listing.format_names(cases))

    else:
        # Full listing with metadata
        console.out(
            listing.format_text_listing(
                file_path,
                cases,
                pretty=getattr(args, "pretty", False),
                header=not getattr(args, "quiet", False),
            )
        )
    return exit_codes.SUCCESS


if __name__ == "__main__":
    sys.exit(main(parse_arguments()))
