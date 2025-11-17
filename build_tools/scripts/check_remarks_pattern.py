#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Checks if a pattern exists in a remarks YAML file.
#
# This script is used to verify that expected ukernels or patterns appear in
# the compiler's remark output. It performs a simple string search in the
# remarks file.
#
# To check if a pattern exists in a remarks file:
#   python check_remarks_pattern.py remarks.yaml "pingpong_medium_f8E4M3FNUZ"

import argparse
import pathlib
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description="Remarks pattern checker")
    parser.add_argument(
        "remarks_file", help="Path to the remarks YAML file", type=pathlib.Path
    )
    parser.add_argument(
        "pattern", help="Pattern to search for (e.g., ukernel name)", type=str
    )
    args = parser.parse_args()
    return args


def main(args):
    # Check if file exists.
    if not args.remarks_file.exists():
        print(
            f"ERROR: Remarks file not found: {args.remarks_file}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Read file and search for pattern.
    try:
        with open(args.remarks_file, "r", encoding="utf-8") as f:
            content = f.read()
            if args.pattern in content:
                print(f"✓ Found expected pattern: {args.pattern}")
            else:
                print(
                    f"ERROR: Expected pattern '{args.pattern}' not found in remarks",
                    file=sys.stderr,
                )
                sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read remarks file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(parse_arguments())
