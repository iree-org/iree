#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generate Unicode lookup tables for IREE's unicode utilities.

This script parses the Unicode Character Database (UCD) and generates
the unicode_tables.c file with category ranges, whitespace codepoints,
case mappings, and NFD decomposition tables.

Usage:
  # Generate tables (downloads UCD if needed):
  python unicode_tables_gen.py

  # Check that existing tables match what would be generated:
  python unicode_tables_gen.py --check

  # Use a specific UCD version:
  python unicode_tables_gen.py --unicode-version 15.1.0

  # Use local UCD files:
  python unicode_tables_gen.py --ucd-dir /path/to/ucd
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# Unicode Character Database URL template.
UCD_URL_TEMPLATE = "https://www.unicode.org/Public/{version}/ucd/{file}"

# Default Unicode version.
DEFAULT_UNICODE_VERSION = "15.1.0"

# Output file path (relative to repo root).
OUTPUT_PATH = "runtime/src/iree/base/internal/unicode_tables.c"

# Category flag values (must match unicode.h).
CATEGORY_FLAGS = {
    "L": "CAT_L",  # Letter
    "M": "CAT_M",  # Mark
    "N": "CAT_N",  # Number
    "P": "CAT_P",  # Punctuation
    "S": "CAT_S",  # Symbol
    "Z": "CAT_Z",  # Separator
    "C": "CAT_C",  # Other
}


def download_ucd_file(version: str, filename: str, cache_dir: Path) -> Path:
    """Download a UCD file if not cached."""
    cache_path = cache_dir / version / filename
    if cache_path.exists():
        return cache_path

    url = UCD_URL_TEMPLATE.format(version=version, file=filename)
    print(f"Downloading {url}...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, cache_path)
    except Exception as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        sys.exit(1)

    return cache_path


def parse_unicode_data(path: Path) -> dict:
    """Parse UnicodeData.txt to extract categories and case mappings."""
    data = {
        "categories": {},  # codepoint -> category
        "uppercase": {},  # codepoint -> uppercase mapping
        "lowercase": {},  # codepoint -> lowercase mapping
    }

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split(";")
            if len(fields) < 15:
                continue

            codepoint = int(fields[0], 16)
            name = fields[1]
            category = fields[2]
            uppercase = fields[12]
            lowercase = fields[13]

            # Handle ranges (e.g., "<CJK Ideograph, First>").
            if name.endswith(", First>"):
                range_start = codepoint
                continue
            elif name.endswith(", Last>"):
                range_end = codepoint
                for cp in range(range_start, range_end + 1):
                    data["categories"][cp] = category
                continue

            data["categories"][codepoint] = category

            if uppercase:
                data["uppercase"][codepoint] = int(uppercase, 16)
            if lowercase:
                data["lowercase"][codepoint] = int(lowercase, 16)

    return data


def parse_prop_list(path: Path) -> set:
    """Parse PropList.txt to extract White_Space property."""
    whitespace = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Remove comments.
            if "#" in line:
                line = line[: line.index("#")].strip()

            parts = line.split(";")
            if len(parts) < 2:
                continue

            codepoint_range = parts[0].strip()
            prop = parts[1].strip()

            if prop != "White_Space":
                continue

            if ".." in codepoint_range:
                start, end = codepoint_range.split("..")
                for cp in range(int(start, 16), int(end, 16) + 1):
                    whitespace.add(cp)
            else:
                whitespace.add(int(codepoint_range, 16))

    return whitespace


def parse_unicode_data_for_decomposition(path: Path, categories: dict) -> dict:
    """Parse UnicodeData.txt to extract NFD decompositions (simple 1:1 only).

    Args:
        path: Path to UnicodeData.txt
        categories: Dict mapping codepoint -> category (from parse_unicode_data)
    """
    nfd = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split(";")
            if len(fields) < 6:
                continue

            codepoint = int(fields[0], 16)
            decomposition = fields[5].strip()

            if not decomposition:
                continue

            # Skip compatibility decompositions (start with <tag>).
            if decomposition.startswith("<"):
                continue

            # Parse canonical decomposition.
            parts = decomposition.split()
            if len(parts) >= 1:
                # Take the first codepoint as the base character.
                base = int(parts[0], 16)
                # Only include if it's a simple 1:1 mapping to a base character.
                # Skip if the base is a combining mark (category M: Mn, Mc, Me).
                base_category = categories.get(base, "")
                if not base_category.startswith("M"):
                    nfd[codepoint] = base

    return nfd


def build_category_ranges(categories: dict) -> list:
    """Build compact category ranges from individual codepoint categories."""
    # Group by major category (first letter).
    by_major = {}
    for cp, cat in sorted(categories.items()):
        major = cat[0] if cat else "C"
        if major not in by_major:
            by_major[major] = []
        by_major[major].append(cp)

    # Build ranges.
    ranges = []
    for major, codepoints in by_major.items():
        codepoints = sorted(set(codepoints))
        if not codepoints:
            continue

        # Merge consecutive codepoints into ranges.
        start = codepoints[0]
        end = start
        for cp in codepoints[1:]:
            if cp == end + 1:
                end = cp
            else:
                ranges.append((start, end, major))
                start = cp
                end = cp
        ranges.append((start, end, major))

    # Sort by start codepoint.
    ranges.sort(key=lambda r: r[0])

    # Filter to only include non-ASCII ranges (ASCII is handled inline).
    ranges = [(s, e, c) for s, e, c in ranges if s >= 0x80]

    return ranges


def hex_compact(value: int) -> str:
    """Format a hex value compactly (no unnecessary leading zeros, 0 for zero)."""
    if value == 0:
        return "0"
    return f"0x{value:X}"


def pack_entries(
    entries: list, indent: str = "    ", max_line_length: int = 100
) -> list:
    """Pack entries into lines up to max_line_length."""
    lines = []
    current_line = indent
    for entry in entries:
        if len(current_line) + len(entry) + 1 > max_line_length:
            lines.append(current_line.rstrip())
            current_line = indent
        current_line += entry + " "
    if current_line.strip():
        lines.append(current_line.rstrip())
    return lines


def generate_tables_c(
    version: str,
    category_ranges: list,
    whitespace: set,
    case_mappings: dict,
    nfd_mappings: dict,
) -> str:
    """Generate the unicode_tables.c file content."""
    lines = []

    # Header with prominent disclaimer and clang-format off.
    lines.append(
        """\
// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off

//===----------------------------------------------------------------------===//
// **********************    DO NOT EDIT THIS FILE    **********************
//===----------------------------------------------------------------------===//
//
// Unicode lookup tables generated from the Unicode Character Database.
//
// Generated by: build_tools/scripts/unicode_tables_gen.py
// Unicode version: {version}
//
// To regenerate:
//   python3 build_tools/scripts/unicode_tables_gen.py
//
//===----------------------------------------------------------------------===//

#include "iree/base/internal/unicode.h"

//===----------------------------------------------------------------------===//
// Category range table
//===----------------------------------------------------------------------===//

#define L (1 << 0)
#define M (1 << 1)
#define N (1 << 2)
#define P (1 << 3)
#define S (1 << 4)
#define Z (1 << 5)
#define C (1 << 6)

const iree_unicode_category_range_t iree_unicode_category_ranges[] = {{""".format(
            version=version
        )
    )

    # Category ranges - packed.
    cat_entries = []
    for start, end, category in category_ranges:
        cat_entries.append(f"{{{hex_compact(start)},{hex_compact(end)},{category}}},")
    lines.extend(pack_entries(cat_entries))

    lines.append(
        """\
};
#undef L
#undef M
#undef N
#undef P
#undef S
#undef Z
#undef C

const iree_host_size_t iree_unicode_category_ranges_count =
    sizeof(iree_unicode_category_ranges) / sizeof(iree_unicode_category_ranges[0]);

//===----------------------------------------------------------------------===//
// Whitespace codepoints (White_Space property)
//===----------------------------------------------------------------------===//

const uint32_t iree_unicode_whitespace_codepoints[] = {"""
    )

    # Whitespace (non-ASCII only, ASCII is handled inline) - packed.
    ws_entries = [f"{hex_compact(cp)}," for cp in sorted(whitespace) if cp >= 0x80]
    lines.extend(pack_entries(ws_entries))

    lines.append(
        """\
};

const iree_host_size_t iree_unicode_whitespace_count =
    sizeof(iree_unicode_whitespace_codepoints) / sizeof(iree_unicode_whitespace_codepoints[0]);

//===----------------------------------------------------------------------===//
// Case mapping table
//===----------------------------------------------------------------------===//

const iree_unicode_case_mapping_t iree_unicode_case_mappings[] = {"""
    )

    # Case mappings (non-ASCII only) - packed.
    case_codepoints = sorted(
        set(case_mappings["uppercase"].keys()) | set(case_mappings["lowercase"].keys())
    )
    case_entries = []
    for cp in case_codepoints:
        if cp < 0x80:
            continue
        lower = case_mappings["lowercase"].get(cp, 0)
        upper = case_mappings["uppercase"].get(cp, 0)
        case_entries.append(
            f"{{{hex_compact(cp)},{hex_compact(lower)},{hex_compact(upper)}}},"
        )
    lines.extend(pack_entries(case_entries))

    lines.append(
        """\
};

const iree_host_size_t iree_unicode_case_mappings_count =
    sizeof(iree_unicode_case_mappings) / sizeof(iree_unicode_case_mappings[0]);

//===----------------------------------------------------------------------===//
// NFD base character table (simple 1:1 decomposition)
//===----------------------------------------------------------------------===//

const iree_unicode_nfd_mapping_t iree_unicode_nfd_mappings[] = {"""
    )

    # NFD mappings - packed.
    nfd_entries = []
    for cp in sorted(nfd_mappings.keys()):
        base = nfd_mappings[cp]
        nfd_entries.append(f"{{{hex_compact(cp)},{hex_compact(base)}}},")
    lines.extend(pack_entries(nfd_entries))

    lines.append(
        """\
};

const iree_host_size_t iree_unicode_nfd_mappings_count =
    sizeof(iree_unicode_nfd_mappings) / sizeof(iree_unicode_nfd_mappings[0]);
"""
    )

    return "\n".join(lines)


def find_repo_root() -> Path:
    """Find the IREE repository root."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "runtime" / "src" / "iree").exists():
            return current
        current = current.parent
    print("Error: Could not find IREE repository root", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate Unicode lookup tables")
    parser.add_argument(
        "--unicode-version",
        default=DEFAULT_UNICODE_VERSION,
        help=f"Unicode version (default: {DEFAULT_UNICODE_VERSION})",
    )
    parser.add_argument(
        "--ucd-dir",
        type=Path,
        help="Path to local UCD directory (downloads if not specified)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that existing tables match what would be generated",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: auto-detect from repo root)",
    )
    args = parser.parse_args()

    repo_root = find_repo_root()
    output_path = args.output or repo_root / OUTPUT_PATH

    # Determine UCD file locations.
    if args.ucd_dir:
        ucd_dir = args.ucd_dir
        unicode_data_path = ucd_dir / "UnicodeData.txt"
        prop_list_path = ucd_dir / "PropList.txt"
    else:
        cache_dir = Path.home() / ".cache" / "iree" / "ucd"
        unicode_data_path = download_ucd_file(
            args.unicode_version, "UnicodeData.txt", cache_dir
        )
        prop_list_path = download_ucd_file(
            args.unicode_version, "PropList.txt", cache_dir
        )

    print(f"Parsing UnicodeData.txt...")
    unicode_data = parse_unicode_data(unicode_data_path)

    print(f"Parsing PropList.txt...")
    whitespace = parse_prop_list(prop_list_path)

    print(f"Parsing NFD decompositions...")
    nfd_mappings = parse_unicode_data_for_decomposition(
        unicode_data_path, unicode_data["categories"]
    )

    print(f"Building category ranges...")
    category_ranges = build_category_ranges(unicode_data["categories"])

    print(f"Generating tables...")
    case_mappings = {
        "uppercase": unicode_data["uppercase"],
        "lowercase": unicode_data["lowercase"],
    }
    content = generate_tables_c(
        args.unicode_version,
        category_ranges,
        whitespace,
        case_mappings,
        nfd_mappings,
    )

    if args.check:
        if not output_path.exists():
            print(f"Error: {output_path} does not exist", file=sys.stderr)
            sys.exit(1)
        existing = output_path.read_text()
        if existing == content:
            print("Tables are up to date.")
            sys.exit(0)
        else:
            print(f"Error: {output_path} is out of date", file=sys.stderr)
            print("Run 'python unicode_tables_gen.py' to regenerate", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Writing {output_path}...")
        output_path.write_text(content)
        print(
            f"Done. Generated {len(category_ranges)} category ranges, "
            f"{len(whitespace)} whitespace codepoints, "
            f"{len(case_mappings['uppercase']) + len(case_mappings['lowercase'])} case mappings, "
            f"{len(nfd_mappings)} NFD mappings."
        )


if __name__ == "__main__":
    main()
