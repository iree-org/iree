#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generate Unicode lookup tables for IREE's unicode utilities.

This script parses the Unicode Character Database (UCD) and generates
the unicode_tables.c file with category ranges, whitespace codepoints,
case mappings, NFD decomposition, CCC (Canonical Combining Class),
NFC composition tables, and NFKD compatibility decomposition tables.

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

# Default Unicode version for main tables.
DEFAULT_UNICODE_VERSION = "15.1.0"

# Unicode version for HuggingFace-compatible Mn detection.
# HuggingFace tokenizers uses the unicode_categories crate which has older data.
# Mn characters added after this version should NOT be stripped by strip_accents.
HUGGINGFACE_UNICODE_VERSION = "8.0.0"

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

# Mark subcategory flag for Mn (Nonspacing Mark).
# This is combined with CAT_M, so Mn codepoints have both bits set.
MARK_NONSPACING_FLAG = "MN"


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
    """Parse UnicodeData.txt to extract categories, case mappings, and CCC."""
    data = {
        "categories": {},  # codepoint -> category
        "uppercase": {},  # codepoint -> uppercase mapping
        "lowercase": {},  # codepoint -> lowercase mapping
        "ccc": {},  # codepoint -> canonical combining class (non-zero only)
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
            ccc = int(fields[3]) if fields[3] else 0  # Canonical Combining Class
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

            # Store CCC only for non-zero values (combining marks).
            if ccc > 0:
                data["ccc"][codepoint] = ccc

            if uppercase:
                data["uppercase"][codepoint] = int(uppercase, 16)
            if lowercase:
                data["lowercase"][codepoint] = int(lowercase, 16)

    return data


def parse_legacy_mn_codepoints(path: Path) -> set:
    """Parse UnicodeData.txt to extract Mn (Mark, Nonspacing) codepoints.

    This is used to identify which characters were Mn in an older Unicode
    version, for HuggingFace tokenizers compatibility. Characters that became
    Mn in later Unicode versions should NOT be stripped during accent stripping.
    """
    mn_codepoints = set()

    with open(path, "r", encoding="utf-8") as f:
        range_start = None
        range_category = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split(";")
            if len(fields) < 3:
                continue

            codepoint = int(fields[0], 16)
            name = fields[1]
            category = fields[2]

            # Handle ranges (e.g., "<CJK Ideograph, First>").
            if name.endswith(", First>"):
                range_start = codepoint
                range_category = category
                continue
            elif name.endswith(", Last>"):
                if range_category == "Mn":
                    for cp in range(range_start, codepoint + 1):
                        mn_codepoints.add(cp)
                range_start = None
                range_category = None
                continue

            if category == "Mn":
                mn_codepoints.add(codepoint)

    return mn_codepoints


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
    """Parse UnicodeData.txt to extract NFD decompositions.

    Returns a dict mapping codepoint -> (base, combining, is_singleton) where:
    - base: The base character (first codepoint of decomposition)
    - combining: The combining mark (second codepoint), or 0 if singleton
    - is_singleton: True if decomposition has exactly 1 codepoint

    Singleton decompositions (like CJK Compatibility Ideographs) are 1:1
    mappings that should be applied during NFC normalization. Non-singletons
    (like é → e + combining acute) include both base and combining mark for
    full NFD decomposition.

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
                # Skip if the base is a combining mark (category M: Mn, Mc, Me).
                base_category = categories.get(base, "")
                if not base_category.startswith("M"):
                    is_singleton = len(parts) == 1
                    # Get the combining mark if present (second codepoint).
                    combining = int(parts[1], 16) if len(parts) >= 2 else 0
                    nfd[codepoint] = (base, combining, is_singleton)

    return nfd


def parse_composition_exclusions(path: Path) -> set:
    """Parse CompositionExclusions.txt to get excluded codepoints.

    These are codepoints that have canonical decompositions but should NOT
    be composed during NFC normalization.
    """
    exclusions = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Remove inline comments.
            if "#" in line:
                line = line[: line.index("#")].strip()

            if not line:
                continue

            # Each line is a single codepoint.
            exclusions.add(int(line, 16))

    return exclusions


def parse_derived_normalization_props(path: Path) -> set:
    """Parse DerivedNormalizationProps.txt to get Full_Composition_Exclusion.

    This includes all characters that should be excluded from NFC composition:
    - Characters from CompositionExclusions.txt
    - Singletons (decompose to single character)
    - Non-starter decompositions
    """
    exclusions = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Remove inline comments.
            if "#" in line:
                line = line[: line.index("#")].strip()

            if not line:
                continue

            parts = line.split(";")
            if len(parts) < 2:
                continue

            prop = parts[1].strip()
            if prop != "Full_Composition_Exclusion":
                continue

            codepoint_range = parts[0].strip()
            if ".." in codepoint_range:
                start, end = codepoint_range.split("..")
                for cp in range(int(start, 16), int(end, 16) + 1):
                    exclusions.add(cp)
            else:
                exclusions.add(int(codepoint_range, 16))

    return exclusions


def parse_nfc_qc_no(path: Path) -> set:
    """Parse DerivedNormalizationProps.txt to get NFC_QC=No codepoints.

    These are characters that can NEVER appear in NFC-normalized text.
    They must be canonically decomposed during NFC normalization.
    """
    nfc_qc_no = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Remove inline comments.
            if "#" in line:
                line = line[: line.index("#")].strip()

            if not line:
                continue

            parts = line.split(";")
            if len(parts) < 2:
                continue

            prop = parts[1].strip()
            if prop != "NFC_QC":
                continue

            # Check for "N" value (NFC_QC=No).
            if len(parts) < 3 or parts[2].strip() != "N":
                continue

            codepoint_range = parts[0].strip()
            if ".." in codepoint_range:
                start, end = codepoint_range.split("..")
                for cp in range(int(start, 16), int(end, 16) + 1):
                    nfc_qc_no.add(cp)
            else:
                nfc_qc_no.add(int(codepoint_range, 16))

    return nfc_qc_no


def parse_all_canonical_decompositions(path: Path) -> dict:
    """Parse UnicodeData.txt to extract ALL canonical decompositions.

    Returns a dict mapping codepoint -> list of codepoints (the decomposition).
    Unlike parse_unicode_data_for_decomposition, this does NOT filter by
    base category — it captures every canonical decomposition.
    """
    decompositions = {}

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
            decompositions[codepoint] = [int(p, 16) for p in parts]

    return decompositions


def parse_compatibility_decompositions(
    path: Path, canonical_decompositions: dict
) -> dict:
    """Parse UnicodeData.txt to extract compatibility decompositions for NFKD.

    Returns a dict mapping codepoint -> list of fully-expanded codepoints.
    The expansion is fully recursive: compatibility decompositions are applied,
    then canonical decompositions are applied to each result, recursively.

    Compatibility decompositions are marked in UnicodeData.txt with a tag prefix
    like <compat>, <font>, <circle>, <wide>, etc.
    """
    # First, collect raw compatibility decompositions (with tag prefix).
    raw_compat = {}

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

            # Only process compatibility decompositions (start with <tag>).
            if not decomposition.startswith("<"):
                continue

            # Parse: "<tag> codepoint1 codepoint2 ..."
            # Skip the tag (everything up to and including first '>').
            tag_end = decomposition.find(">")
            if tag_end == -1:
                continue

            parts = decomposition[tag_end + 1 :].strip().split()
            if not parts:
                continue

            raw_compat[codepoint] = [int(p, 16) for p in parts]

    # Now fully expand each compatibility decomposition.
    # NFKD = apply compatibility decomposition, then recursively apply
    # canonical decomposition to each result.
    def expand_canonical(cp, visited=None):
        """Recursively expand canonical decompositions."""
        if visited is None:
            visited = set()
        if cp in visited:
            return [cp]  # Prevent infinite recursion.
        visited.add(cp)

        if cp not in canonical_decompositions:
            return [cp]

        result = []
        for target in canonical_decompositions[cp]:
            result.extend(expand_canonical(target, visited.copy()))
        return result

    def expand_nfkd(cp, visited=None):
        """Recursively expand NFKD (compatibility + canonical)."""
        if visited is None:
            visited = set()
        if cp in visited:
            return [cp]
        visited.add(cp)

        # Check for compatibility decomposition first.
        if cp in raw_compat:
            result = []
            for target in raw_compat[cp]:
                # Recursively apply NFKD to each target.
                result.extend(expand_nfkd(target, visited.copy()))
            return result

        # No compatibility decomposition - apply canonical decomposition.
        if cp in canonical_decompositions:
            result = []
            for target in canonical_decompositions[cp]:
                result.extend(expand_nfkd(target, visited.copy()))
            return result

        return [cp]

    # Build fully-expanded NFKD decompositions.
    nfkd_decomps = {}
    for cp in sorted(raw_compat.keys()):
        expanded = expand_nfkd(cp)
        if len(expanded) == 1 and expanded[0] == cp:
            # No decomposition - skip.
            continue
        nfkd_decomps[cp] = expanded

    return nfkd_decomps


def build_nfc_canonical_decompositions(
    nfc_qc_no: set, all_decompositions: dict, existing_nfd_singletons: set
) -> dict:
    """Build fully-expanded canonical decompositions for NFC_QC=No characters.

    For each NFC_QC=No character, recursively expands the canonical
    decomposition until all codepoints are themselves in NFC form.

    Excludes:
    - Hangul syllables (U+AC00-U+D7A3): handled algorithmically at runtime
    - Characters already in the existing NFD singleton table

    Returns a dict mapping codepoint -> list of fully-expanded codepoints.
    """
    HANGUL_S_BASE = 0xAC00
    HANGUL_S_COUNT = 11172

    def expand(cp, visited=None):
        """Recursively expand a codepoint to its NFC canonical form."""
        if visited is None:
            visited = set()
        if cp in visited:
            return [cp]  # Prevent infinite recursion.
        visited.add(cp)

        if cp not in all_decompositions:
            return [cp]

        result = []
        for target in all_decompositions[cp]:
            # Recursively expand if the target is also NFC_QC=No.
            if target in nfc_qc_no:
                result.extend(expand(target, visited))
            else:
                result.append(target)
        return result

    nfc_decomps = {}
    for cp in sorted(nfc_qc_no):
        # Skip Hangul syllables (handled algorithmically).
        if HANGUL_S_BASE <= cp < HANGUL_S_BASE + HANGUL_S_COUNT:
            continue

        # Skip characters already handled by the existing NFD singleton table.
        if cp in existing_nfd_singletons:
            continue

        # Character must have a canonical decomposition.
        if cp not in all_decompositions:
            continue

        expanded = expand(cp)
        if len(expanded) > 3:
            # Safety: max 3 codepoints per entry (current Unicode max is 3).
            raise ValueError(
                f"U+{cp:04X} expands to {len(expanded)} codepoints: "
                f"{[f'U+{c:04X}' for c in expanded]}"
            )
        nfc_decomps[cp] = expanded

    return nfc_decomps


def build_nfc_composition_pairs(path: Path, exclusions: set) -> dict:
    """Build NFC composition table from canonical decompositions.

    This creates the reverse mapping: (base, combining) -> composed.
    Only includes "Primary Composites" - characters that:
    - Have a canonical decomposition to exactly 2 codepoints
    - Are not in the Full_Composition_Exclusion set
    - The first codepoint (base) has CCC=0 (is a starter)

    Args:
        path: Path to UnicodeData.txt
        exclusions: Set of codepoints excluded from composition
    """
    pairs = {}  # (base, combining) -> composed

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split(";")
            if len(fields) < 6:
                continue

            composed = int(fields[0], 16)
            decomposition = fields[5].strip()

            if not decomposition:
                continue

            # Skip compatibility decompositions.
            if decomposition.startswith("<"):
                continue

            # Skip excluded codepoints.
            if composed in exclusions:
                continue

            # Parse canonical decomposition.
            parts = decomposition.split()

            # Only include 2-character decompositions.
            if len(parts) != 2:
                continue

            base = int(parts[0], 16)
            combining = int(parts[1], 16)

            pairs[(base, combining)] = composed

    return pairs


def build_category_ranges(categories: dict, legacy_mn_codepoints: set) -> list:
    """Build compact category ranges from individual codepoint categories.

    Returns a list of (start, end, flags_string) tuples, where flags_string
    is the C expression for the category flags (e.g., "M|MN" for Mn).

    The Mn (Mark, Nonspacing) subcategory is tracked separately so that
    accent stripping can distinguish Mn from Mc (Spacing Combining) and
    Me (Enclosing). This is required for HuggingFace compatibility.

    Only codepoints that were Mn in the legacy Unicode version (8.0) get the
    MN flag. Characters that became Mn in later Unicode versions should NOT
    be stripped during accent stripping to match HuggingFace behavior.
    """
    # Group by (major_category, is_legacy_mn) tuple.
    # is_legacy_mn is True only if the codepoint was Mn in Unicode 8.0.
    by_key = {}
    for cp, cat in sorted(categories.items()):
        major = cat[0] if cat else "C"
        # Only set MN flag for codepoints that were Mn in the legacy version.
        is_legacy_mn = cp in legacy_mn_codepoints
        key = (major, is_legacy_mn)
        if key not in by_key:
            by_key[key] = []
        by_key[key].append(cp)

    # Build ranges.
    ranges = []
    for (major, is_legacy_mn), codepoints in by_key.items():
        codepoints = sorted(set(codepoints))
        if not codepoints:
            continue

        # Determine the flags string for this category.
        if is_legacy_mn:
            flags = f"{major}|MN"  # e.g., "M|MN"
        else:
            flags = major  # e.g., "M", "L", etc.

        # Merge consecutive codepoints into ranges.
        start = codepoints[0]
        end = start
        for cp in codepoints[1:]:
            if cp == end + 1:
                end = cp
            else:
                ranges.append((start, end, flags))
                start = cp
                end = cp
        ranges.append((start, end, flags))

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


def format_array_compact(values: list) -> str:
    """Format an array of integers, stripping trailing zeros.

    Arrays in C are automatically zero-initialized for omitted elements,
    so we can strip trailing zeros to save space. Always keeps at least
    one element to avoid empty arrays.

    Examples:
        [0x641, 0, 0, 0] -> "0x641"
        [0x12, 0x34, 0, 0] -> "0x12,0x34"
        [0, 0, 0, 0] -> "0"
    """
    # Strip trailing zeros but keep at least one element.
    while len(values) > 1 and values[-1] == 0:
        values = values[:-1]
    return ",".join(hex_compact(v) for v in values)


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
    ccc_mappings: dict,
    nfc_pairs: dict,
    nfc_decompositions: dict,
    nfkd_decompositions: dict,
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
#define MN (1 << 7)  // Mn (Nonspacing Mark) subcategory

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
#undef MN

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
// Case mapping tables (split by direction for cache efficiency)
//===----------------------------------------------------------------------===//
// Split into separate lowercase and uppercase tables to eliminate redundant
// zero fields. 99.9% of codepoints map in only one direction, so the unified
// table wasted 33% of its space (one field always zero).
//
// Splitting reduces struct size from 12 to 8 bytes (33% smaller), improving
// cache efficiency by 60% (8 entries per 64-byte cache line vs 5)."""
    )

    # Separate case mappings by direction (non-ASCII only).
    lowercase_only = []
    uppercase_only = []
    dual_mappings = []

    all_codepoints = sorted(
        set(case_mappings["uppercase"].keys()) | set(case_mappings["lowercase"].keys())
    )

    for cp in all_codepoints:
        if cp < 0x80:
            continue
        lower = case_mappings["lowercase"].get(cp, 0)
        upper = case_mappings["uppercase"].get(cp, 0)

        if lower != 0 and upper != 0:
            # Both directions (rare - only 4 in Unicode 15.1).
            dual_mappings.append((cp, lower, upper))
        elif lower != 0:
            # Lowercase only.
            lowercase_only.append((cp, lower))
        elif upper != 0:
            # Uppercase only.
            uppercase_only.append((cp, upper))

    # Generate lowercase mappings table.
    lines.append(
        """

const iree_unicode_case_mapping_simple_t iree_unicode_lowercase_mappings[] = {"""
    )
    lowercase_entries = [
        f"{{{hex_compact(cp)},{hex_compact(target)}}}," for cp, target in lowercase_only
    ]
    lines.extend(pack_entries(lowercase_entries))

    lines.append(
        """\
};

const iree_host_size_t iree_unicode_lowercase_mappings_count =
    sizeof(iree_unicode_lowercase_mappings) / sizeof(iree_unicode_lowercase_mappings[0]);"""
    )

    # Generate uppercase mappings table.
    lines.append(
        """

const iree_unicode_case_mapping_simple_t iree_unicode_uppercase_mappings[] = {"""
    )
    uppercase_entries = [
        f"{{{hex_compact(cp)},{hex_compact(target)}}}," for cp, target in uppercase_only
    ]
    lines.extend(pack_entries(uppercase_entries))

    lines.append(
        """\
};

const iree_host_size_t iree_unicode_uppercase_mappings_count =
    sizeof(iree_unicode_uppercase_mappings) / sizeof(iree_unicode_uppercase_mappings[0]);

//===----------------------------------------------------------------------===//
// NFD decomposition table (base + combining mark)
//===----------------------------------------------------------------------===//

const iree_unicode_nfd_mapping_t iree_unicode_nfd_mappings[] = {"""
    )

    # NFD mappings - packed.
    # The base field has the singleton flag (0x80000000) OR'd in for 1:1 decompositions.
    # The combining field is 0 for singletons, else the combining mark codepoint.
    nfd_entries = []
    for cp in sorted(nfd_mappings.keys()):
        base, combining, is_singleton = nfd_mappings[cp]
        if is_singleton:
            base |= 0x80000000  # Set singleton flag
        nfd_entries.append(
            f"{{{hex_compact(cp)},{hex_compact(base)},{hex_compact(combining)}}},"
        )
    lines.extend(pack_entries(nfd_entries))

    lines.append(
        """\
};

const iree_host_size_t iree_unicode_nfd_mappings_count =
    sizeof(iree_unicode_nfd_mappings) / sizeof(iree_unicode_nfd_mappings[0]);

//===----------------------------------------------------------------------===//
// Canonical Combining Class (CCC) table
//===----------------------------------------------------------------------===//

const iree_unicode_ccc_entry_t iree_unicode_ccc_entries[] = {"""
    )

    # CCC entries - packed.
    ccc_entries = []
    for cp in sorted(ccc_mappings.keys()):
        ccc = ccc_mappings[cp]
        ccc_entries.append(f"{{{hex_compact(cp)},{ccc}}},")
    lines.extend(pack_entries(ccc_entries))

    lines.append(
        """\
};

const iree_host_size_t iree_unicode_ccc_entries_count =
    sizeof(iree_unicode_ccc_entries) / sizeof(iree_unicode_ccc_entries[0]);

//===----------------------------------------------------------------------===//
// NFC composition pairs table
//===----------------------------------------------------------------------===//

const iree_unicode_nfc_pair_t iree_unicode_nfc_pairs[] = {"""
    )

    # NFC pairs - sorted by (base, combining) for binary search.
    nfc_entries = []
    for (base, combining), composed in sorted(nfc_pairs.items()):
        nfc_entries.append(
            f"{{{hex_compact(base)},{hex_compact(combining)},{hex_compact(composed)}}},"
        )
    lines.extend(pack_entries(nfc_entries))

    lines.append(
        """\
};

const iree_host_size_t iree_unicode_nfc_pairs_count =
    sizeof(iree_unicode_nfc_pairs) / sizeof(iree_unicode_nfc_pairs[0]);

//===----------------------------------------------------------------------===//
// NFC canonical decomposition table (NFC_QC=No characters)
//===----------------------------------------------------------------------===//
// Characters that can never appear in NFC-normalized text and must be
// decomposed. Each entry contains the fully-expanded recursive canonical
// decomposition (max 3 codepoints). Sorted by source codepoint for binary
// search.
//
// Excludes:
// - Hangul syllables (algorithmic decomposition at runtime)
// - CJK Compatibility Ideographs (already in the NFD singleton table)

const iree_unicode_nfc_decomp_t iree_unicode_nfc_decompositions[] = {"""
    )

    # NFC decompositions - sorted by codepoint for binary search.
    nfc_decomp_entries = []
    for cp in sorted(nfc_decompositions.keys()):
        targets = nfc_decompositions[cp]
        # Pad to 3 entries (trailing zeros stripped by format_array_compact).
        padded = targets + [0] * (3 - len(targets))
        target_str = format_array_compact(padded)
        nfc_decomp_entries.append(f"{{{hex_compact(cp)},{{{target_str}}}}},")
    lines.extend(pack_entries(nfc_decomp_entries))

    lines.append(
        """\
};

const iree_host_size_t iree_unicode_nfc_decompositions_count =
    sizeof(iree_unicode_nfc_decompositions) / sizeof(iree_unicode_nfc_decompositions[0]);

//===----------------------------------------------------------------------===//
// NFKD compatibility decomposition table
//===----------------------------------------------------------------------===//
// Characters with compatibility decompositions (marked with <tag> in UnicodeData.txt).
// Each entry contains the fully-expanded NFKD decomposition (compatibility + canonical).
// Sorted by source codepoint for binary search.
//
// Structure:
// - Short decompositions (length <= 4): stored inline in the entry
// - Long decompositions (length > 4): stored as offset into overflow array
//
// Max decomposition length is 18 codepoints (U+FDFA: Arabic ligature).
// Length distribution: ~70% length 1, ~17% length 2, ~11% length 3, ~2% length 4+."""
    )

    # Separate short (inline) and long (overflow) decompositions.
    # Short: length <= 4, stored inline.
    # Long: length > 4, stored with offset into overflow array.
    INLINE_MAX = 4
    overflow_targets = []
    nfkd_entries = []

    for cp in sorted(nfkd_decompositions.keys()):
        targets = nfkd_decompositions[cp]
        length = len(targets)

        if length <= INLINE_MAX:
            # Inline storage: pad to 4 entries (trailing zeros stripped by format_array_compact).
            padded = targets + [0] * (INLINE_MAX - length)
            inline_str = format_array_compact(padded)
            nfkd_entries.append(f"{{{hex_compact(cp)},{length},0,{{{inline_str}}}}},")
        else:
            # Overflow storage: store offset into overflow array.
            offset = len(overflow_targets)
            overflow_targets.extend(targets)
            # Inline array is all zeros for overflow entries (compacts to single 0).
            inline_str = format_array_compact([0, 0, 0, 0])
            nfkd_entries.append(
                f"{{{hex_compact(cp)},{length},{offset},{{{inline_str}}}}},"
            )

    lines.append(
        """
const iree_unicode_nfkd_mapping_t iree_unicode_nfkd_mappings[] = {"""
    )
    lines.extend(pack_entries(nfkd_entries))

    lines.append(
        """\
};

const iree_host_size_t iree_unicode_nfkd_mappings_count =
    sizeof(iree_unicode_nfkd_mappings) / sizeof(iree_unicode_nfkd_mappings[0]);"""
    )

    # Generate overflow array for long decompositions.
    if overflow_targets:
        lines.append(
            """
// Overflow storage for NFKD decompositions with length > 4.
// Entries in iree_unicode_nfkd_mappings with length > 4 use the offset field
// to index into this array.

const uint32_t iree_unicode_nfkd_overflow[] = {"""
        )
        overflow_entries = [f"{hex_compact(t)}," for t in overflow_targets]
        lines.extend(pack_entries(overflow_entries))
        lines.append(
            """\
};

const iree_host_size_t iree_unicode_nfkd_overflow_count =
    sizeof(iree_unicode_nfkd_overflow) / sizeof(iree_unicode_nfkd_overflow[0]);"""
        )
    else:
        lines.append(
            """
// No long NFKD decompositions (all fit inline).
const uint32_t* iree_unicode_nfkd_overflow = NULL;
const iree_host_size_t iree_unicode_nfkd_overflow_count = 0;"""
        )

    lines.append("")  # Trailing newline.

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
        derived_norm_props_path = ucd_dir / "DerivedNormalizationProps.txt"
    else:
        cache_dir = Path.home() / ".cache" / "iree" / "ucd"
        unicode_data_path = download_ucd_file(
            args.unicode_version, "UnicodeData.txt", cache_dir
        )
        prop_list_path = download_ucd_file(
            args.unicode_version, "PropList.txt", cache_dir
        )
        derived_norm_props_path = download_ucd_file(
            args.unicode_version, "DerivedNormalizationProps.txt", cache_dir
        )

    print(f"Parsing UnicodeData.txt...")
    unicode_data = parse_unicode_data(unicode_data_path)

    print(f"Parsing PropList.txt...")
    whitespace = parse_prop_list(prop_list_path)

    print(f"Parsing NFD decompositions...")
    nfd_mappings = parse_unicode_data_for_decomposition(
        unicode_data_path, unicode_data["categories"]
    )

    print(f"Parsing composition exclusions...")
    composition_exclusions = parse_derived_normalization_props(derived_norm_props_path)

    print(f"Building NFC composition pairs...")
    nfc_pairs = build_nfc_composition_pairs(unicode_data_path, composition_exclusions)

    print(f"Parsing NFC_QC=No characters...")
    nfc_qc_no = parse_nfc_qc_no(derived_norm_props_path)

    print(f"Parsing all canonical decompositions...")
    all_decompositions = parse_all_canonical_decompositions(unicode_data_path)

    # Determine which codepoints are already handled by the existing NFD
    # singleton table (singletons in nfd_mappings).
    existing_nfd_singletons = {
        cp
        for cp, (base, combining, is_singleton) in nfd_mappings.items()
        if is_singleton
    }

    print(f"Building NFC canonical decompositions...")
    nfc_decompositions = build_nfc_canonical_decompositions(
        nfc_qc_no, all_decompositions, existing_nfd_singletons
    )

    print(f"Parsing NFKD compatibility decompositions...")
    nfkd_decompositions = parse_compatibility_decompositions(
        unicode_data_path, all_decompositions
    )
    # Report statistics on decomposition lengths.
    max_length = (
        max(len(v) for v in nfkd_decompositions.values()) if nfkd_decompositions else 0
    )
    long_count = sum(1 for v in nfkd_decompositions.values() if len(v) > 4)
    print(
        f"  Found {len(nfkd_decompositions)} NFKD entries, "
        f"max length {max_length}, {long_count} need overflow storage"
    )

    # Download legacy Unicode data for HuggingFace-compatible Mn detection.
    # HuggingFace tokenizers uses older Unicode data, so characters that became
    # Mn in later versions should NOT be stripped during accent stripping.
    print(
        f"Downloading legacy Unicode {HUGGINGFACE_UNICODE_VERSION} for Mn compatibility..."
    )
    if args.ucd_dir:
        # Local mode doesn't support legacy version - use current version's Mn.
        legacy_mn_codepoints = {
            cp for cp, cat in unicode_data["categories"].items() if cat == "Mn"
        }
    else:
        legacy_unicode_data_path = download_ucd_file(
            HUGGINGFACE_UNICODE_VERSION, "UnicodeData.txt", cache_dir
        )
        legacy_mn_codepoints = parse_legacy_mn_codepoints(legacy_unicode_data_path)
    print(
        f"  Found {len(legacy_mn_codepoints)} Mn codepoints in Unicode {HUGGINGFACE_UNICODE_VERSION}"
    )

    print(f"Building category ranges...")
    category_ranges = build_category_ranges(
        unicode_data["categories"], legacy_mn_codepoints
    )

    print(f"Generating tables...")
    case_mappings = {
        "uppercase": unicode_data["uppercase"],
        "lowercase": unicode_data["lowercase"],
    }
    ccc_mappings = unicode_data["ccc"]
    content = generate_tables_c(
        args.unicode_version,
        category_ranges,
        whitespace,
        case_mappings,
        nfd_mappings,
        ccc_mappings,
        nfc_pairs,
        nfc_decompositions,
        nfkd_decompositions,
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
            f"{len(nfd_mappings)} NFD mappings, "
            f"{len(ccc_mappings)} CCC entries, "
            f"{len(nfc_pairs)} NFC composition pairs, "
            f"{len(nfc_decompositions)} NFC decompositions, "
            f"{len(nfkd_decompositions)} NFKD decompositions."
        )


if __name__ == "__main__":
    main()
