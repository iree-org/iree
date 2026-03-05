#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generate golden token IDs for tiktoken encodings.

Loads tiktoken encodings via OpenAI's tiktoken library, encodes a set of test
inputs, and writes the golden IDs into tokenizer_corpus.json alongside the
existing HuggingFace model entries.

Usage:
    uvx --with tiktoken python generate_tiktoken_golden_ids.py

    # Or with pip:
    pip install tiktoken
    python generate_tiktoken_golden_ids.py

    # Update only a specific encoding:
    python generate_tiktoken_golden_ids.py --encoding cl100k_base
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import tiktoken
except ImportError:
    print(
        "Error: tiktoken library not installed.\n"
        "Install with: pip install tiktoken\n"
        "Or run with: uvx --with tiktoken python generate_tiktoken_golden_ids.py",
        file=sys.stderr,
    )
    sys.exit(1)

# =============================================================================
# Constants
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
CORPUS_PATH = SCRIPT_DIR / "testdata" / "tokenizer_corpus.json"

# Tiktoken encoding names and their descriptions.
ENCODINGS = {
    "tiktoken/cl100k_base": {
        "encoding": "cl100k_base",
        "description": "OpenAI cl100k_base (GPT-4, GPT-3.5-turbo) - 100K BPE vocab",
    },
    "tiktoken/o200k_base": {
        "encoding": "o200k_base",
        "description": "OpenAI o200k_base (GPT-4o) - 200K BPE vocab",
    },
    "tiktoken/r50k_base": {
        "encoding": "r50k_base",
        "description": "OpenAI r50k_base (GPT-3) - 50K BPE vocab",
    },
    "tiktoken/p50k_base": {
        "encoding": "p50k_base",
        "description": "OpenAI p50k_base (Codex) - 50K BPE vocab",
    },
}

# Test inputs covering diverse text patterns. These are the same categories
# used in the HuggingFace corpus entries for consistent cross-validation.
TEST_INPUTS = [
    {
        "name": "simple_ascii",
        "input": "Hello, world!",
    },
    {
        "name": "code_snippet",
        "input": "def hello():\n    print('Hello, world!')",
    },
    {
        "name": "numbers",
        "input": "The year 2026 has 365 days.",
    },
    {
        "name": "punctuation",
        "input": "Wait... Really?! Yes: it's true.",
    },
    {
        "name": "mixed_case",
        "input": "HeLLo WoRLD",
    },
    {
        "name": "unicode_cjk",
        "input": "你好世界",
    },
    {
        "name": "unicode_accents",
        "input": "café résumé naïve",
    },
    {
        "name": "unicode_emoji",
        "input": "Hello 🌍! How are you? 😊",
    },
    {
        "name": "whitespace_variations",
        "input": "  hello   world  \n\ttabs\there  ",
    },
    {
        "name": "empty_string",
        "input": "",
    },
    {
        "name": "repeated_chars",
        "input": "aaaaaaaaaa",
    },
    {
        "name": "special_chars",
        "input": "!@#$%^&*()_+-=[]{}|;':\",./<>?",
    },
    {
        "name": "leading_space",
        "input": " hello world",
        "description": "Leading space (byte-level BPE encodes this differently)",
    },
    {
        "name": "no_special_tokens",
        "input": "hello world",
        "add_special_tokens": False,
    },
    {
        "name": "mixed_script",
        "input": "English, Français, Deutsch, 中文, 日本語, العربية",
    },
    {
        "name": "long_word",
        "input": "antidisestablishmentarianism supercalifragilisticexpialidocious",
    },
    {
        "name": "carriage_return_middle",
        "input": "hello\rworld",
        "description": "Carriage return in middle of string",
    },
    {
        "name": "crlf_sequence",
        "input": "line1\r\nline2",
        "description": "Windows-style line endings",
    },
    {
        "name": "special_token_endoftext",
        "input": "hello<|endoftext|>world",
        "description": "Special token in middle of text (exercises special token matching and ID assignment)",
    },
]


def generate_golden_ids(
    encoding_name: str,
) -> list[dict]:
    """Generate golden IDs for all test inputs using tiktoken."""
    enc = tiktoken.get_encoding(encoding_name)
    results = []

    for test in TEST_INPUTS:
        entry = {"name": test["name"], "input": test["input"]}
        if "description" in test:
            entry["description"] = test["description"]
        if "add_special_tokens" in test:
            entry["add_special_tokens"] = test["add_special_tokens"]

        # tiktoken.encode() does not add special tokens by default.
        # When add_special_tokens is not False, we use allowed_special="all"
        # to let special token patterns be recognized in the input.
        add_special = test.get("add_special_tokens", True)
        if add_special:
            ids = enc.encode(test["input"], allowed_special="all")
        else:
            ids = enc.encode(test["input"], disallowed_special=())

        entry["golden_ids"] = ids
        results.append(entry)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate tiktoken golden IDs for tokenizer_corpus.json"
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        help="Update only this encoding (e.g., cl100k_base). Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print golden IDs to stdout instead of updating corpus file.",
    )
    args = parser.parse_args()

    # Determine which encodings to generate.
    if args.encoding:
        key = f"tiktoken/{args.encoding}"
        if key not in ENCODINGS:
            print(
                f"Unknown encoding: {args.encoding}. "
                f"Available: {', '.join(e['encoding'] for e in ENCODINGS.values())}",
                file=sys.stderr,
            )
            sys.exit(1)
        targets = {key: ENCODINGS[key]}
    else:
        targets = ENCODINGS

    # Generate golden IDs for each encoding.
    results = {}
    for model_key, info in targets.items():
        encoding_name = info["encoding"]
        print(f"Generating golden IDs for {encoding_name}...")
        tests = generate_golden_ids(encoding_name)
        results[model_key] = {
            "description": info["description"],
            "tests": tests,
        }
        print(
            f"  {len(tests)} test cases, "
            f"{sum(len(t['golden_ids']) for t in tests)} total tokens"
        )

    if args.dry_run:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    # Load existing corpus and merge.
    if not CORPUS_PATH.exists():
        print(f"Error: Corpus file not found: {CORPUS_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    for model_key, data in results.items():
        corpus["models"][model_key] = data

    # Write back. Use indent=2 for structure, then compact integer arrays
    # to single lines so golden_ids don't bloat the file (one ID per line
    # triples the file size for no readability benefit).
    text = json.dumps(corpus, indent=2, ensure_ascii=False, sort_keys=False)
    text = re.sub(
        r"\[ *\n *-?\d+(?: *, *\n *-?\d+)* *\n *\]",
        lambda m: "[" + ", ".join(re.findall(r"-?\d+", m.group(0))) + "]",
        text,
    )
    with open(CORPUS_PATH, "w") as f:
        f.write(text)
        f.write("\n")

    print(f"\nUpdated {CORPUS_PATH}")
    print(f"Added/updated {len(results)} tiktoken encoding(s).")


if __name__ == "__main__":
    main()
