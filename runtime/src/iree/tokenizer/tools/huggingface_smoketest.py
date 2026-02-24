#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""HuggingFace integration tests for IREE tokenizer.

This tool compares iree-tokenize output against HuggingFace tokenizers to verify
correctness. It supports three operational modes:

  dev     - Download tokenizers from HuggingFace, compare against live HF output
  verify  - Download tokenizers from HuggingFace, compare against stored goldens
  ci      - Load tokenizers from cache directory, compare against stored goldens

Usage:
    # Use the wrapper script (handles all uvx dependencies):
    ./run_smoketest.sh --iree-tokenize /path/to/binary

    # Regenerate goldens after fixing bugs
    ./run_smoketest.sh --iree-tokenize /path/to/binary --update-goldens

    # Test specific model
    ./run_smoketest.sh --iree-tokenize /path/to/binary --model gpt2

    # Run fuzz tests
    ./run_smoketest.sh --iree-tokenize /path/to/binary --fuzz

    # CI mode - fully offline with cached tokenizers (no uvx deps needed)
    python huggingface_smoketest.py --iree-tokenize /path/to/binary \\
        --mode=ci --tokenizer-cache=/tmp/tokenizers

See run_smoketest.sh for the full list of Python dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# =============================================================================
# Section 1: Constants
# =============================================================================

# Default test corpus location (relative to script).
DEFAULT_CORPUS_PATH = Path(__file__).parent / "testdata" / "tokenizer_corpus.json"

# Hint for error messages - tells users to use the wrapper script.
UV_HINT = "Use ./run_smoketest.sh instead of calling this script directly."


# =============================================================================
# Section 2: Data Types
# =============================================================================


@dataclass
class TestResult:
    """Result of a single test."""

    name: str
    passed: bool
    expected: dict | None = None
    actual: dict | None = None
    diff: str = ""
    input_text: str | None = None  # For encode tests.
    decode_ids: list[int] | None = None  # For decode tests.


@dataclass
class ModelResult:
    """Result of all tests for a single model."""

    model: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[TestResult] = field(default_factory=list)
    load_time: float = 0.0
    test_time: float = 0.0
    error: str | None = None


# =============================================================================
# Section 3: Tokenizer Loading
# =============================================================================

# Lazy imports to avoid slow startup.
_tokenizer_lib = None
_hf_hub = None
_auto_tokenizer = None


def _get_tokenizers_lib():
    """Lazy import of tokenizers library."""
    global _tokenizer_lib
    if _tokenizer_lib is None:
        try:
            import tokenizers

            _tokenizer_lib = tokenizers
        except ImportError:
            print(
                f"Error: tokenizers library not installed.\n" f"Hint: {UV_HINT}",
                file=sys.stderr,
            )
            sys.exit(1)
    return _tokenizer_lib


def _get_hf_hub():
    """Lazy import of huggingface_hub."""
    global _hf_hub
    if _hf_hub is None:
        try:
            import huggingface_hub

            _hf_hub = huggingface_hub
        except ImportError:
            print(
                f"Error: huggingface_hub library not installed.\n" f"Hint: {UV_HINT}",
                file=sys.stderr,
            )
            sys.exit(1)
    return _hf_hub


def _get_auto_tokenizer():
    """Lazy import of transformers.AutoTokenizer (slow due to PyTorch init)."""
    global _auto_tokenizer
    if _auto_tokenizer is None:
        try:
            from transformers import AutoTokenizer

            _auto_tokenizer = AutoTokenizer
        except ImportError:
            print(
                f"Error: transformers library not installed.\n" f"Hint: {UV_HINT}",
                file=sys.stderr,
            )
            sys.exit(1)
    return _auto_tokenizer


def load_hf_tokenizer(model_name: str) -> tuple:
    """Load HuggingFace tokenizer by model name.

    Returns (tokenizer, is_fast) where is_fast indicates if we got a
    tokenizers.Tokenizer (True) or transformers tokenizer (False).
    """
    tokenizers = _get_tokenizers_lib()
    hf_hub = _get_hf_hub()

    # Try fast tokenizers library first.
    try:
        tokenizer_path = hf_hub.hf_hub_download(
            repo_id=model_name, filename="tokenizer.json"
        )
        tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
        tokenizer.no_padding()
        return tokenizer, True
    except Exception:
        pass

    # Fall back to transformers (handles vocab.txt, spiece.model, etc.).
    AutoTokenizer = _get_auto_tokenizer()
    return AutoTokenizer.from_pretrained(model_name), False


def save_tokenizer_json(tokenizer, output_path: str, is_fast: bool) -> Path | None:
    """Save tokenizer to tokenizer.json file. Returns path on success."""
    try:
        tokenizer_json = Path(output_path) / "tokenizer.json"
        if is_fast:
            tokenizer.save(str(tokenizer_json))
        else:
            tokenizer.save_pretrained(output_path)
        if tokenizer_json.exists():
            return tokenizer_json
    except Exception as e:
        print(f"Error: Failed to save tokenizer: {e}", file=sys.stderr)
    return None


def load_tokenizer_from_cache(model_name: str, cache_dir: Path) -> Path:
    """Load tokenizer.json from cache directory.

    Raises FileNotFoundError if not found.
    """
    # Convert model name to filename: "BAAI/bge-large-en" -> "BAAI_bge-large-en.json"
    safe_name = model_name.replace("/", "_") + ".json"
    cache_path = cache_dir / safe_name
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Tokenizer for {model_name} not found in cache.\n"
            f"Expected: {cache_path}\n"
            f"Run with --mode=dev --save-tokenizers={cache_dir} to populate cache."
        )
    return cache_path


# =============================================================================
# Section 4: Test Corpus Management
# =============================================================================


def load_corpus(path: Path) -> dict:
    """Load test corpus from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Test corpus not found: {path}")
    with open(path) as f:
        return json.load(f)


def save_corpus(corpus: dict, path: Path):
    """Save test corpus to JSON file with stable ordering."""
    # Sort models alphabetically.
    sorted_models = {}
    for model_name in sorted(corpus.get("models", {}).keys()):
        model_config = corpus["models"][model_name]
        # Sort tests by name.
        if "tests" in model_config:
            model_config["tests"] = sorted(
                model_config["tests"], key=lambda t: t.get("name", "")
            )
        sorted_models[model_name] = model_config

    output = {"version": corpus.get("version", 1), "models": sorted_models}

    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        f.write("\n")


def filter_corpus(corpus: dict, model_pattern: str | None) -> dict:
    """Filter corpus to models matching pattern."""
    if not model_pattern:
        return corpus

    filtered_models = {}
    pattern_lower = model_pattern.lower()
    for model_name, model_config in corpus.get("models", {}).items():
        if pattern_lower in model_name.lower():
            filtered_models[model_name] = model_config

    return {"version": corpus.get("version", 1), "models": filtered_models}


# =============================================================================
# Section 5: Expected Output
# =============================================================================


def get_expected_encode(test: dict, tokenizer, is_fast: bool, mode: str) -> dict:
    """Get expected encode output.

    In dev mode: calls HuggingFace tokenizer.
    In verify/ci mode: returns golden from test dict.
    """
    if mode in ("verify", "ci"):
        if "golden_ids" not in test:
            return {"error": f"Missing golden_ids (required in {mode} mode)"}
        return {"ids": test["golden_ids"]}

    # Dev mode: call HuggingFace.
    input_text = test.get("input", "")
    add_special = test.get("add_special_tokens", True)
    try:
        if is_fast:
            encoding = tokenizer.encode(input_text, add_special_tokens=add_special)
            return {"ids": encoding.ids}
        else:
            ids = tokenizer.encode(input_text, add_special_tokens=add_special)
            return {"ids": ids}
    except Exception as e:
        return {"error": str(e)}


def get_expected_decode(test: dict, tokenizer, is_fast: bool, mode: str) -> dict:
    """Get expected decode output.

    In dev mode: calls HuggingFace tokenizer.
    In verify/ci mode: returns golden from test dict.
    """
    if mode in ("verify", "ci"):
        if "golden_text" not in test:
            return {"error": f"Missing golden_text (required in {mode} mode)"}
        return {"text": test["golden_text"]}

    # Dev mode: call HuggingFace.
    decode_ids = test.get("decode_ids", [])
    try:
        text = tokenizer.decode(decode_ids, skip_special_tokens=True)
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Section 6: Test Execution
# =============================================================================


def run_iree_tokenize(
    binary_path: str,
    tokenizer_json: Path,
    input_text: str = None,
    decode_ids: list[int] = None,
    add_special_tokens: bool = True,
    timeout: int = 30,
) -> dict:
    """Run iree-tokenize and return parsed JSON output."""
    cmd = [binary_path, f"--tokenizer={tokenizer_json}", "--json"]

    if decode_ids is not None:
        cmd.append("--decode")
        cmd.append("--special=false")  # Skip special tokens in decode output.
        cmd.append(",".join(str(x) for x in decode_ids))
    else:
        if not add_special_tokens:
            cmd.append("--special=false")
        cmd.append(input_text or "")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return {
                "error": f"Exit code {result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
            }
        return json.loads(result.stdout.strip())
    except subprocess.TimeoutExpired:
        return {"error": f"Timeout after {timeout}s"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}
    except Exception as e:
        return {"error": str(e)}


def run_iree_tokenize_batch(
    binary_path: str,
    tokenizer_json: Path,
    inputs: list[str],
    add_special_tokens: bool = True,
    timeout: int = 60,
) -> list[dict]:
    """Run iree-tokenize in batch mode for multiple encode inputs."""
    if not inputs:
        return []

    cmd = [binary_path, f"--tokenizer={tokenizer_json}", "--json", "--batch"]
    if not add_special_tokens:
        cmd.append("--special=false")

    batch_input = "\n".join(inputs)

    try:
        result = subprocess.run(
            cmd, input=batch_input, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            error = {"error": f"Exit code {result.returncode}: {result.stderr.strip()}"}
            return [error] * len(inputs)

        lines = [line for line in result.stdout.strip().split("\n") if line]
        if len(lines) != len(inputs):
            error = {
                "error": f"Output count mismatch: expected {len(inputs)}, got {len(lines)}"
            }
            return [error] * len(inputs)

        results = []
        for line in lines:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                results.append({"error": f"JSON parse error: {e}"})
        return results
    except subprocess.TimeoutExpired:
        return [{"error": f"Timeout after {timeout}s"}] * len(inputs)
    except Exception as e:
        return [{"error": str(e)}] * len(inputs)


def compare_results(expected: dict, actual: dict) -> tuple[bool, str]:
    """Compare expected and actual results. Returns (match, diff_description)."""
    if "error" in expected:
        return False, f"Expected error: {expected['error']}"
    if "error" in actual:
        return False, f"IREE error: {actual['error']}"

    if "ids" in expected:
        expected_ids = expected["ids"]
        actual_ids = actual.get("ids", [])
        if expected_ids == actual_ids:
            return True, ""

        # Find first difference.
        for i, (e, a) in enumerate(zip(expected_ids, actual_ids)):
            if e != a:
                return False, f"Mismatch at position {i}: expected={e}, actual={a}"
        if len(expected_ids) != len(actual_ids):
            return (
                False,
                f"Length mismatch: expected={len(expected_ids)}, actual={len(actual_ids)}",
            )
        return False, "Unknown mismatch"

    if "text" in expected:
        expected_text = expected["text"]
        actual_text = actual.get("text", "")
        if expected_text == actual_text:
            return True, ""
        return (
            False,
            f"Text mismatch:\n  expected: {repr(expected_text)}\n  actual:   {repr(actual_text)}",
        )

    return False, "Unknown result format"


# =============================================================================
# Section 7: Model Test Runner
# =============================================================================


def run_model_tests(
    model_name: str,
    model_config: dict,
    binary_path: str,
    mode: str,
    tokenizer_cache_dir: Path | None,
    save_tokenizers_dir: Path | None,
    update_goldens: bool,
    verbose: bool,
) -> tuple[ModelResult, dict | None]:
    """Run all tests for a single model.

    Returns (ModelResult, updated_model_config or None if no updates).
    """
    result = ModelResult(model=model_name)
    tests = model_config.get("tests", [])
    updated_config = None

    if model_config.get("xfail"):
        result.skipped = len(tests)
        return result, None

    # Load tokenizer.
    load_start = time.time()
    try:
        if mode == "ci":
            if not tokenizer_cache_dir:
                raise ValueError("--tokenizer-cache required for ci mode")
            tokenizer_json = load_tokenizer_from_cache(model_name, tokenizer_cache_dir)
            tokenizer = None
            is_fast = True  # Not used in ci mode.
        else:
            tokenizer, is_fast = load_hf_tokenizer(model_name)
    except Exception as e:
        error_msg = str(e)
        # Check for common missing dependency errors and provide helpful message.
        if (
            "tiktoken" in error_msg
            or "sentencepiece" in error_msg
            or "protobuf" in error_msg
        ):
            error_msg = (
                f"{error_msg}\n"
                f"Hint: Some models require additional Python dependencies.\n"
                f"Hint: {UV_HINT}"
            )
        result.error = error_msg
        result.failed = len(tests)
        return result, None

    result.load_time = time.time() - load_start

    # Save tokenizer if needed.
    with tempfile.TemporaryDirectory() as tmpdir:
        if mode != "ci":
            tokenizer_json = save_tokenizer_json(tokenizer, tmpdir, is_fast)
            if not tokenizer_json:
                result.error = "Failed to save tokenizer.json"
                result.failed = len(tests)
                return result, None

            if save_tokenizers_dir:
                import shutil

                save_tokenizers_dir.mkdir(parents=True, exist_ok=True)
                safe_name = model_name.replace("/", "_") + ".json"
                shutil.copy(tokenizer_json, save_tokenizers_dir / safe_name)

        # Group tests.
        encode_tests_special = []
        encode_tests_no_special = []
        encode_tests_individual = []  # Contains newlines, can't batch.
        decode_tests = []

        for test in tests:
            if "decode_ids" in test:
                decode_tests.append(test)
            else:
                input_text = test.get("input", "")
                add_special = test.get("add_special_tokens", True)
                if "\n" in input_text:
                    encode_tests_individual.append(test)
                elif add_special:
                    encode_tests_special.append(test)
                else:
                    encode_tests_no_special.append(test)

        test_start = time.time()
        updated_tests = list(tests) if update_goldens else None

        # Run batched encode tests.
        def run_encode_batch(test_group: list[dict], add_special: bool):
            if not test_group:
                return

            inputs = [t.get("input", "") for t in test_group]

            # Get expected results.
            if mode == "ci" or (mode == "verify" and not update_goldens):
                expected_results = [
                    get_expected_encode(t, None, True, mode) for t in test_group
                ]
            else:
                if is_fast:
                    encodings = tokenizer.encode_batch(
                        inputs, add_special_tokens=add_special
                    )
                    expected_results = [{"ids": enc.ids} for enc in encodings]
                else:
                    batch_output = tokenizer(
                        inputs,
                        add_special_tokens=add_special,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                    )
                    expected_results = [
                        {"ids": ids} for ids in batch_output["input_ids"]
                    ]

            # Run IREE.
            actual_results = run_iree_tokenize_batch(
                binary_path, tokenizer_json, inputs, add_special
            )

            # Compare.
            for test, expected, actual in zip(
                test_group, expected_results, actual_results
            ):
                test_name = test.get("name", "unnamed")
                match, diff = compare_results(expected, actual)

                test_result = TestResult(
                    name=test_name,
                    passed=match,
                    expected=expected,
                    actual=actual,
                    diff=diff,
                    input_text=test.get("input", ""),
                )
                result.results.append(test_result)

                if match:
                    result.passed += 1
                    if verbose:
                        print(f"  ✓ {test_name}")
                else:
                    result.failed += 1
                    print(f"  ✗ {test_name}")
                    print(
                        f"    expected: {json.dumps(expected, separators=(',', ':'))}"
                    )
                    print(f"    actual:   {json.dumps(actual, separators=(',', ':'))}")
                    if diff:
                        print(f"    {diff}")

                # Update golden if requested.
                if update_goldens and "ids" in expected and "error" not in expected:
                    for t in updated_tests:
                        if t.get("name") == test_name and "input" in t:
                            t["golden_ids"] = expected["ids"]

        run_encode_batch(encode_tests_special, add_special=True)
        run_encode_batch(encode_tests_no_special, add_special=False)

        # Run individual encode tests (with newlines).
        for test in encode_tests_individual:
            test_name = test.get("name", "unnamed")
            input_text = test.get("input", "")
            add_special = test.get("add_special_tokens", True)

            expected = get_expected_encode(test, tokenizer, is_fast, mode)
            actual = run_iree_tokenize(
                binary_path, tokenizer_json, input_text, None, add_special
            )
            match, diff = compare_results(expected, actual)

            test_result = TestResult(
                name=test_name,
                passed=match,
                expected=expected,
                actual=actual,
                diff=diff,
                input_text=input_text,
            )
            result.results.append(test_result)

            if match:
                result.passed += 1
                if verbose:
                    print(f"  ✓ {test_name}")
            else:
                result.failed += 1
                print(f"  ✗ {test_name}")
                print(f"    expected: {json.dumps(expected, separators=(',', ':'))}")
                print(f"    actual:   {json.dumps(actual, separators=(',', ':'))}")

            if update_goldens and "ids" in expected and "error" not in expected:
                for t in updated_tests:
                    if t.get("name") == test_name and "input" in t:
                        t["golden_ids"] = expected["ids"]

        # Run decode tests.
        for test in decode_tests:
            test_name = test.get("name", "unnamed")
            decode_ids = test.get("decode_ids", [])

            expected = get_expected_decode(test, tokenizer, is_fast, mode)
            actual = run_iree_tokenize(
                binary_path, tokenizer_json, None, decode_ids, True
            )
            match, diff = compare_results(expected, actual)

            test_result = TestResult(
                name=test_name,
                passed=match,
                expected=expected,
                actual=actual,
                diff=diff,
                decode_ids=decode_ids,
            )
            result.results.append(test_result)

            if match:
                result.passed += 1
                if verbose:
                    print(f"  ✓ {test_name}")
            else:
                result.failed += 1
                print(f"  ✗ {test_name}")
                print(f"    expected: {json.dumps(expected, separators=(',', ':'))}")
                print(f"    actual:   {json.dumps(actual, separators=(',', ':'))}")

            if update_goldens and "text" in expected and "error" not in expected:
                for t in updated_tests:
                    if t.get("name") == test_name and "decode_ids" in t:
                        t["golden_text"] = expected["text"]

        result.test_time = time.time() - test_start

        if update_goldens:
            updated_config = dict(model_config)
            updated_config["tests"] = updated_tests

    return result, updated_config


# =============================================================================
# Section 8: Fuzz Generators
# =============================================================================
# These generators produce test inputs for differential fuzzing. Adapted from
# llama.cpp's test-tokenizer-random.py which has found many real tokenizer bugs.

import random
import unicodedata


def generator_custom_text() -> Iterator[str]:
    """Basic sanity tests covering common tokenization scenarios."""
    yield from [
        "",
        " ",
        "  ",
        "   ",
        "\t",
        "\n",
        "\n\n",
        "\n\n\n",
        "\t\n",
        "Hello world",
        " Hello world",
        "Hello World",
        " Hello World",
        " Hello World!",
        "Hello, world!",
        " Hello, world!",
        " this is 🦙.cpp",
        "w048 7tuijk dsdfhu",
        "нещо на Български",
        "កាន់តែពិសេសអាចខលចេញ",
        "🚀 (normal) 😶‍🌫️ (multiple emojis) ✅ (emoji)",
        "Hello",
        " Hello",
        "  Hello",
        "   Hello",
        "    Hello",
        "    Hello\n    Hello",
        " (",
        "\n =",
        "' era",
        "Hello, y'all! How are you 😁 ?我想在apple工作1314151天～",
        "3",
        "33",
        "333",
        "3333",
        "33333",
        "333333",
        "3333333",
        "33333333",
        "333333333",
        "a" * 100,
        " " * 50,
        "🎉" * 10,
        "Test\x00String",
        "Line1\r\nLine2",
        "Tab\there",
    ]


def generator_edge_cases() -> Iterator[str]:
    """Known edge cases from llama.cpp that have caused tokenizer bugs."""
    yield from [
        "\x1f-a",
        "¼-a",
        "½-a",
        "¾-a",
        "a 〇b",
        "Ⅵ-a",
        "\ufeff//",
        "Cửa Việt",
        "<s>a",
        "<unk><|endoftext|><s>",
        "a\na",
        '"`',
        " \u2e4e",
        "\n\x0b  ",
        "a\xa0\xa0\x00b",
        "one <mask>",
        "a </s> b",
        "a <mask> b",
        "\xa0aC",
        "\u2029 \ua3e4",
        "a ?",
        "å",
        "\U000ac517",
        "\U000522f4",
        "<s><s><unk><s>a<s>b<s>c<unk>d<unk></s>",
        "<s> <s> <unk><s>a<s>b<s>c<unk>d<unk></s>",
        "café",
        "cafe\u0301",
        "\u200b",
        "\u200c",
        "\u200d",
        "👨‍👩‍👧‍👦",
        "🏳️‍🌈",
        "\ufeff",
        "test\uffffreplace",
        "𝕳𝖊𝖑𝖑𝖔",
        "ﬁ",
        "①②③",
        "™®©",
    ]


def generator_ascii_lr_strip(seed: int, limit: int) -> Iterator[str]:
    """ASCII characters with leading/trailing whitespace combinations."""
    whitespaces = ["", " ", "  "]
    characters = [chr(i) for i in range(1, 0x80)] + [""]

    def all_combinations():
        for char1 in characters:
            for char2 in characters:
                for lstrip in whitespaces:
                    for rstrip in whitespaces:
                        yield lstrip + char1 + char2 + rstrip
                        yield lstrip + char1 + rstrip + char2
                        yield char1 + lstrip + char2 + rstrip

    rand = random.Random(seed)
    all_combos = list(all_combinations())
    rand.shuffle(all_combos)
    yield from all_combos[:limit]


def generator_apostrophe(seed: int, limit: int) -> Iterator[str]:
    """Apostrophe placement edge cases."""
    whitespaces = ["", " ", "  "]
    characters = [chr(i) for i in range(1, 0x80)] + [""]

    def all_combinations():
        for char1 in characters:
            for char2 in characters:
                for lstrip in whitespaces:
                    for rstrip in whitespaces:
                        yield char1 + lstrip + "'" + rstrip + char2
                        yield char1 + char2 + lstrip + "'" + rstrip + "z"
                        yield "a" + lstrip + "'" + rstrip + char1 + char2

    rand = random.Random(seed)
    all_combos = list(all_combinations())
    rand.shuffle(all_combos)
    yield from all_combos[:limit]


def generator_unicodes(seed: int, limit: int) -> Iterator[str]:
    """Individual Unicode codepoints."""
    max_codepoints = 0x30000

    def is_valid_codepoint(codepoint: int) -> bool:
        if codepoint >= 0x30000:
            return False
        try:
            char = chr(codepoint)
            category = unicodedata.category(char)
            return category not in ("Cn", "Cs", "Co")
        except (ValueError, OverflowError):
            return False

    valid_codepoints = [
        chr(cp) for cp in range(max_codepoints) if is_valid_codepoint(cp)
    ]

    if len(valid_codepoints) > limit:
        rand = random.Random(seed)
        edges = valid_codepoints[:100] + valid_codepoints[-100:]
        if limit <= len(edges):
            rand.shuffle(edges)
            yield from edges[:limit]
        else:
            middle = valid_codepoints[100:-100]
            rand.shuffle(middle)
            remaining = limit - len(edges)
            yield from edges
            yield from middle[:remaining]
    else:
        yield from valid_codepoints


def generator_added_tokens(
    tokenizer, is_fast: bool, seed: int, limit: int
) -> Iterator[str]:
    """Special/added tokens with whitespace variations."""
    whitespaces = ["", " ", "  ", "\n", "\r\n", "\n\n", "\t", "\t\t"]

    try:
        if is_fast:
            added_tokens = [
                token.content for token in tokenizer.get_added_tokens_decoder().values()
            ]
        else:
            added_tokens = list(tokenizer.added_tokens_encoder.keys())
            added_tokens.extend(tokenizer.all_special_tokens)
    except Exception:
        added_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>", "[CLS]", "[SEP]"]

    all_tokens = list(set(added_tokens))
    if not all_tokens:
        return

    def all_combinations():
        for token in all_tokens:
            for lstrip in whitespaces:
                for rstrip in whitespaces:
                    yield lstrip + token + rstrip
                    yield "a" + lstrip + token + rstrip
                    yield lstrip + token + rstrip + "z"
                    yield "a" + lstrip + token + rstrip + "z"

    rand = random.Random(seed)
    all_combos = list(all_combinations())
    rand.shuffle(all_combos)
    yield from all_combos[:limit]


# Registry of available fuzz generators.
FUZZ_GENERATORS = {
    "custom_text": (generator_custom_text, False),
    "edge_cases": (generator_edge_cases, False),
    "ascii_lr_strip": (generator_ascii_lr_strip, False),
    "apostrophe": (generator_apostrophe, False),
    "unicodes": (generator_unicodes, False),
    "added_tokens": (generator_added_tokens, True),
}


def create_fuzz_generator(
    name: str,
    seed: int,
    limit: int,
    tokenizer=None,
    is_fast: bool = True,
) -> Iterator[str]:
    """Factory function to create a fuzz generator by name."""
    if name not in FUZZ_GENERATORS:
        raise ValueError(f"Unknown generator: {name}")

    gen_func, needs_tokenizer = FUZZ_GENERATORS[name]

    if needs_tokenizer:
        if tokenizer is None:
            raise ValueError(f"Generator {name} requires tokenizer")
        return gen_func(tokenizer, is_fast, seed, limit)
    elif name in ("custom_text", "edge_cases"):
        return gen_func()
    else:
        return gen_func(seed, limit)


def run_fuzz_tests(
    model_name: str,
    model_config: dict,
    binary_path: str,
    generators: list[str],
    seed: int,
    limit: int,
    add_failures: bool,
    verbose: bool,
) -> tuple[int, int, list[dict], dict | None]:
    """Run fuzz generators against a model.

    Returns (passed, total, failures, updated_config or None).
    """
    if model_config.get("xfail"):
        return 0, 0, [], None

    # Load tokenizer.
    try:
        tokenizer, is_fast = load_hf_tokenizer(model_name)
    except Exception as e:
        print(f"  ERROR: Could not load tokenizer: {e}")
        return 0, 0, [], None

    passed = 0
    total = 0
    failures = []
    new_tests = []
    existing_inputs = {t.get("input") for t in model_config.get("tests", [])}

    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer_json = save_tokenizer_json(tokenizer, tmpdir, is_fast)
        if not tokenizer_json:
            print(f"  ERROR: Could not save tokenizer.json")
            return 0, 0, [], None

        for gen_name in generators:
            try:
                inputs = list(
                    create_fuzz_generator(gen_name, seed, limit, tokenizer, is_fast)
                )
            except ValueError as e:
                print(f"  WARNING: {e}")
                continue

            if not inputs:
                continue

            print(f"  [{gen_name}] Testing {len(inputs)} inputs...", end="", flush=True)

            # Filter batchable inputs (no newlines).
            batchable = [inp for inp in inputs if "\n" not in inp]
            non_batchable = [inp for inp in inputs if "\n" in inp]

            gen_passed = 0
            gen_failures = []

            # Run batchable inputs.
            if batchable:
                try:
                    if is_fast:
                        encodings = tokenizer.encode_batch(
                            batchable, add_special_tokens=True
                        )
                        hf_results = [{"ids": enc.ids} for enc in encodings]
                    else:
                        batch_output = tokenizer(
                            batchable,
                            add_special_tokens=True,
                            return_attention_mask=False,
                            return_token_type_ids=False,
                        )
                        hf_results = [{"ids": ids} for ids in batch_output["input_ids"]]
                except Exception as e:
                    hf_results = [{"error": str(e)}] * len(batchable)

                iree_results = run_iree_tokenize_batch(
                    binary_path, tokenizer_json, batchable, True
                )

                for idx, (inp, hf_result, iree_result) in enumerate(
                    zip(batchable, hf_results, iree_results)
                ):
                    total += 1
                    match, diff = compare_results(hf_result, iree_result)
                    if match:
                        gen_passed += 1
                    else:
                        gen_failures.append(
                            {
                                "generator": gen_name,
                                "index": idx,
                                "input": inp,
                                "hf_result": hf_result,
                                "iree_result": iree_result,
                                "diff": diff,
                            }
                        )

            # Run non-batchable inputs.
            for idx, inp in enumerate(non_batchable):
                total += 1
                try:
                    if is_fast:
                        encoding = tokenizer.encode(inp, add_special_tokens=True)
                        hf_result = {"ids": encoding.ids}
                    else:
                        ids = tokenizer.encode(inp, add_special_tokens=True)
                        hf_result = {"ids": ids}
                except Exception as e:
                    hf_result = {"error": str(e)}

                iree_result = run_iree_tokenize(
                    binary_path, tokenizer_json, inp, None, True
                )
                match, diff = compare_results(hf_result, iree_result)
                if match:
                    gen_passed += 1
                else:
                    gen_failures.append(
                        {
                            "generator": gen_name,
                            "index": len(batchable) + idx,
                            "input": inp,
                            "hf_result": hf_result,
                            "iree_result": iree_result,
                            "diff": diff,
                        }
                    )

            passed += gen_passed
            failures.extend(gen_failures)

            if gen_failures:
                print(f" {len(gen_failures)} FAILED")
                for failure in gen_failures:
                    inp = failure["input"]
                    print(f"\n  BUG: {repr(inp[:50])}")
                    print(
                        f"    HF:   {json.dumps(failure['hf_result'], separators=(',', ':'))}"
                    )
                    print(
                        f"    IREE: {json.dumps(failure['iree_result'], separators=(',', ':'))}"
                    )
                    if failure["diff"]:
                        print(f"    {failure['diff']}")

                    if inp not in existing_inputs:
                        test_name = f"fuzz_{failure['generator']}_{failure['index']}"
                        new_tests.append({"name": test_name, "input": inp})
                        existing_inputs.add(inp)
            else:
                print(" OK")

    # Return updated config if adding failures.
    updated_config = None
    if add_failures and new_tests:
        updated_config = dict(model_config)
        updated_config["tests"] = list(model_config.get("tests", [])) + new_tests

    return passed, total, failures, updated_config


# =============================================================================
# Section 9: JSON Report Generation
# =============================================================================


def serialize_test_result(result: TestResult) -> dict:
    """Convert TestResult to JSON-serializable dict."""
    data = {
        "name": result.name,
        "passed": result.passed,
    }
    if not result.passed:
        if result.input_text is not None:
            data["input"] = result.input_text
        if result.decode_ids is not None:
            data["decode_ids"] = result.decode_ids
        if result.expected:
            data["expected"] = result.expected
        if result.actual:
            data["actual"] = result.actual
        if result.diff:
            data["diff"] = result.diff
    return data


def serialize_model_result(result: ModelResult) -> dict:
    """Convert ModelResult to JSON-serializable dict."""
    data = {
        "model": result.model,
        "passed": result.passed,
        "failed": result.failed,
        "skipped": result.skipped,
        "load_time": result.load_time,
        "test_time": result.test_time,
    }
    if result.error:
        data["error"] = result.error
    # Only include failed tests in the output to keep it manageable.
    failures = [serialize_test_result(r) for r in result.results if not r.passed]
    if failures:
        data["failures"] = failures
    return data


def write_json_report(
    output_path: Path,
    model_results: list[ModelResult],
    fuzz_failures: list[dict] | None = None,
    summary: dict | None = None,
):
    """Write JSON report for triage."""
    report = {
        "summary": summary or {},
        "models": [serialize_model_result(r) for r in model_results],
    }

    # Categorize failures by error type for easier triage.
    categories = categorize_failures(model_results, fuzz_failures)
    if categories:
        report["categories"] = categories

    if fuzz_failures:
        report["fuzz_failures"] = fuzz_failures

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nJSON report written to: {output_path}")


def categorize_failures(
    model_results: list[ModelResult],
    fuzz_failures: list[dict] | None = None,
) -> dict[str, list[dict]]:
    """Categorize failures by error type for triage.

    Categories:
      - error: IREE returned an error (unimplemented, resource exhausted, etc.)
      - mismatch: Token IDs don't match
      - decode_mismatch: Decoded text doesn't match
      - length_mismatch: Different number of tokens
      - hf_error: HuggingFace returned an error (golden issue)
    """
    categories: dict[str, list[dict]] = {
        "error": [],
        "mismatch": [],
        "decode_mismatch": [],
        "length_mismatch": [],
        "hf_error": [],
    }

    for model_result in model_results:
        for test_result in model_result.results:
            if test_result.passed:
                continue

            failure_info = {
                "model": model_result.model,
                "test": test_result.name,
                "diff": test_result.diff,
            }

            # Add input if available from expected/actual.
            if test_result.expected:
                failure_info["expected"] = test_result.expected
            if test_result.actual:
                failure_info["actual"] = test_result.actual

            # Categorize based on the diff message and actual/expected.
            diff = test_result.diff or ""
            actual = test_result.actual or {}

            if "error" in actual:
                categories["error"].append(failure_info)
            elif test_result.expected and "error" in test_result.expected:
                categories["hf_error"].append(failure_info)
            elif "Length mismatch" in diff:
                categories["length_mismatch"].append(failure_info)
            elif "Text mismatch" in diff:
                categories["decode_mismatch"].append(failure_info)
            else:
                categories["mismatch"].append(failure_info)

    # Also categorize fuzz failures if provided.
    if fuzz_failures:
        for failure in fuzz_failures:
            failure_info = {
                "model": failure.get("model", "unknown"),
                "generator": failure.get("generator"),
                "input": failure.get("input"),
                "diff": failure.get("diff"),
            }

            iree_result = failure.get("iree_result", {})
            hf_result = failure.get("hf_result", {})

            if "error" in iree_result:
                categories["error"].append(failure_info)
            elif "error" in hf_result:
                categories["hf_error"].append(failure_info)
            else:
                categories["mismatch"].append(failure_info)

    # Remove empty categories.
    return {k: v for k, v in categories.items() if v}


# =============================================================================
# Section 10: Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace integration tests for IREE tokenizer.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required.
    parser.add_argument(
        "--iree-tokenize",
        required=True,
        help="Path to iree-tokenize binary",
    )

    # Mode selection.
    parser.add_argument(
        "--mode",
        choices=["dev", "verify", "ci"],
        default="dev",
        help="Operating mode (default: dev)",
    )
    parser.add_argument(
        "--tokenizer-cache",
        type=Path,
        help="Directory containing cached tokenizer.json files (required for ci mode)",
    )

    # Golden management.
    parser.add_argument(
        "--update-goldens",
        action="store_true",
        help="Regenerate goldens from HuggingFace (dev mode only)",
    )
    parser.add_argument(
        "--update-model",
        help="Update goldens for specific model only (implies --update-goldens)",
    )

    # Filtering.
    parser.add_argument(
        "--model",
        help="Filter tests by model name substring",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to first N models (for debugging)",
    )

    # Output control.
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all test details, not just failures",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write detailed test results to JSON file for triage",
    )
    parser.add_argument(
        "--save-tokenizers",
        type=Path,
        help="Save downloaded tokenizer.json files to this directory",
    )

    # Corpus path.
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"Path to test corpus (default: {DEFAULT_CORPUS_PATH})",
    )

    # Fuzzing.
    parser.add_argument(
        "--fuzz",
        action="store_true",
        help="Run fuzz generators instead of regular tests",
    )
    parser.add_argument(
        "--fuzz-generators",
        nargs="+",
        default=list(FUZZ_GENERATORS.keys()),
        metavar="GEN",
        help=f"Which generators to run (default: all). Available: {', '.join(FUZZ_GENERATORS.keys())}",
    )
    parser.add_argument(
        "--fuzz-seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42)",
    )
    parser.add_argument(
        "--fuzz-limit",
        type=int,
        default=1000,
        help="Max inputs per generator (default: 1000)",
    )
    parser.add_argument(
        "--fuzz-add-failures",
        action="store_true",
        help="Add failing inputs to the corpus as regression tests",
    )

    args = parser.parse_args()

    # Validate args.
    if not os.path.isfile(args.iree_tokenize):
        print(f"Error: iree-tokenize not found: {args.iree_tokenize}", file=sys.stderr)
        sys.exit(1)
    if not os.access(args.iree_tokenize, os.X_OK):
        print(
            f"Error: iree-tokenize not executable: {args.iree_tokenize}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.mode == "ci" and not args.tokenizer_cache:
        print("Error: --tokenizer-cache required for ci mode", file=sys.stderr)
        sys.exit(1)

    if args.update_goldens and args.mode != "dev":
        print("Error: --update-goldens only valid with --mode=dev", file=sys.stderr)
        sys.exit(1)

    if args.update_model:
        args.update_goldens = True
        args.model = args.update_model

    # Load corpus.
    try:
        corpus = load_corpus(args.corpus)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Filter corpus for iteration (keep original for saving).
    filtered = filter_corpus(corpus, args.model)
    models = list(filtered.get("models", {}).items())

    if not models:
        print(f"No models found matching filter: {args.model}")
        sys.exit(1)

    if args.limit > 0:
        models = models[: args.limit]

    # Print header.
    if args.fuzz:
        print(f"Fuzzing {len(models)} model(s)")
        print(f"  generators: {', '.join(args.fuzz_generators)}")
        print(f"  seed={args.fuzz_seed}, limit={args.fuzz_limit}")
    else:
        print(f"Running {len(models)} model(s) in {args.mode} mode")
        if args.update_goldens:
            print("  Goldens will be updated")

    # Fuzz mode: run fuzz generators.
    if args.fuzz:
        total_passed = 0
        total_failed = 0
        all_bugs = []
        updated_models = {}

        for model_name, model_config in models:
            print(f"\n=== FUZZ: {model_name} ===")

            passed, total, failures, updated_config = run_fuzz_tests(
                model_name,
                model_config,
                args.iree_tokenize,
                args.fuzz_generators,
                args.fuzz_seed,
                args.fuzz_limit,
                args.fuzz_add_failures,
                args.verbose,
            )

            total_passed += passed
            total_failed += len(failures)
            all_bugs.extend(failures)

            if updated_config:
                updated_models[model_name] = updated_config

            print(f"  {passed}/{total} passed")

        # Update corpus if adding failures.
        if args.fuzz_add_failures and updated_models:
            for model_name, updated_config in updated_models.items():
                corpus["models"][model_name] = updated_config
            save_corpus(corpus, args.corpus)
            print(f"\nAdded failing tests to {args.corpus}")

        # Summary.
        print(f"\n{'=' * 50}")
        total = total_passed + total_failed
        if not all_bugs:
            print(f"SUCCESS: {total_passed}/{total} fuzz tests passed (no bugs)")
        else:
            print(f"BUGS FOUND: {len(all_bugs)} failures in {total} tests")
            if args.fuzz_add_failures:
                print("  Failing tests have been added to corpus.")
            else:
                print("  Use --fuzz-add-failures to add them as regression tests.")

        sys.exit(0 if not all_bugs else 1)

    # Run regular tests.
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    all_failures = []
    updated_models = {}
    model_results: list[ModelResult] = []

    for model_name, model_config in models:
        description = model_config.get("description", "")
        xfail = model_config.get("xfail", False)
        test_count = len(model_config.get("tests", []))

        if xfail:
            print(f"\n=== {model_name} (xfail, {test_count} tests) ===")
            total_skipped += test_count
            # Create a placeholder result for skipped models.
            skipped_result = ModelResult(model=model_name, skipped=test_count)
            model_results.append(skipped_result)
            continue

        print(f"\n=== {model_name} ({test_count} tests) ===")
        if description and args.verbose:
            print(f"  {description}")

        model_result, updated_config = run_model_tests(
            model_name,
            model_config,
            args.iree_tokenize,
            args.mode,
            args.tokenizer_cache,
            args.save_tokenizers,
            args.update_goldens,
            args.verbose,
        )

        model_results.append(model_result)

        if model_result.error:
            print(f"  ERROR: {model_result.error}")
            all_failures.append(f"{model_name}: {model_result.error}")

        total_passed += model_result.passed
        total_failed += model_result.failed
        total_skipped += model_result.skipped

        if model_result.failed == 0 and model_result.error is None:
            timing = f"[load: {model_result.load_time:.2f}s, test: {model_result.test_time:.2f}s]"
            print(f"  ✓ {model_result.passed}/{test_count} tests passed {timing}")
        else:
            for test_result in model_result.results:
                if not test_result.passed:
                    all_failures.append(f"{model_name}/{test_result.name}")

        if updated_config:
            updated_models[model_name] = updated_config

    # Update corpus if goldens were regenerated.
    if args.update_goldens and updated_models:
        for model_name, updated_config in updated_models.items():
            corpus["models"][model_name] = updated_config
        save_corpus(corpus, args.corpus)
        print(f"\nUpdated goldens in {args.corpus}")

    # Summary.
    total = total_passed + total_failed
    print(f"\n{'=' * 50}")
    if total_failed == 0:
        print(f"SUCCESS: {total_passed}/{total} tests passed")
        if total_skipped:
            print(f"  ({total_skipped} skipped due to xfail)")
    else:
        print(f"FAILED: {total_passed}/{total} tests passed ({total_failed} failed)")
        print("\nFailures:")
        for failure in all_failures:
            print(f"  - {failure}")

    # Write JSON report if requested.
    if args.json_output:
        summary = {
            "mode": args.mode,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_skipped": total_skipped,
            "total_models": len(models),
        }
        write_json_report(args.json_output, model_results, summary=summary)

    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
