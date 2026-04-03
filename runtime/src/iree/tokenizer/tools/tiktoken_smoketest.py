#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tiktoken integration tests for IREE tokenizer.

This tool compares iree-tokenize output against OpenAI's tiktoken library to
verify correctness of the tiktoken format loader. It also cross-validates
against HuggingFace tokenizer.json files to prove format equivalence.

It supports three operational modes:

  dev     - Download .tiktoken files, compare against live tiktoken output
  verify  - Download .tiktoken files, compare against stored goldens
  ci      - Load .tiktoken files from cache directory, compare against goldens

Usage:
    # Use the wrapper script (handles all uvx dependencies):
    ./run_tiktoken_smoketest.sh --iree-tokenize /path/to/binary

    # Update goldens after fixing bugs
    ./run_tiktoken_smoketest.sh --iree-tokenize /path/to/binary --update-goldens

    # Test specific encoding
    ./run_tiktoken_smoketest.sh --iree-tokenize /path/to/binary --encoding cl100k_base

    # Cross-validate tiktoken vs HuggingFace (both through IREE)
    ./run_tiktoken_smoketest.sh --iree-tokenize /path/to/binary --cross-validate

    # CI mode - fully offline with cached files (no uvx deps needed)
    python tiktoken_smoketest.py --iree-tokenize /path/to/binary \\
        --mode=ci --tokenizer-cache=/tmp/tiktoken

See run_tiktoken_smoketest.sh for the full list of Python dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# =============================================================================
# Section 1: Constants
# =============================================================================

DEFAULT_CORPUS_PATH = Path(__file__).parent / "testdata" / "tokenizer_corpus.json"

UV_HINT = "Use ./run_tiktoken_smoketest.sh instead of calling this script directly."

# Canonical download URLs for tiktoken encoding files.
# These are the same URLs used by the tiktoken Python library internally.
TIKTOKEN_URLS = {
    "cl100k_base": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    "o200k_base": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    "r50k_base": "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
    "p50k_base": "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
}

# Equivalent HuggingFace repos for cross-validation. When loaded through IREE's
# HF JSON loader and tiktoken loader respectively, these should produce
# identical token sequences for the same input.
CROSS_VALIDATION_HF_REPOS = {
    "cl100k_base": "Xenova/gpt-4",
    "o200k_base": "Xenova/gpt-4o",
    "r50k_base": "gpt2",
    "p50k_base": "Xenova/text-davinci-003",
}

# Corpus key prefix for tiktoken entries in tokenizer_corpus.json.
CORPUS_KEY_PREFIX = "tiktoken/"


# =============================================================================
# Section 2: Data Types
# =============================================================================


@dataclass
class TestResult:
    """Result of a single test."""

    name: str
    passed: bool
    expected_ids: list[int] | None = None
    actual_ids: list[int] | None = None
    diff: str = ""
    input_text: str | None = None


@dataclass
class EncodingResult:
    """Result of all tests for a single encoding."""

    encoding: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[TestResult] = field(default_factory=list)
    load_time: float = 0.0
    test_time: float = 0.0
    error: str | None = None


# =============================================================================
# Section 3: Tiktoken File Management
# =============================================================================


def download_tiktoken_file(encoding: str, cache_dir: Path) -> Path:
    """Download a .tiktoken file to the cache directory.

    Returns the path to the cached file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{encoding}.tiktoken"

    if dest.exists():
        return dest

    url = TIKTOKEN_URLS[encoding]
    print(f"  Downloading {encoding}.tiktoken from {url}...")
    try:
        urllib.request.urlretrieve(url, str(dest) + ".tmp")
        (dest.parent / (dest.name + ".tmp")).rename(dest)
    except Exception as e:
        (dest.parent / (dest.name + ".tmp")).unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {encoding}: {e}") from e

    size_kb = dest.stat().st_size // 1024
    print(f"  Downloaded {encoding}.tiktoken ({size_kb}KB)")
    return dest


def load_tiktoken_from_cache(encoding: str, cache_dir: Path) -> Path:
    """Load a .tiktoken file from the cache directory."""
    dest = cache_dir / f"{encoding}.tiktoken"
    if not dest.exists():
        raise FileNotFoundError(
            f"Tiktoken file not found in cache: {dest}\n"
            f"Run in dev or verify mode first to download, or provide --tokenizer-cache."
        )
    return dest


# =============================================================================
# Section 4: Reference Encoding (tiktoken Python library)
# =============================================================================

_tiktoken_lib = None


def _get_tiktoken():
    """Lazy import of tiktoken library."""
    global _tiktoken_lib
    if _tiktoken_lib is None:
        try:
            import tiktoken

            _tiktoken_lib = tiktoken
        except ImportError:
            print(
                f"Error: tiktoken library not installed.\nHint: {UV_HINT}",
                file=sys.stderr,
            )
            sys.exit(1)
    return _tiktoken_lib


def tiktoken_encode(encoding_name: str, text: str) -> list[int]:
    """Encode text using tiktoken Python library (ground truth)."""
    tiktoken = _get_tiktoken()
    enc = tiktoken.get_encoding(encoding_name)
    # tiktoken.encode does not insert special tokens by default.
    # Use disallowed_special=() to allow all text (no special token matching).
    return enc.encode(text, disallowed_special=())


# =============================================================================
# Section 5: IREE Invocation
# =============================================================================


def run_iree_tokenize(
    binary_path: str,
    tokenizer_path: Path,
    input_text: str,
    timeout: int = 30,
) -> dict:
    """Run iree-tokenize and return parsed JSON output."""
    cmd = [
        binary_path,
        f"--tokenizer={tokenizer_path}",
        "--json",
        "--special=false",
        "--match_special=false",
        input_text,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return {
                "error": f"Exit code {result.returncode}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
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
    tokenizer_path: Path,
    inputs: list[str],
    timeout: int = 60,
) -> list[dict]:
    """Run iree-tokenize in batch mode for multiple encode inputs.

    Batch mode reads one input per line from stdin and writes one JSON object
    per line to stdout. Inputs containing newlines or carriage returns cannot
    be batched and are run individually.
    """
    if not inputs:
        return []

    cmd = [
        binary_path,
        f"--tokenizer={tokenizer_path}",
        "--json",
        "--batch",
        "--special=false",
        "--match_special=false",
    ]

    # Batch mode is line-delimited on stdin, so inputs with newlines can't
    # be batched. Run those individually.
    batchable = [inp for inp in inputs if "\n" not in inp and "\r" not in inp]
    if not batchable:
        return [
            run_iree_tokenize(binary_path, tokenizer_path, inp, timeout)
            for inp in inputs
        ]

    batch_input = "\n".join(batchable)

    try:
        result = subprocess.run(
            cmd,
            input=batch_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            error = {
                "error": f"Exit code {result.returncode}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            }
            return [error] * len(batchable)

        lines = [line for line in result.stdout.strip().split("\n") if line]
        if len(lines) != len(batchable):
            error = {
                "error": f"Output count mismatch: expected {len(batchable)}, got {len(lines)}"
            }
            return [error] * len(batchable)

        results = []
        for line in lines:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                results.append({"error": f"JSON parse error: {e}"})
        return results
    except subprocess.TimeoutExpired:
        return [{"error": f"Timeout after {timeout}s"}] * len(batchable)
    except Exception as e:
        return [{"error": str(e)}] * len(batchable)


# =============================================================================
# Section 6: Test Comparison
# =============================================================================


def compare_ids(expected: list[int], actual: list[int]) -> tuple[bool, str]:
    """Compare expected and actual token ID lists."""
    if expected == actual:
        return True, ""

    for i, (e, a) in enumerate(zip(expected, actual)):
        if e != a:
            return False, f"Mismatch at position {i}: expected={e}, actual={a}"

    if len(expected) != len(actual):
        return False, f"Length mismatch: expected={len(expected)}, actual={len(actual)}"

    return False, "Unknown mismatch"


# =============================================================================
# Section 7: Test Runner
# =============================================================================


def run_encoding_tests(
    encoding: str,
    binary_path: str,
    mode: str,
    corpus: dict,
    tokenizer_cache_dir: Path,
    update_goldens: bool,
    verbose: bool,
) -> tuple[EncodingResult, dict | None]:
    """Run all tests for a single tiktoken encoding.

    Returns (EncodingResult, updated_corpus_entry or None).
    """
    result = EncodingResult(encoding=encoding)
    corpus_key = f"{CORPUS_KEY_PREFIX}{encoding}"
    model_config = corpus.get("models", {}).get(corpus_key, {})
    tests = model_config.get("tests", [])

    if not tests:
        print(f"  No test entries found for {corpus_key} in corpus.")
        print(f"  Run generate_tiktoken_golden_ids.py first.")
        result.error = "No test entries in corpus"
        return result, None

    # Download or load tiktoken file.
    load_start = time.time()
    try:
        if mode == "ci":
            tiktoken_path = load_tiktoken_from_cache(encoding, tokenizer_cache_dir)
        else:
            tiktoken_path = download_tiktoken_file(encoding, tokenizer_cache_dir)
    except Exception as e:
        result.error = str(e)
        result.failed = len(tests)
        return result, None
    result.load_time = time.time() - load_start

    test_start = time.time()
    updated_tests = list(tests) if update_goldens else None

    # Separate tests that can be batched from those that can't.
    batchable_tests = []
    individual_tests = []
    for test in tests:
        input_text = test.get("input", "")
        if "\n" in input_text or "\r" in input_text:
            individual_tests.append(test)
        else:
            batchable_tests.append(test)

    # Run batchable tests.
    if batchable_tests:
        inputs = [t.get("input", "") for t in batchable_tests]
        actual_results = run_iree_tokenize_batch(binary_path, tiktoken_path, inputs)

        for test, actual in zip(batchable_tests, actual_results):
            _process_test_result(
                test,
                actual,
                encoding,
                mode,
                result,
                updated_tests,
                update_goldens,
                verbose,
            )

    # Run individual tests (contain newlines).
    for test in individual_tests:
        input_text = test.get("input", "")
        actual = run_iree_tokenize(binary_path, tiktoken_path, input_text)
        _process_test_result(
            test,
            actual,
            encoding,
            mode,
            result,
            updated_tests,
            update_goldens,
            verbose,
        )

    result.test_time = time.time() - test_start

    updated_config = None
    if update_goldens and updated_tests:
        updated_config = dict(model_config)
        updated_config["tests"] = updated_tests

    return result, updated_config


def _process_test_result(
    test: dict,
    actual: dict,
    encoding: str,
    mode: str,
    result: EncodingResult,
    updated_tests: list[dict] | None,
    update_goldens: bool,
    verbose: bool,
):
    """Process a single test result: compare, record, optionally update golden."""
    test_name = test.get("name", "unnamed")
    input_text = test.get("input", "")

    if "error" in actual:
        test_result = TestResult(
            name=test_name,
            passed=False,
            diff=actual["error"],
            input_text=input_text,
        )
        result.results.append(test_result)
        result.failed += 1
        print(f"  FAIL {test_name}: {actual['error']}")
        return

    actual_ids = actual.get("ids", [])

    # Determine expected IDs.
    if mode in ("ci", "verify") and not update_goldens:
        # Use stored golden IDs.
        expected_ids = test.get("golden_ids")
        if expected_ids is None:
            test_result = TestResult(
                name=test_name,
                passed=False,
                diff="No golden_ids in corpus (run generate_tiktoken_golden_ids.py)",
                input_text=input_text,
            )
            result.results.append(test_result)
            result.skipped += 1
            return
    else:
        # Use live tiktoken library output as ground truth.
        try:
            expected_ids = tiktoken_encode(encoding, input_text)
        except Exception as e:
            test_result = TestResult(
                name=test_name,
                passed=False,
                diff=f"tiktoken encode error: {e}",
                input_text=input_text,
            )
            result.results.append(test_result)
            result.failed += 1
            print(f"  FAIL {test_name}: tiktoken error: {e}")
            return

    match, diff = compare_ids(expected_ids, actual_ids)

    test_result = TestResult(
        name=test_name,
        passed=match,
        expected_ids=expected_ids,
        actual_ids=actual_ids,
        diff=diff,
        input_text=input_text,
    )
    result.results.append(test_result)

    if match:
        result.passed += 1
        if verbose:
            print(f"  PASS {test_name} ({len(actual_ids)} tokens)")
    else:
        result.failed += 1
        print(f"  FAIL {test_name}")
        print(f"    input:    {repr(input_text)[:80]}")
        print(
            f"    expected: {expected_ids[:20]}{'...' if len(expected_ids) > 20 else ''}"
        )
        print(f"    actual:   {actual_ids[:20]}{'...' if len(actual_ids) > 20 else ''}")
        if diff:
            print(f"    {diff}")

    # Update golden if requested.
    if update_goldens and updated_tests is not None and expected_ids is not None:
        for t in updated_tests:
            if t.get("name") == test_name:
                t["golden_ids"] = expected_ids


# =============================================================================
# Section 8: Cross-Validation
# =============================================================================

_hf_hub = None


def _get_hf_hub():
    """Lazy import of huggingface_hub."""
    global _hf_hub
    if _hf_hub is None:
        try:
            import huggingface_hub

            _hf_hub = huggingface_hub
        except ImportError:
            print(
                f"Error: huggingface_hub not installed.\nHint: {UV_HINT}",
                file=sys.stderr,
            )
            sys.exit(1)
    return _hf_hub


def download_hf_tokenizer_json(repo: str, cache_dir: Path) -> Path:
    """Download tokenizer.json from a HuggingFace repo."""
    hf_hub = _get_hf_hub()
    try:
        path = hf_hub.hf_hub_download(repo_id=repo, filename="tokenizer.json")
        return Path(path)
    except Exception as e:
        raise RuntimeError(f"Failed to download tokenizer.json from {repo}: {e}") from e


def cross_validate_encoding(
    encoding: str,
    binary_path: str,
    tiktoken_cache_dir: Path,
    verbose: bool,
) -> tuple[int, int]:
    """Cross-validate tiktoken vs HuggingFace JSON through IREE.

    Encodes the same texts through both loaders and verifies identical output.
    Returns (pass_count, fail_count).
    """
    hf_repo = CROSS_VALIDATION_HF_REPOS.get(encoding)
    if not hf_repo:
        print(f"  No HuggingFace equivalent configured for {encoding}")
        return 0, 0

    # Get file paths.
    tiktoken_path = tiktoken_cache_dir / f"{encoding}.tiktoken"
    if not tiktoken_path.exists():
        tiktoken_path = download_tiktoken_file(encoding, tiktoken_cache_dir)

    print(f"  Downloading HF tokenizer.json from {hf_repo}...")
    try:
        hf_path = download_hf_tokenizer_json(hf_repo, tiktoken_cache_dir)
    except Exception as e:
        print(f"  SKIP cross-validation: {e}")
        return 0, 0

    # Test inputs for cross-validation.
    test_inputs = [
        ("hello_world", "Hello, world!"),
        ("code", "def hello():\n    print('Hello!')"),
        ("numbers", "The year 2026 has 365 days."),
        ("cjk", "你好世界"),
        ("mixed", "English and 中文 mixed"),
        ("emoji", "Hello 🌍!"),
        ("whitespace", "  hello   world  "),
        ("empty", ""),
        ("punctuation", "!@#$%^&*()"),
        ("long_word", "antidisestablishmentarianism"),
    ]

    pass_count = 0
    fail_count = 0

    for name, text in test_inputs:
        tiktoken_result = run_iree_tokenize(binary_path, tiktoken_path, text)
        hf_result = run_iree_tokenize(binary_path, hf_path, text)

        tiktoken_ids = tiktoken_result.get("ids", [])
        hf_ids = hf_result.get("ids", [])

        if "error" in tiktoken_result:
            print(f"  FAIL {name}: tiktoken loader error: {tiktoken_result['error']}")
            fail_count += 1
            continue
        if "error" in hf_result:
            print(f"  FAIL {name}: HF loader error: {hf_result['error']}")
            fail_count += 1
            continue

        if tiktoken_ids == hf_ids:
            pass_count += 1
            if verbose:
                print(f"  PASS {name}: {len(tiktoken_ids)} tokens match")
        else:
            fail_count += 1
            print(f"  FAIL {name}: outputs differ")
            print(
                f"    tiktoken: {tiktoken_ids[:15]}{'...' if len(tiktoken_ids) > 15 else ''}"
            )
            print(f"    hf json:  {hf_ids[:15]}{'...' if len(hf_ids) > 15 else ''}")

            # Find first difference.
            for i, (t, h) in enumerate(zip(tiktoken_ids, hf_ids)):
                if t != h:
                    print(f"    First diff at position {i}: tiktoken={t}, hf={h}")
                    break
            if len(tiktoken_ids) != len(hf_ids):
                print(f"    Length: tiktoken={len(tiktoken_ids)}, hf={len(hf_ids)}")

    return pass_count, fail_count


# =============================================================================
# Section 9: Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Tiktoken integration tests for IREE tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--iree-tokenize",
        required=True,
        help="Path to iree-tokenize binary",
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "verify", "ci"],
        default="dev",
        help="Test mode: dev (download+live), verify (download+goldens), "
        "ci (cached+goldens). Default: dev",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        help="Test only this encoding (e.g., cl100k_base). Default: all.",
    )
    parser.add_argument(
        "--tokenizer-cache",
        type=str,
        default=None,
        help="Cache directory for .tiktoken files. Default: ~/.cache/iree-tiktoken",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help=f"Path to test corpus JSON. Default: {DEFAULT_CORPUS_PATH}",
    )
    parser.add_argument(
        "--update-goldens",
        action="store_true",
        help="Update golden IDs in corpus file from live tiktoken output.",
    )
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Cross-validate tiktoken vs HuggingFace JSON through IREE.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print passing tests too.",
    )
    args = parser.parse_args()

    # Resolve paths.
    binary_path = args.iree_tokenize
    if not os.path.isfile(binary_path):
        print(f"Error: iree-tokenize binary not found: {binary_path}", file=sys.stderr)
        sys.exit(1)

    corpus_path = Path(args.corpus) if args.corpus else DEFAULT_CORPUS_PATH
    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    cache_dir = (
        Path(args.tokenizer_cache)
        if args.tokenizer_cache
        else (
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "iree-tiktoken"
        )
    )

    # Load corpus.
    with open(corpus_path) as f:
        corpus = json.load(f)

    # Determine which encodings to test.
    if args.encoding:
        if args.encoding not in TIKTOKEN_URLS:
            print(
                f"Unknown encoding: {args.encoding}. "
                f"Available: {', '.join(TIKTOKEN_URLS.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)
        encodings = [args.encoding]
    else:
        encodings = list(TIKTOKEN_URLS.keys())

    # Run tests.
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    corpus_updated = False

    print(f"Mode: {args.mode}")
    print(f"Cache: {cache_dir}")
    print(f"Encodings: {', '.join(encodings)}")
    print()

    for encoding in encodings:
        print(f"{'=' * 60}")
        print(f"  {encoding}")
        print(f"{'=' * 60}")

        enc_result, updated_config = run_encoding_tests(
            encoding=encoding,
            binary_path=binary_path,
            mode=args.mode,
            corpus=corpus,
            tokenizer_cache_dir=cache_dir,
            update_goldens=args.update_goldens,
            verbose=args.verbose,
        )

        total_passed += enc_result.passed
        total_failed += enc_result.failed
        total_skipped += enc_result.skipped

        if enc_result.error:
            print(f"  ERROR: {enc_result.error}")

        status = (
            "PASS" if enc_result.failed == 0 and enc_result.error is None else "FAIL"
        )
        print(
            f"  {status}: {enc_result.passed} passed, {enc_result.failed} failed"
            f"{f', {enc_result.skipped} skipped' if enc_result.skipped else ''}"
            f" (load: {enc_result.load_time:.1f}s, test: {enc_result.test_time:.1f}s)"
        )
        print()

        if updated_config:
            corpus_key = f"{CORPUS_KEY_PREFIX}{encoding}"
            corpus["models"][corpus_key] = updated_config
            corpus_updated = True

    # Cross-validation (optional).
    if args.cross_validate:
        print(f"{'=' * 60}")
        print(f"  Cross-Validation: tiktoken vs HuggingFace JSON")
        print(f"{'=' * 60}")

        xv_passed = 0
        xv_failed = 0

        for encoding in encodings:
            print(f"\n  --- {encoding} ---")
            p, f = cross_validate_encoding(
                encoding,
                binary_path,
                cache_dir,
                args.verbose,
            )
            xv_passed += p
            xv_failed += f

        print(f"\n  Cross-validation: {xv_passed} passed, {xv_failed} failed")
        total_passed += xv_passed
        total_failed += xv_failed
        print()

    # Save updated corpus if goldens were updated.
    if corpus_updated:
        with open(corpus_path, "w") as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False, sort_keys=False)
            f.write("\n")
        print(f"Updated goldens in {corpus_path}")

    # Summary.
    print(f"{'=' * 60}")
    print(
        f"  TOTAL: {total_passed} passed, {total_failed} failed"
        f"{f', {total_skipped} skipped' if total_skipped else ''}"
    )
    print(f"{'=' * 60}")

    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    main()
