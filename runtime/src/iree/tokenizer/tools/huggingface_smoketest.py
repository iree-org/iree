#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Smoketest comparing iree-tokenize against HuggingFace tokenizers.

This tool runs test cases through both HuggingFace's tokenizers and iree-tokenize,
comparing the outputs to verify correctness. Test cases are defined in JSON files
in the testdata directory.

Installation:
    pip install transformers tokenizers huggingface_hub protobuf sentencepiece

    Or use uv (recommended - no installation required):
    uv run --with tokenizers --with huggingface_hub --with transformers \\
           --with protobuf --with sentencepiece python huggingface_smoketest.py ...

Usage:
    # Run all smoketests (with uv)
    uv run --with tokenizers --with huggingface_hub --with transformers \\
           --with protobuf --with sentencepiece python \\
           huggingface_smoketest.py --iree-tokenize /path/to/build/tools/iree-tokenize

    # Or if dependencies are already installed:
    python huggingface_smoketest.py --iree-tokenize /path/to/build/tools/iree-tokenize

    # Run tests for a specific model only
    python huggingface_smoketest.py --iree-tokenize /path/to/binary --model bert-base-uncased

    # Verbose output (show all test details)
    python huggingface_smoketest.py --iree-tokenize /path/to/binary --verbose

    # Save tokenizer.json files for manual inspection
    python huggingface_smoketest.py --iree-tokenize /path/to/binary --save-tokenizers /tmp/tokenizers

Adding New Test Cases:
    1. Create a new JSON file in the testdata/ directory (e.g., testdata/my_model.json)
    2. Set "model" to the HuggingFace model identifier
    3. Add test cases to the "tests" array
    4. Run this script to verify

Expected Failures (xfail):
    Files prefixed with 'xfail_' are expected to fail due to unimplemented features.
    These are skipped by default but can be included with --include-xfail.
    Use xfail for missing features (UNIMPLEMENTED errors), NOT for bugs.

Testdata JSON Schema:
    {
      "model": "bert-base-uncased",           # HuggingFace model identifier (required)
      "description": "BERT base uncased",      # Human-readable description (optional)
      "tests": [
        {
          "name": "simple_ascii",              # Test name for reporting (required)
          "input": "Hello, world!",            # Text to encode
          "add_special_tokens": true           # Whether to add special tokens (default: true)
        },
        {
          "name": "decode_test",
          "decode_ids": [101, 7592, 102],      # Token IDs to decode
        }
      ]
    }
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Recommended uv command for running this script with all dependencies.
UV_COMMAND = (
    "uv run --with tokenizers --with huggingface_hub --with transformers "
    "--with protobuf --with sentencepiece python"
)

# Import tokenizers library (fast, no PyTorch dependency).
try:
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
except ImportError:
    print(
        "Error: tokenizers or huggingface_hub library not installed.\n"
        "Install with: pip install tokenizers huggingface_hub\n"
        f"Or run with uv: {UV_COMMAND} {__file__} --iree-tokenize <path>",
        file=sys.stderr,
    )
    sys.exit(1)

# Lazy import for transformers (slow due to PyTorch/HIP init).
# Only imported when a model doesn't have tokenizer.json.
_transformers_auto_tokenizer = None


def _get_transformers_auto_tokenizer():
    """Lazy import of transformers.AutoTokenizer."""
    global _transformers_auto_tokenizer
    if _transformers_auto_tokenizer is None:
        try:
            from transformers import AutoTokenizer

            _transformers_auto_tokenizer = AutoTokenizer
        except ImportError:
            print(
                "Error: transformers library not installed.\n"
                "Install with: pip install transformers\n"
                f"Or run with uv: {UV_COMMAND} {__file__} --iree-tokenize <path>",
                file=sys.stderr,
            )
            sys.exit(1)
    return _transformers_auto_tokenizer


def load_hf_tokenizer(model_name: str):
    """Load HuggingFace tokenizer by model name.

    First tries the fast tokenizers library. Falls back to transformers
    (which can convert vocab.txt format) if tokenizer.json is not available.

    Returns (tokenizer, is_fast) where is_fast indicates if we got a
    tokenizers.Tokenizer (True) or transformers tokenizer (False).
    """
    # Try tokenizers library first (fast path).
    try:
        tokenizer_path = hf_hub_download(repo_id=model_name, filename="tokenizer.json")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        # Disable padding - we want raw token IDs like iree-tokenize.
        tokenizer.no_padding()
        return tokenizer, True
    except Exception:
        pass  # Fall back to transformers.

    # Fall back to transformers (can handle vocab.txt, spiece.model, etc.).
    AutoTokenizer = _get_transformers_auto_tokenizer()
    return AutoTokenizer.from_pretrained(model_name), False


def save_tokenizer_json(tokenizer, output_path: str, is_fast: bool) -> bool:
    """Save tokenizer to tokenizer.json file. Returns True on success."""
    try:
        tokenizer_json = os.path.join(output_path, "tokenizer.json")
        if is_fast:
            # tokenizers.Tokenizer - save directly.
            tokenizer.save(tokenizer_json)
        else:
            # transformers tokenizer - use save_pretrained.
            tokenizer.save_pretrained(output_path)
        return os.path.exists(tokenizer_json)
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}", file=sys.stderr)
        return False


def run_iree_tokenize(
    binary_path: str,
    tokenizer_json: str,
    input_text: str = None,
    decode_ids: list[int] = None,
    add_special_tokens: bool = True,
    skip_special_tokens: bool = True,
) -> dict | None:
    """Run iree-tokenize and return parsed JSON output."""
    cmd = [binary_path, tokenizer_json]

    if decode_ids is not None:
        cmd.append("--decode")
        # --no_special in decode mode means skip special tokens in output.
        # Default to True to match HuggingFace behavior.
        if skip_special_tokens:
            cmd.append("--no_special")
        cmd.append(",".join(str(x) for x in decode_ids))
    else:
        if not add_special_tokens:
            cmd.append("--no_special")
        cmd.append(input_text or "")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {"error": f"Exit code {result.returncode}: {result.stderr}"}
        return json.loads(result.stdout.strip())
    except subprocess.TimeoutExpired:
        return {"error": "Timeout"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}
    except Exception as e:
        return {"error": str(e)}


def run_iree_tokenize_batch(
    binary_path: str,
    tokenizer_json: str,
    inputs: list[str],
    add_special_tokens: bool = True,
) -> list[dict]:
    """Run iree-tokenize in batch mode for multiple encode inputs.

    This is much faster than spawning a process per input because we only
    load and compile the tokenizer once.
    """
    if not inputs:
        return []

    cmd = [binary_path, tokenizer_json, "--batch"]
    if not add_special_tokens:
        cmd.append("--no_special")

    # Join inputs with newlines for batch mode.
    batch_input = "\n".join(inputs)

    try:
        result = subprocess.run(
            cmd, input=batch_input, capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            error = {"error": f"Exit code {result.returncode}: {result.stderr}"}
            return [error] * len(inputs)

        # Parse each line of output as JSON.
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
        return [{"error": "Timeout"}] * len(inputs)
    except Exception as e:
        return [{"error": str(e)}] * len(inputs)


def run_hf_tokenize(
    tokenizer,
    is_fast: bool,
    input_text: str = None,
    decode_ids: list[int] = None,
    add_special_tokens: bool = True,
) -> dict:
    """Run HuggingFace tokenizer and return result in iree-tokenize format.

    Handles both tokenizers.Tokenizer (is_fast=True) and transformers tokenizers.
    """
    try:
        if decode_ids is not None:
            # skip_special_tokens=True to match IREE's --no_special behavior.
            # Both fast (tokenizers library) and slow (transformers) tokenizers
            # support this parameter.
            text = tokenizer.decode(decode_ids, skip_special_tokens=True)
            return {"text": text}
        else:
            if is_fast:
                encoding = tokenizer.encode(
                    input_text or "", add_special_tokens=add_special_tokens
                )
                ids = encoding.ids
            else:
                ids = tokenizer.encode(
                    input_text or "", add_special_tokens=add_special_tokens
                )
            return {"ids": ids}
    except Exception as e:
        return {"error": str(e)}


def compare_results(hf_result: dict, iree_result: dict) -> tuple[bool, str]:
    """Compare HF and IREE results. Returns (match, diff_description)."""
    if "error" in hf_result:
        return False, f"HF error: {hf_result['error']}"
    if "error" in iree_result:
        return False, f"IREE error: {iree_result['error']}"

    if "ids" in hf_result:
        hf_ids = hf_result["ids"]
        iree_ids = iree_result.get("ids", [])
        if hf_ids == iree_ids:
            return True, ""

        # Find first difference.
        for i, (h, r) in enumerate(zip(hf_ids, iree_ids)):
            if h != r:
                return False, f"Mismatch at position {i}: HF={h}, IREE={r}"
        if len(hf_ids) != len(iree_ids):
            return False, f"Length mismatch: HF={len(hf_ids)}, IREE={len(iree_ids)}"
        return False, "Unknown mismatch"

    if "text" in hf_result:
        hf_text = hf_result["text"]
        iree_text = iree_result.get("text", "")
        if hf_text == iree_text:
            return True, ""
        return (
            False,
            f"Text mismatch:\n  HF:   {repr(hf_text)}\n  IREE: {repr(iree_text)}",
        )

    return False, "Unknown result format"


def run_test_file(
    test_file: Path,
    iree_tokenize_path: str,
    verbose: bool,
    save_tokenizers_dir: str | None,
) -> tuple[int, int, list[str]]:
    """Run all tests from a testdata JSON file.

    Returns (passed, total, failures).
    """
    with open(test_file) as f:
        test_data = json.load(f)

    model_name = test_data["model"]
    description = test_data.get("description", model_name)
    tests = test_data.get("tests", [])

    print(f"\n=== {model_name} ({description}) ===")

    # Load HuggingFace tokenizer.
    load_start = time.time()
    try:
        tokenizer, is_fast = load_hf_tokenizer(model_name)
    except Exception as e:
        print(f"  ERROR: Could not load tokenizer: {e}")
        return 0, len(tests), [f"{model_name}: Failed to load tokenizer"]
    load_time = time.time() - load_start
    fast_marker = "fast" if is_fast else "slow"
    print(f"  [load: {load_time:.2f}s {fast_marker}]", end="")

    # Save tokenizer.json for iree-tokenize.
    save_start = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        if not save_tokenizer_json(tokenizer, tmpdir, is_fast):
            print(f"  ERROR: Could not save tokenizer.json")
            return 0, len(tests), [f"{model_name}: Failed to save tokenizer.json"]
        save_time = time.time() - save_start
        print(f" [save: {save_time:.2f}s]")

        tokenizer_json = os.path.join(tmpdir, "tokenizer.json")

        # Copy to save directory if requested.
        if save_tokenizers_dir:
            save_path = os.path.join(
                save_tokenizers_dir, f"{model_name.replace('/', '_')}.json"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            import shutil

            shutil.copy(tokenizer_json, save_path)
            print(f"  Saved tokenizer to: {save_path}")

        passed = 0
        failures = []
        hf_total = 0.0
        iree_total = 0.0

        # Group tests by type for batching: encode tests can be batched,
        # decode tests must run individually.
        # NOTE: Batch mode uses newlines as delimiters, so inputs containing
        # newlines must be run individually.
        encode_tests_special = []  # (test, input_text) with add_special_tokens=True
        encode_tests_no_special = []  # (test, input_text) with add_special_tokens=False
        encode_tests_individual = []  # Inputs with newlines (can't batch)
        decode_tests = []  # (test, decode_ids)

        for test in tests:
            decode_ids = test.get("decode_ids")
            if decode_ids is not None:
                decode_tests.append((test, decode_ids))
            else:
                input_text = test.get("input", "")
                add_special = test.get("add_special_tokens", True)
                if "\n" in input_text:
                    # Can't batch inputs with newlines.
                    encode_tests_individual.append((test, input_text, add_special))
                elif add_special:
                    encode_tests_special.append((test, input_text))
                else:
                    encode_tests_no_special.append((test, input_text))

        # Run encode tests in batch (one subprocess per group).
        def run_encode_batch(test_group, add_special):
            nonlocal passed, hf_total, iree_total
            if not test_group:
                return
            inputs = [t[1] for t in test_group]

            # HF encode - use batch API for performance.
            hf_start = time.time()
            try:
                if is_fast:
                    # tokenizers library supports encode_batch.
                    encodings = tokenizer.encode_batch(
                        inputs, add_special_tokens=add_special
                    )
                    hf_results = [{"ids": enc.ids} for enc in encodings]
                else:
                    # transformers tokenizer - use batch call.
                    batch_output = tokenizer(
                        inputs,
                        add_special_tokens=add_special,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                    )
                    hf_results = [{"ids": ids} for ids in batch_output["input_ids"]]
            except Exception as e:
                hf_results = [{"error": str(e)}] * len(inputs)
            hf_total += time.time() - hf_start

            # IREE encode in batch.
            iree_start = time.time()
            iree_results = run_iree_tokenize_batch(
                iree_tokenize_path, tokenizer_json, inputs, add_special
            )
            iree_total += time.time() - iree_start

            # Compare.
            for (test, _), hf_result, iree_result in zip(
                test_group, hf_results, iree_results
            ):
                test_name = test.get("name", "unnamed")
                input_text = test.get("input", "")
                match, diff = compare_results(hf_result, iree_result)
                if match:
                    passed += 1
                    if verbose:
                        print(f"  ✓ {test_name}: {repr(input_text[:40])}")
                else:
                    print(f"  ✗ {test_name}: {repr(input_text[:40])}")
                    print(f"    HF:   {json.dumps(hf_result, separators=(',', ':'))}")
                    print(f"    IREE: {json.dumps(iree_result, separators=(',', ':'))}")
                    if diff:
                        print(f"    {diff}")
                    failures.append(f"{model_name}/{test_name}")

        run_encode_batch(encode_tests_special, add_special=True)
        run_encode_batch(encode_tests_no_special, add_special=False)

        # Run encode tests with newlines individually.
        for test, input_text, add_special in encode_tests_individual:
            test_name = test.get("name", "unnamed")

            hf_start = time.time()
            hf_result = run_hf_tokenize(
                tokenizer, is_fast, input_text, None, add_special
            )
            hf_total += time.time() - hf_start

            iree_start = time.time()
            iree_result = run_iree_tokenize(
                iree_tokenize_path, tokenizer_json, input_text, None, add_special
            )
            iree_total += time.time() - iree_start

            match, diff = compare_results(hf_result, iree_result)
            if match:
                passed += 1
                if verbose:
                    print(f"  ✓ {test_name}: {repr(input_text[:40])}")
            else:
                print(f"  ✗ {test_name}: {repr(input_text[:40])}")
                print(f"    HF:   {json.dumps(hf_result, separators=(',', ':'))}")
                print(f"    IREE: {json.dumps(iree_result, separators=(',', ':'))}")
                if diff:
                    print(f"    {diff}")
                failures.append(f"{model_name}/{test_name}")

        # Run decode tests individually (no batch support).
        for test, decode_ids in decode_tests:
            test_name = test.get("name", "unnamed")

            hf_start = time.time()
            hf_result = run_hf_tokenize(tokenizer, is_fast, None, decode_ids, True)
            hf_total += time.time() - hf_start

            iree_start = time.time()
            iree_result = run_iree_tokenize(
                iree_tokenize_path, tokenizer_json, None, decode_ids, True
            )
            iree_total += time.time() - iree_start

            match, diff = compare_results(hf_result, iree_result)
            if match:
                passed += 1
                if verbose:
                    print(f"  ✓ {test_name}: decode({decode_ids})")
            else:
                print(f"  ✗ {test_name}: decode({decode_ids})")
                print(f"    HF:   {json.dumps(hf_result, separators=(',', ':'))}")
                print(f"    IREE: {json.dumps(iree_result, separators=(',', ':'))}")
                if diff:
                    print(f"    {diff}")
                failures.append(f"{model_name}/{test_name}")

        if not verbose and passed == len(tests):
            print(f"  ✓ {passed}/{len(tests)} tests passed", end="")
        print(f"  [hf: {hf_total:.2f}s, iree: {iree_total:.2f}s]")

        return passed, len(tests), failures


def discover_testdata(
    testdata_dir: Path, model_filter: str | None, include_xfail: bool
) -> tuple[list[Path], list[Path]]:
    """Discover testdata JSON files.

    Returns (test_files, xfail_files).
    When include_xfail is True, xfail_files are still returned separately
    so we can track which ones now pass.
    """
    all_files = sorted(testdata_dir.glob("*.json"))

    test_files = []
    xfail_files = []

    for f in all_files:
        is_xfail = f.name.startswith("xfail_")

        # Apply model filter if specified.
        if model_filter:
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    if model_filter.lower() not in data.get("model", "").lower():
                        continue
            except Exception:
                continue

        if is_xfail:
            xfail_files.append(f)
        else:
            test_files.append(f)

    return test_files, xfail_files


def main():
    parser = argparse.ArgumentParser(
        description="Smoketest comparing iree-tokenize against HuggingFace tokenizers.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--iree-tokenize",
        required=True,
        help="Path to iree-tokenize binary",
    )
    parser.add_argument(
        "--testdata-dir",
        type=Path,
        default=None,
        help="Directory containing testdata JSON files (default: script_dir/testdata)",
    )
    parser.add_argument(
        "--model",
        help="Run tests only for models matching this string",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show details for all tests, not just failures",
    )
    parser.add_argument(
        "--save-tokenizers",
        metavar="DIR",
        help="Save downloaded tokenizer.json files to this directory",
    )
    parser.add_argument(
        "--include-xfail",
        action="store_true",
        help="Include xfail_ prefixed tests (expected failures due to unimplemented features)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to first N test files (for debugging performance)",
    )

    args = parser.parse_args()

    # Verify iree-tokenize exists.
    if not os.path.isfile(args.iree_tokenize):
        print(f"Error: iree-tokenize not found: {args.iree_tokenize}", file=sys.stderr)
        sys.exit(1)
    if not os.access(args.iree_tokenize, os.X_OK):
        print(
            f"Error: iree-tokenize is not executable: {args.iree_tokenize}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Find testdata directory.
    if args.testdata_dir:
        testdata_dir = args.testdata_dir
    else:
        script_dir = Path(__file__).parent
        testdata_dir = script_dir / "testdata"

    if not testdata_dir.is_dir():
        print(f"Error: Testdata directory not found: {testdata_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover test files.
    test_files, xfail_files = discover_testdata(
        testdata_dir, args.model, args.include_xfail
    )
    if not test_files and not xfail_files:
        print(f"No testdata files found in: {testdata_dir}")
        if args.model:
            print(f"  (filtered by model: {args.model})")
        sys.exit(1)

    # Apply limit if specified.
    if args.limit > 0:
        test_files = test_files[: args.limit]
        xfail_files = xfail_files[: max(0, args.limit - len(test_files))]

    # Build run message.
    if args.include_xfail:
        xfail_msg = f" + {len(xfail_files)} xfail" if xfail_files else ""
    else:
        xfail_msg = f" (skipping {len(xfail_files)} xfail)" if xfail_files else ""
    limit_msg = f" (limited to {args.limit})" if args.limit > 0 else ""
    print(
        f"Running {len(test_files)} test file(s) from {testdata_dir}{xfail_msg}{limit_msg}"
    )

    # Create save directory if needed.
    if args.save_tokenizers:
        os.makedirs(args.save_tokenizers, exist_ok=True)

    # Run regular tests.
    total_passed = 0
    total_tests = 0
    all_failures = []

    for test_file in test_files:
        passed, total, failures = run_test_file(
            test_file, args.iree_tokenize, args.verbose, args.save_tokenizers
        )
        total_passed += passed
        total_tests += total
        all_failures.extend(failures)

    # Run xfail tests if requested.
    xfail_passed_files = []
    xfail_still_failing = []
    if args.include_xfail and xfail_files:
        print(f"\n--- Running {len(xfail_files)} xfail file(s) ---")
        for test_file in xfail_files:
            passed, total, failures = run_test_file(
                test_file,
                args.iree_tokenize,
                args.verbose,
                args.save_tokenizers,
            )
            total_passed += passed
            total_tests += total
            all_failures.extend(failures)
            if passed == total:
                xfail_passed_files.append(test_file)
            else:
                xfail_still_failing.append(test_file)

    # Summary.
    print(f"\n{'=' * 50}")
    if total_passed == total_tests:
        print(f"SUCCESS: {total_passed}/{total_tests} tests passed")
    else:
        print(
            f"FAILED: {total_passed}/{total_tests} tests passed ({total_tests - total_passed} failed)"
        )
        print("\nFailures:")
        for failure in all_failures:
            print(f"  - {failure}")

    # Report xfail files that now pass.
    if xfail_passed_files:
        print(f"\n\u2705 XFAIL files that now PASS ({len(xfail_passed_files)}):")
        print("   These can have the 'xfail_' prefix removed:")
        for f in xfail_passed_files:
            print(f"     - {f.name}")

    if xfail_still_failing:
        print(
            f"\n\u26a0\ufe0f  XFAIL files still failing ({len(xfail_still_failing)}):"
        )
        for f in xfail_still_failing:
            print(f"     - {f.name}")

    if xfail_files and not args.include_xfail:
        print(
            f"\nSkipped {len(xfail_files)} xfail file(s) (use --include-xfail to run):"
        )
        for f in xfail_files:
            print(f"  - {f.name}")

    sys.exit(0 if total_passed == total_tests else 1)


if __name__ == "__main__":
    main()
