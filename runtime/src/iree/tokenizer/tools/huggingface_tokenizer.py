#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""HuggingFace tokenizer CLI wrapper with iree-tokenize compatible output.

This tool wraps HuggingFace's tokenizers to produce JSON output matching
iree-tokenize, enabling direct comparison between implementations.

Installation:
    pip install transformers tokenizers

Examples:
    # Encode text using a HuggingFace model (downloads/caches automatically)
    python huggingface_tokenizer.py bert-base-uncased "Hello, world!"
    # Output: {"ids":[101,7592,1010,2088,999,102]}

    # Encode using a local tokenizer.json file
    python huggingface_tokenizer.py /path/to/tokenizer.json "Hello, world!"

    # Decode token IDs back to text
    python huggingface_tokenizer.py bert-base-uncased --decode "101,7592,1010,2088,999,102"
    # Output: {"text":"[CLS] hello, world! [SEP]"}

    # Encode without special tokens (no [CLS]/[SEP] or BOS/EOS)
    python huggingface_tokenizer.py bert-base-uncased --no_special "Hello"

    # Show tokenizer metadata
    python huggingface_tokenizer.py bert-base-uncased --info

    # Batch mode: encode multiple lines from stdin
    echo -e "hello\\nworld" | python huggingface_tokenizer.py bert-base-uncased --batch

    # Compare with iree-tokenize
    diff <(python huggingface_tokenizer.py bert-base-uncased "hello") \\
         <(iree-tokenize tokenizer.json "hello")
"""

import argparse
import json
import os
import sys

# Recommended uv command for running this script with all dependencies.
UV_COMMAND = "uv run --with transformers --with tokenizers python"


def load_tokenizer(model_or_path: str):
    """Load tokenizer from HuggingFace model name or local tokenizer.json path."""
    # Import here to give helpful error message if not installed.
    try:
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
    except ImportError:
        print(
            "Error: transformers library not installed.\n"
            "Install with: pip install transformers tokenizers\n"
            f"Or run with uv: {UV_COMMAND} {__file__} <model> <text>",
            file=sys.stderr,
        )
        sys.exit(1)

    # Detect if input is a local path or HuggingFace model name.
    if os.path.isfile(model_or_path):
        # Local tokenizer.json file.
        return PreTrainedTokenizerFast(tokenizer_file=model_or_path)
    elif os.path.isdir(model_or_path):
        # Local model directory - use AutoTokenizer to handle various formats
        # (tokenizer.json, vocab.txt, spiece.model, etc.).
        return AutoTokenizer.from_pretrained(model_or_path)
    else:
        # HuggingFace model name - downloads and caches automatically.
        return AutoTokenizer.from_pretrained(model_or_path)


def get_special_token_id(tokenizer, attr_name: str, fallback_token: str = None) -> int:
    """Get special token ID, returning -1 if not defined."""
    # Try attribute first (e.g., tokenizer.cls_token_id).
    token_id = getattr(tokenizer, attr_name, None)
    if token_id is not None:
        return token_id

    # Try looking up fallback token in vocab.
    if fallback_token and hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
        if fallback_token in vocab:
            return vocab[fallback_token]

    return -1


def get_model_type(tokenizer) -> str:
    """Determine model type (BPE or WordPiece) from tokenizer."""
    # Check backend tokenizer if available.
    if hasattr(tokenizer, "backend_tokenizer"):
        model = tokenizer.backend_tokenizer.model
        model_type = type(model).__name__
        if "BPE" in model_type:
            return "BPE"
        if "WordPiece" in model_type:
            return "WordPiece"

    # Fallback: check for merge rules (BPE has them, WordPiece doesn't).
    if hasattr(tokenizer, "backend_tokenizer"):
        # BPE models have merges.
        try:
            model_str = str(tokenizer.backend_tokenizer.model)
            if "merges" in model_str.lower():
                return "BPE"
        except Exception:
            pass

    # Check for WordPiece indicators.
    if hasattr(tokenizer, "wordpiece_tokenizer"):
        return "WordPiece"

    # Default based on common patterns.
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    if any(k.startswith("##") for k in vocab.keys()):
        return "WordPiece"

    return "BPE"  # Default assumption.


def print_info(tokenizer):
    """Print tokenizer metadata as JSON (matching iree-tokenize --info output)."""
    info = {
        "vocab_size": tokenizer.vocab_size,
        "model_type": get_model_type(tokenizer),
    }

    # Add merge count for BPE models.
    if info["model_type"] == "BPE" and hasattr(tokenizer, "backend_tokenizer"):
        try:
            # Access the BPE model's merges.
            model = tokenizer.backend_tokenizer.model
            if hasattr(model, "get_trainer"):
                # This is a rough approximation.
                pass
        except Exception:
            pass

    # Add special token IDs (only if defined, matching iree-tokenize behavior).
    special_tokens = [
        ("bos_id", "bos_token_id", None),
        ("eos_id", "eos_token_id", None),
        ("unk_id", "unk_token_id", "[UNK]"),
        ("pad_id", "pad_token_id", "[PAD]"),
        ("cls_id", "cls_token_id", "[CLS]"),
        ("sep_id", "sep_token_id", "[SEP]"),
        ("mask_id", "mask_token_id", "[MASK]"),
    ]

    for key, attr, fallback in special_tokens:
        token_id = get_special_token_id(tokenizer, attr, fallback)
        if token_id >= 0:
            info[key] = token_id

    print(json.dumps(info, separators=(",", ":")))


def encode_text(tokenizer, text: str, add_special_tokens: bool) -> list[int]:
    """Encode text to token IDs."""
    return tokenizer.encode(text, add_special_tokens=add_special_tokens)


def decode_ids(tokenizer, ids: list[int], skip_special_tokens: bool = True) -> str:
    """Decode token IDs to text."""
    return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


def parse_ids(ids_str: str) -> list[int] | str:
    """Parse comma-separated token IDs. Returns error string on failure."""
    ids_str = ids_str.strip()
    if not ids_str:
        return []
    try:
        return [int(x.strip()) for x in ids_str.split(",")]
    except ValueError as e:
        return f"Invalid token ID: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace tokenizer CLI with iree-tokenize compatible output.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Decode mode: input is comma-separated token IDs",
    )
    parser.add_argument(
        "--no_special",
        action="store_true",
        help="Don't add special tokens ([CLS]/[SEP], BOS/EOS)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: read inputs from stdin, one per line",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print tokenizer metadata and exit",
    )

    # Use parse_known_args to separate flags from positional args.
    # This allows flags to appear anywhere (like iree-tokenize).
    args, remaining = parser.parse_known_args()

    # Remaining args: model [input...]
    if not remaining:
        parser.error("model argument is required")
    args.model = remaining[0]
    args.input = remaining[1:] if len(remaining) > 1 else []

    # Load the tokenizer.
    tokenizer = load_tokenizer(args.model)

    # Info mode.
    if args.info:
        print_info(tokenizer)
        return

    # Determine input source.
    if args.batch:
        # Batch mode: read from stdin.
        for line in sys.stdin:
            line = line.rstrip("\n\r")
            if args.decode:
                ids = parse_ids(line)
                if isinstance(ids, str):
                    print(json.dumps({"error": ids}, separators=(",", ":")))
                else:
                    text = decode_ids(
                        tokenizer, ids, skip_special_tokens=args.no_special
                    )
                    print(json.dumps({"text": text}, separators=(",", ":")))
            else:
                ids = encode_text(tokenizer, line, not args.no_special)
                print(json.dumps({"ids": ids}, separators=(",", ":")))
    else:
        # Single input mode.
        input_text = " ".join(args.input) if args.input else ""

        if args.decode:
            ids = parse_ids(input_text)
            if isinstance(ids, str):
                print(json.dumps({"error": ids}, separators=(",", ":")))
                sys.exit(1)
            text = decode_ids(tokenizer, ids, skip_special_tokens=args.no_special)
            print(json.dumps({"text": text}, separators=(",", ":")))
        else:
            ids = encode_text(tokenizer, input_text, not args.no_special)
            print(json.dumps({"ids": ids}, separators=(",", ":")))


if __name__ == "__main__":
    main()
