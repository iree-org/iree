#!/bin/bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Wrapper script for running tiktoken_smoketest.py with all required
# Python dependencies via uvx.
#
# Usage:
#   ./run_tiktoken_smoketest.sh --iree-tokenize /path/to/binary [other args...]
#
# Examples:
#   # Run all tests
#   ./run_tiktoken_smoketest.sh --iree-tokenize $(iree-bazel-run --print-path //tools:iree-tokenize)
#
#   # Update goldens after fixing bugs
#   ./run_tiktoken_smoketest.sh --iree-tokenize /path/to/binary --update-goldens
#
#   # Test specific encoding
#   ./run_tiktoken_smoketest.sh --iree-tokenize /path/to/binary --encoding cl100k_base
#
#   # Cross-validate tiktoken vs HuggingFace
#   ./run_tiktoken_smoketest.sh --iree-tokenize /path/to/binary --cross-validate

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python dependencies:
#   tiktoken        - OpenAI's tiktoken library for reference encoding
#   huggingface_hub - Download equivalent HuggingFace tokenizer.json files
#   tokenizers      - HuggingFace tokenizers for cross-validation
#   requests        - HTTP downloads for .tiktoken files

DEPS=(
    --with tiktoken
    --with huggingface_hub
    --with tokenizers
    --with requests
)

exec uvx "${DEPS[@]}" python "$SCRIPT_DIR/tiktoken_smoketest.py" "$@"
