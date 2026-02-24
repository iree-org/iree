#!/bin/bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Wrapper script for running huggingface_smoketest.py with all required
# Python dependencies via uvx.
#
# Usage:
#   ./run_smoketest.sh --iree-tokenize /path/to/binary [other args...]
#
# Examples:
#   # Run all tests
#   ./run_smoketest.sh --iree-tokenize $(iree-bazel-run --print-path //tools:iree-tokenize)
#
#   # Update goldens after fixing bugs
#   ./run_smoketest.sh --iree-tokenize /path/to/binary --update-goldens
#
#   # Test specific model
#   ./run_smoketest.sh --iree-tokenize /path/to/binary --model gpt2
#
#   # Run fuzz tests
#   ./run_smoketest.sh --iree-tokenize /path/to/binary --fuzz

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# All Python dependencies needed for full model coverage.
# Core deps:
#   tokenizers      - HuggingFace tokenizers library
#   huggingface_hub - Model downloads
#   transformers    - Required for some models
# Serialization:
#   sentencepiece   - Required for T5, XLM-R, DeBERTa
#   protobuf        - Required for DeBERTa
# Language-specific:
#   sacremoses      - Required for FlauBERT (French)
#   fugashi         - Required for Japanese BERT (MeCab wrapper)
#   ipadic          - Required for Japanese BERT (MeCab dictionary)
#   unidic_lite     - Required for Japanese BERT (word segmentation)
# Vision:
#   tiktoken        - Required for some vision models

DEPS=(
    --with tokenizers
    --with huggingface_hub
    --with transformers
    --with sentencepiece
    --with protobuf
    --with sacremoses
    --with fugashi
    --with ipadic
    --with unidic_lite
    --with tiktoken
)

exec uvx "${DEPS[@]}" python "$SCRIPT_DIR/huggingface_smoketest.py" "$@"
