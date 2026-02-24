#!/bin/bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Downloads popular HuggingFace tokenizers and runs comprehensive benchmarks.
# All downloads are cached; subsequent runs reuse cached files.
#
# Usage:
#   ./run_benchmarks.sh [benchmark flags...]
#
# Examples:
#   # Run all tokenizers with 1s minimum benchmark time
#   ./run_benchmarks.sh --benchmark_min_time=1s
#
#   # Run only encode benchmarks for a specific tokenizer
#   ./run_benchmarks.sh --benchmark_filter=".*Encode.*" --benchmark_min_time=2s
#
#   # Output JSON results
#   ./run_benchmarks.sh --benchmark_min_time=1s --benchmark_format=json
#
#   # Run with defaults (1s min_time per benchmark)
#   ./run_benchmarks.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/iree-tokenizer-benchmarks"
HF_BASE="https://huggingface.co"

# Ordered list of tokenizers to benchmark.
# Each entry: "name|repo" where the tokenizer.json URL is:
#   ${HF_BASE}/${repo}/resolve/main/tokenizer.json
TOKENIZERS=(
  "gpt2|gpt2"
  "llama3|NousResearch/Meta-Llama-3-8B"
  "gemma-2b|unsloth/gemma-2b"
  "qwen2.5|Qwen/Qwen2.5-7B"
  "bloom|bigscience/bloom-560m"
  "mistral-nemo|mistralai/Mistral-Nemo-Instruct-2407"
  "deepseek-v3|deepseek-ai/DeepSeek-V3"
  "whisper|openai/whisper-large-v3"
  "bert|google-bert/bert-base-uncased"
  "t5|google-t5/t5-base"
)

# Download tokenizer.json files (cached).
echo "Checking tokenizer cache at $CACHE_DIR ..."
mkdir -p "$CACHE_DIR"

download_count=0
for entry in "${TOKENIZERS[@]}"; do
  name="${entry%%|*}"
  repo="${entry#*|}"
  dest="$CACHE_DIR/$name.json"
  if [ ! -f "$dest" ]; then
    url="$HF_BASE/$repo/resolve/main/tokenizer.json"
    echo "  Downloading $name from $repo ..."
    if ! curl -sL --fail -o "$dest.tmp" "$url"; then
      echo "  WARNING: Failed to download $name ($url), skipping."
      rm -f "$dest.tmp"
      continue
    fi
    mv "$dest.tmp" "$dest"
    download_count=$((download_count + 1))
  fi
done

if [ "$download_count" -gt 0 ]; then
  echo "  Downloaded $download_count new tokenizer(s)."
else
  echo "  All tokenizers cached."
fi
echo ""

# Download text corpora (cached).
# These provide realistic input for benchmarks. The embedded ~1KB seeds in the
# benchmark binary are fallbacks for CI only — real benchmarking needs real text.
TEXT_DIR="$CACHE_DIR/text"
mkdir -p "$TEXT_DIR"

ASCII_TEXT="$TEXT_DIR/ascii.txt"
CJK_TEXT="$TEXT_DIR/cjk.txt"
CODE_TEXT="$TEXT_DIR/code.txt"

echo "Checking text corpus cache at $TEXT_DIR ..."
text_download_count=0

# ASCII: Project Gutenberg — Adventures of Sherlock Holmes (~594KB, UTF-8).
if [ ! -f "$ASCII_TEXT" ]; then
  url="https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
  echo "  Downloading ASCII corpus (Sherlock Holmes) ..."
  if curl -sL --fail -o "$ASCII_TEXT.tmp" "$url"; then
    mv "$ASCII_TEXT.tmp" "$ASCII_TEXT"
    text_download_count=$((text_download_count + 1))
  else
    echo "  WARNING: Failed to download ASCII corpus ($url)."
    rm -f "$ASCII_TEXT.tmp"
  fi
fi

# CJK: Project Gutenberg — Chinese classic text (~100KB, UTF-8).
if [ ! -f "$CJK_TEXT" ]; then
  url="https://www.gutenberg.org/cache/epub/23950/pg23950.txt"
  echo "  Downloading CJK corpus ..."
  if curl -sL --fail -o "$CJK_TEXT.tmp" "$url"; then
    mv "$CJK_TEXT.tmp" "$CJK_TEXT"
    text_download_count=$((text_download_count + 1))
  else
    echo "  WARNING: Failed to download CJK corpus ($url)."
    echo "           Benchmarks will use embedded CJK seed text."
    rm -f "$CJK_TEXT.tmp"
  fi
fi

# Code: Concatenate .c files from the repo's tokenizer source.
# Always available (no download), real C code, updated on each run.
if [ ! -f "$CODE_TEXT" ] || \
   [ "$(find "$REPO_ROOT/runtime/src/iree/tokenizer" -name '*.c' -newer "$CODE_TEXT" 2>/dev/null | head -1)" ]; then
  echo "  Generating code corpus from tokenizer source ..."
  find "$REPO_ROOT/runtime/src/iree/tokenizer" -name '*.c' -print0 | \
    sort -z | xargs -0 cat > "$CODE_TEXT.tmp"
  mv "$CODE_TEXT.tmp" "$CODE_TEXT"
  text_download_count=$((text_download_count + 1))
fi

if [ "$text_download_count" -gt 0 ]; then
  echo "  Updated $text_download_count text corpus file(s)."
else
  echo "  All text corpora cached."
fi

# Report corpus sizes.
for f in "$ASCII_TEXT" "$CJK_TEXT" "$CODE_TEXT"; do
  if [ -f "$f" ]; then
    size_kb=$(( $(wc -c < "$f") / 1024 ))
    echo "  $(basename "$f"): ${size_kb}KB"
  fi
done
echo ""

# Build text flags for the benchmark binary.
TEXT_FLAGS=""
[ -f "$ASCII_TEXT" ] && TEXT_FLAGS="$TEXT_FLAGS --ascii_text=$ASCII_TEXT"
[ -f "$CJK_TEXT" ] && TEXT_FLAGS="$TEXT_FLAGS --cjk_text=$CJK_TEXT"
[ -f "$CODE_TEXT" ] && TEXT_FLAGS="$TEXT_FLAGS --code_text=$CODE_TEXT"

# Build benchmark binary with full optimization.
echo "Building benchmark binary (O3 + march=native + thin_lto)..."
TARGET="//runtime/src/iree/tokenizer/tools:comprehensive_benchmark"
"$REPO_ROOT/build_tools/bin/iree-bazel-build" \
  --copt=-O3 --copt=-march=native --features=thin_lto \
  "$TARGET" 2>&1 | tail -1
echo ""

BENCHMARK_BIN="$REPO_ROOT/bazel-bin/runtime/src/iree/tokenizer/tools/comprehensive_benchmark"
if [ ! -x "$BENCHMARK_BIN" ]; then
  echo "ERROR: Benchmark binary not found at $BENCHMARK_BIN"
  exit 1
fi

# Run benchmarks for each tokenizer.
failed_count=0
for entry in "${TOKENIZERS[@]}"; do
  name="${entry%%|*}"
  dest="$CACHE_DIR/$name.json"
  if [ ! -f "$dest" ]; then
    echo "================================================================"
    echo "  SKIPPED: $name (download failed)"
    echo "================================================================"
    echo ""
    continue
  fi

  echo "================================================================"
  echo "  $name"
  echo "================================================================"
  if ! "$BENCHMARK_BIN" --tokenizer_json="$dest" --rotate $TEXT_FLAGS \
    --benchmark_min_time=1s "$@"; then
    echo "  FAILED: $name (exit code $?)"
    failed_count=$((failed_count + 1))
  fi
  echo ""
done

echo "Done. Results for $(echo "${TOKENIZERS[@]}" | tr ' ' '\n' | wc -l) tokenizers."
if [ "$failed_count" -gt 0 ]; then
  echo "WARNING: $failed_count tokenizer(s) failed."
  exit 1
fi
echo "Cache directory: $CACHE_DIR"
