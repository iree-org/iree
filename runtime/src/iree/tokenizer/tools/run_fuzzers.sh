#!/bin/bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Orchestrates fuzzing of the tokenizer library using iree-bazel-fuzz.
#
# Two modes:
#
#   Integration fuzzing (--tokenizer_json):
#     Downloads HuggingFace tokenizer JSONs, then runs each integration fuzz
#     target against each real tokenizer. Tests the full encode/decode pipeline
#     with real vocabularies, merge tables, normalizers, and segmenters.
#
#   Component fuzzing (standalone):
#     Runs component-level fuzz targets (normalizer, segmenter, decoder, model,
#     format) using iree-bazel-fuzz's multi-target pattern support. These
#     targets build their own adversarial configurations from fuzz input.
#
# All fuzzing goes through iree-bazel-fuzz — no direct binary execution,
# no duplicate build config. Persistent corpus, dictionaries, and artifacts
# are managed by iree-bazel-fuzz under ~/.cache/iree-fuzz-{cache,corpus}/.
#
# Usage:
#   ./run_fuzzers.sh                          # All fuzzers, 120s per combo
#   ./run_fuzzers.sh --duration=300           # 5 minutes per combo
#   ./run_fuzzers.sh --forever                # Loop until crash
#   ./run_fuzzers.sh --integration-only       # Only real-tokenizer tests
#   ./run_fuzzers.sh --component-only         # Only component tests
#   ./run_fuzzers.sh --jobs=16                # 16 parallel libfuzzer jobs
#
# See https://iree.dev/developers/debugging/fuzzing/ for more information.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

# Reuse the same cache directory as run_benchmarks.sh.
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/iree-tokenizer-benchmarks"

HF_BASE="https://huggingface.co"

#===----------------------------------------------------------------------===#
# Configuration
#===----------------------------------------------------------------------===#

DURATION=120
FOREVER=0
JOBS=$(( $(nproc) / 2 ))
RUN_INTEGRATION=1
RUN_COMPONENT=1

# Integration fuzz targets (use fuzzing_util, accept --tokenizer_json).
INTEGRATION_TARGETS=(
  "tokenizer_encode_fuzz"
  "tokenizer_decode_fuzz"
  "tokenizer_ringbuffer_fuzz"
  "tokenizer_roundtrip_fuzz"
  "tokenizer_batch_encode_fuzz"
  "tokenizer_batch_decode_fuzz"
)

# Component fuzz target patterns (iree-bazel-fuzz expands ... to all *_fuzz).
COMPONENT_PATTERNS=(
  "//runtime/src/iree/tokenizer/normalizer/..."
  "//runtime/src/iree/tokenizer/segmenter/..."
  "//runtime/src/iree/tokenizer/decoder/..."
  "//runtime/src/iree/tokenizer/model/..."
  "//runtime/src/iree/tokenizer/format/..."
  "//runtime/src/iree/tokenizer/vocab/..."
  "//runtime/src/iree/tokenizer:special_tokens_fuzz"
  "//runtime/src/iree/tokenizer:postprocessor_fuzz"
)

# HuggingFace tokenizers to download and fuzz against.
# Same list as run_benchmarks.sh. Format: "name|repo".
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

#===----------------------------------------------------------------------===#
# Flag Parsing
#===----------------------------------------------------------------------===#

show_help() {
  cat << 'EOF'
run_fuzzers.sh - Orchestrate tokenizer fuzzing via iree-bazel-fuzz

USAGE
    ./run_fuzzers.sh [options]

OPTIONS
    --duration=N          Seconds per (tokenizer x target) combo (default: 120)
    --forever             Loop all combos indefinitely until a crash
    --jobs=N              Parallel libfuzzer jobs per target (default: nproc/2)
    --integration-only    Only run integration fuzzers (with real tokenizers)
    --component-only      Only run component fuzzers
    -h, --help            Show this help

INTEGRATION TARGETS
    Encode, decode, roundtrip, batch encode/decode, ringbuffer fuzzers.
    Each runs against every downloaded HuggingFace tokenizer JSON with
    --tokenizer_json and --track_offsets flags.

COMPONENT TARGETS
    Normalizer, segmenter, decoder, model, format, vocab, postprocessor,
    and special_tokens fuzzers. These build adversarial configs from fuzz
    input and don't need a real tokenizer.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --duration=*) DURATION="${arg#*=}" ;;
    --forever) FOREVER=1 ;;
    --jobs=*) JOBS="${arg#*=}" ;;
    --integration-only) RUN_COMPONENT=0 ;;
    --component-only) RUN_INTEGRATION=0 ;;
    -h|--help) show_help; exit 0 ;;
    *)
      echo "ERROR: Unknown flag: $arg"
      echo "Run with --help for usage."
      exit 1
      ;;
  esac
done

if [ "$JOBS" -lt 1 ]; then
  JOBS=1
fi

IREE_BAZEL_FUZZ="$REPO_ROOT/build_tools/bin/iree-bazel-fuzz"
if [ ! -x "$IREE_BAZEL_FUZZ" ]; then
  echo "ERROR: iree-bazel-fuzz not found at $IREE_BAZEL_FUZZ"
  exit 1
fi

#===----------------------------------------------------------------------===#
# Tokenizer Download
#===----------------------------------------------------------------------===#

download_tokenizers() {
  echo "Checking tokenizer cache at $CACHE_DIR ..."
  mkdir -p "$CACHE_DIR"

  local download_count=0
  for entry in "${TOKENIZERS[@]}"; do
    local name="${entry%%|*}"
    local repo="${entry#*|}"
    local dest="$CACHE_DIR/$name.json"
    if [ ! -f "$dest" ]; then
      local url="$HF_BASE/$repo/resolve/main/tokenizer.json"
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
}

#===----------------------------------------------------------------------===#
# Integration Fuzzing
#===----------------------------------------------------------------------===#

# Runs each integration target against each real tokenizer JSON.
# Returns non-zero if any fuzzer crashes.
run_integration() {
  echo "================================================================"
  echo "  INTEGRATION FUZZING"
  echo "  ${#TOKENIZERS[@]} tokenizers x ${#INTEGRATION_TARGETS[@]} targets"
  echo "  Duration: ${DURATION}s per combo, Jobs: ${JOBS}"
  echo "================================================================"
  echo ""

  local combo_index=0
  local total_combos=$(( ${#TOKENIZERS[@]} * ${#INTEGRATION_TARGETS[@]} ))

  for entry in "${TOKENIZERS[@]}"; do
    local name="${entry%%|*}"
    local json_path="$CACHE_DIR/$name.json"
    if [ ! -f "$json_path" ]; then
      echo "  SKIPPED: $name (not downloaded)"
      continue
    fi

    for target in "${INTEGRATION_TARGETS[@]}"; do
      combo_index=$((combo_index + 1))
      echo "--- [$combo_index/$total_combos] $name x $target ---"

      if ! "$IREE_BAZEL_FUZZ" \
        "//runtime/src/iree/tokenizer:${target}" \
        -- \
        "--tokenizer_json=${json_path}" \
        "--track_offsets" \
        "-max_total_time=${DURATION}" \
        "-jobs=${JOBS}"; then
        echo ""
        echo "CRASH DETECTED: $name x $target"
        echo "Tokenizer: $json_path"
        return 1
      fi
      echo ""
    done
  done

  echo "Integration fuzzing complete: $combo_index combos, no crashes."
  return 0
}

#===----------------------------------------------------------------------===#
# Component Fuzzing
#===----------------------------------------------------------------------===#

# Runs component fuzz targets using iree-bazel-fuzz's multi-target support.
# For multi-target patterns (...), iree-bazel-fuzz's exit code is unreliable
# (known issue with `local` outside functions under set -e). We check for
# crash artifacts instead of relying on exit codes.
run_component() {
  echo "================================================================"
  echo "  COMPONENT FUZZING"
  echo "  Duration: ${DURATION}s, Jobs: ${JOBS}"
  echo "================================================================"
  echo ""

  local fuzz_cache="${IREE_FUZZ_CACHE:-${HOME}/.cache/iree-fuzz-cache}"

  for pattern in "${COMPONENT_PATTERNS[@]}"; do
    echo "--- $pattern ---"

    # Snapshot artifact counts before fuzzing.
    local artifact_count_before
    artifact_count_before=$(find "$fuzz_cache" -path "*/artifacts/crash-*" -o -path "*/artifacts/leak-*" 2>/dev/null | wc -l)

    # Run fuzzer (ignore exit code for multi-target patterns).
    "$IREE_BAZEL_FUZZ" \
      "$pattern" \
      -- \
      "-max_total_time=${DURATION}" \
      "-jobs=${JOBS}" || true

    # Check for new crash/leak artifacts.
    local artifact_count_after
    artifact_count_after=$(find "$fuzz_cache" -path "*/artifacts/crash-*" -o -path "*/artifacts/leak-*" 2>/dev/null | wc -l)

    if [ "$artifact_count_after" -gt "$artifact_count_before" ]; then
      echo ""
      echo "CRASH DETECTED in: $pattern"
      echo "New artifacts:"
      find "$fuzz_cache" -path "*/artifacts/crash-*" -newer "$0" -o -path "*/artifacts/leak-*" -newer "$0" 2>/dev/null | head -10
      return 1
    fi
    echo ""
  done

  echo "Component fuzzing complete: no crashes."
  return 0
}

#===----------------------------------------------------------------------===#
# Main
#===----------------------------------------------------------------------===#

# Download tokenizers if we need them.
if [ "$RUN_INTEGRATION" -eq 1 ]; then
  download_tokenizers
fi

pass_number=0
while true; do
  pass_number=$((pass_number + 1))

  if [ "$FOREVER" -eq 1 ]; then
    echo "==============================="
    echo "  Pass $pass_number"
    echo "==============================="
    echo ""
  fi

  if [ "$RUN_INTEGRATION" -eq 1 ]; then
    if ! run_integration; then
      echo ""
      echo "Stopping due to crash (pass $pass_number)."
      exit 1
    fi
  fi

  if [ "$RUN_COMPONENT" -eq 1 ]; then
    if ! run_component; then
      echo ""
      echo "Stopping due to crash (pass $pass_number)."
      exit 1
    fi
  fi

  if [ "$FOREVER" -eq 0 ]; then
    break
  fi

  echo ""
  echo "Pass $pass_number complete, starting next pass..."
  echo ""
done

echo ""
echo "All fuzzing complete."
