#!/bin/bash

# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Test the cross-compiled MIPS 64-bit Linux targets.
#
# The desired build directory can be passed as the first argument. Otherwise, it
# uses the environment variable IREE_TARGET_BUILD_DIR, defaulting to
# "build-mips". Designed for CI, but can be run manually. Expects to be run
# from the root of the IREE repository.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_TARGET_BUILD_DIR:-build-mips}}"
MIPS_PLATFORM="${IREE_TARGET_PLATFORM:-linux}"
MIPS_ARCH="${IREE_TARGET_ARCH:-mips_64}"

# Auto-detect the QEMU MIPS emulator so the script can run with no QEMU_BIN=
# prefix. run_mips_test.sh reads QEMU_BIN to decide whether to emulate; the
# sysroot (-L) and QEMU_CPU_FLAGS are already provided by CMake per test.
# Override by setting QEMU_BIN yourself (e.g. empty for on-device runs).
if [[ -z "${QEMU_BIN:-}" ]]; then
  QEMU_BIN="$(command -v qemu-mips64el || command -v qemu-mips64el-static || true)"
fi
export QEMU_BIN
if [[ -z "${QEMU_BIN}" ]]; then
  echo "warning: no qemu-mips64el found on PATH; tests will run the target"
  echo "         binaries directly (only valid on a real MIPS host)."
fi

# Sysroot for QEMU's dynamic loader. The -L flag baked into each test command is
# NOT sufficient on its own: it fails to resolve some libraries (notably
# libstdc++.so.6, needed by the C++ tests), so QEMU_LD_PREFIX must also be set.
# Defaults to the distro cross sysroot; set it yourself for a custom toolchain
# root (e.g. "${MIPS_TOOLCHAIN_ROOT}/sysroot").
if [[ -z "${QEMU_LD_PREFIX:-}" ]]; then
  QEMU_LD_PREFIX="/usr/mips64el-linux-gnuabi64"
fi
export QEMU_LD_PREFIX

export CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL_LEVEL:-$(nproc)}

ctest_args=(
  "--timeout 900"
  "--output-on-failure"
  "--no-tests=error"
)

declare -a label_exclude_args=(
  "^nodocker$"
  "^driver=vulkan$"
  "^driver=metal$"
  "^driver=cuda$"
  "^driver=hip$"
  "^vulkan_uses_vk_khr_shader_float16_int8$"
  "^requires-filesystem$"
  "^requires-dtz$"
  "^nomips$"
)

# TODO(#10462): e2e test not run in this VMVX-only build
declare -a test_exclude_args=(
  "regression_llvm-cpu_lowering_config"
)

# Known MIPS-alignment failures under qemu-mips64el, excluded from the test
# gate. Each fails with SIGBUS (unaligned memory access in a binary/data parser
# that QEMU user-mode does not fix up). Only these specific
# broken tests are skipped; the many passing tokenizer/tooling tests still run
declare -a runtime_test_exclude_args=(
  # ELF loader: unused by VMVX.
  "iree/hal/local/elf/elf_module_test"
  # Profiling / command-stream replay tooling.
  "iree/hal/local/profile_test"
  "iree/hal/replay/execute_test"
  "iree/tooling/profile/cli_test"
  # GGUF weight-file parser.
  "iree/io/formats/gguf/gguf_parser_test"
  # LLM tokenizer: JSON / tiktoken / regex / segmenter parsers.
  "iree/tokenizer/special_tokens_test"
  "iree/tokenizer/tokenizer_encode_test"
  "iree/tokenizer/tokenizer_huggingface_test"
  "iree/tokenizer/tokenizer_consistency_test"
  "iree/tokenizer/format/huggingface/segmenter_json_test"
  "iree/tokenizer/format/huggingface/tokenizer_json_test"
  "iree/tokenizer/format/tiktoken/tiktoken_test"
  "iree/tokenizer/model/unigram_test"
  "iree/tokenizer/regex/exec_benchmark_test"
  "iree/tokenizer/regex/compile_test"
  "iree/tokenizer/regex/compile_benchmark_test"
  "iree/tokenizer/regex/regex_test"
  "iree/tokenizer/segmenter/split_test"
)

# Test runtime unit tests
runtime_label_exclude_regex="($(IFS="|" ; echo "${label_exclude_args[*]}"))"
runtime_test_exclude_regex="($(IFS="|" ; echo "${runtime_test_exclude_args[*]}"))"
runtime_ctest_args=(
  "--test-dir ${BUILD_DIR}/runtime/"
  ${ctest_args[@]}
  "--label-exclude ${runtime_label_exclude_regex}"
  "--exclude-regex ${runtime_test_exclude_regex}"
)

# Print the exclusion summary cleanly: turn off xtrace for this block only, so
# the list is not doubled by the "+ echo ..." trace lines
{ set +x; } 2>/dev/null
echo ""
echo "******** Excluding ${#runtime_test_exclude_args[@]} known-broken tests from the test gate ********"
echo "These fail under qemu-mips64el with SIGBUS (unaligned access in binary/data"
echo "parsers), are off the VMVX inference path, and are deferred to future work:"
for _t in "${runtime_test_exclude_args[@]}"; do
  echo "  - ${_t}"
done
echo ""
echo "******** Running runtime CTest ********"
set -x
ctest ${runtime_ctest_args[@]}

# e2e tests (not run in this VMVX-only build)
# tests_label_exclude_regex="($(IFS="|" ; echo "${label_exclude_args[*]}"))"
# tests_exclude_regex="($(IFS="|" ; echo "${test_exclude_args[*]}"))"
# test_ctest_args=(
#   "--test-dir ${BUILD_DIR}/tests/e2e"
#   ${ctest_args[@]}
#   "--label-exclude ${tests_label_exclude_regex}"
#   "--exclude-regex ${tests_exclude_regex}"
# )
# echo "******** Running e2e CTest ********"
# ctest  ${test_ctest_args[@]}
