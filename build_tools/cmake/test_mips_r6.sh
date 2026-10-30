#!/bin/bash

# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Test the cross-compiled MIPS64 Release 6 + MSA (native llvm-cpu codegen)
# targets, under QEMU -cpu I6400. Counterpart to test_mips.sh (scalar r2 / VMVX).
#
# The build directory can be passed as the first argument; otherwise it uses
# IREE_TARGET_BUILD_DIR, defaulting to "build-mips-r6". Expects a tree built by
# build_mips_r6.sh. Run from the root of the IREE repository.

# Note: no `set -x` here (it floods the log with every assignment and the full
# ctest command line). The ******** echoes below mark progress, and ctest's
# --output-on-failure surfaces details for any failing test.
set -euo pipefail

BUILD_DIR="${1:-${IREE_TARGET_BUILD_DIR:-build-mips-r6}}"

# Auto-detect the little-endian QEMU MIPS emulator.
if [[ -z "${QEMU_BIN:-}" ]]; then
  QEMU_BIN="$(command -v qemu-mips64el || command -v qemu-mips64el-static || true)"
fi
export QEMU_BIN
if [[ -z "${QEMU_BIN}" ]]; then
  echo "warning: no qemu-mips64el found on PATH; tests will run the target"
  echo "         binaries directly (only valid on a real MIPS r6 host)."
fi

# Sysroot for QEMU's dynamic loader. The r6 toolchain's per-multilib sysroot
# (mipsel-r6-hard) carries libc and, after build_mips_r6.sh, the C++/gcc runtime
# libs. The -L baked into each test command is not sufficient on its own, so
# QEMU_LD_PREFIX must also be set. Default it from MIPS_TOOLCHAIN_ROOT; set it
# yourself for a custom layout.
if [[ -z "${QEMU_LD_PREFIX:-}" ]]; then
  MIPS_R6_SYSROOT_VARIANT="${MIPS_R6_SYSROOT_VARIANT:-mipsel-r6-hard}"
  if [[ -n "${MIPS_TOOLCHAIN_ROOT:-}" ]]; then
    QEMU_LD_PREFIX="${MIPS_TOOLCHAIN_ROOT}/sysroot/${MIPS_R6_SYSROOT_VARIANT}"
  else
    echo "error: set QEMU_LD_PREFIX (or MIPS_TOOLCHAIN_ROOT) to the r6 sysroot," >&2
    echo "       e.g. <toolchain>/sysroot/mipsel-r6-hard" >&2
    exit 1
  fi
fi
export QEMU_LD_PREFIX

# QEMU user-mode emulation is CPU-bound, so running one emulated test per host
# core over-subscribes the machine and makes the heavy (bf16) data-tiled
# matmuls miss their baked 60s per-test timeout even though they pass with
# headroom. Cap the parallelism for a reliable gate; override as needed.
export CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL_LEVEL:-4}

ctest_args=(
  "--timeout 900"
  "--output-on-failure"
  "--no-tests=error"
)

# NOTE: unlike the scalar r2 gate, `nomips` is NOT excluded here: on r6/MSA the
# data-tiling / winograd suites compile and run (IREE_MIPS_ENABLE_MSA lifts the
# build-time exclusion), so they are part of this gate.
declare -a label_exclude_args=(
  "^nodocker$"
  "^driver=vulkan$"
  "^driver=metal$"
  "^driver=cuda$"
  "^driver=hip$"
  "^vulkan_uses_vk_khr_shader_float16_int8$"
  "^requires-filesystem$"
  "^requires-dtz$"
)

# e2e tests skipped on the MIPS r6/MSA gate. These are known-broken on mips64el
# codegen for reasons orthogonal to MSA; skipped (not labelled) so they
# re-enable automatically once fixed. NOTE: this list is finalized empirically
# by re-probing on r6 (some r2 skips, e.g. bf16/i8 data-tiling SIGBUS, may be
# fixed by MSA) - trim to the confirmed-failing set.
declare -a test_exclude_args=(
  "regression_llvm-cpu_lowering_config"

  # f16 is broadly supported on r6/MSA: arith, conversions, and exp all pass
  # (e.g. exponential_fp16), and the f16 data-tiled matmuls are numerically
  # correct. bf16 and i8 data-tiled matmul, which SIGBUS'd on scalar r2, also
  # now pass under MSA and are intentionally NOT skipped. The following are
  # skipped for two distinct reasons (tracked in <ISSUE>):
  #
  # LLVM MIPS backend gaps (genuine):
  #  - inner-tiled data-tiling with an f16 accumulator crashes instruction
  #    selection ("Do not know how to soft promote this operator's operand" on
  #    a v8f16 scalar_to_vector); the f32-accumulator variant is fine.
  #  - erf/cbrt are miscompiled under MSA (no vector libm, so they stay
  #    polynomial-approximated and the MSA approximation is wrong) -> math_ops.
  "e2e_matmul_cpu_dt_inner_tiled_f16_f16"
  "math_ops_llvm-cpu"
  #
  # Emulation speed, NOT a codegen bug: the default-shape f16 data-tiled
  # matmuls are numerically correct (verified in isolation) but take ~46s each
  # under qemu soft-float emulation, so they exceed the baked 60s per-test
  # timeout under parallel gate load. Skipped on the QEMU gate; they would run
  # fine on hardware.
  "e2e_matmul_cpu_dt_f16_f16"
  "e2e_matmul_cpu_dt_uk_f16_f16"
  "e2e_matmul_cpu_experimental_dt_f16_f16"
  "e2e_matmul_cpu_experimental_dt_uk_f16_f16"
)

# Known MIPS-alignment failures under qemu (SIGBUS in binary/data parsers that
# QEMU user-mode does not fix up); off the inference path. Re-probe on r6.
declare -a runtime_test_exclude_args=(
  "iree/hal/local/elf/elf_module_test"
  "iree/hal/local/profile_test"
  "iree/hal/replay/execute_test"
  "iree/tooling/profile/cli_test"
  "iree/io/formats/gguf/gguf_parser_test"
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

runtime_label_exclude_regex="($(IFS="|" ; echo "${label_exclude_args[*]}"))"
runtime_test_exclude_regex="($(IFS="|" ; echo "${runtime_test_exclude_args[*]}"))"
runtime_ctest_args=(
  "--test-dir ${BUILD_DIR}/runtime/"
  ${ctest_args[@]}
  "--label-exclude ${runtime_label_exclude_regex}"
  "--exclude-regex ${runtime_test_exclude_regex}"
)

tests_label_exclude_regex="($(IFS="|" ; echo "${label_exclude_args[*]}"))"
tests_exclude_regex="($(IFS="|" ; echo "${test_exclude_args[*]}"))"
test_ctest_args=(
  "--test-dir ${BUILD_DIR}/tests/e2e"
  ${ctest_args[@]}
  "--label-exclude ${tests_label_exclude_regex}"
  "--exclude-regex ${tests_exclude_regex}"
)

# Run both suites regardless of each other's result, then fail if either did,
# so a runtime failure does not hide the e2e (codegen) result.
rc=0
echo "******** Running runtime CTest ********"
{ set +e; } 2>/dev/null
ctest ${runtime_ctest_args[@]} || rc=1
echo "******** Running e2e CTest ********"
ctest ${test_ctest_args[@]} || rc=1
{ set -e; } 2>/dev/null
exit ${rc}
