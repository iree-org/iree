#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Test the cross-compiled RISCV 64-bit Linux targets.
#
# The desired build directory can be passed as the first argument. Otherwise, it
# uses the environment variable IREE_TARGET_BUILD_DIR, defaulting to
# "build-riscv". Designed for CI, but can be run manually. Expects to be run
# from the root of the IREE repository.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_TARGET_BUILD_DIR:-build-riscv}}"
RISCV_PLATFORM="${IREE_TARGET_PLATFORM:-linux}"
RISCV_ARCH="${IREE_TARGET_ARCH:-riscv_64}"
BUILD_PRESET="${BUILD_PRESET:-test}"

# Environment variable used by the emulator.
export RISCV_TOOLCHAIN_ROOT="${RISCV_RV64_LINUX_TOOLCHAIN_ROOT}"

export CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL_LEVEL:-$(nproc)}

ctest_args=(
  "--timeout 900"
  "--output-on-failure"
  "--no-tests=error"
)

declare -a label_exclude_args=(
  "^nodocker$"
  "^driver=vulkan$"
  "^driver=cuda$"
  "^vulkan_uses_vk_khr_shader_float16_int8$"
  "^requires-filesystem$"
  "^requires-dtz$"
  "^noriscv$"
)

# Excluding mobilebert, fp16, and lowering_config regression
# tests for now.
# TODO(#10462): Investigate the lowering_config test issue.
declare -a test_exclude_args=(
  "bert"
  "fp16"
  "regression_llvm-cpu_lowering_config"
)

if [[ "${BUILD_PRESET}" == "benchmark-suite-test" ]]; then
  declare -a label_args=(
    "^test-type=run-module-test$"
  )
  label_include_regex="($(IFS="|" ; echo "${label_args[*]}"))"
  echo "******** Running run-module CTest ********"
  ctest --test-dir ${BUILD_DIR} ${ctest_args[@]} \
    --label-regex ${label_include_regex}
  exit 0
fi

# Test runtime unit tests
runtime_label_exclude_regex="($(IFS="|" ; echo "${label_exclude_args[*]}"))"
runtime_ctest_args=(
  "--test-dir ${BUILD_DIR}/runtime/"
  ${ctest_args[@]}
  "--label-exclude ${runtime_label_exclude_regex}"
)
echo "******** Running runtime CTest ********"
ctest ${runtime_ctest_args[@]}

tools_ctest_args=(
  "--test-dir ${BUILD_DIR}/tools/test"
  ${ctest_args[@]}
  "--label-exclude ${runtime_label_exclude_regex}"
)
echo "******** Running tools CTest ********"
ctest ${tools_ctest_args[@]}

if [[ "${RISCV_PLATFORM}-${RISCV_ARCH}" == "linux-riscv_32" ]]; then
  # stablehlo.power is also disabled because musl math library is not compiled for
  # 32-bit.
  test_exclude_args+=(
    "stablehlo.*llvm-cpu.*pow"
  )
fi

test_exclude_args+=(
  # TODO(#12703): Enable the tests.
  "iree/tests/e2e/matmul/e2e_matmul_mmt4d_i8_intrinsics_small_llvm-cpu_local-task"
  "iree/tests/e2e/matmul/e2e_matmul_mmt4d_i8_small_llvm-cpu_local-task"
  "iree/tests/e2e/tensor_ops/check_llvm-cpu_local-task_pack.mlir"
  "iree/tests/e2e/tensor_ops/check_llvm-cpu_local-task_pack_dynamic_inner_tiles.mlir"
  # TODO(#13421): Enable the tests
  "iree/tests/e2e/stablehlo_ops/check_llvm-cpu_local-task_dot.mlir"
  "iree/tests/e2e/matmul/e2e_matmul_direct_i8_small_llvm-cpu_local-task"
  "iree/tests/e2e/matmul/e2e_matmul_direct_f32_small_llvm-cpu_local-task"
  "iree/tests/e2e/regression/check_regression_llvm-cpu_strided_slice.mlir"
)

tests_label_exclude_regex="($(IFS="|" ; echo "${label_exclude_args[*]}"))"
tests_exclude_regex="($(IFS="|" ; echo "${test_exclude_args[*]}"))"
test_ctest_args=(
  "--test-dir ${BUILD_DIR}/tests/e2e"
  ${ctest_args[@]}
  "--label-exclude ${tests_label_exclude_regex}"
  "--exclude-regex ${tests_exclude_regex}"
)
echo "******** Running e2e CTest ********"
ctest  ${test_ctest_args[@]}
