#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Run all ctest tests in a build directory, adding some automatic test filtering
# based on environment variables and defaulting to running as many test jobs as
# there are processes. The build directory to test is passed as the first
# argument. Designed for CI, but can be run manually.

set -euo pipefail

BUILD_DIR="$1"

get_default_parallel_level() {
  if [[ "$(uname)" == "Darwin" ]]; then
    echo "$(sysctl -n hw.logicalcpu)"
  else
    echo "$(nproc)"
  fi
}

# Respect the user setting, but default to as many jobs as we have cores.
export CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL_LEVEL:-$(get_default_parallel_level)}

# Respect the user setting, but default to turning on Vulkan.
export IREE_VULKAN_DISABLE=${IREE_VULKAN_DISABLE:-0}
# Respect the user setting, but default to turning off CUDA.
export IREE_CUDA_DISABLE=${IREE_CUDA_DISABLE:-1}
# The VK_KHR_shader_float16_int8 extension is optional prior to Vulkan 1.2.
# We test on SwiftShader as a baseline, which does not support this extension.
export IREE_VULKAN_F16_DISABLE=${IREE_VULKAN_F16_DISABLE:-1}
# Respect the user setting, but default to skipping tests that require Nvidia GPU.
export IREE_NVIDIA_GPU_TESTS_DISABLE=${IREE_NVIDIA_GPU_TESTS_DISABLE:-1}
# Respect the user setting, default to no --repeat-until-fail.
export IREE_CTEST_REPEAT_UNTIL_FAIL_COUNT=${IREE_CTEST_REPEAT_UNTIL_FAIL_COUNT:-}
# Respect the user setting, default to no --tests-regex.
export IREE_CTEST_TESTS_REGEX=${IREE_CTEST_TESTS_REGEX:-}

# Tests to exclude by label. In addition to any custom labels (which are carried
# over from Bazel tags), every test should be labeled with its directory.
declare -a label_exclude_args=(
  # Exclude specific labels.
  # Put the whole label with anchors for exact matches.
  # For example:
  #   ^nokokoro$
  # TODO: update label name as part of dropping Kokoro
  ^nokokoro$

  # Exclude all tests in a directory.
  # Put the whole directory with anchors for exact matches.
  # For example:
  #   ^bindings/python/iree/runtime$

  # Exclude all tests in some subdirectories.
  # Put the whole parent directory with only a starting anchor.
  # Use a trailing slash to avoid prefix collisions.
  # For example:
  #   ^bindings/
)

if [[ "${IREE_VULKAN_DISABLE}" == 1 ]]; then
  label_exclude_args+=("^driver=vulkan$")
fi
if [[ "${IREE_CUDA_DISABLE}" == 1 ]]; then
  label_exclude_args+=("^driver=cuda$")
fi
if [[ "${IREE_VULKAN_F16_DISABLE}" == 1 ]]; then
  label_exclude_args+=("^vulkan_uses_vk_khr_shader_float16_int8$")
fi
if [[ "${IREE_NVIDIA_GPU_TESTS_DISABLE}" == 1 ]]; then
  label_exclude_args+=("^requires-gpu-nvidia$")
fi

IFS=',' read -ra extra_label_exclude_args <<< "${IREE_EXTRA_COMMA_SEPARATED_CTEST_LABELS_TO_EXCLUDE:-}"
label_exclude_args+=(${extra_label_exclude_args[@]})

# Some tests are just failing on some platforms and this filtering lets us
# exclude any type of test. Ideally each test would be tagged with the
# platforms it doesn't support, but that would require editing through layers
# of CMake functions. Hopefully this list stays very short.
declare -a excluded_tests=()
if [[ "$OSTYPE" =~ ^msys ]]; then
  # These tests are failing on Windows.
  excluded_tests+=(
    # TODO(#11077): INVALID_ARGUMENT: argument/result signature mismatch
    "iree/tests/e2e/matmul/e2e_matmul_direct_i8_small_ukernel_vmvx_local-task"
    "iree/tests/e2e/matmul/e2e_matmul_direct_f32_small_ukernel_vmvx_local-task"
    "iree/tests/e2e/matmul/e2e_matmul_mmt4d_i8_small_ukernel_vmvx_local-task"
    "iree/tests/e2e/matmul/e2e_matmul_mmt4d_f32_small_ukernel_vmvx_local-task"
    # TODO: Regressed when `pack` ukernel gained a uint64_t parameter in #13264.
    "iree/tests/e2e/tensor_ops/check_vmvx_ukernel_local-task_pack.mlir"
    "iree/tests/e2e/tensor_ops/check_vmvx_ukernel_local-task_pack_dynamic_inner_tiles.mlir"
    # TODO: Fix equality mismatch
    "iree/tests/e2e/tensor_ops/check_vmvx_ukernel_local-task_unpack.mlir"
    # TODO(#11070): Fix argument/result signature mismatch
    "iree/tests/e2e/tosa_ops/check_vmvx_local-sync_microkernels_fully_connected.mlir"
  )
elif [[ "$OSTYPE" =~ ^darwin ]]; then
  excluded_tests+=(
    #TODO(#12496): Remove after fixing the test on macOS
    "iree/compiler/bindings/c/loader_test"
    #TODO(#12496): Remove after fixing the test on macOS
    "iree/compiler/bindings/python/test/transforms/ireec/compile_sample_module"
  )
fi

# TODO(#12305): figure out how to run samples with custom binary outputs
# on the CI. $IREE_BINARY_DIR may not be setup right or the object files may
# not be getting deployed to the test_all/test_gpu bots.
excluded_tests+=(
  "iree/samples/custom_dispatch/cpu/embedded/example_hal.mlir.test"
  "iree/samples/custom_dispatch/cpu/embedded/example_stream.mlir.test"
)

ctest_args=(
  "--test-dir ${BUILD_DIR}"
  "--timeout 900"
  "--output-on-failure"
  "--no-tests=error"
)

if [[ -n "${IREE_CTEST_TESTS_REGEX}" ]]; then
  ctest_args+=("--tests-regex ${IREE_CTEST_TESTS_REGEX}")
fi

if [[ -n "${IREE_CTEST_REPEAT_UNTIL_FAIL_COUNT}" ]]; then
  ctest_args+=("--repeat-until-fail ${IREE_CTEST_REPEAT_UNTIL_FAIL_COUNT}")
fi

if (( ${#label_exclude_args[@]} )); then
  # Join on "|"
  label_exclude_regex="($(IFS="|" ; echo "${label_exclude_args[*]}"))"
  ctest_args+=("--label-exclude ${label_exclude_regex}")
fi

if (( ${#excluded_tests[@]} )); then
  # Prefix with `^` anchor
  excluded_tests=( "${excluded_tests[@]/#/^}" )
  # Suffix with `$` anchor
  excluded_tests=( "${excluded_tests[@]/%/$}" )
  # Join on `|` and wrap in parens
  excluded_tests_regex="($(IFS="|" ; echo "${excluded_tests[*]?}"))"
  ctest_args+=("--exclude-regex ${excluded_tests_regex}")
fi

echo "*************** Running CTest ***************"

set -x
ctest ${ctest_args[@]}
