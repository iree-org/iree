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

# Respect the user setting, but default to as many jobs as we have cores.
export CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL_LEVEL:-$(nproc)}

# Respect the user setting, but default to turning on Vulkan.
export IREE_VULKAN_DISABLE=${IREE_VULKAN_DISABLE:-0}
# Respect the user setting, but default to turning off CUDA.
export IREE_CUDA_DISABLE=${IREE_CUDA_DISABLE:-1}
# The VK_KHR_shader_float16_int8 extension is optional prior to Vulkan 1.2.
# We test on SwiftShader as a baseline, which does not support this extension.
export IREE_VULKAN_F16_DISABLE=${IREE_VULKAN_F16_DISABLE:-1}

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

# Join on "|"
label_exclude_regex="($(IFS="|" ; echo "${label_exclude_args[*]}"))"

echo "******************** Running main project ctests ************************"
ctest \
  --test-dir "${BUILD_DIR}" \
  --timeout 900 \
  --output-on-failure \
  --no-tests=error \
  --label-exclude "${label_exclude_regex}"
