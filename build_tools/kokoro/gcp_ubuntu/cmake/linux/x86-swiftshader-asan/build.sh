#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build the project with cmake using Kokoro.

set -e
set -x

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

# Check these exist and print the versions for later debugging
export CMAKE_BIN="$(which cmake)"
"${CMAKE_BIN?}" --version
"${CC?}" --version
"${CXX?}" --version
python3 --version

echo "Initializing submodules"
git submodule update --init --jobs 8 --depth 1

./build_tools/kokoro/gcp_ubuntu/check_vulkan.sh

# Print SwiftShader git commit
cat /swiftshader/git-commit

CMAKE_BUILD_DIR="${CMAKE_BUILD_DIR:-$HOME/build}"

CMAKE_ARGS=(
  "-G" "Ninja"
  "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
  "-DIREE_ENABLE_ASAN=ON"
  "-B" "${CMAKE_BUILD_DIR?}"
)

echo "Configuring CMake"
"${CMAKE_BIN?}" "${CMAKE_ARGS[@]?}"

echo "Building all"
echo "------------"
"${CMAKE_BIN?}" --build "${CMAKE_BUILD_DIR?}" -- -k 0

echo "Building test deps"
echo "------------------"
"${CMAKE_BIN?}" --build "${CMAKE_BUILD_DIR?}" --target iree-test-deps -- -k 0

# Respect the user setting, but default to as many jobs as we have cores.
export CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL_LEVEL:-$(nproc)}

# Respect the user setting, but default to turning off the vulkan tests.
# TODO(#5716): Fix and enable Vulkan tests.
export IREE_VULKAN_DISABLE=${IREE_VULKAN_DISABLE:-1}
# CUDA is off by default.
export IREE_CUDA_DISABLE=${IREE_CUDA_DISABLE:-1}
# The VK_KHR_shader_float16_int8 extension is optional prior to Vulkan 1.2.
# We test on SwiftShader, which does not support this extension.
export IREE_VULKAN_F16_DISABLE=${IREE_VULKAN_F16_DISABLE:-1}

# Tests to exclude by label. In addition to any custom labels (which are carried
# over from Bazel tags), every test should be labeled with the directory it is
# in.
declare -a label_exclude_args=(
  # Exclude specific labels.
  # Put the whole label with anchors for exact matches.
  # For example:
  #   ^nokokoro$
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

if [[ "${IREE_VULKAN_DISABLE?}" == 1 ]]; then
  label_exclude_args+=("^driver=vulkan$")
fi
if [[ "${IREE_CUDA_DISABLE?}" == 1 ]]; then
  label_exclude_args+=("^driver=cuda$")
  label_exclude_args+=("^uses_cuda_runtime$")
fi
if [[ "${IREE_VULKAN_F16_DISABLE?}" == 1 ]]; then
  label_exclude_args+=("^vulkan_uses_vk_khr_shader_float16_int8$")
fi

# Join on "|"
label_exclude_regex="($(IFS="|" ; echo "${label_exclude_args[*]?}"))"

# These tests currently have asan failures
# TODO(#5715): Fix these
declare -a excluded_tests=(
  "iree/samples/simple_embedding/simple_embedding_vulkan_test"
)

# Prefix with `^` anchor
excluded_tests=( "${excluded_tests[@]/#/^}" )
# Suffix with `$` anchor
excluded_tests=( "${excluded_tests[@]/%/$}" )

# Join on `|` and wrap in parens
excluded_tests_regex="($(IFS="|" ; echo "${excluded_tests[*]?}"))"

cd ${CMAKE_BUILD_DIR?}

echo "Testing with ctest"
ctest --timeout 900 --output-on-failure \
  --label-exclude "${label_exclude_regex}" \
  --exclude-regex "${excluded_tests_regex?}"
