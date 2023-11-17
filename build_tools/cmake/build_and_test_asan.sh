#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build and test, using CMake/CTest, with AddressSanitizer instrumentation.
#
# See https://clang.llvm.org/docs/AddressSanitizer.html. Components that don't
# work with ASan (e.g. Python bindings) are disabled.
#
# The desired build directory can be passed as the first argument. Otherwise, it
# uses the environment variable IREE_ASAN_BUILD_DIR, defaulting to "build-asan".
# Designed for CI, but can be run manually. It reuses the build directory if it
# already exists. Expects to be run from the root of the IREE repository.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_ASAN_BUILD_DIR:-build-asan}}"
IREE_ENABLE_ASSERTIONS="${IREE_ENABLE_ASSERTIONS:-ON}"

source build_tools/cmake/setup_build.sh
source build_tools/cmake/setup_ccache.sh

# We run everything twice, for each boolean value of `asan_in_bytecode_modules_ON_OFF`:
#  - When asan_in_bytecode_modules_ON_OFF=OFF, ASAN is only enabled in the C/C++ code
#    (compiler and runtime), not in bytecode modules. The default embedded-ELF
#    path is used.
#  - When asan_in_bytecode_modules_ON_OFF=ON, ASAN is also enabled in bytecode
#    modules. The system-ELF path is used (required for ASAN-in-modules).
for asan_in_bytecode_modules_ON_OFF in OFF ON; do
    # It's enough to do that in either of the two loop iterations.
    build_microbenchmarks_ON_OFF=${asan_in_bytecode_modules_ON_OFF}

  CMAKE_ARGS=(
    "-G" "Ninja"
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
    "-DIREE_ENABLE_ASAN=ON"
    "-B" "${BUILD_DIR?}"
    "-DPython3_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"
    "-DPYTHON_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"
    "-DIREE_ENABLE_ASSERTIONS=${IREE_ENABLE_ASSERTIONS}"

    # The main difference between the two loop iterations: conditionally enable
    # ASAN in bytecode modules and use the system-ELF path.
    "-DIREE_BYTECODE_MODULE_ENABLE_ASAN=${asan_in_bytecode_modules_ON_OFF}"
    "-DIREE_BYTECODE_MODULE_FORCE_LLVM_SYSTEM_LINKER=${asan_in_bytecode_modules_ON_OFF}"

    # Also check if microbenchmarks are buildable. It's enough to do that in one
    # of the two loop iterations.
    "-DIREE_BUILD_MICROBENCHMARKS=${build_microbenchmarks_ON_OFF}"
  )

  echo "*** Configuring CMake (IREE_BYTECODE_MODULE_ENABLE_ASAN=${asan_in_bytecode_modules_ON_OFF}) ***"
  echo "------------------"
  "${CMAKE_BIN?}" "${CMAKE_ARGS[@]?}"

  echo "*** Building all (IREE_BYTECODE_MODULE_ENABLE_ASAN=${asan_in_bytecode_modules_ON_OFF}) ***"
  echo "------------------"
  "${CMAKE_BIN?}" --build "${BUILD_DIR?}" -- -k 0

  echo "*** Building test deps (IREE_BYTECODE_MODULE_ENABLE_ASAN=${asan_in_bytecode_modules_ON_OFF}) ***"
  echo "------------------"
  "${CMAKE_BIN?}" --build "${BUILD_DIR?}" --target iree-test-deps -- -k 0

  if [[ "${build_microbenchmarks_ON_OFF}" == "ON" ]]; then
    echo "*** Building microbenchmark suites (IREE_BYTECODE_MODULE_ENABLE_ASAN=${asan_in_bytecode_modules_ON_OFF}) ***"
    echo "------------------"
    "${CMAKE_BIN?}" --build "${BUILD_DIR?}" --target iree-microbenchmark-suites -- -k 0
  fi

  if (( IREE_USE_CCACHE == 1 )); then
    ccache --show-stats
  fi

  # Respect the user setting, but default to as many jobs as we have cores.
  export CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL_LEVEL:-$(nproc)}

  # Respect the user setting, but default to turning on Vulkan.
  export IREE_VULKAN_DISABLE=${IREE_VULKAN_DISABLE:-0}
  # Respect the user setting, but default to turning off Metal.
  export IREE_METAL_DISABLE="${IREE_METAL_DISABLE:-1}"
  # Respect the user setting, but default to turning off CUDA.
  export IREE_CUDA_DISABLE=${IREE_CUDA_DISABLE:-1}
  # The VK_KHR_shader_float16_int8 extension is optional prior to Vulkan 1.2.
  # We test on SwiftShader as a baseline, which does not support this extension.
  export IREE_VULKAN_F16_DISABLE=${IREE_VULKAN_F16_DISABLE:-1}
  # Respect the user setting, but default to skipping tests that require Nvidia GPU.
  export IREE_NVIDIA_GPU_TESTS_DISABLE=${IREE_NVIDIA_GPU_TESTS_DISABLE:-1}

  # Tests to exclude by label. In addition to any custom labels (which are carried
  # over from Bazel tags), every test should be labeled with its directory.
  declare -a label_exclude_args=(
    # Exclude specific labels.
    # Put the whole label with anchors for exact matches.
    ^noasan$
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

  # IREE_VULKAN_DISABLE is handled separately as we run Vulkan and non-Vulkan
  # tests in separate ctest commands anyway.
  if (( IREE_CUDA_DISABLE == 1 )); then
    label_exclude_args+=("^driver=cuda$")
  fi
  if (( IREE_VULKAN_F16_DISABLE == 1 )); then
    label_exclude_args+=("^vulkan_uses_vk_khr_shader_float16_int8$")
  fi
  if (( IREE_NVIDIA_GPU_TESTS_DISABLE == 1 )); then
    label_exclude_args+=("^requires-gpu")
  fi

  label_exclude_regex="($(IFS="|" ; echo "${label_exclude_args[*]?}"))"

  vulkan_label_regex='^driver=vulkan$'

  pushd ${BUILD_DIR?}

  echo "*** Running main project ctests that do not use the Vulkan driver (IREE_BYTECODE_MODULE_ENABLE_ASAN=${asan_in_bytecode_modules_ON_OFF}) *******"
  echo "------------------"
  ASAN_OPTIONS=check_initialization_order=true \
    ctest \
      --timeout 900 \
      --output-on-failure \
      --no-tests=error \
      --label-exclude "${label_exclude_regex}|${vulkan_label_regex}"

  echo "*** Running llvm-external-projects tests (IREE_BYTECODE_MODULE_ENABLE_ASAN=${asan_in_bytecode_modules_ON_OFF}) ***"
  echo "------------------"
  cmake --build . --target check-iree-dialects -- -k 0

  if (( IREE_VULKAN_DISABLE == 0 )); then
    echo "*** Running ctests that use the Vulkan driver, with LSAN disabled (IREE_BYTECODE_MODULE_ENABLE_ASAN=${asan_in_bytecode_modules_ON_OFF}) ***"
    echo "------------------"
    # Disable LeakSanitizer (LSAN) because of a history of issues with Swiftshader
    # (#5716, #8489, #11203).
    ASAN_OPTIONS=detect_leaks=0:check_initialization_order=true \
      ctest \
        --timeout 900 \
        --output-on-failure \
        --no-tests=error \
        --label-regex "${vulkan_label_regex}" \
        --label-exclude "${label_exclude_regex}"
  fi

  popd
done
