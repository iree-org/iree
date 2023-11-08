#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build and test, using CMake/CTest, with ThreadSanitizer instrumentation.
#
# See https://clang.llvm.org/docs/ThreadSanitizer.html. Some tests are run many
# times to flush out non-determinstic failures.
#
# The desired build directory can be passed as the first argument. Otherwise, it
# uses the environment variable IREE_TSAN_BUILD_DIR, defaulting to "build-tsan".
# Designed for CI, but can be run manually. It reuses the build directory if it
# already exists. Expects to be run from the root of the IREE repository.

set -euo pipefail

BUILD_DIR="${1:-${IREE_TSAN_BUILD_DIR:-build-tsan}}"

source build_tools/cmake/setup_build.sh
source build_tools/cmake/setup_ccache.sh

CMAKE_ARGS=(
  "-G" "Ninja"
  "-DPython3_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"
  "-DPYTHON_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"

  # The debug information will help get more helpful TSan reports (stacks).
  "-DCMAKE_BUILD_TYPE=RelWithDebInfo"

  # Let's make linking fast. Also, there's the off chance that TSan might be
  # better supported with LLD.
  "-DIREE_ENABLE_LLD=ON"

  # Enable TSan in all C/C++ targets, including IREE runtime, compiler, tests.
  "-DIREE_ENABLE_TSAN=ON"

  # Enable TSan in iree_bytecode_module's. Otherwise, our TSan-enabled IREE
  # runtime won't be able to call into these modules.
  "-DIREE_BYTECODE_MODULE_ENABLE_TSAN=ON"

  # Link iree_bytecode_module's with system linker, not embedded-ELF linker.
  # Necessary with TSan.
  "-DIREE_BYTECODE_MODULE_FORCE_LLVM_SYSTEM_LINKER=ON"

  # Don't build samples: they assume embedded-ELF so don't work with
  # IREE_BYTECODE_MODULE_FORCE_LLVM_SYSTEM_LINKER=ON.
  "-DIREE_BUILD_SAMPLES=OFF"

  # Building the compiler with TSan hangs on arm64 builders.
  # Anyway, the issue we're looking to reproduce in #15488 is in the runtime.
  "-DIREE_BUILD_COMPILER=OFF"
)

"${CMAKE_BIN}" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]?}"

echo "Building all"
echo "------------"
"$CMAKE_BIN" --build "${BUILD_DIR}" -- -k 0

# We're disabled the compiler above.
# echo "Building test deps"
# echo "------------------"
# "$CMAKE_BIN" --build "${BUILD_DIR}" --target iree-test-deps -- -k 0

if (( IREE_USE_CCACHE == 1 )); then
  ccache --show-stats
fi

# Disable actually running GPU tests. This tends to yield TSan reports that are
# specific to one's particular GPU driver and therefore hard to reproduce across
# machines and often un-actionable anyway.
# See e.g. https://github.com/openxla/iree/issues/9393
export IREE_VULKAN_DISABLE=1
export IREE_METAL_DISABLE=1
export IREE_CUDA_DISABLE=1

# Honor the "notsan" label on tests.
export IREE_EXTRA_COMMA_SEPARATED_CTEST_LABELS_TO_EXCLUDE=notsan

# Run all tests, once
build_tools/cmake/ctest_all.sh "${BUILD_DIR}"

# Re-run many times certain tests that are cheap and prone to nondeterministic
# failure (typically, IREE runtime tests exercising multi-threading features).
export IREE_CTEST_TESTS_REGEX="(^iree/base/|^iree/task/)"
export IREE_CTEST_REPEAT_UNTIL_FAIL_COUNT=32
build_tools/cmake/ctest_all.sh "${BUILD_DIR}"
