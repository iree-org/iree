#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build and test, using CMake/CTest, with ThreadSanitizer instrumentation.
#
# The desired build directory can be passed as
# the first argument. Otherwise, it uses the environment variable
# IREE_TSAN_BUILD_DIR, defaulting to "build-tsan". Designed for CI, but
# can be run manually. This reuses the build directory if it already exists.

set -euo pipefail

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";

BUILD_DIR="${1:-${IREE_TSAN_BUILD_DIR:-build-tsan}}"

source "${SCRIPT_DIR}/setup_build.sh"

CMAKE_ARGS=(
  "-G" "Ninja"

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

  # Enable CUDA compiler and runtime builds unconditionally. Our CI images all
  # have enough deps to at least build CUDA support and compile CUDA binaries.
  # We will skip running GPU tests below but this still yields a bit more TSan
  # coverage, at least in the compiler, and regarding the runtime it's at least
  # checking that it builds with TSan.
  "-DIREE_HAL_DRIVER_CUDA=ON"
  "-DIREE_TARGET_BACKEND_CUDA=ON"
)

"${CMAKE_BIN}" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]?}"

echo "Building all"
echo "------------"
"$CMAKE_BIN" --build "${BUILD_DIR}" -- -k 0

echo "Building test deps"
echo "------------------"
"$CMAKE_BIN" --build "${BUILD_DIR}" --target iree-test-deps -- -k 0

# Disable actually running GPU tests. This tends to yield TSan reports that are
# specific to one's particular GPU driver and therefore hard to reproduce across
# machines and often un-actionable anyway.
# See e.g. https://github.com/iree-org/iree/issues/9393
export IREE_VULKAN_DISABLE=1
export IREE_CUDA_DISABLE=1

# Honor the "notsan" label on tests.
export IREE_EXTRA_COMMA_SEPARATED_CTEST_LABELS_TO_EXCLUDE=notsan

# Run all tests, once
"${SCRIPT_DIR}/ctest_all.sh" "${BUILD_DIR}"

# Re-run many times certain tests that are cheap and prone to nondeterministic
# failure (typically, IREE runtime tests exercising multi-threading features).
export IREE_CTEST_TESTS_REGEX="(^iree/base/|^iree/task/)"
export IREE_CTEST_REPEAT_UNTIL_FAIL_COUNT=32
"${SCRIPT_DIR}/ctest_all.sh" "${BUILD_DIR}"
