#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile the IREE project towards Android with CMake. Designed for CI,
# but can be run manually.
#
# Requires pre-compiled IREE and TF integrations host tools. Also requires that
# ANDROID_ABI and ANDROID_NDK variables be set

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
cd "${ROOT_DIR}"

CMAKE_BIN="${CMAKE_BIN:-$(which cmake)}"
IREE_HOST_BINARY_ROOT="$(realpath ${IREE_HOST_BINARY_ROOT})"
BUILD_ANDROID_DIR="${BUILD_ANDROID_DIR:-$ROOT_DIR/build-android}"


if [[ -d "${BUILD_ANDROID_DIR}" ]]; then
  echo "${BUILD_ANDROID_DIR} directory already exists. Will use cached results there."
else
  echo "${BUILD_ANDROID_DIR} directory does not already exist. Creating a new one."
  mkdir "${BUILD_ANDROID_DIR}"
fi

declare -a common_args=(
  -G Ninja
  -B "${BUILD_ANDROID_DIR}"
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake"
  -DANDROID_ABI="${ANDROID_ABI}"
  -DANDROID_PLATFORM=android-29
  -DIREE_HOST_BINARY_ROOT="${IREE_HOST_BINARY_ROOT}"
  -DIREE_BUILD_COMPILER=OFF
  -DIREE_BUILD_SAMPLES=OFF
)

# Settings for the minimal build used for code size / c++ symbols checks.
# See "### C++ symbols checks" section below.
declare -a minimal_build_args=(
  -DIREE_BUILD_TESTS=OFF
  # A bunch of instrumentation is tied to IREE_ENABLE_ASSERTIONS - this isn't
  # just about eliding assertions.
  -DIREE_ENABLE_ASSERTIONS=OFF
  # The Vulkan driver is C++.
  -DIREE_HAL_DRIVER_VULKAN=OFF
)

# These should at least undo any non-default settings in minimal_build_args.
declare -a full_build_args=(
  -DIREE_BUILD_TESTS=ON
  -DIREE_ENABLE_ASSERTIONS=ON
  -DIREE_HAL_DRIVER_VULKAN=ON
)

#############################################################################
### C++ symbols checks
#############################################################################

# Checks that a minimal C program exercising the IREE runtime doesn't contain
# any C++ symbols.
#
# This matters specifically on Android because Android does not
# treat the C++ runtime as a "system library" --- it's up to each application to
# distribute its own, and the CMake default, which we are using, is to
# statically link it. See https://developer.android.com/ndk/guides/cpp-support.
#
# To make this worse, even though IREE C++ code is built with -fno-rtti and
# -fno-exceptions, the libc++ that we are statically linking against in the
# Android NDK is the same as everyone else's --- just because we have those
# compilation flags doesn't change what we link against. As a result, a single
# `operator new` is enough to drag in a large amount of exception-handling,
# stack-unwinding, demangling code.

echo
echo "Start of minimal code size and symbols checks. The actual build and tests will follow below."
echo 

echo "Configuring for device (minimal build, for code size / symbols checks)"
echo "------------"
"${CMAKE_BIN}" "${common_args[@]}" "${minimal_build_args[@]}"

TARGET="iree-minimal-invoke-module"

echo "Building target ${TARGET} for device"
echo "------------"
"${CMAKE_BIN}" --build "${BUILD_ANDROID_DIR}" --target "${TARGET}" -- -k 0

TARGET_BINARY="${BUILD_ANDROID_DIR}/tools/${TARGET}"

TOOLCHAIN_TOOLS_DIR="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin"
NM_TOOL="${TOOLCHAIN_TOOLS_DIR}/llvm-nm"
STRIP_TOOL="${TOOLCHAIN_TOOLS_DIR}/llvm-strip"
NM_ARGS="-S --radix=d"
NM_OUTPUT="$(${NM_TOOL} ${NM_ARGS} ${TARGET_BINARY})"

# If we ever start stripping binaries in the Release build type, that will cause
# nm to report "no symbols" as a 1-line output but nm doesn't return an error
# code in that case.
NM_OUTPUT_LINES="$(wc -l <<< "${NM_OUTPUT}")"
if (( NM_OUTPUT_LINES < 100 ))
then
  echo "FATAL error: unexpectedly short nm output:"
  echo "${NM_OUTPUT}"
  exit 1
fi

NM_OUTPUT_ALL_CXX_SYMBOLS="$(grep '\b_Z' <<< "${NM_OUTPUT}" || true)"
# The standard C library we're linking against has a snprintf implementation
# that internally calls a C++ snprintf. As a result we get those C++ snprintf
# symbols from plain C functions in .c files doing plain snprintf calls.
# Example: iree_status_format (status.c) -> snprintf -> C++ snprintf.
# We have no choice but to tolerate that and it's not a problem anyway as this
# C++ code isn't dragging in any other symbol.
NM_OUTPUT_CXX_SYMBOLS="$(grep -v '\b_ZL8snprintf' <<< "${NM_OUTPUT_ALL_CXX_SYMBOLS}"|| true)"

if [[ ! -z "${NM_OUTPUT_CXX_SYMBOLS}" ]]
then
  CXX_SYMBOLS_SIZE="$(echo "${NM_OUTPUT_CXX_SYMBOLS}" | awk '{sum += $2} END {print sum}')"
  echo "*********************************************************************"
  echo "FATAL error: target ${TARGET} contains C++ symbols."
  echo "*********************************************************************"
  echo
  echo "Total size of C++ symbols: ${CXX_SYMBOLS_SIZE} bytes"
  echo
  echo "Output of nm ${NM_ARGS} for these C++ symbols:"
  # We could have `nm` demangle right away with -C, but that would make it
  # harder to tell which symbols are C++. We would have to run `nm` twice.
  echo "${NM_OUTPUT_CXX_SYMBOLS}" | c++filt
  echo
  exit 1
fi

${STRIP_TOOL} ${TARGET_BINARY}

TARGET_BINARY_SIZE="$(size -A ${TARGET_BINARY} | grep Total | grep -o '[0-9]\+')"
echo "Sections size of ${TARGET_BINARY}: ${TARGET_BINARY_SIZE} bytes"

# As of October 2022, the current size is 397827, so this gives 20% growth room.
TARGET_BINARY_SIZE_LIMIT=480000

if ((TARGET_BINARY_SIZE > TARGET_BINARY_SIZE_LIMIT ))
then
  echo "*********************************************************************"
  echo "FATAL error: ${TARGET_BINARY} exceeds the current arbitrary limit."
  echo "*********************************************************************"
  echo
  echo "Actual size:     ${TARGET_BINARY_SIZE}"
  echo "Arbitrary limit: ${TARGET_BINARY_SIZE_LIMIT}"
  echo
  echo "If this is normal, feel free to bump the arbitrary limit."
  echo
  exit 1
fi

##############################################################################
### End of C++ symbols checks
##############################################################################

echo
echo "End of minimal code size and symbols checks. The actual build and tests start below."
echo 

echo "Configuring for device (full build, for actually running tests)"
echo "------------"
"${CMAKE_BIN}" "${common_args[@]}" "${full_build_args[@]}"

echo "Building all for device"
echo "------------"
"${CMAKE_BIN}" --build "${BUILD_ANDROID_DIR}" -- -k 0

echo "Building test deps for device"
echo "------------------"
"${CMAKE_BIN}" --build "${BUILD_ANDROID_DIR}" --target iree-test-deps -- -k 0
