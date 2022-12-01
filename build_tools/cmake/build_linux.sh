#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Compile the runtime using CMake targeting Linux
#
# The required IREE_HOST_BINARY_ROOT environment variable indicates the location
# of the precompiled IREE binaries. The BUILD_PRESET environment variable
# indicates how the project should be configured: "benchmark" or
# "benchmark-with-tracing". Defaults to "benchmark".
#
# The desired build directory can be passed as the first argument. Otherwise, it
# uses the environment variable IREE_LINUX_BUILD_DIR, defaulting to
# "build-linux". Designed for CI, but can be run manually. It reuses the build
# directory if it already exists. Expects to be run from the root of the IREE
# repository.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_BUILD_LINUX_DIR:-build-linux}}"
LINUX_ARCH="${LINUX_ARCH:-x86_64}"
IREE_HOST_BINARY_ROOT="$(realpath ${IREE_HOST_BINARY_ROOT})"
BUILD_PRESET="${BUILD_PRESET:-benchmark}"

source build_tools/cmake/setup_build.sh
source build_tools/cmake/setup_ccache.sh

declare -a args
args=(
  "-G" "Ninja"
  "-B" "${BUILD_DIR}"
  -DIREE_HOST_BINARY_ROOT="${IREE_HOST_BINARY_ROOT}"
  -DIREE_BUILD_COMPILER=OFF
  # CUDA driver can be enabled even we don't run on a GPU machine. Enable it so
  # the same tools can be used with both CPU and GPU.
  -DIREE_HAL_DRIVER_CUDA=ON
)

if [[ "${LINUX_ARCH}" == "x86_64" ]]; then
  # Assume the CI builder is Linux x86_64.
  if [[ $(uname -m) != "x86_64" ]]; then
    echo "${LINUX_ARCH} must be built on an x86_64 host."
    return -1
  fi
else
  echo "${LINUX_ARCH} not supported yet."
  return -1
fi

case "${BUILD_PRESET}" in
  benchmark)
    args+=(
      -DIREE_ENABLE_ASSERTIONS=OFF
      -DIREE_BUILD_SAMPLES=OFF
      -DIREE_BUILD_TESTS=OFF
    )
    ;;
  benchmark-with-tracing)
    args+=(
      -DIREE_ENABLE_ASSERTIONS=OFF
      -DIREE_BUILD_SAMPLES=OFF
      -DIREE_BUILD_TESTS=OFF
      -DIREE_ENABLE_RUNTIME_TRACING=ON
    )
    ;;
  *)
    echo "Unknown build preset: ${BUILD_PRESET}"
    exit 1
    ;;
esac

"${CMAKE_BIN}" "${args[@]}"

case "${BUILD_PRESET}" in
  benchmark|benchmark-with-tracing)
    "${CMAKE_BIN}" --build "${BUILD_DIR}" --target iree-benchmark-module -- -k 0
    ;;
  *)
    echo "Unknown build preset: ${BUILD_PRESET}"
    exit 1
    ;;
esac
