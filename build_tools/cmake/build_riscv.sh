#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile the runtime using CMake targeting RISC-V
#
# The required IREE_HOST_BINARY_ROOT environment variable indicates the location
# of the precompiled IREE binaries. The BUILD_PRESET environment variable
# indicates how the project should be configured: "test", "benchmark",
# "benchmark-with-tracing", or "benchmark-suite-test". Defaults to "test".
#
# The desired build directory can be passed as the first argument. Otherwise, it
# uses the environment variable IREE_RISCV_BUILD_DIR, defaulting to
# "build-riscv". Designed for CI, but can be run manually. It reuses the build
# directory if it already exists. Expects to be run from the root of the IREE
# repository.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_BUILD_RISCV_DIR:-build-riscv}}"
RISCV_ARCH="${RISCV_ARCH:-rv64}"
RISCV_COMPILER_FLAGS="${RISCV_COMPILER_FLAGS:--O3}"
IREE_HOST_BINARY_ROOT="$(realpath ${IREE_HOST_BINARY_ROOT})"
BUILD_BENCHMARK_SUITE_DIR="${BUILD_BENCHMARK_SUITE_DIR:-build-benchmarks/benchmark_suites}"
BUILD_PRESET="${BUILD_PRESET:-test}"

source build_tools/cmake/setup_build.sh

echo "Build riscv target with the config of ${RISCV_ARCH}"
TOOLCHAIN_FILE="$(realpath build_tools/cmake/riscv.toolchain.cmake)"
declare -a args
args=(
  "-G" "Ninja"
  "-B" "${BUILD_DIR}"
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}"
  -DIREE_HOST_BINARY_ROOT="${IREE_HOST_BINARY_ROOT}"
  -DRISCV_CPU="${RISCV_ARCH}"
  -DRISCV_COMPILER_FLAGS="${RISCV_COMPILER_FLAGS}"
  -DIREE_BUILD_COMPILER=OFF
  # CPU info doesn't work on RISCV
  -DIREE_ENABLE_CPUINFO=OFF
)

if [[ "${RISCV_ARCH}" == "rv64" || "${RISCV_ARCH}" == "rv32-linux" ]]; then
  args+=(
    -DRISCV_TOOLCHAIN_ROOT="${RISCV_RV64_LINUX_TOOLCHAIN_ROOT}"
  )
elif [[ "${RISCV_ARCH}" == "rv32-baremetal" ]]; then
  args+=(
    # TODO(#6353): Off until tools/ are refactored to support threadless config.
    -DIREE_BUILD_TESTS=OFF
    -DRISCV_TOOLCHAIN_ROOT="${RISCV_RV32_NEWLIB_TOOLCHAIN_ROOT}"
  )
else
  echo "riscv config not supported yet"
  return -1
fi

case "${BUILD_PRESET}" in
  test)
    args+=(
      -DIREE_ENABLE_ASSERTIONS=ON
      -DIREE_BUILD_SAMPLES=ON
    )
    ;;
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
  benchmark-suite-test)
    BUILD_BENCHMARK_SUITE_DIR="$(realpath ${BUILD_BENCHMARK_SUITE_DIR})"
    args+=(
      -DIREE_BENCHMARK_SUITE_DIR="${BUILD_BENCHMARK_SUITE_DIR}"
    )
    ;;
  *)
    echo "Unknown build preset: ${BUILD_PRESET}"
    exit 1
    ;;
esac

"${CMAKE_BIN}" "${args[@]}"

if [[ "${BUILD_PRESET}" == "benchmark-suite-test" ]] && \
   [[ "${RISCV_ARCH}" == "rv64" || "${RISCV_ARCH}" == "rv32-linux" ]]; then
  echo "Building iree-run-module and run-module-test deps for RISC-V"
  echo "------------------------------------------------------------"
  "${CMAKE_BIN}" --build "${BUILD_DIR}" --target iree-run-module \
    iree-run-module-test-deps -- -k 0
else
  "${CMAKE_BIN}" --build "${BUILD_DIR}" -- -k 0

  if [[ "${RISCV_ARCH}" == "rv64" || "${RISCV_ARCH}" == "rv32-linux" ]]; then
    echo "Building test deps for RISC-V"
    echo "-----------------------------"
    "${CMAKE_BIN}" --build "${BUILD_DIR}" --target iree-test-deps -- -k 0
  fi
fi
