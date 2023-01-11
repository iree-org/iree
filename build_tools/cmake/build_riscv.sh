#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile the IREE project towards RISCV with CMake. Designed for CI,
# but can be run manually. This uses previously cached build results and does
# not clear build directories.
#
# Host binaries (e.g. compiler tools) will be built and installed in build-host/
# RISCV binaries (e.g. tests) will be built in build-riscv/.
#
# BUILD_PRESET can be: test, benchmark, benchmark-with-tracing to build with
# different flags.

set -xeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
cd "${ROOT_DIR}"

CMAKE_BIN="${CMAKE_BIN:-$(which cmake)}"
RISCV_ARCH="${RISCV_ARCH:-rv64}"
RISCV_COMPILER_FLAGS="${RISCV_COMPILER_FLAGS:--O3}"
IREE_HOST_BINARY_ROOT="$(realpath ${IREE_HOST_BINARY_ROOT})"
BUILD_BENCHMARK_SUITE_DIR="${BUILD_BENCHMARK_SUITE_DIR:-$ROOT_DIR/build-benchmarks/benchmark_suites}"
BUILD_RISCV_DIR="${BUILD_RISCV_DIR:-$ROOT_DIR/build-riscv}"
BUILD_PRESET="${BUILD_PRESET:-test}"

# --------------------------------------------------------------------------- #
# Build for the target (riscv).
if [[ -d "${BUILD_RISCV_DIR}" ]]; then
  echo "build-riscv directory already exists. Will use cached results there."
else
  echo "build-riscv directory does not already exist. Creating a new one."
  mkdir -p "${BUILD_RISCV_DIR}"
fi

echo "Build riscv target with the config of ${RISCV_ARCH}"
TOOLCHAIN_FILE="$(realpath ${ROOT_DIR}/build_tools/cmake/riscv.toolchain.cmake)"
declare -a args
args=(
  "-G" "Ninja"
  "-B" "${BUILD_RISCV_DIR}"
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

"${CMAKE_BIN}" "${args[@]}" "${ROOT_DIR}"

if [[ "${BUILD_PRESET}" == "benchmark-suite-test" ]] && \
   [[ "${RISCV_ARCH}" == "rv64" || "${RISCV_ARCH}" == "rv32-linux" ]]; then
  echo "Building iree-run-module and run-module-test deps for RISC-V"
  echo "------------------------------------------------------------"
  "${CMAKE_BIN}" --build "${BUILD_RISCV_DIR}" --target iree-run-module \
    iree-run-module-test-deps -- -k 0
else
  "${CMAKE_BIN}" --build "${BUILD_RISCV_DIR}" -- -k 0

  if [[ "${RISCV_ARCH}" == "rv64" || "${RISCV_ARCH}" == "rv32-linux" ]]; then
    echo "Building test deps for RISC-V"
    echo "-----------------------------"
    "${CMAKE_BIN}" --build "${BUILD_RISCV_DIR}" --target iree-test-deps -- -k 0
  fi
fi
