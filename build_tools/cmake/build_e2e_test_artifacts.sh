#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build e2e test artifacts using a host tools directory.
#
# This is copied and modified from build_tools/cmake/build_benchmarks.sh. We
# will remove build_tools/cmake/build_benchmarks.sh once everything has been
# migrated.
#
# The required IREE_HOST_BIN_DIR environment variable indicates the location
# of the precompiled IREE binaries. The IREE_BENCHMARK_PRESETS environment 
# variable can be set to build required artifacts for the comma-separated
# benchmark presets. By default `iree-benchmark-suites` is built for sanity
# check and e2e model testing. It can be disabled with the environment variable
# `IREE_BUILD_DEFAULT_BENCHMARK_SUITES=0`.
#
# Designed for CI, but can be run locally. The desired build directory can be
# passed as the first argument. Otherwise, it uses the environment variable
# IREE_BUILD_E2E_TEST_ARTIFACTS_DIR, defaulting to "build-e2e-test-artifacts".
# It reuses the build directory if it already exists. Expects to be run from the
# root of the IREE repository.


set -xeuo pipefail

BUILD_DIR="${1:-${IREE_BUILD_E2E_TEST_ARTIFACTS_DIR:-build-e2e-test-artifacts}}"
IREE_HOST_BIN_DIR="$(realpath ${IREE_HOST_BIN_DIR})"
BENCHMARK_PRESETS="${IREE_BENCHMARK_PRESETS:-}"
BUILD_DEFAULT_BENCHMARK_SUITES="${IREE_BUILD_DEFAULT_BENCHMARK_SUITES:-1}"

source build_tools/cmake/setup_build.sh
source build_tools/cmake/setup_tf_python.sh

declare -a BUILD_TARGETS

if (( "${BUILD_DEFAULT_BENCHMARK_SUITES}" ==  1 )); then
  BUILD_TARGETS+=("iree-benchmark-suites")
fi

# Separate the presets into the execution and compilation benchmark presets to
# export different configs with export_benchmark_config.py.
COMPILATION_PRESETS=""
EXECUTION_PRESETS=""
if [[ -n "${BENCHMARK_PRESETS}" ]]; then
  IFS=, read -r -a PRESET_ARRAY <<< "${BENCHMARK_PRESETS}"
  for PRESET in "${PRESET_ARRAY[@]}"; do
    case "${PRESET}" in
      comp-stats)
        BUILD_TARGETS+=(iree-e2e-compile-stats-suites)
        COMPILATION_PRESETS="${COMPILATION_PRESETS},${PRESET}"
        ;;
      comp-stats-long)
        BUILD_TARGETS+=(iree-e2e-compile-stats-suites-long)
        COMPILATION_PRESETS="${COMPILATION_PRESETS},${PRESET}"
        ;;
      *-long)
        BUILD_TARGETS+=(iree-benchmark-suites-long)
        EXECUTION_PRESETS="${EXECUTION_PRESETS},${PRESET}"
        ;;
      *)
        # Build target of the default preset has been added above.
        EXECUTION_PRESETS="${EXECUTION_PRESETS},${PRESET}"
        ;;
    esac
  done
  COMPILATION_PRESETS="${COMPILATION_PRESETS#,}"
  EXECUTION_PRESETS="${EXECUTION_PRESETS#,}"
fi

if (( "${#BUILD_TARGETS[@]}" == 0 )); then
  echo "No target to build."
  exit 1
fi

echo "Configuring to build e2e test artifacts"
"${CMAKE_BIN}" -B "${BUILD_DIR}" \
  -G Ninja \
  -DPython3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DPYTHON_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DIREE_HOST_BIN_DIR="${IREE_HOST_BIN_DIR}" \
  -DIREE_BUILD_E2E_TEST_ARTIFACTS=ON \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_TESTS=OFF

echo "Building e2e test artifacts"
"${CMAKE_BIN}" \
  --build "${BUILD_DIR}" \
  --target "${BUILD_TARGETS[@]}" \
  -- -k 0

E2E_TEST_ARTIFACTS_DIR="${BUILD_DIR}/e2e_test_artifacts"
COMPILATION_CONFIG="${E2E_TEST_ARTIFACTS_DIR}/compilation-benchmark-config.json"
EXECUTION_CONFIG="${E2E_TEST_ARTIFACTS_DIR}/execution-benchmark-config.json"
FLAG_DUMP="${E2E_TEST_ARTIFACTS_DIR}/benchmark-flag-dump.txt"
./build_tools/benchmarks/export_benchmark_config.py \
  compilation \
  --benchmark_presets="${COMPILATION_PRESETS}" \
  --output="${COMPILATION_CONFIG}"
./build_tools/benchmarks/export_benchmark_config.py \
  execution \
  --benchmark_presets="${EXECUTION_PRESETS}" \
  --output="${EXECUTION_CONFIG}"
./build_tools/benchmarks/benchmark_helper.py dump-cmds \
  --execution_benchmark_config="${EXECUTION_CONFIG}" \
  --compilation_benchmark_config="${COMPILATION_CONFIG}" \
  > "${FLAG_DUMP}"
