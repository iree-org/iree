#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Script to run benchmarks on CI with the proper docker image and benchmark tool
# based on the IREE_DEVICE_NAME. This script can also run locally, but some
# devices require docker to run benchmarks. By default it uses the wrapper
# build_tools/docker/docker_run.sh if IREE_DOCKER_WRAPPER is not specified. See
# the script to learn about the required setup.
#
# IREE_BENCHMARK_TOOLS_DIR needs to point to a directory contains IREE
# benchmark tools. See benchmarks/README.md for more information.
#
# Command line arguments:
# 1. The path of e2e test artifacts directory
# 2. The path of IREE benchmark run config
# 3. The target device name
# 4. The shard index
# 5. The path to write benchmark results

set -euo pipefail

DOCKER_WRAPPER="${IREE_DOCKER_WRAPPER:-./build_tools/docker/docker_run.sh}"
BENCHMARK_TOOLS_DIR="${IREE_BENCHMARK_TOOLS_DIR}"
E2E_TEST_ARTIFACTS_DIR="${1:-${IREE_E2E_TEST_ARTIFACTS_DIR}}"
EXECUTION_BENCHMARK_CONFIG="${2:-${IREE_EXECUTION_BENCHMARK_CONFIG}}"
TARGET_DEVICE_NAME="${3:-${IREE_TARGET_DEVICE_NAME}}"
SHARD_INDEX="${4:-${IREE_SHARD_INDEX}}"
BENCHMARK_RESULTS="${5:-${IREE_BENCHMARK_RESULTS}}"

if [[ "${TARGET_DEVICE_NAME}" == "c2-standard-60" ]]; then
  ${DOCKER_WRAPPER} \
    gcr.io/iree-oss/base-bleeding-edge@sha256:cf2e78194e64fd0166f4141317366261d7a62432b72e9a324cb8c2ff4e1a515a \
      ./build_tools/benchmarks/run_benchmarks_on_linux.py \
        --benchmark_tool_dir="${BENCHMARK_TOOLS_DIR}" \
        --e2e_test_artifacts_dir="${E2E_TEST_ARTIFACTS_DIR}" \
        --execution_benchmark_config="${EXECUTION_BENCHMARK_CONFIG}" \
        --target_device_name="${TARGET_DEVICE_NAME}" \
        --shard_index="${SHARD_INDEX}" \
        --output="${BENCHMARK_RESULTS}" \
        --device_model="GCP-${TARGET_DEVICE_NAME}" \
        --cpu_uarch=CascadeLake \
        --verbose
elif [[ "${TARGET_DEVICE_NAME}" =~ ^(pixel-4|pixel-6-pro|moto-edge-x30)$ ]]; then
  ./build_tools/benchmarks/run_benchmarks_on_android.py \
    --benchmark_tool_dir="${BENCHMARK_TOOLS_DIR}" \
    --e2e_test_artifacts_dir="${E2E_TEST_ARTIFACTS_DIR}" \
    --execution_benchmark_config="${EXECUTION_BENCHMARK_CONFIG}" \
    --target_device_name="${TARGET_DEVICE_NAME}" \
    --shard_index="${SHARD_INDEX}" \
    --output="${BENCHMARK_RESULTS}" \
    --pin-cpu-freq \
    --pin-gpu-freq \
    --verbose
else
  echo "${TARGET_DEVICE_NAME} is not supported yet."
  exit 1
fi
