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
# IREE_NORMAL_BENCHMARK_TOOLS_DIR needs to point to a directory contains IREE
# benchmark tools. See benchmarks/README.md for more information.
#
# Command line arguments:
# 1. The path of e2e test artifacts directory
# 2. The path of IREE benchmark run config
# 3. The target device name
# 4. The shard index
# 5. The path to write benchmark results
# 6. The path to write benchmark traces

set -euo pipefail

DOCKER_WRAPPER="${IREE_DOCKER_WRAPPER:-./build_tools/docker/docker_run.sh}"
NORMAL_BENCHMARK_TOOLS_DIR="${IREE_NORMAL_BENCHMARK_TOOLS_DIR}"
TRACED_BENCHMARK_TOOLS_DIR="${IREE_TRACED_BENCHMARK_TOOLS_DIR}"
TRACY_CAPTURE_TOOL="${IREE_TRACY_CAPTURE_TOOL}"
E2E_TEST_ARTIFACTS_DIR="${1:-${IREE_E2E_TEST_ARTIFACTS_DIR}}"
EXECUTION_BENCHMARK_CONFIG="${2:-${IREE_EXECUTION_BENCHMARK_CONFIG}}"
TARGET_DEVICE_NAME="${3:-${IREE_TARGET_DEVICE_NAME}}"
SHARD_INDEX="${4:-${IREE_SHARD_INDEX}}"
BENCHMARK_RESULTS="${5:-${IREE_BENCHMARK_RESULTS}}"
BENCHMARK_TRACES="${6:-${IREE_BENCHMARK_TRACES}}"

if [[ "${TARGET_DEVICE_NAME}" == "a2-highgpu-1g" ]]; then
  ${DOCKER_WRAPPER} \
    --gpus all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    gcr.io/iree-oss/nvidia-bleeding-edge@sha256:2eb17e2e8e0d5d25600e667e3b7a71e8d39d9d12a99ec2ad6bdabbc8919db43b \
      ./build_tools/benchmarks/run_benchmarks_on_linux.py \
        --normal_benchmark_tool_dir="${NORMAL_BENCHMARK_TOOLS_DIR}" \
        --traced_benchmark_tool_dir="${TRACED_BENCHMARK_TOOLS_DIR}" \
        --trace_capture_tool="${TRACY_CAPTURE_TOOL}" \
        --capture_tarball="${BENCHMARK_TRACES}" \
        --e2e_test_artifacts_dir="${E2E_TEST_ARTIFACTS_DIR}" \
        --execution_benchmark_config="${EXECUTION_BENCHMARK_CONFIG}" \
        --target_device_name="${TARGET_DEVICE_NAME}" \
        --shard_index="${SHARD_INDEX}" \
        --output="${BENCHMARK_RESULTS}" \
        --verbose
elif [[ "${TARGET_DEVICE_NAME}" =~ ^(c2-standard-16|c2-standard-60)$ ]]; then
  ${DOCKER_WRAPPER} \
    gcr.io/iree-oss/base-bleeding-edge@sha256:14200dacca3a0f3a66f8aa87c6f64729b83a2eeb403b689c24204074ad157418 \
      ./build_tools/benchmarks/run_benchmarks_on_linux.py \
        --normal_benchmark_tool_dir="${NORMAL_BENCHMARK_TOOLS_DIR}" \
        --traced_benchmark_tool_dir="${TRACED_BENCHMARK_TOOLS_DIR}" \
        --trace_capture_tool="${TRACY_CAPTURE_TOOL}" \
        --capture_tarball="${BENCHMARK_TRACES}" \
        --e2e_test_artifacts_dir="${E2E_TEST_ARTIFACTS_DIR}" \
        --execution_benchmark_config="${EXECUTION_BENCHMARK_CONFIG}" \
        --target_device_name="${TARGET_DEVICE_NAME}" \
        --shard_index="${SHARD_INDEX}" \
        --output="${BENCHMARK_RESULTS}" \
        --device_model="GCP-${TARGET_DEVICE_NAME}" \
        --cpu_uarch=CascadeLake \
        --verbose
elif [[ "${TARGET_DEVICE_NAME}" =~ ^(pixel-4|pixel-6-pro|pixel-8-pro|moto-edge-x30)$ ]]; then
  ./build_tools/benchmarks/run_benchmarks_on_android.py \
    --normal_benchmark_tool_dir="${NORMAL_BENCHMARK_TOOLS_DIR}" \
    --traced_benchmark_tool_dir="${TRACED_BENCHMARK_TOOLS_DIR}" \
    --trace_capture_tool="${TRACY_CAPTURE_TOOL}" \
    --capture_tarball="${BENCHMARK_TRACES}" \
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
