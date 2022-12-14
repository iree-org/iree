#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Script to run benchmarks with a proper docker image and benchmark tool based
# on device name.

set -euo pipefail

DEVICE_NAME="${IREE_DEVICE_NAME}"
NORMAL_BENCHMARK_TOOLS_DIR="${IREE_NORMAL_BENCHMARK_TOOLS_DIR}"
E2E_TEST_ARTIFACTS_DIR="${1:-${IREE_E2E_TEST_ARTIFACTS_DIR}}"
RUN_CONFIG="${2:-${IREE_RUN_CONFIG}}"
BENCHMARK_RESULTS="${3:-${IREE_BENCHMARK_RESULTS}}"

if [[ "${DEVICE_NAME}" == "a2-highgpu-1g" ]]; then
  ./build_tools/github_actions/docker_run.sh \
    --gpus all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    gcr.io/iree-oss/nvidia@sha256:b0751e9f2fcb104d9d3d56fab6c6e79405bdcd2e503e53f2bf4f2b66d13cd88b \
      ./build_tools/benchmarks/run_benchmarks_on_linux.py \
        --normal_benchmark_tool_dir="${NORMAL_BENCHMARK_TOOLS_DIR}" \
        --e2e_test_artifacts_dir="${E2E_TEST_ARTIFACTS_DIR}" \
        --run_config="${RUN_CONFIG}" \
        --output="${BENCHMARK_RESULTS}" \
        --verbose
else
  echo "${DEVICE_NAME} is not supported yet."
  exit 1
fi
