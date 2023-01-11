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
# benchmark tools. See benchmarks/README.md for more information. The first
# argument is the path of e2e test artifacts directory. The second argument is
# the path of IREE benchmark run config. The third argument is the path to write
# benchmark results.

set -euo pipefail

DOCKER_WRAPPER="${IREE_DOCKER_WRAPPER:-./build_tools/docker/docker_run.sh}"
DEVICE_NAME="${IREE_DEVICE_NAME}"
NORMAL_BENCHMARK_TOOLS_DIR="${IREE_NORMAL_BENCHMARK_TOOLS_DIR}"
E2E_TEST_ARTIFACTS_DIR="${1:-${IREE_E2E_TEST_ARTIFACTS_DIR}}"
RUN_CONFIG="${2:-${IREE_RUN_CONFIG}}"
BENCHMARK_RESULTS="${3:-${IREE_BENCHMARK_RESULTS}}"

if [[ "${DEVICE_NAME}" == "a2-highgpu-1g" ]]; then
  ${DOCKER_WRAPPER} \
    --gpus all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    gcr.io/iree-oss/nvidia@sha256:1294591d06d2b5eb03a7214fac040a1ccab890ea62e466843553f7fb7aacdc1d \
      ./build_tools/benchmarks/run_benchmarks_on_linux.py \
        --normal_benchmark_tool_dir="${NORMAL_BENCHMARK_TOOLS_DIR}" \
        --e2e_test_artifacts_dir="${E2E_TEST_ARTIFACTS_DIR}" \
        --run_config="${RUN_CONFIG}" \
        --output="${BENCHMARK_RESULTS}" \
        --verbose
elif [[ "${DEVICE_NAME}" == "c2-standard-16" ]]; then
  ${DOCKER_WRAPPER} \
    gcr.io/iree-oss/base-bleeding-edge@sha256:479eefb76447c865cf58c5be7ca9fe33f48584b474a1da3dfaa125aad2510463 \
      ./build_tools/benchmarks/run_benchmarks_on_linux.py \
        --normal_benchmark_tool_dir="${NORMAL_BENCHMARK_TOOLS_DIR}" \
        --e2e_test_artifacts_dir="${E2E_TEST_ARTIFACTS_DIR}" \
        --run_config="${RUN_CONFIG}" \
        --output="${BENCHMARK_RESULTS}" \
        --device_model=GCP-c2-standard-16 \
        --cpu_uarch=CascadeLake \
        --verbose
else
  echo "${DEVICE_NAME} is not supported yet."
  exit 1
fi
