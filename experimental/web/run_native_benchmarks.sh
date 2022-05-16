#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Run native benchmarks to compare with WebAssembly.
#
# Sample usage (after generate_web_metrics.sh):
#   run_native_benchmarks.sh /tmp/iree/web_metrics/

set -euo pipefail

TARGET_DIR="$1"
if [[ -z "$TARGET_DIR" ]]; then
  >&2 echo "ERROR: Expected target directory (e.g. /tmp/iree/web_metrics/)"
  exit 1
fi
echo "Working in directory '$TARGET_DIR'"
mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

###############################################################################
# Run benchmarks                                                              #
###############################################################################

# Either build from source (setting this path), or use from the python packages.
# IREE_BENCHMARK_MODULE_PATH=~/code/iree-build/iree/tools/iree-benchmark-module
# IREE_BENCHMARK_MODULE_PATH="D:\dev\projects\iree-build\iree\tools\iree-benchmark-module"
IREE_BENCHMARK_MODULE_PATH=iree-benchmark-module

echo "Benchmarking DeepLabV3..."
"${IREE_BENCHMARK_MODULE_PATH?}" \
    --module_file=./deeplabv3_native.vmfb \
    --driver=dylib \
    --task_topology_group_count=1 \
    --entry_function=main \
    --function_input=1x257x257x3xf32 \
    --benchmark_min_time=3

echo ""
echo "Benchmarking MobileSSD..."
"${IREE_BENCHMARK_MODULE_PATH?}" \
    --module_file=./mobile_ssd_v2_float_coco_native.vmfb \
    --driver=dylib \
    --task_topology_group_count=1 \
    --entry_function=main \
    --function_input=1x320x320x3xf32 \
    --benchmark_min_time=3

echo ""
echo "Benchmarking PoseNet..."
"${IREE_BENCHMARK_MODULE_PATH?}" \
    --module_file=./posenet_native.vmfb \
    --driver=dylib \
    --task_topology_group_count=1 \
    --entry_function=main \
    --function_input=1x353x257x3xf32 \
    --benchmark_min_time=3

echo ""
echo "Benchmarking MobileBertSquad..."
"${IREE_BENCHMARK_MODULE_PATH?}" \
    --module_file=./mobilebertsquad_native.vmfb \
    --driver=dylib \
    --task_topology_group_count=1 \
    --entry_function=main \
    --function_input=1x384xi32 \
    --function_input=1x384xi32 \
    --function_input=1x384xi32 \
    --benchmark_min_time=10

echo ""
echo "Benchmarking MobileNetV2..."
"${IREE_BENCHMARK_MODULE_PATH?}" \
    --module_file=./mobilenet_v2_1.0_224_native.vmfb \
    --driver=dylib \
    --task_topology_group_count=1 \
    --entry_function=main \
    --function_input=1x224x224x3xf32 \
    --benchmark_min_time=3

echo ""
echo "Benchmarking MobileNetV3Small..."
"${IREE_BENCHMARK_MODULE_PATH?}" \
    --module_file=./MobileNetV3SmallStaticBatch_native.vmfb \
    --driver=dylib \
    --task_topology_group_count=1 \
    --entry_function=main \
    --function_input=1x224x224x3xf32 \
    --benchmark_min_time=3
