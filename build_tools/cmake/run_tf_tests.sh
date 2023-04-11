#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -xeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
cd "${ROOT_DIR}"

BUILD_DIR="$1"
IREE_VULKAN_DISABLE="${IREE_VULKAN_DISABLE:-0}"
IREE_LLVM_CPU_DISABLE="${IREE_LLVM_CPU_DISABLE:-0}"

source "${BUILD_DIR}/.env" && export PYTHONPATH
source build_tools/cmake/setup_local_python.sh

echo "***** Running TensorFlow integration tests *****"
# TODO: Use "--timeout 900" instead of --max-time below. Requires that
# `psutil` python package be installed in the VM for per test timeout.
LIT_SCRIPT="${ROOT_DIR}/third_party/llvm-project/llvm/utils/lit/lit.py"

CMD=(
  python3
  "${LIT_SCRIPT}"
  -v integrations/tensorflow/test
  --max-time 1800
)

if (( ${IREE_VULKAN_DISABLE} != 1 )); then
  CMD+=(-D FEATURES=vulkan)
fi

if (( ${IREE_LLVM_CPU_DISABLE} == 1 )); then
  CMD+=(-D DISABLE_FEATURES=llvmcpu)
fi

if "${CMD[@]}"; then
  tests_passed=1
else
  tests_passed=0
fi

if (( ${tests_passed} != 1 )); then
   echo "Some tests failed!!!"
   exit 1
fi
