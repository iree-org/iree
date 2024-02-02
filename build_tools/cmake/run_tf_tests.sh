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
IREE_VULKAN_DISABLE="${IREE_VULKAN_DISABLE:-1}"
IREE_LLVM_CPU_DISABLE="${IREE_LLVM_CPU_DISABLE:-0}"

# VMVX codegen is for reference and less optimized than other target backends.
# Disable the tests by default to reduce the test time.
IREE_VMVX_DISABLE="${IREE_VMVX_DISABLE:-1}"

# TODO(scotttodd): use build_tools/pkgci/setup_venv.py here instead of BUILD_DIR
source "${BUILD_DIR}/.env" && export PYTHONPATH
source build_tools/cmake/setup_tf_python.sh

python3 -m pip install lit
LIT_SCRIPT=$(which lit)

echo "***** Running TensorFlow integration tests *****"

# TODO: Use "--timeout 900" instead of --max-time below. Requires that
# `psutil` python package be installed in the VM for per test timeout.
CMD=(
  python3
  "${LIT_SCRIPT}"
  -v integrations/tensorflow/test
  --max-time 1800
)

declare -a TARGET_BACKENDS=()

if (( ${IREE_VULKAN_DISABLE} != 1 )); then
  TARGET_BACKENDS+=(vulkan)
fi
if (( ${IREE_VMVX_DISABLE} != 1 )); then
  TARGET_BACKENDS+=(vmvx)
fi

if [[ -n "${TARGET_BACKENDS[*]}" ]]; then
  TARGET_BACKENDS_STR="$(IFS="," ; echo "${TARGET_BACKENDS[*]}")"
  CMD+=(-D FEATURES=${TARGET_BACKENDS_STR})
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
