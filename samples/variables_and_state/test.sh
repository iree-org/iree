#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script runs the steps laid out in the README for this sample. It is
# intended for use on continuous integration servers and as a reference for
# users, but can also be run manually.

set -x
set -e

ROOT_DIR=$(git rev-parse --show-toplevel)
BUILD_DIR=${ROOT_DIR}/build-samples
ARTIFACTS_DIR=/tmp/iree/colab_artifacts

# 1. Run the notebook to generate `counter.mlir` and `counter_vmvx.vmfb`
${ROOT_DIR}/build_tools/testing/run_python_notebook.sh \
  ${ROOT_DIR}/samples/variables_and_state/variables_and_state.ipynb
test -f ${ARTIFACTS_DIR}/counter.mlir && echo "counter.mlir exists"
test -f ${ARTIFACTS_DIR}/counter_vmvx.vmfb && echo "counter_vmvx.vmfb exists"

# 2. Build the `iree_samples_variables_and_state` CMake target.
cmake -B ${BUILD_DIR} -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo ${ROOT_DIR}
cmake --build ${BUILD_DIR} --target iree_samples_variables_and_state -- -k 0

# 3. Run the sample binary.
${BUILD_DIR}/samples/variables_and_state/variables-and-state \
  ${ARTIFACTS_DIR}/counter_vmvx.vmfb local-task
