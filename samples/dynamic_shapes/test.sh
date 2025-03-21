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

# 1. Run the notebook to generate `dynamic_shapes.mlir` and
#    `dynamic_shapes_cpu.vmfb`
# TODO(scotttodd): Test pytorch_dynamic_shapes.ipynb instead/also
${ROOT_DIR}/build_tools/testing/run_python_notebook.sh \
  ${ROOT_DIR}/samples/dynamic_shapes/tensorflow_dynamic_shapes.ipynb
test -f ${ARTIFACTS_DIR}/dynamic_shapes.mlir && echo "dynamic_shapes.mlir exists"

# Setup Python venv for step 2.
python3 --version
python3 -m venv .venv
source .venv/bin/activate
trap "deactivate 2> /dev/null" EXIT

# 2. Install the `iree-compile` tool from pip (or build from source).
python -m pip install \
  --find-links https://iree.dev/pip-release-links.html \
  --upgrade \
  --pre \
  iree-base-compiler

# 3. Compile `dynamic_shapes.mlir` using `iree-compile`.
iree-compile \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  ${ARTIFACTS_DIR}/dynamic_shapes.mlir -o ${ARTIFACTS_DIR}/dynamic_shapes_cpu.vmfb

# 4. Build the `iree_samples_dynamic_shapes` CMake target.
cmake -B ${BUILD_DIR} -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DIREE_BUILD_COMPILER=OFF .
cmake --build ${BUILD_DIR} --target iree_samples_dynamic_shapes -- -k 0

# 5. Run the sample binary.
${BUILD_DIR}/samples/dynamic_shapes/dynamic-shapes \
  ${ARTIFACTS_DIR}/dynamic_shapes_cpu.vmfb local-task
