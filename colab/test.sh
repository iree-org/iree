#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Test each notebook in this directory.

set -x
set -e

SCRIPT_DIR=$(dirname "$0")
ROOT_DIR=$(git rev-parse --show-toplevel)

FAILING_NOTEBOOKS=(
  # Low level invoke function notebook throws an exception:
  #   KeyError: "Function 'simple_mul' not found in module
  low_level_invoke_function.ipynb \

  # MNIST training notebook fails to import/compile:
  # https://github.com/google/iree/issues/6163
  mnist_training.ipynb \

  # Text classification notebook has a dep with non-standard install steps:
  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
  tflite_text_classification.ipynb  \
)

# TODO(scotttodd): improve script ergonomics:
#   * move this script to Python
#   * summarize pass/fail (with expected status) at the end of the script
#   * wrap in a CMake target?

for NOTEBOOK_FILE in ${SCRIPT_DIR}/*.ipynb
do
  NOTEBOOK_NAME=$(basename ${NOTEBOOK_FILE})
  printf "Running notebook '%s'\n" ${NOTEBOOK_NAME}

  if [[ "${FAILING_NOTEBOOKS[*]}" =~ .*${NOTEBOOK_NAME}.* ]]; then
    printf "  (Expected fail)\n"
    !(${ROOT_DIR}/build_tools/testing/run_python_notebook.sh ${NOTEBOOK_FILE})
  else
    printf "  (Expected pass)\n"
    ${ROOT_DIR}/build_tools/testing/run_python_notebook.sh ${NOTEBOOK_FILE}
  fi
done
