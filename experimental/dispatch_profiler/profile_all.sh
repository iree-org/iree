#!/bin/bash

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Sets up a venv suitable for running IREE Dispatch Profiler and executes
# a suite of runs. This is invoked by a Github workflow and can be invoked
# locally.
#
# Recommend getting default 'python' to be python 3. For example on Debian:
#   sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1
# Or launch with python=/some/path
#
# Arg 1: The directory where iree-compile, iree-benchmark-module, etc. are
#        located. If unset, uses IREE Dispatch Profiler defaults.
# Arg 2: The directory where output is written. If unset, uses
#        `dispatch_profiler_output` in current working directory.

set -euo pipefail

TD="$(cd $(dirname $0) && pwd)"

PYTHON="${PYTHON:-python3}"

DISPATCH_PROFILER_IREE_BIN_DIR=${1:-""}
if [[ -z "${DISPATCH_PROFILER_IREE_BIN_DIR}" ]]; then
  DISPATCH_PROFILER_IREE_BIN_DIR_FLAG=""
else
  DISPATCH_PROFILER_IREE_BIN_DIR_FLAG="--iree-bin-dir=${DISPATCH_PROFILER_IREE_BIN_DIR}"
fi

DISPATCH_PROFILER_OUTPUT_DIR="${2:-"dispatch_profiler_output"}"
DISPATCH_PROFILER_GENERATED_DIR="."
VENV_DIR="dispatch-profiler.venv"

echo "Setting up venv dir: ${VENV_DIR}"
echo "Python: ${PYTHON}"
echo "Python version: $("${PYTHON}" --version)"
echo "Dispatch Profiler IREE bin dir flag: ${DISPATCH_PROFILER_IREE_BIN_DIR_FLAG}"
echo "Dispatch Profiler output dir: ${DISPATCH_PROFILER_OUTPUT_DIR}"
echo "Dispatch profiler generated dir: ${DISPATCH_PROFILER_GENERATED_DIR}"

${PYTHON} -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# Upgrade pip and install requirements. 'python' is used here in order to
# reference to the python executable from the venv.
python -m pip install --upgrade pip
python -m pip install --upgrade -r "${TD}/requirements.txt"

mkdir -p "${DISPATCH_PROFILER_OUTPUT_DIR}"

python "${TD}/generator.py" \
  --generated-dir "${DISPATCH_PROFILER_GENERATED_DIR}"
python "${TD}/compile.py" \
  ${DISPATCH_PROFILER_IREE_BIN_DIR_FLAG} \
  --generated-dir "${DISPATCH_PROFILER_GENERATED_DIR}"
python "${TD}/profiler.py" \
  ${DISPATCH_PROFILER_IREE_BIN_DIR_FLAG} \
  --generated-dir "${DISPATCH_PROFILER_GENERATED_DIR}" \
  --dispatches="matmul_3456x1024x2048_f16t_f16t_f16t_tile_config_128x128_32x5_tensorcore_mmasync,matmul_3456x1024x2048_f32t_f32t_f32t_tile_config_128x128_16x5_tensorcore_mmasync" \
  --output "${DISPATCH_PROFILER_OUTPUT_DIR}/matmul_perf_tensor_core_a100.csv"
