#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Generate metrics for a set of programs using IREE's WebAssembly backend.
#
# Intended for interactive developer use, so this generates lots of output
# files for manual inspection outside of the script.
# Some hardcoded paths should be edited before use, and you might want to
# comment out specific portions (such as python package installs) depending
# on your environment or workflow choices.
#
# Script steps:
#   * Install IREE tools into a Python virtual environment (venv)
#   * Download program source files (.tflite files from GCS)
#   * Import programs into MLIR (.tflite -> .mlir)
#   * Compile programs for WebAssembly (.mlir -> .vmfb, intermediates)
#   * (TODO) Print statistics that can be produced by a script (i.e. no
#     launching a webpage and waiting for benchmark results there)
#
# Sample usage 1:
#   generate_web_metrics.sh /tmp/iree/web_metrics/
#
#   then look in that directory for any files you want to process further,
#   and/or run run_native_benchmarks.sh to get native metrics for comparison
#
# Sample usage 2:
#   sample_dynamic/build_sample.sh (optional install path)
#   generate_web_metrics.sh /tmp/iree/web_metrics/
#   sample_dynamic/serve_sample.sh
#
#   then open http://localhost:8000/benchmarks.html (for automated benchmarks)
#   or open http://localhost:8000/ (for interactive benchmarks)

set -eo pipefail

TARGET_DIR="$1"
if [[ -z "$TARGET_DIR" ]]; then
  >&2 echo "ERROR: Expected target directory (e.g. /tmp/iree/web_metrics/)"
  exit 1
fi

# If used in conjunction with sample_dynamic/build_sample.sh, this script can
# copy compiled programs into that sample's build directory for benchmarking.
ROOT_DIR="$(git rev-parse --show-toplevel)"
BUILD_DIR="${ROOT_DIR?}"/build-emscripten
SAMPLE_BINARY_DIR="${BUILD_DIR}"/experimental/web/sample_dynamic/

echo "Working in directory '${TARGET_DIR}'"
mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

###############################################################################
# Set up Python virtual environment                                           #
###############################################################################

python -m venv .venv
source .venv/bin/activate
trap "deactivate 2> /dev/null" EXIT

# Skip package installs when you want by commenting this out. Freezing to a
# specific version when iterating on metrics is useful, and fetching is slow.

python -m pip install --upgrade \
  --find-links https://iree.dev/pip-release-links.html \
  iree-compiler iree-tools-tflite

###############################################################################
# Download program source files                                               #
###############################################################################

wget -nc https://storage.googleapis.com/iree-model-artifacts/mobile_ssd_v2_float_coco.tflite
wget -nc https://storage.googleapis.com/iree-model-artifacts/deeplabv3.tflite
wget -nc https://storage.googleapis.com/iree-model-artifacts/posenet.tflite
wget -nc https://storage.googleapis.com/iree-model-artifacts/mobilebert-baseline-tf2-float.tflite
wget -nc https://storage.googleapis.com/iree-model-artifacts/mobilenet_v2_1.0_224.tflite
wget -nc https://storage.googleapis.com/iree-model-artifacts/MobileNetV3SmallStaticBatch.tflite

###############################################################################
# Import programs into MLIR                                                   #
###############################################################################

# Note: you can also download imported programs from CI runs:
# https://github.com/openxla/iree/blob/main/docs/developers/developing_iree/benchmark_suites.md#fetching-benchmark-artifacts-from-ci

IREE_IMPORT_TFLITE_PATH=iree-import-tflite

# import_program helper
#   Args: program_name, tflite_source_path
#   Imports tflite_source_path to program_name.tflite.mlir
function import_program {
  OUTPUT_FILE=./$1.tflite.mlir
  echo "Importing '$1' to '${OUTPUT_FILE}'..."
  "${IREE_IMPORT_TFLITE_PATH?}" "$2" -o "${OUTPUT_FILE}"
}

import_program "deeplabv3" "deeplabv3.tflite"
import_program "mobile_ssd_v2_float_coco" "mobile_ssd_v2_float_coco.tflite"
import_program "posenet" "posenet.tflite"
import_program "mobilebertsquad" "./mobilebert-baseline-tf2-float.tflite"
import_program "mobilenet_v2_1.0_224" "./mobilenet_v2_1.0_224.tflite"
import_program "MobileNetV3SmallStaticBatch" "MobileNetV3SmallStaticBatch.tflite"

###############################################################################
# Compile programs                                                            #
###############################################################################

# Either build from source (setting this path), or use from the python packages.
# IREE_COMPILE_PATH=~/code/iree-build/tools/iree-compile
# IREE_COMPILE_PATH="D:\dev\projects\iree-build\tools\iree-compile"
IREE_COMPILE_PATH=iree-compile

# compile_program_wasm helper
#   Args: program_name
#   Compiles program_name.tflite.mlir to program_name_wasm.vmfb, dumping
#   statistics and intermediate files to disk.
function compile_program_wasm {
  INPUT_FILE=./"$1".tflite.mlir
  OUTPUT_FILE=./"$1"_wasm.vmfb
  echo "Compiling '${INPUT_FILE}' to '${OUTPUT_FILE}'..."

  ARTIFACTS_DIR=./"$1"-wasm_artifacts/
  mkdir -p "${ARTIFACTS_DIR}"

  # Compile from .mlir to .vmfb, dumping all the intermediate files and
  # compile-time statistics that we can.
  "${IREE_COMPILE_PATH?}" "${INPUT_FILE}" \
    --iree-input-type=tosa \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=wasm32-unknown-emscripten \
    --iree-hal-dump-executable-sources-to="${ARTIFACTS_DIR}" \
    --iree-hal-dump-executable-binaries-to="${ARTIFACTS_DIR}" \
    --iree-scheduling-dump-statistics-format=csv \
    --iree-scheduling-dump-statistics-file=${ARTIFACTS_DIR}/$1_statistics.csv \
    --o ${OUTPUT_FILE}

  # Compress the .vmfb file (ideally it would be compressed already, but we can
  # expect some compression support from platforms like the web).
  gzip -k -f "${OUTPUT_FILE}"

  if [[ -d "$SAMPLE_BINARY_DIR" ]]; then
    echo "Copying '${OUTPUT_FILE}' to '${SAMPLE_BINARY_DIR}' for benchmarking"
    cp "${OUTPUT_FILE}" "${SAMPLE_BINARY_DIR}"
  fi
}

# compile_program_native helper
#   Args: program_name
#   Compiles program_name.tflite.mlir to program_name_native.vmfb, dumping
#   statistics and intermediate files to disk.
function compile_program_native {
  INPUT_FILE=./"$1".tflite.mlir
  OUTPUT_FILE=./"$1"_native.vmfb
  echo "Compiling '${INPUT_FILE}' to '${OUTPUT_FILE}'..."

  ARTIFACTS_DIR=./"$1"-native_artifacts/
  mkdir -p "${ARTIFACTS_DIR}"

  # Compile from .mlir to .vmfb, dumping all the intermediate files and
  # compile-time statistics that we can.
  "${IREE_COMPILE_PATH?}" "${INPUT_FILE}" \
    --iree-input-type=tosa \
    --iree-hal-target-backends=llvm-cpu \
    --iree-hal-dump-executable-sources-to="${ARTIFACTS_DIR}" \
    --iree-hal-dump-executable-binaries-to="${ARTIFACTS_DIR}" \
    --iree-scheduling-dump-statistics-format=csv \
    --iree-scheduling-dump-statistics-file="${ARTIFACTS_DIR}"/$1_statistics.csv \
    --o "${OUTPUT_FILE}"

  # Compress the .vmfb file (ideally it would be compressed already, but we can
  # expect some compression support from platforms like the web).
  gzip -k -f "${OUTPUT_FILE}"
}

# compile_program helper
#   Args: program_name
#   Wraps compile_program_wasm and compile_program_native.
function compile_program {
  compile_program_wasm $1
  compile_program_native $1
}

compile_program "deeplabv3"
compile_program "mobile_ssd_v2_float_coco"
compile_program "posenet"
compile_program "mobilebertsquad"
compile_program "mobilenet_v2_1.0_224"
compile_program "MobileNetV3SmallStaticBatch"

###############################################################################
# TODO: collect/summarize statistics (manual inspection or scripted)
#   * .vmfb size
#   * number of executables
#   * size of each executable (data size)
#   * size of constants
