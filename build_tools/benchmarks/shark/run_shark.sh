#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Runs SHARK tank using both SHARK-Runtime and IREE-Runtime, producing benchmark
# numbers.
#
# Usage:
#    ./run_shark.sh \
#        --sha <SHA of https://github.com/nod-ai/SHARK.git to pin to> \
#        --pytest-regex e.g. "cpu", "cuda", "cuda and torch". \
#        --driver e.g. "cpu", "cuda", "vulkan" \
#        --save-version-info <true|false> \
#        --iree-source-dir <source directory IREE if a local version is used> \
#        --output-dir <local output dir>

set -xeuo pipefail

export SAVE_VERSION_INFO=false
export BENCHMARK_REGEX="cpu"
export DRIVER="cpu"
export IREE_SOURCE_DIR=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --sha)
      SHARK_SHA="$2"
      shift;;
    --pytest-regex)
      BENCHMARK_REGEX="$2"
      shift;;
    -d|--driver)
      DRIVER="$2"
      shift;;
    --save-version-info)
      SAVE_VERSION_INFO="$2"
      shift;;
    -o|--output-dir)
      SHARK_OUTPUT_DIR="$(realpath $2)"
      shift;;
    --iree-source-dir)
      if [[ ! -z "$2" ]]; then
        IREE_SOURCE_DIR="$(realpath $2)"
      fi
      shift;;
    *)
      echo "Unknown parameter passed: $1"
      exit 1;;
esac
shift
done

mkdir "${SHARK_OUTPUT_DIR}"

git clone https://github.com/nod-ai/SHARK.git
pushd SHARK
git reset --hard ${SHARK_SHA}

# Remove existing data.
rm -rf ./shark_tmp
rm -rf ~/.local/shark_tank

declare -a args=(
  --benchmark="native"
  --update_tank
  --maxfail=500
  -k "${BENCHMARK_REGEX}"
)

if [[ ${DRIVER} == "cuda" ]]; then
  args+=(--tf32)
fi

# Run with SHARK-Runtime.
PYTHON=python3.11 VENV_DIR=shark.venv BENCHMARK=1 IMPORTER=1 ./setup_venv.sh
source shark.venv/bin/activate

export SHARK_VERSION=`pip show iree-compiler | grep Version | sed -e "s/^Version: \(.*\)$/\1/g"`
pytest "${args[@]}" tank/test_models.py || true

echo "######################################################"
echo "Benchmarks for SHARK-Runtime Complete"
cat bench_results.csv
mv bench_results.csv "${SHARK_OUTPUT_DIR}/${DRIVER}_shark_${SHARK_VERSION}.csv"
echo "######################################################"
deactivate

# Remove existing data.
rm -rf ./shark_tmp
rm -rf ~/.local/shark_tank

# Run with IREE.
PYTHON=python3.11 VENV_DIR=iree.venv USE_IREE=1 ./setup_venv.sh
source iree.venv/bin/activate

export IREE_VERSION=$(pip show iree-compiler | grep Version | sed -e "s/^Version: \(.*\)$/\1/g")

if [[ ! -z "${IREE_SOURCE_DIR}" ]]; then
  # `setup_venv.sh` would have installed a nightly release of IREE so we uninstall here.
  pip uninstall -y iree-compiler iree-runtime

  pushd "${IREE_SOURCE_DIR}"

  git submodule update --init
  pip install -r ./runtime/bindings/python/iree/runtime/build_requirements.txt

  export IREE_VERSION="sha_$(git rev-parse --short=10 HEAD)"

  # We build using the same Python Virtual Environment as SHARK to ensure compatibility.
  cmake -GNinja -B iree-build -S . \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DIREE_ENABLE_ASSERTIONS=ON \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DIREE_ENABLE_LLD=ON \
      -DIREE_BUILD_PYTHON_BINDINGS=ON \
      -DPython3_EXECUTABLE="$(which python)" \
      -DIREE_HAL_DRIVER_CUDA=ON \
      -DIREE_TARGET_BACKEND_CUDA=ON

  cmake --build iree-build

  # Install Python bindings.
  pushd iree-build
  pip install -e compiler/
  pip install -e runtime/
  popd # iree-build
  popd # ${IREE_SOURCE_DIR}
fi

pytest "${args[@]}" tank/test_models.py || true

echo "######################################################"
echo "Benchmarks for IREE Complete"
cat bench_results.csv
mv bench_results.csv "${SHARK_OUTPUT_DIR}/${DRIVER}_iree_${IREE_VERSION}.csv"
echo "######################################################"

# Save version info.
if [[ "${SAVE_VERSION_INFO}" == true ]]; then
  touch "${SHARK_OUTPUT_DIR}/version_info.txt"
  echo "iree=${IREE_VERSION}" >> "${SHARK_OUTPUT_DIR}/version_info.txt"
  echo "shark=${SHARK_VERSION}" >> "${SHARK_OUTPUT_DIR}/version_info.txt"

  export TF_VERSION=$(pip show tensorflow | grep "Version" | awk '{print $NF}')
  echo "tf=Tensorflow XLA ${TF_VERSION}" >> "${SHARK_OUTPUT_DIR}/version_info.txt"

  export TORCH_VERSION=$(pip show torch | grep "Version" | awk '{print $NF}')
  echo "torch=Torch Inductor ${TORCH_VERSION}" >> "${SHARK_OUTPUT_DIR}/version_info.txt"
fi

deactivate
