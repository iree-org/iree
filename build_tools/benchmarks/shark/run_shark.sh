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
#        <SHA of https://github.com/nod-ai/SHARK.git to pin to> \
#        <pytest regex> e.g. "cpu", "cuda", "cuda and torch".
#        <driver> e.g. "cpu", "cuda", "vulkan"
#        <output directory>

set -xeuo pipefail

export SHARK_SHA=$1
export BENCHMARK_REGEX=$2
export DRIVER=$3
export SHARK_OUTPUT_DIR=`pwd`/$4

mkdir "${SHARK_OUTPUT_DIR}"

git clone https://github.com/nod-ai/SHARK.git
cd SHARK
git reset --hard ${SHARK_SHA}

# Remove existing data.
rm -rf ./shark_tmp
rm -rf ~/.local/shark_tank

declare -a args=(
  --benchmark
  --update_tank
  --maxfail=500
  -k "${BENCHMARK_REGEX}"
)

if [[ ${DRIVER} == "cuda" ]]; then
  args+=(--tf32)
fi

# Run with SHARK-Runtime.
PYTHON=python3.10 VENV_DIR=shark.venv BENCHMARK=1 IMPORTER=1 ./setup_venv.sh
source shark.venv/bin/activate

# Revert Torch version due to https://github.com/nod-ai/SHARK/issues/824.
pip uninstall -y torch torchvision
pip install --no-deps https://download.pytorch.org/whl/nightly/cu117/torch-2.0.0.dev20230113%2Bcu117-cp310-cp310-linux_x86_64.whl https://download.pytorch.org/whl/nightly/cu117/torchvision-0.15.0.dev20230114%2Bcu117-cp310-cp310-linux_x86_64.whl

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
PYTHON=python3.10 VENV_DIR=iree.venv BENCHMARK=1 IMPORTER=1 USE_IREE=1 ./setup_venv.sh
source iree.venv/bin/activate

# Revert Torch version due to https://github.com/nod-ai/SHARK/issues/824.
pip uninstall -y torch torchvision
pip install --no-deps https://download.pytorch.org/whl/nightly/cu117/torch-2.0.0.dev20230113%2Bcu117-cp310-cp310-linux_x86_64.whl https://download.pytorch.org/whl/nightly/cu117/torchvision-0.15.0.dev20230114%2Bcu117-cp310-cp310-linux_x86_64.whl

export IREE_VERSION=`pip show iree-compiler | grep Version | sed -e "s/^Version: \(.*\)$/\1/g"`
pytest "${args[@]}" tank/test_models.py || true

echo "######################################################"
echo "Benchmarks for IREE Complete"
cat bench_results.csv
mv bench_results.csv "${SHARK_OUTPUT_DIR}/${DRIVER}_iree_${IREE_VERSION}.csv"
echo "######################################################"
deactivate
