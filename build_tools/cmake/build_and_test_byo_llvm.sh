#!/bin/bash

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build and test, using CMake/CTest, with the bring-your-own-LLVM path.

set -xeuo pipefail

source build_tools/cmake/setup_ccache.sh

# Environment variables read by byo_llvm.sh.
export IREE_BYOLLVM_BUILD_DIR="${1:-build-byo-llvm}"
export IREE_BYOLLVM_INSTALL_DIR="${IREE_BYOLLVM_BUILD_DIR}/install"

python3 --version

# We've been instructed to set up a venv.
VENV_DIR="$IREE_BYOLLVM_BUILD_DIR/.venv"
echo "Setting up venv at $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -r runtime/bindings/python/iree/runtime/build_requirements.txt
python -m pip install -r third_party/llvm-project/mlir/python/requirements.txt
# Note: IREE's Python bindings for Python 3.13 are build with support for
# free-threading for which support was added to pybind with version 2.13.0.
# Therefore, we upgrade to a more recent version and avoid mixing of different
# pybind versions.
python -m pip install pybind11==2.13.6

# Note: by using the `build_llvm` action here, we are exercising byo_llvm.sh's
# ability to build LLVM... from our own third_party/llvm-project. That's not
# the most intuitive interpretation of "bring your own LLVM", since as far as
# the source repository is concerned, that's just the standard IREE-bundled one,
# but that exists for good reasons as some users need control more over the
# building and packaging than over the source repository, and that's good to
# have test coverage for, and of course that's more convenient for us to test.
build_tools/llvm/byo_llvm.sh build_llvm

build_tools/llvm/byo_llvm.sh build_mlir
build_tools/llvm/byo_llvm.sh build_iree

echo "*********************** TESTING IREE **********************************"
iree_build_dir="${IREE_BYOLLVM_BUILD_DIR}/iree"
echo "Build Directory: ${iree_build_dir}"
cmake --build "${iree_build_dir}" --target iree-test-deps
build_tools/cmake/ctest_all.sh "${iree_build_dir}"
