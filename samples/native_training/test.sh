#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script runs the steps laid out in the README for this sample. It is
# intended for use on continuous integration servers and as a reference for
# users, but can also be run manually.

set -x
set -e

# Run under a virtual environment to isolate Python packages.
#
# For more information, see `build_tools/testing/run_python_notebook.sh`
python3 -m venv .script.venv --system-site-packages --clear
source .script.venv/bin/activate 2> /dev/null
trap "deactivate 2> /dev/null" EXIT

# Update pip within the venv (you'd think this wouldn't be needed, but it is).
python3 -m pip install --quiet --upgrade pip

# Install script requirements, reusing system versions if possible.
python3 -m pip install --quiet \
  -f https://iree-org.github.io/iree/pip-release-links.html iree-compiler
python3 -m pip install --quiet \
  -f https://llvm.github.io/torch-mlir/package-index/ torch-mlir
python3 -m pip install --quiet \
  git+https://github.com/iree-org/iree-torch.git

# Update submodules in this repo.
(cd $(git rev-parse --show-toplevel) && git submodule update --init)

# Build the IREE runtime.
(cd $(git rev-parse --show-toplevel) && cmake -GNinja -B /tmp/iree-build-runtime/ .)
cmake --build /tmp/iree-build-runtime/ --target iree_runtime_unified

# Build the example.
cd $(git rev-parse --show-toplevel)/samples/native_training
make

# Generate the VM bytecode.
python native_training.py /tmp/native_training.vmfb

# Run the native training model.
./native-training /tmp/native_training.vmfb