#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Run a Python (Colab/Jupyter) notebook in an isolated virtual environment.
# Fails if the notebook has errors.
#
# Usage: run_python_notebook.sh /path/to/notebook.ipynb
#
# This is intended for use on continuous integration servers and as a reference
# for users, but it can also be run manually.

set -x
set -e

# Run under a virtual environment to isolate Python packages.
# TODO(scotttodd): clean the venv first? `rm -rf`?
# python3 -m venv .venv
# source .venv/bin/activate
# trap deactivate EXIT

# Install general Jupyter notebook requirements.
# python3 -m pip install --upgrade pip
pip3 install --user --quiet jupyter_core nbconvert ipykernel

# Install common notebook requirements.
# TODO(scotttodd): refactor so not all deps are always installed
pip3 install --user --quiet \
  numpy \
  matplotlib \
  tensorflow \
  tensorflow_hub

# Run the notebook, discarding output (still fails if an exception is thrown).
jupyter nbconvert --to notebook --execute $1 --stdout > /dev/null
printf "Notebook exit code: %d\n" $?
