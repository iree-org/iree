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
# This is intended for use on continuous integration servers within a Docker
# container and as a reference for users, but it can also be run locally.

set -x
set -e

# Run under a virtual environment to isolate Python packages.
#
# Note: --without-pip and --system-site-packages are workarounds for Docker
#  * https://stackoverflow.com/a/63805343
#  * https://askubuntu.com/a/897004
#
# Note: --clear deletes existing venv contents, which helps make these tests
# more hermetic, at the cost of time spent performing redundant installs.
# TODO(scotttodd): evaluate if we can do without --clear
python3 -m venv .venv --without-pip --system-site-packages --clear
source .venv/bin/activate
trap deactivate EXIT

# Install general Jupyter notebook requirements.
python3 -m pip install --quiet jupyter_core nbconvert ipykernel

# Install common notebook requirements.
# TODO(scotttodd): refactor so not all deps are always installed
python3 -m pip install --quiet \
  numpy \
  matplotlib \
  tensorflow \
  tensorflow_hub \
  bottleneck

# https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
python3 -m pip install --quiet \
  --index-url https://google-coral.github.io/py-repo/ tflite_runtime

# Tone down TensorFlow's logging by default.
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}

# Run the notebook, discarding output (still fails if an exception is thrown).
jupyter nbconvert --to notebook --execute $1 --stdout > /dev/null
