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
# This is intended for use on continuous integration servers (possibly within a
# Docker container) and as a reference for users, but it can also be run
# locally.

set -e
set -x

# Run under a virtual environment to isolate Python packages.
#
# This is informed by Docker workarounds and we're walking a thin line here.
# --system-site-packages: leverage system packages for common notebook
#     requirements (tensorflow, numpy, and other large packages)
# --clear: delete existing venv contents (package installs within notebooks)
#
# See also:
#   * https://stackoverflow.com/a/63805343
#   * https://askubuntu.com/a/897004
python3 -m venv .notebook.venv --system-site-packages --clear
source .notebook.venv/bin/activate 2> /dev/null
trap "deactivate 2> /dev/null" EXIT

# Update pip within the venv (you'd think this wouldn't be needed, but it is).
python3 -m pip install --quiet --upgrade pip

# Install general Jupyter notebook requirements, ignoring any system versions
# This ensures that the `jupyter` command runs within the venv.
#
# See also:
#   * https://stackoverflow.com/q/42449814
#   * https://stackoverflow.com/a/19459977
python3 -m pip install --ignore-installed --quiet \
  jupyter_core nbconvert ipykernel

# Install common notebook requirements, reusing system versions if possible.
# To match behavior with online Colab notebooks, the versions used here could
# track closely with https://us-docker.pkg.dev/colab-images/public/runtime.
python3 -m pip install --quiet \
  numpy \
  matplotlib \
  tensorflow \
  tensorflow_hub \
  bottleneck

# Tone down TensorFlow's logging by default.
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}

# Run the notebook, discarding output (still fails if an exception is thrown).
jupyter nbconvert --to notebook --execute $1 --stdout > /dev/null
