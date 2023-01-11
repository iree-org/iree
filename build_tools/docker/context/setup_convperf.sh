#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Sets up `convperf` (https://github.com/nod-ai/convperf).
#
# `convperf` benchmarks convolution workloads on IREE and other backends such
# as libxsmm. IREE is included as a submodule.
#
# Usage:
#    ./setup_convperf.sh \
#        <convperf repo dir> \
#        <convperf sha>

set -xeuo pipefail

export REPO_DIR="$1"
export REPO_SHA="$2"

pushd "${REPO_DIR}"

mkdir convperf
pushd convperf
git init
git fetch --depth 1 https://github.com/nod-ai/convperf.git "${REPO_SHA}"
git checkout "${REPO_SHA}"
git submodule update --init --recursive --jobs 8 --depth 1

# Checkout a specific commit.
git checkout "${REPO_SHA}"

# Create virtual environment.
python3 -m venv convperf.venv
source convperf.venv/bin/activate
pip install -r requirements.txt

# Since the root user clones the convperf repo, we update permissions so that a
# runner can access this repo, but we don't want to set the executable bit for
# non-executables because git tracks this, so we then restore any git-tracked
# changes.
chmod -R 777 .
git restore .
git submodule foreach --recursive git restore .

popd # convperf
popd # "${REPO_DIR}"
