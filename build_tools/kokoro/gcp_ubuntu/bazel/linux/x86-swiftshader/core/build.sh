#!/bin/bash

# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# For use within a IREE bazel docker image on a Kokoro VM.
# Log some information about the environment, initialize the submodules and then
# run the bazel core tests.

set -e
set -x

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

# Check these exist and print the versions for later debugging
bazel --version
"${CXX?}" --version
"${CC?}" --version

echo "Initializing submodules"
git submodule update --init --jobs 8 --depth 1

echo "Building and testing with bazel"
./build_tools/bazel/build_core.sh
