#!/bin/bash

# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build the IREE project with CMake. Designed for CI, but can be run manually.
# This is equivalent to the build_tools/cmake/rebuild.sh script except it
# first deletes the build directory to deal with any faulty caching.

set -x
set -e

ROOT_DIR=$(git rev-parse --show-toplevel)

cd ${ROOT_DIR?}
rm -rf build/
./build_tools/cmake/rebuild.sh "$@"
