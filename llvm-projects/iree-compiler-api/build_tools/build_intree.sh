#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Simple script that does a CMake configure and build of the project.
# This is meant for CI's and project maintainers.

set -eu -o errtrace

project_dir="$(cd $(dirname $0)/.. && pwd)"
build_dir="$project_dir/build"

cmake -GNinja -B"$build_dir" "$project_dir" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_BUILD_TYPE=Release \
  "$@"

cd $build_dir
ninja
