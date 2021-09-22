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
workspace_dir="$project_dir/../.."
build_dir="$project_dir/build"

# Write out a .env file to the workspace.
echo "PYTHONPATH=$build_dir/python_package:$build_dir/iree/bindings/python" > $workspace_dir/.env

cmake -GNinja -B"$build_dir" "$project_dir" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  "$@"

cd $build_dir
ninja all iree/bindings/python/all
