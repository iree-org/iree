#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Simple script that does a CMake configure of this project as an external
# LLVM project so it can be tested in isolation to larger assemblies.
# This is meant for CI's and project maintainers.

set -eu -o errtrace

project_dir="$(cd $(dirname $0)/.. && pwd)"
repo_root="$(cd "$project_dir"/../.. && pwd)"
llvm_project_dir="$repo_root/third_party/llvm-project"
build_dir="$project_dir/build"

cmake -GNinja -B"$build_dir" "$llvm_project_dir/llvm" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=iree-dialects \
  -DLLVM_EXTERNAL_IREE_DIALECTS_SOURCE_DIR="$project_dir" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON

cd "$build_dir"
ninja tools/iree-dialects/all
