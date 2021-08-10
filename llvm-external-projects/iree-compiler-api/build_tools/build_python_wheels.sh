#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -e

if [ -z "$PYTHON" ]; then
  PYTHON="$(which python)"
fi
version="$("$PYTHON" --version)"
echo "Using python: $PYTHON (version $version)"

repo_root="$(cd $(dirname $0)/.. && pwd)"
wheelhouse="$repo_root/wheels"
mkdir -p "$wheelhouse"

echo "---- BUILDING iree-compiler-api ----"
if [ -x "$(command -v ccache)" ]; then
  echo "Using ccache"
  export CMAKE_C_COMPILER_LAUNCHER=ccache
  export CMAKE_CXX_COMPILER_LAUNCHER=ccache
fi
if [ -x "$(command -v ninja)" ]; then
  echo "Using ninja"
  export CMAKE_GENERATOR=Ninja
fi
$PYTHON -m pip wheel "${repo_root}" \
  --use-feature=in-tree-build \
  -w "$wheelhouse" -v

echo "---- INSTALLING iree-compiler-api ----"
$PYTHON -m pip install -f "$wheelhouse" --force-reinstall iree-compiler-api

echo "---- QUICK SMOKE TEST ----"
$PYTHON $repo_root/build_tools/smoketest.py
