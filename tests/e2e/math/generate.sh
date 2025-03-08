#!/bin/bash

# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Run this script, without arguments, to regenerate the .mlir test files.

set -eu

dir="$(dirname "$0")"

backends=(
  "llvm-cpu"
  "rocm"
)

float_types=(
  "f16"
  "f32"
)

header="// Generated file, do not edit!
// To regenerate, run: $(basename "$0")"

for backend in ${backends[@]}; do
  for type in ${float_types[@]}; do
    python3 ${dir}/generate.py \
      --type="${type}" \
      --backend="${backend}" \
      --header="${header}" \
      > "${dir}/math_ops_${type}_${backend}.mlir"
  done
done
