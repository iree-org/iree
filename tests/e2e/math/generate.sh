#!/bin/bash

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
    python ${dir}/generate.py \
      --type="${type}" \
      --backend="${backend}" \
      --header="${header}" \
      > "${dir}/math_ops_${type}_${backend}.mlir"
  done
done
