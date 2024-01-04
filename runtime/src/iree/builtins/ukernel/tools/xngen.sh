#!/bin/bash

# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build the IREE project with CMake. Designed for CI, but can be run manually.
# This is equivalent to the build_tools/cmake/rebuild.sh script except it
# first deletes the build directory to deal with any faulty caching.

set -euo pipefail

IREE_ROOT="$(realpath "$(git rev-parse --show-toplevel)" --relative-to="$PWD")"

UKERNEL_ROOT="${IREE_ROOT}/runtime/src/iree/builtins/ukernel"

shopt -s globstar

for output in "${UKERNEL_ROOT}"/**/gen/*; do
    output_basename="$(basename "${output}")"
    output_dirname="$(dirname "${output}")"
    output_parent_dir="$(dirname "${output_dirname}")"
    input="${output_parent_dir}/${output_basename}.in"
    echo "Generating ${output} from ${input}"
    if [[ ! -f "${input}" ]]; then
        echo "Error: ${input} does not exist."
        exit 1
    fi
    "${IREE_ROOT}/third_party/XNNPACK/tools/xngen.py" "${input}" -o "${output}"
done
