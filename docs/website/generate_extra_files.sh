#!/usr/bin/env bash

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Generates documentation files that require extra steps outside of mkdocs.
#
# Typical usage:
#   docs/website$ ./generate_extra_files.sh
#   docs/website$ ./mkdocs serve
#
# This script should be re-run whenever the generated files change, e.g. when
# updating MLIR dialect .td files.

set -xeuo pipefail

THIS_DIR="$(cd $(dirname $0) && pwd)"
REPO_ROOT="$(cd $THIS_DIR/../../ && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-docs}"

source ${REPO_ROOT}/build_tools/cmake/setup_build.sh
source ${REPO_ROOT}/build_tools/cmake/setup_ccache.sh

# Build `iree-doc` CMake target. This requires the LLVM submodule and can take
# several minutes, as it builds `iree-tblgen`.
cmake -G Ninja \
  -B "${BUILD_DIR}" "${REPO_ROOT}" \
  -DIREE_BUILD_DOCS=ON
cmake --build "${BUILD_DIR}" --target iree-doc

if (( IREE_USE_CCACHE == 1 )); then
  ccache --show-stats
fi

# Copy from build directory -> source directory (files are .gitignore'd).
cp -r \
  "${BUILD_DIR}/doc/Dialects/." \
  "${THIS_DIR}/docs/reference/mlir-dialects/"

# Delete sample dialects.
rm "${THIS_DIR}/docs/reference/mlir-dialects/SimpleIODialect.md"

# Trim "Dialect" suffix from file names, e.g. FlowDialect.md -> Flow.md.
for f in ${THIS_DIR}/docs/reference/mlir-dialects/*Dialect.md; do
  mv "$f" "${f/%Dialect.md/.md}"
done

# Note: any post-processing on the .md files could go here.
