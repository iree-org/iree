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

# Build `iree-doc` and `iree-dialects-doc` CMake targets. This requires the LLVM
# submodule and can take several minutes with an empty cache, as it builds
# `iree-tblgen` and `mlir-tblgen`.
cmake -G Ninja \
  -B "${BUILD_DIR}" "${REPO_ROOT}" \
  -DIREE_BUILD_DOCS=ON
cmake --build "${BUILD_DIR}" --target iree-doc
cmake --build "${BUILD_DIR}" --target iree-dialects-doc

if (( IREE_USE_CCACHE == 1 )); then
  ccache --show-stats
fi

# Copy into a new directory before making edits, so CMake only runs when needed.
BUILD_DIALECTS_ORIGINAL_DIR="${BUILD_DIR}/doc/Dialects"
BUILD_EXTERNAL_DIALECTS_ORIGINAL_DIR="${BUILD_DIR}/llvm-external-projects/mlir-iree-dialects/docs/Dialects"
BUILD_DIALECTS_PROCESSED_DIR="${BUILD_DIR}/doc/Dialects_for_website"
mkdir -p "${BUILD_DIALECTS_PROCESSED_DIR}"
cp -r "${BUILD_DIALECTS_ORIGINAL_DIR}/." "${BUILD_DIALECTS_PROCESSED_DIR}"
cp -r "${BUILD_EXTERNAL_DIALECTS_ORIGINAL_DIR}/." "${BUILD_DIALECTS_PROCESSED_DIR}"

BUILD_PASSES_ORIGINAL_DIR="${BUILD_DIR}/doc/Passes"
BUILD_PASSES_PROCESSED_DIR="${BUILD_DIR}/doc/Passes_for_website"
mkdir -p "${BUILD_PASSES_PROCESSED_DIR}"
cp -r "${BUILD_PASSES_ORIGINAL_DIR}/." "${BUILD_PASSES_PROCESSED_DIR}"

# Delete any dialect docs we don't want to publish (yet?).
rm "${BUILD_DIALECTS_PROCESSED_DIR}/SimpleIODialect.md" # Sample dialect, just ignore
rm "${BUILD_DIALECTS_PROCESSED_DIR}/StructuredTransformOpsExt.md" # Dialect extensions

# Trim "Dialect"/"Passes" suffix from file names e.g. FlowDialect.md -> Flow.md.
for f in ${BUILD_DIALECTS_PROCESSED_DIR}/*Dialect.md; do
  mv "$f" "${f/%Dialect.md/.md}"
done
for f in ${BUILD_PASSES_PROCESSED_DIR}/*Passes.md; do
  mv "$f" "${f/%Passes.md/.md}"
done

# Postprocess the generated docs (e.g. making tweaks to the markdown source).
python3 "${THIS_DIR}/postprocess_dialect_docs.py" "${BUILD_DIALECTS_PROCESSED_DIR}"
python3 "${THIS_DIR}/postprocess_passes_docs.py" "${BUILD_PASSES_PROCESSED_DIR}"

# Copy from build directory -> source directory (files are .gitignore'd).
cp -r \
  "${BUILD_DIALECTS_PROCESSED_DIR}/." \
  "${THIS_DIR}/docs/reference/mlir-dialects/"
cp -r \
  "${BUILD_PASSES_PROCESSED_DIR}/." \
  "${THIS_DIR}/docs/reference/mlir-passes/"
