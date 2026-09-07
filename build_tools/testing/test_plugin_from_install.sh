#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Proves an out-of-tree plugin can be built against an IREE install tree and
# loaded, which is the point of exporting the plugin rules at all. Nothing here
# reads the IREE sources except to install them.
#
# Usage: test_plugin_from_install.sh <iree-build-dir>

set -euo pipefail

BUILD_DIR="${1:-build}"
if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  echo "error: '${BUILD_DIR}' is not a configured IREE build directory" >&2
  exit 1
fi
BUILD_DIR="$(cd "${BUILD_DIR}" && pwd)"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if ! grep -q "^IREE_COMPILER_DYNAMIC_PLUGINS:BOOL=ON" \
     "${BUILD_DIR}/CMakeCache.txt"; then
  echo "error: ${BUILD_DIR} was configured without" \
       "-DIREE_COMPILER_DYNAMIC_PLUGINS=ON" >&2
  exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
PREFIX="${WORK_DIR}/install"
PLUGIN_BUILD="${WORK_DIR}/plugin"

echo "--- Installing IREE to ${PREFIX}"
for component in IREECMakeExports IREEDevLibraries-Compiler Compiler; do
  cmake --install "${BUILD_DIR}" --prefix "${PREFIX}" \
    --component "${component}" >/dev/null
done

# The install tree must carry everything IREE-side a plugin needs.
for required in \
    "lib/cmake/IREE/IREECompilerConfig.cmake" \
    "lib/cmake/IREE/IREECompilerPluginRules.cmake" \
    "lib/cmake/IREE/gen_rename_map.py" \
    "include/iree/compiler/PluginAPI/PluginEntryPoint.h" \
    "include/iree/compiler/PluginAPI/PluginABIHash.h" \
    "include/iree/compiler/PluginAPI/Client.h" ; do
  if [[ ! -f "${PREFIX}/${required}" ]]; then
    echo "error: install tree is missing ${required}" >&2
    exit 1
  fi
done

echo "--- Building the plugin against the install tree"
# LLVM and MLIR come from the build tree: IREE installs no llvm/mlir headers,
# and a plugin must compile against the ones the compiler was built from.
cmake -G Ninja \
  -S "${REPO_DIR}/build_tools/testing/plugin_from_install" \
  -B "${PLUGIN_BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DIREECompiler_DIR="${PREFIX}/lib/cmake/IREE" \
  -DMLIR_DIR="${BUILD_DIR}/lib/cmake/mlir" \
  -DLLVM_DIR="${BUILD_DIR}/llvm-project/lib/cmake/llvm"
cmake --build "${PLUGIN_BUILD}"

PLUGIN="${PLUGIN_BUILD}/libiree_compiler_plugin_install_tree_probe.so"
if [[ ! -f "${PLUGIN}" ]]; then
  echo "error: the plugin was not built at ${PLUGIN}" >&2
  exit 1
fi

echo "--- Loading it into iree-compile"
OUTPUT="$("${BUILD_DIR}/tools/iree-compile" \
  "--iree-load-plugin=${PLUGIN}" --help 2>&1 || true)"
if ! grep -q "INSTALL_TREE_PLUGIN: renamed MLIRContext ok" <<<"${OUTPUT}"; then
  echo "error: the plugin did not register. Output was:" >&2
  echo "${OUTPUT}" >&2
  exit 1
fi

echo "PASS: a plugin built against the install tree loaded"
