#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Builds and tests IREE samples for CI.
#
# The build directory for the emscripten build is taken from the environment
# variable IREE_EMPSCRIPTEN_BUILD_DIR, defaulting to "build-emscripten".
# Designed for CI, but can be run manually.

set -xeuo pipefail

HOST_TOOLS_BINARY_DIR="$1"
export IREE_EMPSCRIPTEN_BUILD_DIR="${IREE_EMPSCRIPTEN_BUILD_DIR:-build-emscripten}"

# These samples require that all HAL drivers be disabled to avoid linking
# in incompatible system code to the iree_runtime_runtime target. Simply
# setting -DIREE_HAL_DRIVER_DEFAULTS=OFF does not affect existing values, so
# we'll clear the CMake cache just to be safe.
test -f "${IREE_EMPSCRIPTEN_BUILD_DIR}/CMakeCache.txt" \
  && rm "${IREE_EMPSCRIPTEN_BUILD_DIR}/CMakeCache.txt"

experimental/web/sample_static/build_sample.sh ${HOST_TOOLS_BINARY_DIR}
experimental/web/sample_dynamic/build_sample.sh ${HOST_TOOLS_BINARY_DIR}

# TODO(scotttodd): re-enable once release packages include webgpu-spirv compiler target

# # Clear the cache again before building the webgpu sample.
# test -f "${IREE_EMPSCRIPTEN_BUILD_DIR}/CMakeCache.txt" \
#   && rm "${IREE_EMPSCRIPTEN_BUILD_DIR}/CMakeCache.txt"
# experimental/web/sample_webgpu/build_sample.sh ${HOST_TOOLS_BINARY_DIR}
