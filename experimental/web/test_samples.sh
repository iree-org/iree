#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -x
set -e

ROOT_DIR=$(git rev-parse --show-toplevel)

# These samples require that all HAL drivers be disabled to avoid linking
# in incompatible system code to the iree_runtime_runtime target. Simply
# setting -DIREE_HAL_DRIVER_DEFAULTS=OFF does not affect existing values, so
# we'll clear the CMake cache just to be safe.
test -f ${ROOT_DIR?}/build-emscripten/CMakeCache.txt && \
  rm ${ROOT_DIR?}/build-emscripten/CMakeCache.txt

${ROOT_DIR?}/experimental/web/sample_static/build_sample.sh
${ROOT_DIR?}/experimental/web/sample_dynamic/build_sample.sh
