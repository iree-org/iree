#!/bin/bash

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Test the cross-compiled Android targets.
#
# The desired build directory can be passed as the first argument. Otherwise, it
# uses the environment variable IREE_TARGET_BUILD_DIR, defaulting to
# "build-android". The variable LABEL_EXCLUDE can be passed to ctest to skip
# unsupported tests. Designed for CI, but can be run manually. Expects to be run
# from the root of the IREE repository.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_TARGET_BUILD_DIR:-build-android}}"
LABEL_EXCLUDE="${IREE_LABEL_EXCLUDE:-}"

ctest -j 4 \
  --test-dir "${BUILD_DIR}" \
  --timeout=900 \
  --output-on-failure \
  --no-tests=error \
  --label-exclude "${LABEL_EXCLUDE}"
