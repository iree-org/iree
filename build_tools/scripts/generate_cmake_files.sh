#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Run all CMake file generators with proper output paths.

set -euo pipefail

ROOT_DIR=$(git rev-parse --show-toplevel)
cd "${ROOT_DIR}"

./build_tools/testing/generate_cmake_e2e_test_artifacts_suite.py \
  --output_dir ./tests/e2e/test_artifacts

./build_tools/testing/generate_cmake_e2e_model_tests.py \
  --output ./tests/e2e/stablehlo_models/generated_e2e_model_tests.cmake
