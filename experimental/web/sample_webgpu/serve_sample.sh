#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

ROOT_DIR=$(git rev-parse --show-toplevel)
BUILD_DIR=${ROOT_DIR?}/build-emscripten
BINARY_DIR=${BUILD_DIR}/experimental/web/sample_webgpu

echo "=== Running local webserver, open at http://localhost:8000/ ==="

python3 ${ROOT_DIR?}/build_tools/scripts/local_web_server.py --directory ${BINARY_DIR}
