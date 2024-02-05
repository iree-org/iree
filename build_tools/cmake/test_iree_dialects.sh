#!/bin/bash

# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Run the check-iree-dialects tests under llvm-external-projects..

set -euo pipefail

BUILD_DIR="$1"

echo "*** Running llvm-external-projects tests ***"
echo "------------------"
cmake --build ${BUILD_DIR} --target check-iree-dialects -- -k 0
