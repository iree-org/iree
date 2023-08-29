#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

CMAKE_VERSION="$1"

ARCH="$(uname -m)"

curl --silent --fail --show-error --location \
    "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${ARCH}.sh" \
    --output cmake-installer.sh

chmod +x cmake-installer.sh
./cmake-installer.sh --skip-license --prefix=/usr/
