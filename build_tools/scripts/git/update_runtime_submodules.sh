#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Initialize a subset of submodules for runtime only.

set -xeuo pipefail

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")"

readarray -t RUNTIME_SUBMODULES < "${SCRIPT_DIR}/runtime_submodules.txt"

git submodule sync && git submodule update --init ${RUNTIME_SUBMODULES[@]}
