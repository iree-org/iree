#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Initialize a subset of submodules for runtime only.

set -xeuo pipefail

readarray -t RUNTIME_SUBMODULES < \
  "$(dirname $(realpath $0))/runtime_submodules.txt"

git submodule update --init ${RUNTIME_SUBMODULES[@]}
