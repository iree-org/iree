#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Checks that CUDA is working correctly and logs some useful information

set -xeuo pipefail

if ! dpkg -l | grep nvidia; then
  echo "Cannot find installed nvidia package"
  exit 1
fi
if ! nvidia-smi; then
  ret=$?
  echo "nvidia-smi failed"
  exit "${ret}"
fi
