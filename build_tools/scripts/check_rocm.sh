#!/bin/bash

# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Checks that ROCm is working correctly and logs some useful information

set -xeuo pipefail

if ! dpkg -l | grep rocm; then
  echo "Cannot find installed rocm package"
  exit 1
fi
if ! rocminfo; then
  ret=$?
  echo "rocminfo failed"
  exit "${ret}"
fi

echo ""
echo "=== ROCm Version ==="
cat /opt/rocm/.info/version || echo "version file not found"

echo ""
echo "=== Kernel Version ==="
uname -r

echo ""
echo "=== rocminfo ==="
rocminfo
