#!/bin/bash

# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -x
set -e

# Run the test under QEMU if `IREE_ARM_SME_QEMU_AARCH64_BIN` is set.
# It is assumed that if `IREE_ARM_SME_QEMU_AARCH64_BIN` is set then IREE has
# been built for AArch64.
if [[ ! -z "${IREE_ARM_SME_QEMU_AARCH64_BIN}" ]]; then
  "${IREE_ARM_SME_QEMU_AARCH64_BIN}" "-cpu" "max,sme512=on,sme_fa64=off" "--" "$@"
else
  "$@"
fi
