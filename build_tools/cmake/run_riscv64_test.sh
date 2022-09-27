#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Wrapper script to run the artifact on RISC-V 64-bit Linux device.
# This script checks if QEMU emulator is set, and use either the emulator or
# the actual device to run the cross-compiled RISC-V 64-bit linux artifacts.

set -x
set -e

# A QEMU 64 Linux emulator must be available at the path specified by the
# `QEMU_RV64_BIN` environment variable to run the artifacts under the emulator.
if [[ ! -z "${QEMU_RV64_BIN}" ]]; then
  "${QEMU_RV64_BIN}" "-cpu" "rv64,x-v=true,x-k=true,vlen=512,elen=64,vext_spec=v1.0" \
  "-L" "${RISCV_RV64_LINUX_TOOLCHAIN_ROOT}/sysroot" $*
fi

# TODO(dcaballe): Add on-device run commands.
