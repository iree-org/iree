#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Wrapper script to run the artifact on RISC-V Linux device.
# This script checks if QEMU emulator is set, and use either the emulator or
# the actual device to run the cross-compiled RISC-V linux artifacts.

set -x
set -e

# A QEMU Linux emulator must be available Within the system that matches the
# processor architecturue.
if [[ ! -z "${QEMU_BIN}" ]] && [[ ! -z "${QEMU_CPU_FLAGS}" ]]; then
  "${QEMU_BIN}" "-cpu" "${QEMU_CPU_FLAGS}" "$@"
else
# TODO(dcaballe): Add on-device run commands.
  "$@"
fi
