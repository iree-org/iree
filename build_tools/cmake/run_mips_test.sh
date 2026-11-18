#!/bin/bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Wrapper to run a cross-compiled MIPS artifact, under QEMU when available.

set -x
set -e

if [[ ! -z "${QEMU_BIN}" ]] && [[ ! -z "${QEMU_CPU_FLAGS}" ]]; then
  "${QEMU_BIN}" "-cpu" "${QEMU_CPU_FLAGS}" "$@"
else
  "$@"
fi
