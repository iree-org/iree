#!/bin/bash

# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs the baremetal_riscv64 sample runners on qemu-system-riscv64
# (-machine virt, no OS, semihosted I/O). The runners are built by
# build_tools/cmake/build_riscv_baremetal.sh.
#
# Environment:
#   QEMU_BIN              qemu-system-riscv64 binary (default: from PATH).
#   IREE_TARGET_BUILD_DIR build from build_riscv_baremetal.sh
#                         (default: build-riscv-baremetal).

set -xeuo pipefail

QEMU_BIN="${QEMU_BIN:-qemu-system-riscv64}"
IREE_TARGET_BUILD_DIR="${IREE_TARGET_BUILD_DIR:-build-riscv-baremetal}"
SAMPLE_DIR="${IREE_TARGET_BUILD_DIR}/samples/baremetal_riscv64"

for runner in runner_llvmcpu runner_vmvx; do
  timeout 120 "${QEMU_BIN}" -machine virt -m 512M -nographic -semihosting \
    -bios none \
    -device loader,file="${SAMPLE_DIR}/${runner}",cpu-num=0 \
    2>&1 | tee "${SAMPLE_DIR}/${runner}.log"
  grep -q "PASS: single-op inference on bare-metal riscv64" \
    "${SAMPLE_DIR}/${runner}.log"
done
