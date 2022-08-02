#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Smoke test the cross-compiled RISCV-32 targets.

set -xeuo pipefail

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

# Run the embedded_library module loader and simple_embedding under QEMU.
echo "Test elf_module_test_binary"
pushd "${BUILD_RISCV_DIR}/runtime/src/iree/hal/local/elf" > /dev/null
"${QEMU_RV32_BIN}" -cpu rv32 elf_module_test_binary
popd > /dev/null

echo "Test simple_embedding binaries"
pushd "${BUILD_RISCV_DIR}/samples/simple_embedding" > /dev/null

"${QEMU_RV32_BIN}" -cpu rv32 simple_embedding_embedded_sync

"${QEMU_RV32_BIN}" -cpu rv32 simple_embedding_vmvx_sync

popd > /dev/null
