#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Smoke test the cross-compiled RISCV-32 targets.
#
# The build directory containing tests can be passed as the first argument.
# Otherwise, it uses the environment variable IREE_TARGET_BUILD_DIR, defaulting
# to "build-riscv". Designed for CI, but can be run manually.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_TARGET_BUILD_DIR:-build-riscv}}"

# Run the embedded_library module loader and simple_embedding under QEMU.
echo "Test elf_module_test_binary"
pushd "${BUILD_DIR}/runtime/src/iree/hal/local/elf" > /dev/null
# Need to run qemu without memory protection for embedded elf (code in .data)
"${QEMU_RV32_V5_2_BIN}" -cpu rv32 elf_module_test_binary
popd > /dev/null

echo "Test simple_embedding binaries"
pushd "${BUILD_DIR}/samples/simple_embedding" > /dev/null

"${QEMU_RV32_V5_2_BIN}" -cpu rv32 simple_embedding_embedded_sync

"${QEMU_RV32_V5_2_BIN}" -cpu rv32 simple_embedding_vmvx_sync

popd > /dev/null
