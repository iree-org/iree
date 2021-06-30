#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Test the cross-compiled RISCV targets using Kokoro.

set -e
set -x

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

# Docker image has the QEMU installed at /usr/src/qemu-riscv.
# Run the embedded_library module loader and simple_embedding under QEMU.

echo "Test elf_module_test_binary"
pushd "${BUILD_RISCV_DIR?}/iree/hal/local/elf" > /dev/null
/usr/src/qemu-riscv/qemu-riscv32 -cpu rv32,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0 \
elf_module_test_binary
popd > /dev/null

echo "Test simple_embedding binaries"
pushd "${BUILD_RISCV_DIR?}/iree/samples/simple_embedding" > /dev/null

/usr/src/qemu-riscv/qemu-riscv32 -cpu rv32,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0 \
simple_embedding_embedded_sync

/usr/src/qemu-riscv/qemu-riscv32 -cpu rv32,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0 \
simple_embedding_vmvx_sync

popd > /dev/null
