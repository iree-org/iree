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
# Run the simple_embedding binaries under QEMU.

echo "Test simple_embedding binaries"
pushd "${BUILD_RISCV_DIR?}/iree/samples/simple_embedding" > /dev/null

/usr/src/qemu-riscv/qemu-riscv64 -cpu rv64,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0 \
-L "${RISCV_TOOLCHAIN_ROOT?}/sysroot" simple_embedding_dylib

/usr/src/qemu-riscv/qemu-riscv64 -cpu rv64,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0 \
-L "${RISCV_TOOLCHAIN_ROOT?}/sysroot" simple_embedding_local_sync

popd > /dev/null

echo "Test e2e mlir --> bytecode module --> iree-run-module"

"${BUILD_HOST_DIR?}/install/bin/iree-translate" \
  -iree-input-type=mhlo \
  -iree-mlir-to-vm-bytecode-module -iree-hal-target-backends=dylib-llvm-aot \
  -iree-llvm-target-triple=riscv64 \
  -iree-llvm-target-cpu=generic-rv64 \
  -iree-llvm-target-cpu-features="+m,+a,+f,+d,+c" \
  -iree-llvm-target-abi=lp64d \
  "${ROOT_DIR?}/iree/tools/test/iree-run-module.mlir" \
  -o "${BUILD_RISCV_DIR?}/iree-run-module-llvm_aot.vmfb"

IREE_RUN_OUT=$(/usr/src/qemu-riscv/qemu-riscv64 -cpu rv64,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0 \
    -L "${RISCV_TOOLCHAIN_ROOT?}/sysroot" \
    "${BUILD_RISCV_DIR?}/iree/tools/iree-run-module" --driver=dylib \
    --module_file="${BUILD_RISCV_DIR?}/iree-run-module-llvm_aot.vmfb" \
    --entry_function=abs --function_input="i32=-10")

# Check the result of running abs(-10).
if [[ "${IREE_RUN_OUT}" != *"i32=10" ]]; then
    exit 1
fi
