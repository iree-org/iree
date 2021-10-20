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

# Environment variable used by the emulator and iree-translate for the
# dylib-llvm-aot bytecode codegen.
export RISCV_TOOLCHAIN_ROOT=${RISCV_RV64_LINUX_TOOLCHAIN_ROOT?}

# Run the binaries under QEMU.
echo "Test simple_embedding binaries"
pushd "${BUILD_RISCV_DIR?}/iree/samples/simple_embedding" > /dev/null

"${QEMU_RV64_BIN?}" -cpu rv64,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0 \
-L "${RISCV_TOOLCHAIN_ROOT?}/sysroot" simple_embedding_dylib

"${QEMU_RV64_BIN?}" -cpu rv64,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0 \
-L "${RISCV_TOOLCHAIN_ROOT?}/sysroot" simple_embedding_embedded_sync

"${QEMU_RV64_BIN?}" -cpu rv64,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0 \
-L "${RISCV_TOOLCHAIN_ROOT?}/sysroot" simple_embedding_vmvx_sync

popd > /dev/null

echo "Test e2e mlir --> bytecode module --> iree-run-module"

"${BUILD_HOST_DIR?}/install/bin/iree-translate" \
  -iree-input-type=mhlo \
  -iree-mlir-to-vm-bytecode-module -iree-hal-target-backends=dylib-llvm-aot \
  -iree-llvm-link-embedded=false \
  -iree-llvm-target-triple=riscv64 \
  -iree-llvm-target-cpu=generic-rv64 \
  -iree-llvm-target-cpu-features="+m,+a,+f,+d,+c" \
  -iree-llvm-target-abi=lp64d \
  "${ROOT_DIR?}/iree/tools/test/iree-run-module.mlir" \
  -o "${BUILD_RISCV_DIR?}/iree-run-module-llvm_aot.vmfb"

IREE_RUN_OUT=$(${QEMU_RV64_BIN?} -cpu rv64,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0 \
    -L "${RISCV_TOOLCHAIN_ROOT?}/sysroot" \
    "${BUILD_RISCV_DIR?}/iree/tools/iree-run-module" --driver=dylib \
    --module_file="${BUILD_RISCV_DIR?}/iree-run-module-llvm_aot.vmfb" \
    --entry_function=abs --function_input="f32=-10")

# Check the result of running abs(-10).
if [[ "${IREE_RUN_OUT}" != *"f32=10" ]]; then
    exit 1
fi
