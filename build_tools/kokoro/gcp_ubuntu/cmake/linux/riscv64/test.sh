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


function generate_dylib_vmfb {
  local target="${1}"; shift
  local translate_arg=(
    -iree-mlir-to-vm-bytecode-module -iree-hal-target-backends=dylib-llvm-aot
    -iree-llvm-embedded-linker-path=${BUILD_HOST_DIR?}/install/bin/lld
    -iree-llvm-target-triple=riscv64
    -iree-llvm-target-cpu=generic-rv64
    -iree-llvm-target-abi=lp64d
  )
  if [[ "${target}" == "mhlo" ]]; then
    translate_arg+=(
      -iree-input-type=mhlo
      -iree-llvm-target-cpu-features="+m,+a,+f,+d,+c"
    )
  fi
  "${BUILD_HOST_DIR?}/install/bin/iree-translate" "${translate_arg[@]}" "$@"
}

generate_dylib_vmfb mhlo \
  "${ROOT_DIR?}/iree/tools/test/iree-run-module.mlir" \
  -o "${BUILD_RISCV_DIR?}/iree-run-module-llvm_aot.vmfb"

${PYTHON_BIN?} "${ROOT_DIR?}/third_party/llvm-project/llvm/utils/lit/lit.py" \
  -v "${ROOT_DIR?}/build_tools/kokoro/gcp_ubuntu/cmake/linux/riscv64/"
