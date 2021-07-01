#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Updates the checked-in ELF files used for testing the ELF loader.
# In general we try not to check in binary files however these files act as a
# test of binary compatibility for artifacts users may have produced. If a
# build break occurs here we know that we have broken compatibility. Today this
# happens every few months as we are not yet binary-stable but in the future
# will be a bigger issue.
#
# To use, ensure iree-translate and 7z are on your PATH and run the script:
#   $ ./iree/hal/local/elf/testdata/generate.sh

# Uncomment to see the iree-translate commands issued:
# set -x
set -e

ROOT_DIR=$(git rev-parse --show-toplevel)

# $1: file name ("foo_arm_32.so")
# $2: list of iree-translate arguments for targeting
function compile_and_extract_library() {
  local so_name=$1
  shift
  local translate_args=("$@")

  echo "Updating ${ROOT_DIR}/iree/hal/local/elf/testdata/${so_name}"

  # Build a vmfb with the executable we want.
  CMD=(
    iree-translate
      -iree-mlir-to-vm-bytecode-module
      -iree-input-type=mhlo
      ${ROOT_DIR}/iree/samples/simple_embedding/simple_embedding_test.mlir
      -o=simple_embedding_test_dylib_llvm_scratch.vmfb

      -iree-hal-target-backends=dylib-llvm-aot
      -iree-llvm-link-embedded=true
      -iree-llvm-debug-symbols=false

      "${translate_args[@]}"
  )
  "${CMD[@]}"

  # Unzip ELF files from the vmfb.
  # Note that `unzip` can't handle these files so we have to use 7zip.
  7z e -aoa -bb0 simple_embedding_test_dylib_llvm_scratch.vmfb -y >/dev/null || true

  # Move the exacted file to the location in-tree.
  mv \
    _simple_mul_dispatch_0_llvm_binary_ex_elf.so \
    "${ROOT_DIR}/iree/hal/local/elf/testdata/${so_name}"

  rm simple_embedding_test_dylib_llvm_scratch.vmfb
}

ARM_32=(
  -iree-llvm-target-triple=armv7a-pc-linux-elf
  -iree-llvm-target-float-abi=hard
)
compile_and_extract_library "simple_mul_dispatch_arm_32.so" ${ARM_32[@]}

ARM_64=(
  -iree-llvm-target-triple=aarch64-pc-linux-elf
)
compile_and_extract_library "simple_mul_dispatch_arm_64.so" ${ARM_64[@]}

RISCV_32=(
  -iree-llvm-target-triple=riscv32-pc-linux-elf
  -iree-llvm-target-cpu=generic-rv32
  -iree-llvm-target-cpu-features=+m,+a,+c,+f
  -iree-llvm-target-abi=ilp32f
  -iree-llvm-target-float-abi=hard
)
compile_and_extract_library "simple_mul_dispatch_riscv_32.so" ${RISCV_32[@]}

RISCV_64=(
  -iree-llvm-target-triple=riscv64-pc-linux-elf
  -iree-llvm-target-cpu=generic-rv64
  -iree-llvm-target-cpu-features=+m,+a,+f,+d,+c
  -iree-llvm-target-abi=lp64d
)
compile_and_extract_library "simple_mul_dispatch_riscv_64.so" ${RISCV_64[@]}

X86_32=(
  -iree-llvm-target-triple=i686-pc-linux-elf
)
compile_and_extract_library "simple_mul_dispatch_x86_32.so" ${X86_32[@]}

X86_64=(
  -iree-llvm-target-triple=x86_64-pc-linux-elf
)
compile_and_extract_library "simple_mul_dispatch_x86_64.so" ${X86_64[@]}
