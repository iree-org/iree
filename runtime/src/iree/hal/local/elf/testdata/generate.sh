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
# To use, ensure iree-compile and your compiled ld.lld are on your PATH and
# run the script:
#   $ ./runtime/src/iree/hal/local/elf/testdata/generate.sh

# Uncomment to see the iree-compile commands issued:
# set -x
set -e

ROOT_DIR=$(git rev-parse --show-toplevel)
TESTDATA="${ROOT_DIR}/runtime/src/iree/hal/local/elf/testdata"

# $1: file name ("foo_arm_32.so")
# $2: list of iree-compile arguments for targeting
function compile_and_extract_library() {
  local so_name=$1
  shift
  local compile_args=("$@")

  echo "Updating ${TESTDATA}/${so_name}"

  CMD=(
    iree-compile
      --compile-mode=hal-executable
      ${TESTDATA}/elementwise_mul.mlir
      -o="${TESTDATA}/${so_name}"

      --iree-hal-target-backends=llvm-cpu
      --iree-llvmcpu-debug-symbols=false

      "${compile_args[@]}"
  )
  "${CMD[@]}"
}

ARM_32=(
  --iree-llvmcpu-target-triple=armv7a-pc-linux-elf
  --iree-llvmcpu-target-float-abi=hard
)
compile_and_extract_library "elementwise_mul_arm_32.so" ${ARM_32[@]}

ARM_64=(
  --iree-llvmcpu-target-triple=aarch64-pc-linux-elf
)
compile_and_extract_library "elementwise_mul_arm_64.so" ${ARM_64[@]}

RISCV_32=(
  --iree-llvmcpu-target-triple=riscv32-pc-linux-elf
  --iree-llvmcpu-target-cpu=generic-rv32
  --iree-llvmcpu-target-cpu-features=+m,+f
  --iree-llvmcpu-target-abi=ilp32
)
compile_and_extract_library "elementwise_mul_riscv_32.so" ${RISCV_32[@]}

RISCV_64=(
  --iree-llvmcpu-target-triple=riscv64-pc-linux-elf
  --iree-llvmcpu-target-cpu=generic-rv64
  --iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c
  --iree-llvmcpu-target-abi=lp64d
)
compile_and_extract_library "elementwise_mul_riscv_64.so" ${RISCV_64[@]}

X86_32=(
  --iree-llvmcpu-target-triple=i686-pc-linux-elf
)
compile_and_extract_library "elementwise_mul_x86_32.so" ${X86_32[@]}

X86_64=(
  --iree-llvmcpu-target-triple=x86_64-pc-linux-elf
)
compile_and_extract_library "elementwise_mul_x86_64.so" ${X86_64[@]}
