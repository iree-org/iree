# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_check_single_backend_test_suite(
  NAME
    ukernel_mmt4d_riscv_32_test
  SRCS
    "matmul_i8.mlir"
  TARGET_BACKEND
    "llvm-cpu"
  DRIVER
    "local-task"
  DEPENDS
    "${PROJECT_BINARY_DIR}/samples/ukernel_mmt4d_riscv_32/ukernel_bitcode_riscv_32.bc"
  COMPILER_FLAGS
    "--iree-flow-enable-data-tiling"
    "--iree-llvmcpu-enable-microkernels"
    "--iree-llvmcpu-target-triple=riscv32-none-elf"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f"
    "--iree-link-bitcode=${PROJECT_BINARY_DIR}/samples/ukernel_mmt4d_riscv_32/ukernel_bitcode_riscv_32.bc"
)
