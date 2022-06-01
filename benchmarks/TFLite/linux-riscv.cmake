# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

################################################################################
#                                                                              #
# Default benchmark configurations                                             #
#                                                                              #
# Each suite benchmarks a list of modules with configurations specifying a     #
# target architecture and runtime characteristics (e.g. threads/cores). These  #
# benchmarks only configure IREE translation and runtime flags for the target  #
# architecture and do *not* include any non-default flags. No non-default      #
# flags should be added here.                                                  #
#                                                                              #
################################################################################

set(LINUX_RV64_GENERIC_CPU_TRANSLATION_FLAGS
  "--iree-input-type=tosa"
  "--iree-llvm-target-triple=riscv64"
  "--iree-llvm-target-cpu=generic-rv64"
  "--iree-llvm-target-abi=lp64d"
  "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+v"
  "--riscv-v-vector-bits-min=512"
  "--riscv-v-fixed-length-vector-lmul-max=8"
)

set(LINUX_RV32_GENERIC_CPU_TRANSLATION_FLAGS
  "--iree-input-type=tosa"
  "--iree-llvm-target-triple=riscv32-pc-linux-elf"
  "--iree-llvm-target-cpu=generic-rv32"
  "--iree-llvm-target-abi=ilp32"
  "--iree-llvm-target-cpu-features=+m,+a,+f,+zvl512b,+zve32x"
  "--riscv-v-vector-bits-min=512"
  "--riscv-v-fixed-length-vector-lmul-max=8"
)

# CPU, Dylib-Sync, RV64-Generic, full-inference
iree_benchmark_suite(
  GROUP_NAME
    "linux-riscv"

  MODULES
    "${MOBILEBERT_FP32_MODULE}"

  BENCHMARK_MODES
    "full-inference,default-flags"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-RV64-Generic"
  TRANSLATION_FLAGS
    ${LINUX_RV64_GENERIC_CPU_TRANSLATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "dylib-sync"
)
