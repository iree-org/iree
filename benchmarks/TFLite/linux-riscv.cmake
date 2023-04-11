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
# benchmarks only configure IREE compilation and runtime flags for the target  #
# architecture and do *not* include any non-default flags. No non-default      #
# flags should be added here.                                                  #
#                                                                              #
################################################################################

set(LINUX_RV64_GENERIC_CPU_COMPILATION_FLAGS
  "--iree-input-type=tosa"
  "--iree-llvmcpu-target-triple=riscv64"
  "--iree-llvmcpu-target-cpu=generic-rv64"
  "--iree-llvmcpu-target-abi=lp64d"
  "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
  "--riscv-v-fixed-length-vector-lmul-max=8"
)

# TODO(llvm-project/60463): Replace 'zve32f' with 'zve32x'.
set(LINUX_RV32_GENERIC_CPU_COMPILATION_FLAGS
  "--iree-input-type=tosa"
  "--iree-llvmcpu-target-triple=riscv32-pc-linux-elf"
  "--iree-llvmcpu-target-cpu=generic-rv32"
  "--iree-llvmcpu-target-abi=ilp32"
  "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f"
  "--riscv-v-fixed-length-vector-lmul-max=8"
)

# CPU, LLVM, local-sync, RV64-Generic, full-inference
iree_benchmark_suite(
  GROUP_NAME
    "linux-riscv"

  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V1_MODULE}"
    "${MOBILEBERT_INT8_MODULE}"
    "${PERSON_DETECT_INT8_MODULE}"
    "${EFFICIENTNET_INT8_MODULE}"
    "${MOBILENET_V2_INT8_MODULE}"
  BENCHMARK_MODES
    "full-inference,default-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-RV64-Generic"
  COMPILATION_FLAGS
    ${LINUX_RV64_GENERIC_CPU_COMPILATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu-sync"
  DRIVER
    "local-sync"
)

# CPU, LLVM, local-sync, RV32-Generic, full-inference
# Note this target is for codegen only. Inference is only possible with
# the cross-compiled runtime and an emulator.
iree_benchmark_suite(
  GROUP_NAME
    "linux-riscv"

  MODULES
    "${PERSON_DETECT_INT8_MODULE}"
    "${EFFICIENTNET_INT8_MODULE}"
    "${MOBILENET_V2_INT8_MODULE}"
  BENCHMARK_MODES
    "full-inference,default-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-RV32-Generic"
  COMPILATION_FLAGS
    ${LINUX_RV32_GENERIC_CPU_COMPILATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu-sync"
  DRIVER
    "local-sync"
)
