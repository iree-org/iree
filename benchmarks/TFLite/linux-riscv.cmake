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

set(LINUX_RISCV64_GENERIC_RV64_CPU_TRANSLATION_FLAGS
  "--iree-input-type=tosa"
  "--iree-llvm-target-triple=riscv64"
  "--iree-llvm-target-cpu=generic-rv64"
  "--iree-llvm-target-abi=lp64d"
  "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+c"
)

# CPU, Dylib-Sync, RISCV64, full-inference
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "dylib-sync,full-inference,default-flags"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-RISCV64-Generic"
  TRANSLATION_FLAGS
    ${LINUX_RISCV64_GENERIC_RV64_CPU_TRANSLATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "dylib-sync"
)

# CPU, Dylib, RISCV64, full-inference
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "dylib,full-inference,default-flags"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-RISCV64-Generic"
  TRANSLATION_FLAGS
    ${LINUX_RISCV64_GENERIC_RV64_CPU_TRANSLATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "dylib"
)
