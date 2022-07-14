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

set(LINUX_CUDA_SM_70_GPU_COMPILATION_FLAGS
  "--iree-input-type=mhlo"
  "--iree-hal-cuda-llvm-target-arch=sm_70"
)

# GPU, CUDA, SM_70, full-inference
iree_benchmark_suite(
  GROUP_NAME
    "linux-cuda"

  MODULES
    "${MINILM_L12_H384_UNCASED_INT32_MODULE}"

  BENCHMARK_MODES
    "full-inference,default-flags"
  TARGET_BACKEND
    "cuda"
  TARGET_ARCHITECTURE
    "GPU-CUDA-SM_70"
  COMPILATION_FLAGS
    ${LINUX_CUDA_SM_70_GPU_COMPILATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-cuda"
  DRIVER
    "cuda"
)
