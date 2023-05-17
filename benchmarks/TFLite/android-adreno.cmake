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

set(ANDROID_ADRENO_GPU_COMPILATION_FLAGS
  "--iree-input-type=tosa"
  "--iree-vulkan-target-triple=adreno-unknown-android31"
)

# GPU, Vulkan, Adreno, full-inference
iree_benchmark_suite(
  GROUP_NAME
    "android-adreno"

  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "full-inference,default-flags"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Adreno"
  COMPILATION_FLAGS
    ${ANDROID_ADRENO_GPU_COMPILATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-vulkan"
  DRIVER
    "vulkan"
)

################################################################################

################################################################################
#                                                                              #
# Specialized benchmark configurations                                         #
#                                                                              #
# Each suite benchmarks one or more module with configurations that can vary   #
# on model or architecture characteristics. These are intended for providing   #
# continuous benchmarks of experimental features that cannot be turned on by   #
# default yet. It is primarily intended for whoever is actively investigating  #
# optimizations for a feature exemplified in a specific model or architecture. #
# Due to our current benchmark setup, there can only be one experimental       #
# configuration per model and other benchmark mode.                            #
#                                                                              #
################################################################################

# GPU, Vulkan, Adreno, full-inference
iree_benchmark_suite(
  GROUP_NAME
    "android-adreno"

  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "full-inference,experimental-flags"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Adreno"
  COMPILATION_FLAGS
    ${ANDROID_ADRENO_GPU_COMPILATION_FLAGS}
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-vulkan"
  DRIVER
    "vulkan"
)

# kernel-execution

# Note that for kernel-execution benchmarks batch_size/repeat-count need to be
# low enough that the whole dispatch completes within an OS-specific timeout.
# Otherwise you'll get error like:
# ```
# INTERNAL; VK_ERROR_DEVICE_LOST; vkQueueSubmit; while invoking native function
# hal.fence.await; while calling import;
# ```
# With current kernel performance and timeouts on Adreno Pixel 4, this means we
# have no kernel benchmark for the DeepLabV3 and MobileBert models
# TODO: Add kernel-execution config for DEEPLABV3_FP32_MODULE and
# MOBILEBERT_FP32_MODULE when they can run with at least 8 repetitions.

# GPU, Vulkan, Adreno, kernel-execution
iree_benchmark_suite(
  GROUP_NAME
    "android-adreno"

  MODULES
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "kernel-execution,experimental-flags"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Adreno"
  COMPILATION_FLAGS
    ${ANDROID_ADRENO_GPU_COMPILATION_FLAGS}
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=16"
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-vulkan"
  DRIVER
    "vulkan"
  RUNTIME_FLAGS
    "--batch_size=16"
)

################################################################################
