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

set(ANDROID_MALI_GPU_COMPILATION_FLAGS
  "--iree-input-type=tosa"
  "--iree-vulkan-target-triple=valhall-unknown-android31"
)

# GPU, Vulkan, Mali, full-inference
iree_benchmark_suite(
  GROUP_NAME
    "android-mali"

  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"
    "${MOBILEBERT_INT8_MODULE}"
    "${EFFICIENTNET_INT8_MODULE}"
    "${PERSON_DETECT_INT8_MODULE}"

  BENCHMARK_MODES
    "full-inference,default-flags"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Mali-Valhall"
  COMPILATION_FLAGS
    ${ANDROID_MALI_GPU_COMPILATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-vulkan"
  DRIVER
    "vulkan"
)

# GPU, Vulkan, Mali, full-inference
iree_benchmark_suite(
  GROUP_NAME
    "android-mali"

  MODULES
    "${MOBILEBERT_FP16_MODULE}"

  BENCHMARK_MODES
    "full-inference,default-flags"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Mali-Valhall"
  COMPILATION_FLAGS
    ${ANDROID_MALI_GPU_COMPILATION_FLAGS}
    # This isn't a special optimization flag. It's so we can reuse the same f32
    # model file. See comments on MOBILEBERT_FP16_MODULE
    "--iree-flow-demote-f32-to-f16"
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

# GPU, Vulkan, Mali, full-inference
iree_benchmark_suite(
  GROUP_NAME
    "android-mali"

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
    "GPU-Mali-Valhall"
  COMPILATION_FLAGS
    ${ANDROID_MALI_GPU_COMPILATION_FLAGS}
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-vulkan"
  DRIVER
    "vulkan"
)

iree_benchmark_suite(
  GROUP_NAME
    "android-mali"

  MODULES
    "${MOBILEBERT_FP16_MODULE}"

  BENCHMARK_MODES
    "full-inference,experimental-flags"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Mali-Valhall"
  COMPILATION_FLAGS
    "--iree-input-type=tosa"
    "--iree-flow-demote-f32-to-f16"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
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

# GPU, Vulkan, Mali, kernel-execution
iree_benchmark_suite(
  GROUP_NAME
    "android-mali"

  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "kernel-execution,experimental-flags"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Mali-Valhall"
  COMPILATION_FLAGS
    ${ANDROID_MALI_GPU_COMPILATION_FLAGS}
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-vulkan"
  DRIVER
    "vulkan"
  RUNTIME_FLAGS
    "--batch_size=32"
)

iree_benchmark_suite(
  GROUP_NAME
    "android-mali"

  MODULES
    "${MOBILEBERT_FP16_MODULE}"

  BENCHMARK_MODES
    "kernel-execution,experimental-flags"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Mali-Valhall"
  COMPILATION_FLAGS
    "--iree-input-type=tosa"
    "--iree-flow-demote-f32-to-f16"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-vulkan"
  DRIVER
    "vulkan"
  RUNTIME_FLAGS
    "--batch_size=32"
)

################################################################################
