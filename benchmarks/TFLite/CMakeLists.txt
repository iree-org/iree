# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


################################################################################
#                                                                              #
# Benchmark models from TFLite                                                 #
#                                                                              #
# Each module specification should be a list containing alternating keys and   #
# values. The fields are: NAME, TAGS, SOURCE, ENTRY_FUNCTION, and              #
# FUNCTION_INPUTS. See the iree_benchmark_suite definition for details         #
# about these fields. Note that these must be quoted when used as arguments.   #
#                                                                              #
################################################################################

set(DEEPLABV3_FP32_MODULE
  NAME
    "DeepLabV3"
  TAGS
    "fp32"
  SOURCE
    # No significant compression of tflite files by gzip
    "https://storage.googleapis.com/iree-model-artifacts/deeplabv3.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x257x257x3xf32"
)

set(MOBILESSD_FP32_MODULE
  NAME
    "MobileSSD"
  TAGS
    "fp32"
  SOURCE
    # No significant compression of tflite files by gzip
    "https://storage.googleapis.com/iree-model-artifacts/mobile_ssd_v2_float_coco.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x320x320x3xf32"
)

set(POSENET_FP32_MODULE
  NAME
    "PoseNet"
  TAGS
    "fp32"
  SOURCE
    # No significant compression of tflite files by gzip
    "https://storage.googleapis.com/iree-model-artifacts/posenet.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x353x257x3xf32"
)

################################################################################
#                                                                              #
# Common benchmark configurations                                              #
#                                                                              #
# Each suite benchmarks a list of modules with some specific configuration,    #
# typically involving different translation/runtime flags and targeting        #
# different IREE drivers and hardware architectures.                           #
#                                                                              #
################################################################################

# CPU, Dylib-Sync, big/little-core, full-inference
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"

  BENCHMARK_MODES
    "big-core,full-inference"
    "little-core,full-inference"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
    "--iree-flow-inline-constants-max-byte-length=2048"
    "--iree-llvm-loop-unrolling=true"
  DRIVER
    "dylib-sync"
)

# CPU, Dylib, 1-thread, big/little-core, full-inference
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"

  BENCHMARK_MODES
    "1-thread,big-core,full-inference"
    "1-thread,little-core,full-inference"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
    "--iree-flow-inline-constants-max-byte-length=2048"
    "--iree-llvm-loop-unrolling=true"
  DRIVER
    "dylib"
  RUNTIME_FLAGS
    "--task_topology_group_count=1"
)

# GPU, Vulkan, Adreno, full-inference
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"

  BENCHMARK_MODES
    "full-inference"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Adreno"
  TRANSLATION_FLAGS
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android11"
    "--iree-flow-inline-constants-max-byte-length=2048"
    "--iree-enable-fusion-with-reduction-ops"
  DRIVER
    "vulkan"
)

# GPU, Vulkan, Mali, full-inference
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"

  BENCHMARK_MODES
    "full-inference"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Mali-Valhall"
  TRANSLATION_FLAGS
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android11"
    "--iree-flow-inline-constants-max-byte-length=16"
    "--iree-enable-fusion-with-reduction-ops"
  DRIVER
    "vulkan"
)

# GPU, Vulkan, Mali, kernel-execution
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"

  BENCHMARK_MODES
    "kernel-execution"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Mali-Valhall"
  TRANSLATION_FLAGS
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android11"
    "--iree-flow-inline-constants-max-byte-length=16"
    "--iree-enable-fusion-with-reduction-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  DRIVER
    "vulkan"
  RUNTIME_FLAGS
    "--batch_size=32"
)
