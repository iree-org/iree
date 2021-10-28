# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


################################################################################
#                                                                              #
# Benchmark models from TensorFlow                                             #
#                                                                              #
# Each module specification should be a list containing alternating keys and   #
# values. The fields are: NAME, TAGS, MLIR_SOURCE, ENTRY_FUNCTION, and         #
# FUNCTION_INPUTS. See the iree_mlir_benchmark_suite definition for details    #
# about these fields.                                                          #
#                                                                              #
################################################################################

set(MOBILEBERT_FP16_MODULE
  NAME
    "MobileBertSquad"
  TAGS
    "fp16"
  # This uses the same input MLIR source as fp32 to save download time.
  # It requires users to have "--iree-flow-demote-f32-to-f16".
  MLIR_SOURCE
    "https://storage.googleapis.com/iree-model-artifacts/MobileBertSquad-89edfa50d.tar.gz"
  ENTRY_FUNCTION
    "serving_default"
  # The conversion done by "--iree-flow-demote-f32-to-f16" won't change the
  # original input signature.
  FUNCTION_INPUTS
    "1x384xi32,1x384xi32,1x384xi32"
)

set(MOBILEBERT_FP32_MODULE
  NAME
    "MobileBertSquad"
  TAGS
    "fp32"
  MLIR_SOURCE
    "https://storage.googleapis.com/iree-model-artifacts/MobileBertSquad-89edfa50d.tar.gz"
  ENTRY_FUNCTION
    "serving_default"
  FUNCTION_INPUTS
    "1x384xi32,1x384xi32,1x384xi32"
)

set(MOBILENET_V2_MODULE
  NAME
    "MobileNetV2"
  TAGS
    "fp32,imagenet"
  MLIR_SOURCE
    "https://storage.googleapis.com/iree-model-artifacts/MobileNetV2-89edfa50d.tar.gz"
  ENTRY_FUNCTION
    "call"
  FUNCTION_INPUTS
    "1x224x224x3xf32"
)

set(MOBILENET_V3SMALL_MODULE
  NAME
    "MobileNetV3Small"
  TAGS
    "fp32,imagenet"
  MLIR_SOURCE
    "https://storage.googleapis.com/iree-model-artifacts/MobileNetV3Small-89edfa50d.tar.gz"
  ENTRY_FUNCTION
    "call"
  FUNCTION_INPUTS
    "1x224x224x3xf32"
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

# CPU, VMVX, 3-thread, little-core, full-inference
iree_mlir_benchmark_suite(
  MODULES
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "3-thread,little-core,full-inference"
  TARGET_BACKEND
    "vmvx"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    "--iree-input-type=mhlo"
    "--iree-flow-inline-constants-max-byte-length=2048"
  DRIVER
    "vmvx"
  RUNTIME_FLAGS
    "--task_topology_group_count=3"
)

# CPU, Dylib-Sync, big/little-core, full-inference
iree_mlir_benchmark_suite(
  MODULES
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "big-core,full-inference"
    "little-core,full-inference"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    "--iree-input-type=mhlo"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
    "--iree-flow-inline-constants-max-byte-length=2048"
    "--iree-llvm-loop-unrolling=true"
    "--iree-llvm-link-embedded=true"
  DRIVER
    "dylib-sync"
)

# CPU, Dylib, 1-thread, big/little-core, full-inference
iree_mlir_benchmark_suite(
  MODULES
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "1-thread,big-core,full-inference"
    "1-thread,little-core,full-inference"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    "--iree-input-type=mhlo"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
    "--iree-flow-inline-constants-max-byte-length=2048"
    "--iree-llvm-loop-unrolling=true"
  DRIVER
    "dylib"
  RUNTIME_FLAGS
    "--task_topology_group_count=1"
)

# CPU, Dylib, 3-thread, big/little-core, full-inference
iree_mlir_benchmark_suite(
  MODULES
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "3-thread,big-core,full-inference"
    "3-thread,little-core,full-inference"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    "--iree-input-type=mhlo"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
    "--iree-flow-inline-constants-max-byte-length=2048"
    "--iree-llvm-loop-unrolling=true"
  DRIVER
    "dylib"
  RUNTIME_FLAGS
    "--task_topology_group_count=3"
)

# GPU, Vulkan, Adreno, full-inference
iree_mlir_benchmark_suite(
  MODULES
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "full-inference"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Adreno"
  TRANSLATION_FLAGS
    "--iree-input-type=mhlo"
    "--iree-vulkan-target-triple=adreno-unknown-android11"
    "--iree-flow-inline-constants-max-byte-length=2048"
    "--iree-enable-fusion-with-reduction-ops"
  DRIVER
    "vulkan"
)

# GPU, Vulkan, Adreno, kernel-execution
iree_mlir_benchmark_suite(
  MODULES
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "kernel-execution"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Adreno"
  TRANSLATION_FLAGS
    "--iree-input-type=mhlo"
    "--iree-vulkan-target-triple=adreno-unknown-android11"
    "--iree-flow-inline-constants-max-byte-length=2048"
    "--iree-enable-fusion-with-reduction-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=16"
  DRIVER
    "vulkan"
  RUNTIME_FLAGS
    "--batch_size=16"
)

# GPU, Vulkan, Mali, full-inference
iree_mlir_benchmark_suite(
  MODULES
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "full-inference"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Mali-Valhall"
  TRANSLATION_FLAGS
    "--iree-input-type=mhlo"
    "--iree-vulkan-target-triple=valhall-unknown-android11"
    "--iree-flow-inline-constants-max-byte-length=16"
    "--iree-enable-fusion-with-reduction-ops"
  DRIVER
    "vulkan"
)

# GPU, Vulkan, Mali, kernel-execution
iree_mlir_benchmark_suite(
  MODULES
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "kernel-execution"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Mali-Valhall"
  TRANSLATION_FLAGS
    "--iree-input-type=mhlo"
    "--iree-vulkan-target-triple=valhall-unknown-android11"
    "--iree-flow-inline-constants-max-byte-length=16"
    "--iree-enable-fusion-with-reduction-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  DRIVER
    "vulkan"
  RUNTIME_FLAGS
    "--batch_size=32"
)

# GPU, Vulkan, Mali, kernel-execution
iree_mlir_benchmark_suite(
  MODULES
    "${MOBILEBERT_FP16_MODULE}"

  BENCHMARK_MODES
    "kernel-execution"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Mali-Valhall"
  TRANSLATION_FLAGS
    "--iree-input-type=mhlo"
    "--iree-flow-demote-f32-to-f16"
    "--iree-vulkan-target-triple=valhall-unknown-android11"
    "--iree-flow-inline-constants-max-byte-length=16"
    "--iree-enable-fusion-with-reduction-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  DRIVER
    "vulkan"
  RUNTIME_FLAGS
    "--batch_size=32"
)

# GPU, Vulkan, Mali, full-inference
iree_mlir_benchmark_suite(
  MODULES
    "${MOBILEBERT_FP16_MODULE}"

  BENCHMARK_MODES
    "full-inference"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Mali-Valhall"
  TRANSLATION_FLAGS
    "--iree-input-type=mhlo"
    "--iree-flow-demote-f32-to-f16"
    "--iree-vulkan-target-triple=valhall-unknown-android11"
    "--iree-flow-inline-constants-max-byte-length=16"
    "--iree-enable-fusion-with-reduction-ops"
  DRIVER
    "vulkan"
)

################################################################################
#                                                                              #
# Speical benchmark configurations                                             #
#                                                                              #
# These are configurations that can only be enabled for some specific model.   #
# However, THIS SHOULD REALLY BE TEMPORARY; we should strike for uniformity.   #
#                                                                              #
################################################################################

# CPU, Dylib-Sync, big/little-core, full-inference
iree_mlir_benchmark_suite(
  MODULES
    "${MOBILEBERT_FP32_MODULE}"

  BENCHMARK_MODES
    "big-core,full-inference"
    "little-core,full-inference"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    # TODO: Merge this rule once we can use the same flags as the common one.
    "--iree-input-type=mhlo"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
    "--iree-flow-inline-constants-max-byte-length=2048"
    "--iree-llvm-link-embedded=true"
  DRIVER
    "dylib-sync"
)

# CPU, Dylib, 1-thread, big/little-core, full-inference
iree_mlir_benchmark_suite(
  MODULES
    "${MOBILEBERT_FP32_MODULE}"

  BENCHMARK_MODES
    "1-thread,big-core,full-inference"
    "1-thread,little-core,full-inference"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    # TODO: Merge this rule once we can use the same flags as the common one.
    "--iree-input-type=mhlo"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
    "--iree-flow-inline-constants-max-byte-length=2048"
  DRIVER
    "dylib"
  RUNTIME_FLAGS
    "--task_topology_group_count=1"
)

# CPU, Dylib, 3-thread, big/little-core, full-inference
iree_mlir_benchmark_suite(
  MODULES
    "${MOBILEBERT_FP32_MODULE}"

  BENCHMARK_MODES
    "3-thread,big-core,full-inference"
    "3-thread,little-core,full-inference"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    # TODO: Merge this rule once we can use the same flags as the common one.
    "--iree-input-type=mhlo"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
    "--iree-flow-inline-constants-max-byte-length=2048"
  DRIVER
    "dylib"
  RUNTIME_FLAGS
    "--task_topology_group_count=3"
)
