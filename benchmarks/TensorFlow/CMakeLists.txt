# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


################################################################################
#                                                                              #
# Benchmark models                                                             #
#                                                                              #
# Each module specification should be a list that contains the following       #
# fields: MODULE_NAME, MODULE_TAGS, MLIR_SOURCE, ENTRY_FUNCTION,               #
# FUNCTION_INPUTS. See iree_mlir_benchmark_suite definition for details about  #
# these fields.                                                                #
#                                                                              #
################################################################################

set(MOBILEBERT_FP16_MODULE
  "MobileBertSquad"               # MODULE_NAME
  "fp16"                          # MODULE_TAGS
  # This uses the same input MLIR source as fp32 to save download time.
  # It requires users to have "--iree-flow-demote-f32-to-f16".
  "https://storage.googleapis.com/iree-model-artifacts/MobileBertSquad-9c0042d19.tar.gz" # MLIR_SOURCE
  "serving_default"               # ENTRY_FUNCTION
  # The conversion done by "--iree-flow-demote-f32-to-f16" won't change the
  # original input signature.
  "1x384xi32,1x384xi32,1x384xi32" # FUNCTION_INPUTS
)

set(MOBILEBERT_FP32_MODULE
  "MobileBertSquad"               # MODULE_NAME
  "fp32"                          # MODULE_TAGS
  "https://storage.googleapis.com/iree-model-artifacts/MobileBertSquad-9c0042d19.tar.gz" # MLIR_SOURCE
  "serving_default"               # ENTRY_FUNCTION
  "1x384xi32,1x384xi32,1x384xi32" # FUNCTION_INPUTS
)

set(MOBILENET_V2_MODULE
  "MobileNetV2"     # MODULE_NAME
  "fp32,imagenet"   # MODULE_TAGS
  "https://storage.googleapis.com/iree-model-artifacts/MobileNetV2-9c0042d19.tar.gz" # MLIR_SOURCE
  "call"            # ENTRY_FUNCTION
  "1x224x224x3xf32" # FUNCTION_INPUTS
)

set(MOBILENET_V3SMALL_MODULE
  "MobileNetV3Small" # MODULE_NAME
  "fp32,imagenet"    # MODULE_TAGS
  "https://storage.googleapis.com/iree-model-artifacts/MobileNetV3Small-9c0042d19.tar.gz" # MLIR_SOURCE
  "call"             # ENTRY_FUNCTION
  "1x224x224x3xf32"  # FUNCTION_INPUTS
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
    ${MOBILENET_V2_MODULE}
    ${MOBILENET_V3SMALL_MODULE}

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
    ${MOBILENET_V2_MODULE}
    ${MOBILENET_V3SMALL_MODULE}

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
    # TODO(GH-5857): Enable this after fixing segfault.
    #"--iree-flow-dispatch-formation-enable-operand-fusion"
    "--iree-llvm-loop-unrolling=true"
  DRIVER
    "dylib-sync"
)

# CPU, Dylib, 1-thread, big/little-core, full-inference
iree_mlir_benchmark_suite(
  MODULES
    ${MOBILENET_V2_MODULE}
    ${MOBILENET_V3SMALL_MODULE}

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
    # TODO(GH-5857): Enable this after fixing segfault.
    #"--iree-flow-dispatch-formation-enable-operand-fusion"
    "--iree-llvm-loop-unrolling=true"
  DRIVER
    "dylib"
  RUNTIME_FLAGS
    "--task_topology_group_count=1"
)

# CPU, Dylib, 3-thread, big/little-core, full-inference
iree_mlir_benchmark_suite(
  MODULES
    ${MOBILENET_V2_MODULE}
    ${MOBILENET_V3SMALL_MODULE}

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
    # TODO(GH-5857): Enable this after fixing segfault.
    #"--iree-flow-dispatch-formation-enable-operand-fusion"
    "--iree-llvm-loop-unrolling=true"
  DRIVER
    "dylib"
  RUNTIME_FLAGS
    "--task_topology_group_count=3"
)

# GPU, Vulkan, Adreno, full-inference
iree_mlir_benchmark_suite(
  MODULES
    ${MOBILEBERT_FP32_MODULE}
    ${MOBILENET_V2_MODULE}
    ${MOBILENET_V3SMALL_MODULE}

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
    "--iree-flow-dispatch-formation-enable-operand-fusion"
    "--iree-enable-fusion-with-reduction-ops"
  DRIVER
    "vulkan"
)

# GPU, Vulkan, Adreno, kernel-execution
iree_mlir_benchmark_suite(
  MODULES
    ${MOBILENET_V2_MODULE}
    ${MOBILENET_V3SMALL_MODULE}

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
    "--iree-flow-dispatch-formation-enable-operand-fusion"
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
    ${MOBILEBERT_FP32_MODULE}
    ${MOBILENET_V2_MODULE}
    ${MOBILENET_V3SMALL_MODULE}

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
    "--iree-flow-dispatch-formation-enable-operand-fusion"
    "--iree-enable-fusion-with-reduction-ops"
  DRIVER
    "vulkan"
)

# GPU, Vulkan, Mali, kernel-execution
iree_mlir_benchmark_suite(
  MODULES
    ${MOBILENET_V2_MODULE}
    ${MOBILENET_V3SMALL_MODULE}

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
    "--iree-flow-dispatch-formation-enable-operand-fusion"
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
    ${MOBILEBERT_FP16_MODULE}

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
    "--iree-flow-dispatch-formation-enable-operand-fusion"
    "--iree-enable-fusion-with-reduction-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  DRIVER
    "vulkan"
  RUNTIME_FLAGS
    "--batch_size=32"
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
    ${MOBILEBERT_FP32_MODULE}

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
  DRIVER
    "dylib-sync"
)

# CPU, Dylib, 1-thread, big/little-core, full-inference
iree_mlir_benchmark_suite(
  MODULES
    ${MOBILEBERT_FP32_MODULE}

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
    ${MOBILEBERT_FP32_MODULE}

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
