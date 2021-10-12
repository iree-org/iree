# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


################################################################################
#                                                                              #
# Benchmark models for Tosa                                                    #
#                                                                              #
# Each module specification should be a list that contains the following       #
# fields: MODULE_NAME, MODULE_TAGS, MLIR_SOURCE, ENTRY_FUNCTION,               #
# FUNCTION_INPUTS. See iree_mlir_benchmark_suite definition for details about  #
# these fields.                                                                #
#                                                                              #
################################################################################

set(ALBERT_FP32_MODULE
  "Albert"                        # MODULE_NAME
  "fp32"                          # MODULE_TAGS
  "https://storage.googleapis.com/iree-model-artifacts/AlbertTosa-5f984ec.tar.gz" # MLIR_SOURCE
  "main"                          # ENTRY_FUNCTION
  "1x384xi32,1x384xi32,1x384xi32" # FUNCTION_INPUTS
)

set(DEEPLABV3_FP32_MODULE
  "DeepLabV3"                     # MODULE_NAME
  "fp32"                          # MODULE_TAGS
  "https://storage.googleapis.com/iree-model-artifacts/DeepLabV3-2bcafb1.tar.gz" # MLIR_SOURCE
  "main"                          # ENTRY_FUNCTION
  "1x257x257x3xf32"               # FUNCTION_INPUTS
)

set(MOBILEBERT_FP32_MODULE
  "MobileBert"                    # MODULE_NAME
  "fp32"                          # MODULE_TAGS
  "https://storage.googleapis.com/iree-model-artifacts/MobileBertTosa-5f984ec.tar.gz" # MLIR_SOURCE
  "main"                          # ENTRY_FUNCTION
  "1x384xi32,1x384xi32,1x384xi32" # FUNCTION_INPUTS
)

set(MOBILENETV1_FP32_MODULE
  "MobileNetV1"                   # MODULE_NAME
  "fp32"                          # MODULE_TAGS
  "https://storage.googleapis.com/iree-model-artifacts/MobileNetV1Tosa-5f984ec.tar.gz" # MLIR_SOURCE
  "main"                          # ENTRY_FUNCTION
  "1x160x160x3xf32"               # FUNCTION_INPUTS
)

set(MOBILENETV2_FP32_MODULE
  "MobileNetV2"                   # MODULE_NAME
  "fp32"                          # MODULE_TAGS
  "https://storage.googleapis.com/iree-model-artifacts/MobileNetV2Tosa-5f984ec.tar.gz" # MLIR_SOURCE
  "main"                          # ENTRY_FUNCTION
  "1x299x299x3xi8"                # FUNCTION_INPUTS
)

set(MOBILENETV2_I8_MODULE
  "MobileNetV2Quant"              # MODULE_NAME
  "i8"                            # MODULE_TAGS
  "https://storage.googleapis.com/iree-model-artifacts/MobileNetV2QuantTosa-5f984ec.tar.gz" # MLIR_SOURCE
  "main"                          # ENTRY_FUNCTION
  "1x224x224x3xi8"                # FUNCTION_INPUTS
)

set(MOBILENETV3_FP32_MODULE
  "MobileNetV3"                   # MODULE_NAME
  "fp32"                          # MODULE_TAGS
  "https://storage.googleapis.com/iree-model-artifacts/MobileNetV3Tosa-5f984ec.tar.gz" # MLIR_SOURCE
  "main"                          # ENTRY_FUNCTION
  "1x360x360x3xi8"               # FUNCTION_INPUTS
)

set(MOBILESSD_FP32_MODULE
  "MobileSSD"                     # MODULE_NAME
  "fp32"                          # MODULE_TAGS
  "https://storage.googleapis.com/iree-model-artifacts/MobileSSD-2bcafb1.tar.gz" # MLIR_SOURCE
  "main"                          # ENTRY_FUNCTION
  "1x320x320x3xf32"               # FUNCTION_INPUTS
)

set(POSENET_FP32_MODULE
  "PoseNet"                       # MODULE_NAME
  "fp32"                          # MODULE_TAGS
  "https://storage.googleapis.com/iree-model-artifacts/PoseNet-2bcafb1.tar.gz" # MLIR_SOURCE
  "main"                          # ENTRY_FUNCTION
  "1x353x257x3xf32"               # FUNCTION_INPUTS
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
iree_mlir_benchmark_suite(
  MODULES
    ${ALBERT_FP32_MODULE}
    ${DEEPLABV3_FP32_MODULE}
    ${MOBILESSD_FP32_MODULE}
    ${MOBILEBERT_FP32_MODULE}
    ${MOBILENETV1_FP32_MODULE}
    ${MOBILENETV2_FP32_MODULE}
    ${MOBILENETV2_I8_MODULE}
    ${MOBILENETV3_FP32_MODULE}
    ${POSENET_FP32_MODULE}

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
    "--iree-llvm-link-embedded=true"
  DRIVER
    "dylib-sync"
)

# CPU, Dylib, 1-thread, big/little-core, full-inference
iree_mlir_benchmark_suite(
  MODULES
    ${ALBERT_FP32_MODULE}
    ${DEEPLABV3_FP32_MODULE}
    ${MOBILESSD_FP32_MODULE}
    ${MOBILEBERT_FP32_MODULE}
    ${MOBILENETV1_FP32_MODULE}
    ${MOBILENETV2_FP32_MODULE}
    ${MOBILENETV2_I8_MODULE}
    ${MOBILENETV3_FP32_MODULE}
    ${POSENET_FP32_MODULE}

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
iree_mlir_benchmark_suite(
  MODULES
    # ${ALBERT_FP32_MODULE} # Failing on vulkan
    ${DEEPLABV3_FP32_MODULE}
    ${MOBILESSD_FP32_MODULE}
    ${MOBILEBERT_FP32_MODULE}
    ${MOBILENETV1_FP32_MODULE}
    # ${MOBILENETV2_FP32_MODULE} # Failing on vulkan
    # ${MOBILENETV2_I8_MODULE} # Failing on vulkan
    # ${MOBILENETV3_FP32_MODULE} # Failing on vulkan.
    ${POSENET_FP32_MODULE}

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
iree_mlir_benchmark_suite(
  MODULES
    # ${ALBERT_FP32_MODULE} # Failing on vulkan
    ${DEEPLABV3_FP32_MODULE}
    ${MOBILESSD_FP32_MODULE}
    ${MOBILEBERT_FP32_MODULE}
    ${MOBILENETV1_FP32_MODULE}
    # ${MOBILENETV2_FP32_MODULE} # Failing on vulkan
    # ${MOBILENETV2_I8_MODULE} # Failing on vulkan
    # ${MOBILENETV3_FP32_MODULE} # Failing on vulkan
    ${POSENET_FP32_MODULE}

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
iree_mlir_benchmark_suite(
  MODULES
    # ${ALBERT_FP32_MODULE} # Failing on vulkan
    ${DEEPLABV3_FP32_MODULE}
    ${MOBILESSD_FP32_MODULE}
    ${MOBILEBERT_FP32_MODULE}
    ${MOBILENETV1_FP32_MODULE}
    # ${MOBILENETV2_FP32_MODULE} # Failing on vulkan
    # ${MOBILENETV2_I8_MODULE} # Failing on vulkan
    # ${MOBILENETV3_FP32_MODULE} # Failing on vulkan
    ${POSENET_FP32_MODULE}

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
